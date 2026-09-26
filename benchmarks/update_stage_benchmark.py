"""Micro-benchmark of the GRPO/CPPO update stage.

CPPO changes exactly one thing about the optimisation step: how many
completions reach the policy forward and backward pass.  This benchmark
isolates that effect.  It builds a synthetic rollout of ``B`` questions with
``G`` completions each, applies the CPPO retention mask for a range of pruning
rates, and times the resulting forward/backward/optimiser steps.

Unlike a full training run it needs no dataset, no generation and no reward
model, so it produces reproducible update-stage numbers in minutes on a single
device -- including CPU, where it is slow but still meaningful as a relative
measurement.

Example:
    python benchmarks/update_stage_benchmark.py \\
        --model Qwen/Qwen3-0.6B --num-generations 8 --questions 4 \\
        --completion-length 512 --output results/update_stage.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from cppo.profiling import collect_environment, peak_memory_bytes, reset_peak_memory
from cppo.pruning import compute_retained_completions, select_by_absolute_advantage

LOGGER = logging.getLogger(__name__)
_BYTES_PER_GIB = 1024.0**3

__all__ = ["BenchmarkResult", "benchmark_pruning_rate", "main", "render_markdown"]


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing and memory for the update stage at one pruning rate."""

    pruning_rate: float
    num_retained: int
    completions_per_step: int
    tokens_per_step: int
    mean_step_seconds: float
    stdev_step_seconds: float
    peak_memory_gib: float
    peak_reserved_gib: float
    speedup: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the result."""
        return asdict(self)


def _resolve_device(requested: str) -> torch.device:
    """Resolve the ``--device`` flag to a concrete device.

    Args:
        requested: ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.

    Returns:
        The device to benchmark on.  ``"auto"`` prefers CUDA, then Apple MPS,
        then CPU.  MPS is usable here because the benchmark never calls TRL's
        generation path, which is the part that is not device-consistent on
        Apple silicon.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    """Block until queued work on ``device`` has finished.

    Args:
        device: The device to synchronise.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _synthetic_advantages(num_questions: int, num_generations: int, seed: int) -> torch.Tensor:
    """Draw a plausible group-relative advantage vector.

    Rewards are sampled from a two-component mixture -- mostly wrong answers
    with a minority of correct ones -- and then standardised per group exactly
    as GRPO does, which reproduces the heavy-tailed ``|A|`` distribution that
    makes pruning worthwhile.

    Args:
        num_questions: Number of groups.
        num_generations: Completions per group.
        seed: Seed for the generator.

    Returns:
        A 1-D tensor of length ``num_questions * num_generations``.
    """
    generator = torch.Generator().manual_seed(seed)
    correct = (torch.rand(num_questions, num_generations, generator=generator) < 0.35).float()
    formatted = (torch.rand(num_questions, num_generations, generator=generator) < 0.8).float()
    rewards = 2.0 * correct + formatted
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    return (centred / (rewards.std(dim=1, keepdim=True) + 1e-4)).reshape(-1)


def _policy_loss(
    model: nn.Module, input_ids: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    """Compute a GRPO-shaped surrogate loss for the given batch.

    The exact objective is irrelevant to the measurement; what matters is that
    it has the same computational profile as TRL's ``_compute_loss``: one
    forward pass producing per-token log-probabilities, weighted by a
    per-sequence advantage, followed by a backward pass.

    Args:
        model: The policy model.
        input_ids: Token ids of shape ``(batch, length)``.
        advantages: Per-sequence advantages of shape ``(batch,)``.

    Returns:
        A scalar loss.
    """
    logits = model(input_ids=input_ids).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    # Selective log-softmax, as TRL computes it: gather the target logit and
    # subtract the log-partition function. Materialising a full
    # `(batch, length, vocab)` float32 log-probability tensor instead would
    # dominate the measurement -- and exhaust memory on a 150k-token vocab.
    selected = torch.gather(logits, dim=2, index=targets.unsqueeze(2)).squeeze(2)
    per_token = selected - torch.logsumexp(logits, dim=-1)
    # `exp(logp - logp.detach())` is identically 1 in value and reproduces the
    # importance ratio's gradient, matching the real objective's graph.
    ratio = torch.exp(per_token - per_token.detach())
    return -(ratio * advantages.unsqueeze(1)).mean()


def benchmark_pruning_rate(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    pruning_rate: float,
    num_questions: int,
    num_generations: int,
    completion_length: int,
    vocab_size: int,
    steps: int,
    warmup: int,
    device: torch.device,
    seed: int,
) -> BenchmarkResult:
    """Time the update stage for one pruning rate.

    Args:
        model: The policy model, already on ``device``.
        optimizer: Optimiser stepped after every backward pass.
        pruning_rate: CPPO pruning rate ``P``.
        num_questions: Questions per update step.
        num_generations: Completions sampled per question.
        completion_length: Sequence length of each completion.
        vocab_size: Vocabulary size used to draw synthetic tokens.
        steps: Measured steps.
        warmup: Unmeasured steps run first.
        device: Device to run on.
        seed: Seed for the synthetic batch.

    Returns:
        The populated :class:`BenchmarkResult`.
    """
    retained = compute_retained_completions(num_generations, pruning_rate)
    advantages = _synthetic_advantages(num_questions, num_generations, seed)
    mask = select_by_absolute_advantage(advantages, num_generations, retained)

    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (num_questions * num_generations, completion_length),
        generator=generator,
    )
    batch = input_ids[mask].to(device)
    batch_advantages = advantages[mask].to(device)

    reset_peak_memory()
    durations: list[float] = []
    peak_bytes = 0.0
    peak_reserved = 0.0
    for index in range(warmup + steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = _policy_loss(model, batch, batch_advantages)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        _synchronize(device)
        # CUDA reports a true peak; MPS only exposes an instantaneous figure,
        # so sample it every step and keep the largest reading.
        memory = peak_memory_bytes()
        peak_bytes = max(peak_bytes, memory["allocated"])
        peak_reserved = max(peak_reserved, memory["reserved"])
        if index >= warmup:
            durations.append(time.perf_counter() - start)

    mean = statistics.fmean(durations)
    return BenchmarkResult(
        pruning_rate=pruning_rate,
        num_retained=retained,
        completions_per_step=int(batch.shape[0]),
        tokens_per_step=int(batch.numel()),
        mean_step_seconds=mean,
        stdev_step_seconds=statistics.stdev(durations) if len(durations) > 1 else 0.0,
        peak_memory_gib=peak_bytes / _BYTES_PER_GIB,
        peak_reserved_gib=peak_reserved / _BYTES_PER_GIB,
        speedup=0.0,  # filled in by the caller once the baseline is known
    )


def render_markdown(results: Sequence[BenchmarkResult], num_generations: int) -> str:
    """Format benchmark results as a Markdown table.

    Args:
        results: Benchmark results, baseline first.
        num_generations: Group size ``G``, shown in the header.

    Returns:
        A Markdown table ready to paste into the report.
    """
    lines = [
        f"| Pruning rate P | k (of G={num_generations}) | Completions/step | "
        "Step time (s) | Peak mem (GiB) | Speedup |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.pruning_rate:.4f} | {result.num_retained} | "
            f"{result.completions_per_step} | "
            f"{result.mean_step_seconds:.4f} ± {result.stdev_step_seconds:.4f} | "
            f"{result.peak_memory_gib:.3f} | {result.speedup:.2f}× |"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the benchmark."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--completion-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--pruning-rates",
        type=float,
        nargs="+",
        default=None,
        help="defaults to the rates that map onto integer k for the group size",
    )
    parser.add_argument("--output", default="results/update_stage_benchmark.json")
    return parser


def _default_rates(num_generations: int) -> list[float]:
    """Return pruning rates that map onto each achievable ``k``.

    Args:
        num_generations: Group size ``G``.

    Returns:
        One rate per ``k`` from ``G`` down to ``1``, baseline first.
    """
    return [1.0 - k / num_generations for k in range(num_generations, 0, -1)]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its results.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    device = _resolve_device(args.device)

    dtype = torch.float32
    if args.dtype == "bfloat16" or (args.dtype == "auto" and device.type in {"cuda", "mps"}):
        dtype = torch.bfloat16

    LOGGER.info("Loading %s on %s (%s)", args.model, device, dtype)
    loaded = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if args.gradient_checkpointing:
        loaded.gradient_checkpointing_enable()
    model: nn.Module = loaded
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    vocab_size = int(AutoConfig.from_pretrained(args.model).vocab_size)

    rates = args.pruning_rates or _default_rates(args.num_generations)
    results: list[BenchmarkResult] = []
    for rate in rates:
        LOGGER.info("Benchmarking pruning rate %.4f", rate)
        result = benchmark_pruning_rate(
            model,
            optimizer,
            pruning_rate=rate,
            num_questions=args.questions,
            num_generations=args.num_generations,
            completion_length=args.completion_length,
            vocab_size=vocab_size,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
            seed=args.seed,
        )
        results.append(result)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    baseline = results[0].mean_step_seconds
    results = [
        BenchmarkResult(**{**result.as_dict(), "speedup": baseline / result.mean_step_seconds})
        for result in results
    ]

    payload = {
        "model": args.model,
        "num_generations": args.num_generations,
        "questions": args.questions,
        "completion_length": args.completion_length,
        "steps": args.steps,
        "warmup": args.warmup,
        "device": str(device),
        "dtype": str(dtype),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "environment": collect_environment(),
        "results": [result.as_dict() for result in results],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(render_markdown(results, args.num_generations))
    print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
