"""Evaluation pipeline built on EleutherAI's `lm-evaluation-harness`.

The three benchmarks requested for this study are all shipped with the
harness:

=================  ============  =========================================
Task id            Few-shot      Notes
=================  ============  =========================================
``gsm8k``          5             ``strict-match`` and ``flexible-extract``
``minerva_math``   4             Group over the seven MATH subjects
``aime24``         0             30 problems, greedy decoding
=================  ============  =========================================

Because the policy is trained with a chat template and a ``\\boxed{}`` answer
convention, evaluation defaults to ``--apply_chat_template`` with the same
system prompt used during training.  Keeping the two prompt formats aligned is
what makes the before/after comparison meaningful.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Final, Sequence

LOGGER = logging.getLogger(__name__)

DEFAULT_TASKS: Final[tuple[str, ...]] = ("gsm8k", "minerva_math", "aime24")

#: Metric keys reported as the headline number for each task, most preferred
#: first.  `lm_eval` names metrics ``<metric>,<filter>``.
PRIMARY_METRICS: Final[dict[str, tuple[str, ...]]] = {
    "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match"),
    "minerva_math": ("math_verify,none", "exact_match,none"),
    "aime24": ("exact_match,none",),
}

__all__ = [
    "DEFAULT_TASKS",
    "build_model_args",
    "main",
    "run_evaluation",
    "summarize_results",
]


def build_model_args(
    model_path: str,
    *,
    backend: str = "hf",
    dtype: str = "bfloat16",
    device: str | None = None,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
) -> str:
    """Assemble the ``model_args`` string for `lm_eval`.

    Args:
        model_path: Local directory or Hub id of the checkpoint.
        backend: ``"hf"`` for the Transformers backend or ``"vllm"`` for the
            much faster vLLM backend.
        dtype: Torch dtype used to load the weights.
        device: Device to pin the Transformers backend to, e.g. ``"cpu"`` or
            ``"cuda:0"``. ``None`` lets the harness choose. Ignored by vLLM,
            which manages placement itself.
        max_model_len: Maximum sequence length (vLLM only).
        gpu_memory_utilization: Fraction of VRAM vLLM may claim.
        tensor_parallel_size: Number of GPUs for tensor parallelism (vLLM).

    Returns:
        A comma-separated ``key=value`` string.

    Raises:
        ValueError: If ``backend`` is neither ``"hf"`` nor ``"vllm"``.
    """
    if backend == "hf":
        parts = [f"pretrained={model_path}", f"dtype={dtype}", "trust_remote_code=True"]
        if device is not None:
            parts.append(f"device={device}")
    elif backend == "vllm":
        parts = [
            f"pretrained={model_path}",
            f"dtype={dtype}",
            f"max_model_len={max_model_len}",
            f"gpu_memory_utilization={gpu_memory_utilization}",
            f"tensor_parallel_size={tensor_parallel_size}",
            "trust_remote_code=True",
        ]
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'hf' or 'vllm'")
    return ",".join(parts)


def summarize_results(results: dict[str, Any]) -> dict[str, float]:
    """Reduce a raw `lm_eval` result payload to one headline number per task.

    Args:
        results: The dict returned by ``lm_eval.simple_evaluate``.

    Returns:
        A mapping from task id to accuracy in percent, sorted by task id.
    """
    summary: dict[str, float] = {}
    for task, metrics in sorted(results.get("results", {}).items()):
        preferred = PRIMARY_METRICS.get(task, ())
        chosen: float | None = None
        for key in preferred:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                chosen = float(value)
                break
        if chosen is None:
            for key, value in sorted(metrics.items()):
                if key.startswith(("exact_match", "acc", "math_verify")) and isinstance(
                    value, (int, float)
                ):
                    chosen = float(value)
                    break
        if chosen is not None:
            summary[task] = 100.0 * chosen
    return summary


def run_evaluation(
    model_path: str,
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    backend: str = "hf",
    batch_size: str = "auto",
    dtype: str = "bfloat16",
    device: str | None = None,
    apply_chat_template: bool = True,
    system_instruction: str | None = None,
    max_gen_toks: int | None = 2048,
    limit: int | None = None,
    num_fewshot: int | None = None,
    seed: int = 1234,
    output_path: str | Path | None = None,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
) -> dict[str, Any]:
    """Evaluate one checkpoint and return the raw harness payload.

    Args:
        model_path: Local directory or Hub id of the checkpoint.
        tasks: Benchmark ids understood by `lm_eval`.
        backend: ``"hf"`` or ``"vllm"``.
        batch_size: `lm_eval` batch size; ``"auto"`` lets the harness probe.
        dtype: Torch dtype used to load the weights.
        device: Device for the Transformers backend; ``None`` auto-selects.
        apply_chat_template: Wrap each prompt in the tokeniser's chat template.
        system_instruction: System prompt prepended to every request.  Pass the
            training system prompt so that train and test prompts agree.
        max_gen_toks: Cap on generated tokens; ``None`` keeps each task's own
            default (32768 for ``aime24``, which is rarely affordable).
        limit: Optional cap on documents per task, for quick checks.
        num_fewshot: Override the per-task few-shot count.
        seed: Seed forwarded to python, numpy, torch and the fewshot sampler.
        output_path: Where to write the JSON payload; skipped when ``None``.
        max_model_len: Maximum sequence length (vLLM only).
        gpu_memory_utilization: Fraction of VRAM vLLM may claim.
        tensor_parallel_size: Number of GPUs for tensor parallelism (vLLM).

    Returns:
        The full `lm_eval` results dict, augmented with a ``cppo_summary`` key
        holding the per-task headline accuracies in percent.
    """
    # Imported lazily so that `pip install -e .` without the `eval` extra can
    # still import the training package.
    import lm_eval  # pylint: disable=import-outside-toplevel

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_args = build_model_args(
        model_path,
        backend=backend,
        dtype=dtype,
        device=device,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    gen_kwargs = f"max_gen_toks={max_gen_toks}" if max_gen_toks is not None else None

    LOGGER.info("Evaluating %s on %s (backend=%s)", model_path, ", ".join(tasks), backend)
    results = lm_eval.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=list(tasks),
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        apply_chat_template=apply_chat_template,
        system_instruction=system_instruction,
        gen_kwargs=gen_kwargs,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        bootstrap_iters=0,
    )
    if results is None:  # pragma: no cover - only on non-zero ranks
        return {}

    results = dict(results)
    results["cppo_summary"] = summarize_results(results)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        LOGGER.info("Wrote results to %s", destination)
    return results


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a GRPO/CPPO checkpoint with lm-evaluation-harness.",
    )
    parser.add_argument("--model-path", required=True, help="checkpoint directory or Hub id")
    parser.add_argument(
        "--tasks", nargs="+", default=list(DEFAULT_TASKS), help="lm_eval task ids"
    )
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--device", default=None, help="e.g. cpu or cuda:0; hf backend only"
    )
    parser.add_argument("--max-gen-toks", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="documents per task")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--no-chat-template",
        dest="apply_chat_template",
        action="store_false",
        help="evaluate with raw completion prompts instead of the chat template",
    )
    parser.add_argument(
        "--system-instruction",
        default=None,
        help="system prompt; pass the training prompt to keep formats aligned",
    )
    parser.add_argument(
        "--prompt-style",
        default=None,
        choices=["boxed", "think_answer"],
        help="shorthand that fills --system-instruction from the training prompts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for ``cppo-eval``.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    system_instruction = args.system_instruction
    if system_instruction is None and args.prompt_style is not None:
        from .data import system_prompt_for  # pylint: disable=import-outside-toplevel

        system_instruction = system_prompt_for(args.prompt_style)

    results = run_evaluation(
        args.model_path,
        tasks=args.tasks,
        backend=args.backend,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=args.device,
        apply_chat_template=args.apply_chat_template,
        system_instruction=system_instruction,
        max_gen_toks=args.max_gen_toks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        seed=args.seed,
        output_path=args.output_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    for task, accuracy in results.get("cppo_summary", {}).items():
        print(f"{task:>16}: {accuracy:6.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
