"""Training entry point shared by the GRPO baseline and the CPPO runs.

``pruning_rate == 0`` selects :class:`~cppo.trainer.ProfiledGRPOTrainer`,
which is unmodified TRL GRPO; any positive rate selects
:class:`~cppo.trainer.CPPOTrainer`.  Everything else -- data, rewards, model,
optimiser, seeds and instrumentation -- is identical between the two, so the
wall-clock and memory numbers they produce are directly comparable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from accelerate import PartialState
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig

from .config import RunSettings, load_settings
from .data import filter_by_prompt_length, load_training_dataset, system_prompt_for
from .geometry import BatchGeometry, resolve_batch_geometry
from .rewards import build_reward_functions, reward_function_names
from .trainer import CPPOTrainer, ProfiledGRPOTrainer

LOGGER = logging.getLogger(__name__)

__all__ = ["build_grpo_config", "main", "run_training"]

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    """Map a dtype name to a torch dtype.

    Args:
        name: One of ``float32``, ``float16``, ``bfloat16`` or ``auto``.

    Returns:
        The torch dtype; ``auto`` resolves to bf16 on CUDA and fp32 elsewhere.

    Raises:
        ValueError: If ``name`` is not recognised.
    """
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of {sorted(_DTYPES)} or 'auto'"
        ) from exc


def build_grpo_config(settings: RunSettings, geometry: BatchGeometry) -> GRPOConfig:
    """Translate :class:`RunSettings` plus a batch geometry into a ``GRPOConfig``.

    ``steps_per_generation`` and ``generation_batch_size`` are deliberately
    left unset so that TRL derives them as
    ``per_device_train_batch_size * num_processes * gradient_accumulation_steps``
    -- the identity the CPPO geometry relies on.

    Args:
        settings: The resolved run settings.
        geometry: The batch geometry produced by
            :func:`~cppo.geometry.resolve_batch_geometry`.

    Returns:
        A populated ``GRPOConfig``.
    """
    model_init_kwargs: dict[str, Any] = {"dtype": _resolve_dtype(settings.torch_dtype)}
    if settings.attn_implementation:
        model_init_kwargs["attn_implementation"] = settings.attn_implementation

    # `top_k` is only meaningful when set; TRL's own default must win otherwise.
    optional: dict[str, Any] = {}
    if settings.top_k is not None:
        optional["top_k"] = settings.top_k

    return GRPOConfig(
        output_dir=settings.output_dir,
        run_name=settings.run_name,
        seed=settings.seed,
        # batching
        per_device_train_batch_size=geometry.per_device_train_batch_size,
        gradient_accumulation_steps=geometry.gradient_accumulation_steps,
        num_generations=settings.num_generations,
        # rollout
        max_completion_length=settings.max_completion_length,
        chat_template_kwargs={"enable_thinking": settings.enable_thinking},
        temperature=settings.temperature,
        top_p=settings.top_p,
        **optional,
        use_vllm=settings.use_vllm,
        vllm_mode=settings.vllm_mode,
        vllm_gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=settings.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=settings.vllm_enable_sleep_mode,
        vllm_importance_sampling_correction=settings.vllm_importance_sampling_correction,
        # objective
        beta=settings.beta,
        epsilon=settings.epsilon,
        epsilon_high=settings.epsilon_high,
        loss_type=settings.loss_type,
        scale_rewards=settings.scale_rewards,
        mask_truncated_completions=settings.mask_truncated_completions,
        # optimisation
        learning_rate=settings.learning_rate,
        lr_scheduler_type=settings.lr_scheduler_type,
        warmup_steps=settings.warmup_steps,
        weight_decay=settings.weight_decay,
        max_grad_norm=settings.max_grad_norm,
        adam_beta1=settings.adam_beta1,
        adam_beta2=settings.adam_beta2,
        num_train_epochs=settings.num_train_epochs,
        max_steps=settings.max_steps,
        gradient_checkpointing=settings.gradient_checkpointing,
        use_cpu=settings.use_cpu,
        model_init_kwargs=model_init_kwargs,
        # reporting
        logging_steps=settings.logging_steps,
        save_strategy="steps" if settings.save_steps > 0 else "no",
        save_steps=settings.save_steps or 500,
        report_to=list(settings.report_to),
        log_completions=settings.log_completions,
        num_completions_to_print=settings.num_completions_to_print,
        disable_tqdm=False,
    )


def _build_peft_config(settings: RunSettings) -> Any:
    """Return a LoRA config when PEFT is requested, otherwise ``None``.

    Args:
        settings: The resolved run settings.

    Returns:
        A ``peft.LoraConfig`` or ``None``.
    """
    if not settings.use_peft:
        return None
    from peft import LoraConfig  # pylint: disable=import-outside-toplevel

    return LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def run_training(settings: RunSettings) -> dict[str, Any]:
    """Execute one training run and return its measurement report.

    Args:
        settings: The resolved run settings.

    Returns:
        The profiling report, extended with the run configuration and the
        resolved batch geometry.
    """
    set_seed(settings.seed)
    num_processes = PartialState().num_processes
    geometry = resolve_batch_geometry(
        base_per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_generations=settings.num_generations,
        pruning_rate=settings.pruning_rate,
        dynamic_allocation=settings.dynamic_allocation,
        num_processes=num_processes,
    )
    LOGGER.info("Batch geometry: %s", json.dumps(geometry.as_dict(), sort_keys=True))

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name_or_path)
    dataset = load_training_dataset(
        settings.dataset_name,
        settings.dataset_config,
        settings.dataset_split,
        prompt_style=settings.prompt_style,
        max_samples=settings.max_samples,
        shuffle_seed=settings.seed,
    )
    if settings.max_prompt_length > 0:
        dataset = filter_by_prompt_length(dataset, tokenizer, settings.max_prompt_length)

    config = build_grpo_config(settings, geometry)
    reward_funcs = build_reward_functions(settings.prompt_style)
    peft_config = _build_peft_config(settings)

    common: dict[str, Any] = {
        "model": settings.model_name_or_path,
        "reward_funcs": reward_funcs,
        "args": config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }
    if settings.pruning_rate > 0.0:
        trainer: Any = CPPOTrainer(
            pruning_rate=settings.pruning_rate,
            drop_zero_advantage=settings.drop_zero_advantage,
            **common,
        )
    else:
        trainer = ProfiledGRPOTrainer(**common)

    trainer.train()

    if settings.save_final_model:
        trainer.save_model(settings.output_dir)
        tokenizer.save_pretrained(settings.output_dir)

    report = trainer.profiling_report(
        {
            "run_name": settings.run_name,
            "algorithm": "CPPO" if settings.pruning_rate > 0.0 else "GRPO",
            "settings": asdict(settings),
            "geometry": geometry.as_dict(),
            "reward_functions": reward_function_names(settings.prompt_style),
            "system_prompt": system_prompt_for(settings.prompt_style),
            "num_train_problems": len(dataset),
            "global_step": int(trainer.state.global_step),
            "log_history": trainer.state.log_history,
        }
    )
    report = dict(report)
    if trainer.accelerator.is_main_process:
        destination = settings.resolved_profile_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        LOGGER.info("Wrote profiling report to %s", destination)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for ``cppo-train``.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings(argv)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    report = run_training(settings)
    stages = report["stages"]
    print(
        f"[{settings.run_name}] wall={report['wall_clock_seconds']:.1f}s "
        f"rollout={stages['rollout_seconds']:.1f}s update={stages['update_seconds']:.1f}s "
        f"peak={report['peak_memory_allocated_gib']:.2f}GiB "
        f"retention={stages['completion_retention']:.3f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
