"""Batch-geometry arithmetic that maps CPPO onto TRL's batching model.

TRL's :class:`~trl.GRPOTrainer` derives its rollout batch from three knobs::

    generation_batch_size = per_device_train_batch_size
                          * num_processes
                          * steps_per_generation

Each *generation round* samples ``generation_batch_size / num_generations``
distinct questions, repeats every question ``num_generations`` times, generates
one completion per repeat, and then splits the result into
``steps_per_generation`` optimiser micro-batches.

CPPO keeps only ``k`` of the ``G`` completions per question before the policy
forward pass.  Applied naively this shrinks every micro-batch by ``k / G`` and
leaves the accelerator idle.  The paper's *dynamic completion allocation*
strategy refills those slots with completions from additional questions.

The key observation implemented here is that the strategy needs no surgery on
TRL's batching code: raising ``per_device_train_batch_size`` by the allocation
multiplier ``m = G // k`` makes TRL sample ``m`` times more questions per
round, and pruning then shrinks each micro-batch back to exactly the baseline
shape.  Rollout cost per question is unchanged, the update stage sees the same
tensor shapes as the GRPO baseline, and one epoch needs ``m`` times fewer
optimiser steps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .pruning import compute_retained_completions, dynamic_allocation_multiplier

__all__ = ["BatchGeometry", "resolve_batch_geometry"]


@dataclass(frozen=True)
class BatchGeometry:
    """Resolved batch shape for one GRPO or CPPO run.

    Attributes:
        num_generations: Group size ``G``.
        num_retained: Completions kept per group ``k``.
        pruning_rate: The configured pruning rate ``P``.
        allocation_multiplier: ``m``; ``1`` when allocation is disabled.
        per_device_train_batch_size: Value handed to TRL.  Already includes
            the allocation multiplier.
        gradient_accumulation_steps: Value handed to TRL; equals
            ``steps_per_generation``.
        num_processes: Data-parallel world size assumed by the arithmetic.
        questions_per_round: Distinct questions sampled per generation round.
        completions_generated_per_round: ``questions_per_round * G``.
        completions_updated_per_round: ``questions_per_round * k``.
        update_microbatch_per_device: Completions per device in one optimiser
            micro-batch *after* pruning.  This is the number that determines
            update-stage memory, and it matches the baseline whenever dynamic
            allocation is enabled.
    """

    num_generations: int
    num_retained: int
    pruning_rate: float
    allocation_multiplier: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_processes: int
    questions_per_round: int
    completions_generated_per_round: int
    completions_updated_per_round: int
    update_microbatch_per_device: float

    @property
    def generation_batch_size(self) -> int:
        """Completions produced per generation round across all devices."""
        return (
            self.per_device_train_batch_size
            * self.num_processes
            * self.gradient_accumulation_steps
        )

    @property
    def update_compute_ratio(self) -> float:
        """Update-stage work per question relative to GRPO (lower is cheaper)."""
        return self.num_retained / self.num_generations

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view including the derived properties."""
        payload = asdict(self)
        payload["generation_batch_size"] = self.generation_batch_size
        payload["update_compute_ratio"] = self.update_compute_ratio
        return payload


def resolve_batch_geometry(
    *,
    base_per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    pruning_rate: float = 0.0,
    dynamic_allocation: bool = True,
    num_processes: int = 1,
) -> BatchGeometry:
    """Compute the TRL batch configuration for a GRPO or CPPO run.

    Args:
        base_per_device_train_batch_size: Completions per device per optimiser
            micro-batch that the GRPO baseline uses.  CPPO is configured so
            that its *post-pruning* micro-batch matches this number.
        gradient_accumulation_steps: Optimiser micro-batches per step; TRL
            reuses this value as ``steps_per_generation``.
        num_generations: Group size ``G``.
        pruning_rate: Pruning rate ``P``; ``0.0`` reproduces plain GRPO.
        dynamic_allocation: Whether to refill pruned slots with completions
            from additional questions.
        num_processes: Data-parallel world size.

    Returns:
        The resolved :class:`BatchGeometry`.

    Raises:
        ValueError: If any argument is non-positive, or if the resulting
            generation batch is not divisible by ``num_generations`` -- a
            constraint TRL enforces as well.
    """
    if base_per_device_train_batch_size <= 0:
        raise ValueError("base_per_device_train_batch_size must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")

    retained = compute_retained_completions(num_generations, pruning_rate)
    multiplier = (
        dynamic_allocation_multiplier(num_generations, retained) if dynamic_allocation else 1
    )
    per_device = base_per_device_train_batch_size * multiplier
    generation_batch_size = per_device * num_processes * gradient_accumulation_steps

    if generation_batch_size % num_generations != 0:
        raise ValueError(
            f"generation batch of {generation_batch_size} completions is not divisible by "
            f"num_generations={num_generations}; adjust per-device batch size, "
            f"gradient accumulation or the group size"
        )

    questions = generation_batch_size // num_generations
    return BatchGeometry(
        num_generations=num_generations,
        num_retained=retained,
        pruning_rate=pruning_rate,
        allocation_multiplier=multiplier,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_processes=num_processes,
        questions_per_round=questions,
        completions_generated_per_round=questions * num_generations,
        completions_updated_per_round=questions * retained,
        update_microbatch_per_device=per_device * retained / num_generations,
    )
