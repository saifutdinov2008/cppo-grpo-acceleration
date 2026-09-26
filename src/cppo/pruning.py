"""Core completion-pruning primitives for CPPO.

This module is deliberately free of any dependency on TRL or Transformers so
that the selection logic -- the mathematical heart of CPPO -- can be unit
tested in isolation and reused by the benchmark harness.

The reference algorithm is *Completion Pruning Policy Optimization* (CPPO,
Lin et al., arXiv:2503.22342).  Given a group of ``G`` completions sampled for
one question, GRPO assigns every completion ``i`` the group-relative advantage

.. math::  A_i = (r_i - \\mathrm{mean}(r)) / \\mathrm{std}(r)

and back-propagates through all ``G`` completions.  CPPO observes that the
gradient contribution of a completion is proportional to ``|A_i|`` and keeps
only the ``k = floor(G * (1 - P))`` completions with the largest absolute
advantage, where ``P`` is the pruning rate.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

__all__ = [
    "PruningStats",
    "compute_retained_completions",
    "dynamic_allocation_multiplier",
    "select_by_absolute_advantage",
    "summarize_pruning",
]


@dataclass(frozen=True)
class PruningStats:
    """Bookkeeping produced by a single pruning decision.

    Attributes:
        num_groups: Number of questions (groups) in the batch.
        num_generations: Completions sampled per question (``G``).
        num_retained_per_group: Completions kept per question (``k``).
        num_total: Total completions before pruning.
        num_kept: Total completions after pruning.
        num_degenerate_groups: Groups whose advantages are all (numerically)
            zero.  Such groups carry no policy-gradient signal at all.
        mean_abs_advantage_kept: Mean ``|A|`` over kept completions.
        mean_abs_advantage_dropped: Mean ``|A|`` over dropped completions.
        retained_signal_fraction: Share of the batch's total ``|A|`` mass that
            survives pruning.  A value close to 1.0 means the discarded
            completions carried almost no learning signal.
    """

    num_groups: int
    num_generations: int
    num_retained_per_group: int
    num_total: int
    num_kept: int
    num_degenerate_groups: int
    mean_abs_advantage_kept: float
    mean_abs_advantage_dropped: float
    retained_signal_fraction: float

    def as_dict(self, prefix: str = "") -> dict[str, Any]:
        """Return the stats as a flat dict, optionally prefixing every key."""
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


def compute_retained_completions(num_generations: int, pruning_rate: float) -> int:
    """Return ``k = floor(G * (1 - P))`` clamped to ``[1, G]``.

    Args:
        num_generations: Group size ``G`` used during rollout.
        pruning_rate: Pruning rate ``P`` in ``[0, 1)``.

    Returns:
        The number of completions retained per group.

    Raises:
        ValueError: If ``num_generations`` is not positive or ``pruning_rate``
            lies outside ``[0, 1)``.
    """
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    if not 0.0 <= pruning_rate < 1.0:
        raise ValueError(f"pruning_rate must lie in [0, 1), got {pruning_rate}")
    retained = int(num_generations * (1.0 - pruning_rate))
    return max(1, min(num_generations, retained))


def dynamic_allocation_multiplier(num_generations: int, num_retained: int) -> int:
    """Return how many times more questions a device can host after pruning.

    Pruning frees ``G - k`` completion slots per question in the update stage.
    The CPPO *dynamic completion allocation* strategy refills those slots with
    completions sampled for additional questions, so that the post-pruning
    batch is again roughly ``G`` completions wide per original question slot.

    Args:
        num_generations: Group size ``G``.
        num_retained: Completions kept per group ``k``.

    Returns:
        The integer multiplier ``m = max(1, G // k)``.

    Raises:
        ValueError: If ``num_retained`` is not in ``[1, num_generations]``.
    """
    if not 1 <= num_retained <= num_generations:
        raise ValueError(
            f"num_retained must lie in [1, {num_generations}], got {num_retained}"
        )
    return max(1, num_generations // num_retained)


def select_by_absolute_advantage(
    advantages: torch.Tensor,
    num_generations: int,
    num_retained: int,
    *,
    drop_zero_advantage: bool = False,
    zero_tolerance: float = 1e-8,
) -> torch.Tensor:
    """Build the CPPO retention mask for a flat batch of completions.

    The batch is assumed to be laid out group-major, i.e. the completions of a
    single question occupy ``num_generations`` consecutive positions.  This is
    the layout produced by TRL's ``RepeatSampler``.

    Ties in ``|A|`` are broken deterministically in favour of the lower flat
    index, so two processes observing the same advantages always select the
    same completions.

    Args:
        advantages: 1-D tensor of group-relative advantages, length
            ``num_groups * num_generations``.
        num_generations: Group size ``G``.
        num_retained: Completions to keep per group ``k``.
        drop_zero_advantage: Additionally drop completions whose advantage is
            numerically zero.  Such completions contribute exactly zero
            gradient through the policy term, so dropping them is lossless
            whenever the KL penalty is disabled (``beta == 0``).
        zero_tolerance: Absolute threshold below which an advantage counts as
            zero.

    Returns:
        A boolean tensor of the same shape as ``advantages``; ``True`` marks a
        completion that takes part in the gradient update.

    Raises:
        ValueError: If ``advantages`` is not 1-D or its length is not a
            multiple of ``num_generations``.
    """
    if advantages.dim() != 1:
        raise ValueError(f"advantages must be 1-D, got shape {tuple(advantages.shape)}")
    total = advantages.numel()
    if num_generations <= 0 or total % num_generations != 0:
        raise ValueError(
            f"batch of {total} completions is not divisible into groups of "
            f"{num_generations}"
        )
    if not 1 <= num_retained <= num_generations:
        raise ValueError(
            f"num_retained must lie in [1, {num_generations}], got {num_retained}"
        )

    grouped = advantages.detach().reshape(-1, num_generations)
    mask = torch.zeros_like(grouped, dtype=torch.bool)

    if num_retained == num_generations:
        mask.fill_(True)
    else:
        # ``stable=True`` makes tie-breaking reproducible across devices.
        order = torch.argsort(grouped.abs().float(), dim=1, descending=True, stable=True)
        mask.scatter_(1, order[:, :num_retained], True)

    if drop_zero_advantage:
        mask &= grouped.abs() > zero_tolerance

    return mask.reshape(-1)


def summarize_pruning(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    num_generations: int,
    *,
    zero_tolerance: float = 1e-8,
) -> PruningStats:
    """Compute diagnostics describing how much learning signal pruning kept.

    Args:
        advantages: 1-D tensor of group-relative advantages.
        mask: Boolean retention mask returned by
            :func:`select_by_absolute_advantage`.
        num_generations: Group size ``G``.
        zero_tolerance: Threshold below which an advantage counts as zero.

    Returns:
        A populated :class:`PruningStats` record.
    """
    abs_adv = advantages.detach().abs().float()
    grouped = abs_adv.reshape(-1, num_generations)
    kept = abs_adv[mask]
    dropped = abs_adv[~mask]
    total_mass = float(abs_adv.sum().item())

    return PruningStats(
        num_groups=int(grouped.shape[0]),
        num_generations=num_generations,
        num_retained_per_group=int(mask.reshape(-1, num_generations).sum(dim=1).max().item())
        if mask.numel()
        else 0,
        num_total=int(abs_adv.numel()),
        num_kept=int(mask.sum().item()),
        num_degenerate_groups=int((grouped.max(dim=1).values <= zero_tolerance).sum().item()),
        mean_abs_advantage_kept=float(kept.mean().item()) if kept.numel() else 0.0,
        mean_abs_advantage_dropped=float(dropped.mean().item()) if dropped.numel() else 0.0,
        retained_signal_fraction=(
            float(kept.sum().item()) / total_mass if total_mass > 0.0 else 1.0
        ),
    )
