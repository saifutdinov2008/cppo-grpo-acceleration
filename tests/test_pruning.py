"""Unit tests for the CPPO completion-pruning primitives."""

from __future__ import annotations

import pytest
import torch

from cppo.pruning import (
    compute_retained_completions,
    dynamic_allocation_multiplier,
    select_by_absolute_advantage,
    summarize_pruning,
)


@pytest.mark.parametrize(
    ("group_size", "pruning_rate", "expected"),
    [
        (16, 0.0, 16),
        (16, 0.5, 8),
        (16, 0.75, 4),
        (16, 0.875, 2),
        (16, 0.9375, 1),
        (8, 0.5, 4),
        (8, 0.99, 1),  # clamped to at least one completion
        (3, 0.5, 1),  # floor(3 * 0.5) == 1
    ],
)
def test_compute_retained_completions(group_size: int, pruning_rate: float, expected: int) -> None:
    """``k = floor(G * (1 - P))`` matches the values tabulated in the paper."""
    assert compute_retained_completions(group_size, pruning_rate) == expected


@pytest.mark.parametrize("bad_rate", [-0.1, 1.0, 1.5])
def test_compute_retained_completions_rejects_invalid_rate(bad_rate: float) -> None:
    """Pruning rates outside ``[0, 1)`` are rejected."""
    with pytest.raises(ValueError):
        compute_retained_completions(16, bad_rate)


def test_compute_retained_completions_rejects_invalid_group() -> None:
    """A non-positive group size is rejected."""
    with pytest.raises(ValueError):
        compute_retained_completions(0, 0.5)


@pytest.mark.parametrize(
    ("group_size", "retained", "expected"),
    [(16, 16, 1), (16, 8, 2), (16, 4, 4), (16, 2, 8), (16, 1, 16), (16, 3, 5)],
)
def test_dynamic_allocation_multiplier(group_size: int, retained: int, expected: int) -> None:
    """Freed completion slots translate into proportionally more questions."""
    assert dynamic_allocation_multiplier(group_size, retained) == expected


def test_dynamic_allocation_multiplier_rejects_out_of_range() -> None:
    """``k`` must lie within ``[1, G]``."""
    with pytest.raises(ValueError):
        dynamic_allocation_multiplier(8, 0)
    with pytest.raises(ValueError):
        dynamic_allocation_multiplier(8, 9)


def test_selection_keeps_largest_absolute_advantage() -> None:
    """Exactly the top-``k`` completions by ``|A|`` survive, per group."""
    advantages = torch.tensor([0.1, -2.0, 0.5, 1.5, -0.2, 0.3, -3.0, 0.4])
    mask = select_by_absolute_advantage(advantages, num_generations=4, num_retained=2)
    # Group 0: |A| = [0.1, 2.0, 0.5, 1.5] -> keep indices 1 and 3.
    # Group 1: |A| = [0.2, 0.3, 3.0, 0.4] -> keep indices 2 and 3 (flat 6, 7).
    assert mask.tolist() == [False, True, False, True, False, False, True, True]


def test_selection_is_sign_agnostic() -> None:
    """Negative advantages compete on equal terms with positive ones."""
    advantages = torch.tensor([-5.0, 1.0, 2.0, 3.0])
    mask = select_by_absolute_advantage(advantages, num_generations=4, num_retained=1)
    assert mask.tolist() == [True, False, False, False]


def test_selection_is_a_no_op_when_nothing_is_pruned() -> None:
    """With ``P = 0`` CPPO degenerates to GRPO and keeps every completion."""
    advantages = torch.randn(32)
    mask = select_by_absolute_advantage(advantages, num_generations=8, num_retained=8)
    assert bool(mask.all())


def test_selection_breaks_ties_deterministically() -> None:
    """Equal ``|A|`` values resolve to the lowest flat index, reproducibly."""
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    first = select_by_absolute_advantage(advantages, num_generations=4, num_retained=2)
    second = select_by_absolute_advantage(advantages, num_generations=4, num_retained=2)
    assert first.tolist() == second.tolist() == [True, True, False, False]


def test_selection_keeps_exactly_k_per_group() -> None:
    """Every group contributes the same number of completions (bucket effect)."""
    torch.manual_seed(0)
    advantages = torch.randn(6 * 16)
    mask = select_by_absolute_advantage(advantages, num_generations=16, num_retained=4)
    per_group = mask.reshape(-1, 16).sum(dim=1)
    assert per_group.tolist() == [4] * 6


def test_drop_zero_advantage_removes_degenerate_groups() -> None:
    """Groups with identical rewards carry no signal and can be dropped."""
    advantages = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5])
    mask = select_by_absolute_advantage(
        advantages, num_generations=4, num_retained=2, drop_zero_advantage=True
    )
    assert mask.tolist() == [False, False, False, False, True, True, False, False]


def test_selection_rejects_bad_shapes() -> None:
    """Shape and divisibility violations raise rather than silently reshape."""
    with pytest.raises(ValueError):
        select_by_absolute_advantage(torch.randn(4, 4), num_generations=4, num_retained=2)
    with pytest.raises(ValueError):
        select_by_absolute_advantage(torch.randn(7), num_generations=4, num_retained=2)
    with pytest.raises(ValueError):
        select_by_absolute_advantage(torch.randn(8), num_generations=4, num_retained=5)


def test_summary_reports_retained_signal_mass() -> None:
    """Pruning keeps most of the ``|A|`` mass even at a high pruning rate."""
    advantages = torch.tensor([3.0, -2.0, 0.05, -0.05])
    mask = select_by_absolute_advantage(advantages, num_generations=4, num_retained=2)
    stats = summarize_pruning(advantages, mask, num_generations=4)
    assert stats.num_total == 4
    assert stats.num_kept == 2
    assert stats.num_degenerate_groups == 0
    assert stats.retained_signal_fraction == pytest.approx(5.0 / 5.1, rel=1e-6)
    assert stats.mean_abs_advantage_kept == pytest.approx(2.5)
    assert stats.mean_abs_advantage_dropped == pytest.approx(0.05)


def test_summary_flags_degenerate_groups() -> None:
    """All-zero advantage groups are counted so they can be logged."""
    advantages = torch.zeros(8)
    mask = select_by_absolute_advantage(advantages, num_generations=4, num_retained=2)
    stats = summarize_pruning(advantages, mask, num_generations=4)
    assert stats.num_degenerate_groups == 2
    assert stats.retained_signal_fraction == 1.0
