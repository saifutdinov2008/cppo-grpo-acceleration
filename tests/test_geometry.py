"""Unit tests for the CPPO batch-geometry arithmetic."""

from __future__ import annotations

import pytest

from cppo.geometry import resolve_batch_geometry


def test_baseline_geometry_is_plain_grpo() -> None:
    """With ``P = 0`` nothing is pruned and nothing is reallocated."""
    geometry = resolve_batch_geometry(
        base_per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        pruning_rate=0.0,
    )
    assert geometry.num_retained == 8
    assert geometry.allocation_multiplier == 1
    assert geometry.per_device_train_batch_size == 8
    assert geometry.generation_batch_size == 32
    assert geometry.questions_per_round == 4
    assert geometry.completions_generated_per_round == 32
    assert geometry.completions_updated_per_round == 32
    assert geometry.update_compute_ratio == 1.0


def test_allocation_restores_the_baseline_microbatch() -> None:
    """Dynamic allocation makes the post-pruning micro-batch match GRPO's."""
    baseline = resolve_batch_geometry(
        base_per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        pruning_rate=0.0,
    )
    cppo = resolve_batch_geometry(
        base_per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        pruning_rate=0.75,
        dynamic_allocation=True,
    )
    assert cppo.num_retained == 2
    assert cppo.allocation_multiplier == 4
    assert cppo.per_device_train_batch_size == 32
    assert cppo.update_microbatch_per_device == baseline.per_device_train_batch_size
    # Four times more questions per round, same update-stage tensor shape.
    assert cppo.questions_per_round == 4 * baseline.questions_per_round
    assert cppo.completions_updated_per_round == baseline.completions_updated_per_round


def test_pruning_without_allocation_shrinks_the_microbatch() -> None:
    """The paper's ``+ Completion Pruning`` ablation leaves slots empty."""
    cppo = resolve_batch_geometry(
        base_per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=8,
        pruning_rate=0.75,
        dynamic_allocation=False,
    )
    assert cppo.allocation_multiplier == 1
    assert cppo.per_device_train_batch_size == 8
    assert cppo.update_microbatch_per_device == 2.0
    assert cppo.completions_updated_per_round == 8


def test_update_compute_ratio_tracks_the_pruning_rate() -> None:
    """Update-stage work per question falls to ``k / G``."""
    for rate, expected in [(0.5, 0.5), (0.75, 0.25), (0.875, 0.125)]:
        geometry = resolve_batch_geometry(
            base_per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_generations=16,
            pruning_rate=rate,
        )
        assert geometry.update_compute_ratio == pytest.approx(expected)


def test_multi_process_geometry_scales_the_generation_batch() -> None:
    """Data-parallel replicas widen the generation batch linearly."""
    geometry = resolve_batch_geometry(
        base_per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=8,
        pruning_rate=0.5,
        num_processes=4,
    )
    assert geometry.per_device_train_batch_size == 8
    assert geometry.generation_batch_size == 8 * 4 * 2
    assert geometry.questions_per_round == 8


def test_indivisible_geometry_is_rejected() -> None:
    """TRL requires the generation batch to be a multiple of the group size."""
    with pytest.raises(ValueError, match="not divisible"):
        resolve_batch_geometry(
            base_per_device_train_batch_size=3,
            gradient_accumulation_steps=1,
            num_generations=8,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_per_device_train_batch_size": 0},
        {"gradient_accumulation_steps": 0},
        {"num_processes": 0},
    ],
)
def test_non_positive_arguments_are_rejected(kwargs: dict[str, int]) -> None:
    """Every size argument must be strictly positive."""
    base = {
        "base_per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_generations": 8,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        resolve_batch_geometry(**base)  # type: ignore[arg-type]
