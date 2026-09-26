"""End-to-end smoke tests that instantiate a real (tiny) TRL trainer.

These tests run on CPU with a randomly initialised 2-layer Qwen3 model, so
they exercise the full GRPO/CPPO code path -- rollout, rewards, advantages,
pruning, loss and optimiser step -- in a few seconds.  They are marked
``slow`` because they download a small checkpoint from the Hub.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset

from cppo.geometry import resolve_batch_geometry
from cppo.trainer import CPPOMixin, CPPOTrainer, ProfiledGRPOTrainer

pytestmark = pytest.mark.slow

TINY_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
NUM_GENERATIONS = 4


def _reward(completions: list[Any], **_: Any) -> list[float]:
    """Return a deterministic per-completion reward that varies within a group.

    Using the completion length keeps the reward spread non-degenerate without
    depending on what a randomly initialised model happens to emit.

    Args:
        completions: Batch of completions supplied by TRL.
        **_: Remaining dataset columns; unused.

    Returns:
        One float per completion.
    """
    return [float(len(str(completion)) % 5) for completion in completions]


def _dataset(num_problems: int = 32) -> Dataset:
    """Build a tiny prompt-only dataset in TRL's standard format.

    Args:
        num_problems: Number of prompts to generate.

    Returns:
        A dataset with a single ``prompt`` column.
    """
    return Dataset.from_dict({"prompt": [f"Compute {i} + {i}." for i in range(num_problems)]})


def _make_config(tmp_path: Path, **overrides: Any) -> Any:
    """Build a minimal CPU-friendly ``GRPOConfig``.

    Args:
        tmp_path: Directory for trainer outputs.
        **overrides: Fields overriding the defaults below.

    Returns:
        The configured ``GRPOConfig``.
    """
    # Imported here so that test collection does not pay TRL's import cost.
    from trl import GRPOConfig  # pylint: disable=import-outside-toplevel

    kwargs: dict[str, Any] = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": NUM_GENERATIONS,
        "gradient_accumulation_steps": 2,
        "num_generations": NUM_GENERATIONS,
        "max_completion_length": 8,
        "max_steps": 2,
        "learning_rate": 1e-4,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "gradient_checkpointing": False,
        "seed": 0,
        "model_init_kwargs": {"dtype": torch.float32},
        # Apple MPS and TRL's generation path disagree about tensor placement,
        # so the smoke tests pin everything to CPU.
        "use_cpu": True,
    }
    kwargs.update(overrides)
    return GRPOConfig(**kwargs)


@pytest.fixture(name="offline_guard", autouse=True)
def _offline_guard() -> None:
    """Keep the Hub quiet and tokenisers single-threaded during the tests."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def test_baseline_trainer_runs_and_reports(tmp_path: Path) -> None:
    """The profiled GRPO baseline trains and emits a measurement report."""
    trainer = ProfiledGRPOTrainer(
        model=TINY_MODEL,
        reward_funcs=_reward,
        args=_make_config(tmp_path),
        train_dataset=_dataset(),
    )
    trainer.train()
    report = trainer.profiling_report()

    stages = report["stages"]
    assert stages["rollout_calls"] >= 1
    assert stages["loss_calls"] >= 1
    assert stages["completion_retention"] == pytest.approx(1.0)
    assert stages["update_seconds"] > 0.0
    # The two stages partition the training step, so they must be additive.
    assert stages["rollout_seconds"] + stages["update_seconds"] == pytest.approx(
        stages["train_step_seconds"]
    )
    assert report["wall_clock_seconds"] > 0.0


def test_cppo_trainer_updates_on_pruned_completions(tmp_path: Path) -> None:
    """CPPO generates the full group but back-propagates through half of it."""
    geometry = resolve_batch_geometry(
        base_per_device_train_batch_size=NUM_GENERATIONS,
        gradient_accumulation_steps=2,
        num_generations=NUM_GENERATIONS,
        pruning_rate=0.5,
        dynamic_allocation=True,
    )
    assert geometry.num_retained == 2
    assert geometry.allocation_multiplier == 2

    trainer = CPPOTrainer(
        model=TINY_MODEL,
        reward_funcs=_reward,
        args=_make_config(
            tmp_path,
            per_device_train_batch_size=geometry.per_device_train_batch_size,
            gradient_accumulation_steps=geometry.gradient_accumulation_steps,
        ),
        train_dataset=_dataset(),
        pruning_rate=0.5,
    )
    assert trainer.num_retained == 2
    assert trainer.pruning_enabled

    trainer.train()
    stages = trainer.profiling_report()["stages"]

    assert stages["completions_generated"] > stages["completions_updated"]
    assert stages["completion_retention"] == pytest.approx(0.5, abs=1e-6)


def test_cppo_trainer_with_zero_rate_matches_baseline_shapes(tmp_path: Path) -> None:
    """``pruning_rate=0`` turns the CPPO trainer into the GRPO baseline."""
    trainer = CPPOTrainer(
        model=TINY_MODEL,
        reward_funcs=_reward,
        args=_make_config(tmp_path),
        train_dataset=_dataset(),
        pruning_rate=0.0,
    )
    assert not trainer.pruning_enabled

    trainer.train()
    assert trainer.profiling_report()["stages"]["completion_retention"] == pytest.approx(1.0)


def test_index_batch_selects_rows_and_leaves_scalars_alone() -> None:
    """Row selection follows the mask; scalars and mismatched values pass through."""
    batch_size = 4
    mask = torch.tensor([True, False, True, False])
    outputs = {
        "advantages": torch.tensor([3.0, 0.1, -2.0, 0.0]),
        "completion_ids": torch.arange(8).reshape(4, 2),
        "num_items_in_batch": torch.tensor(99.0),
        "prompt_texts": ["a", "b", "c", "d"],
        "unrelated": torch.zeros(7),
    }
    pruned = CPPOMixin._index_batch(outputs, mask, batch_size)  # pylint: disable=protected-access

    assert pruned["advantages"].tolist() == [3.0, -2.0]
    assert pruned["completion_ids"].tolist() == [[0, 1], [4, 5]]
    assert pruned["prompt_texts"] == ["a", "c"]
    assert pruned["num_items_in_batch"].item() == 99.0
    assert pruned["unrelated"].shape == (7,)
