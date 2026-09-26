"""Unit tests for the stage-timing and memory instrumentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cppo.profiling import (
    ProfilingMixin,
    StageTimings,
    collect_environment,
    peak_host_rss_bytes,
    peak_memory_bytes,
)


class _FakeTrainer:
    """Stand-in for ``GRPOTrainer`` exposing only what the mixin wraps."""

    def __init__(self, **_: Any) -> None:
        """Accept and ignore the trainer's constructor arguments."""
        self.calls: list[str] = []

    def _generate_and_score_completions(self, _inputs: Any) -> Any:
        """Pretend to roll out, returning a batch-shaped dict.

        Args:
            _inputs: The prompt batch; ignored by the stand-in.

        Returns:
            A dict with an ``advantages`` entry.
        """
        self.calls.append("rollout")
        return {"advantages": _Column(8)}

    def training_step(self, *_: Any, **__: Any) -> float:
        """Pretend to run one training step.

        Returns:
            A dummy loss.
        """
        self.calls.append("training_step")
        return 0.0

    def compute_loss(self, *_: Any, **__: Any) -> float:
        """Pretend to compute the loss.

        Returns:
            A dummy loss.
        """
        self.calls.append("compute_loss")
        return 0.0


class _Column:
    """Minimal tensor stand-in exposing only ``shape``."""

    def __init__(self, rows: int) -> None:
        """Record the number of rows.

        Args:
            rows: Leading-dimension size.
        """
        self.shape = (rows,)


class _ProfiledFake(ProfilingMixin, _FakeTrainer):
    """The mixin applied to the fake trainer, mirroring the real MRO."""


def test_update_time_excludes_the_rollout() -> None:
    """Rollout happens inside the training step, so update is a difference."""
    timings = StageTimings(train_step_seconds=10.0, rollout_seconds=4.0)
    assert timings.update_seconds == pytest.approx(6.0)
    payload = timings.as_dict()
    assert payload["rollout_seconds"] + payload["update_seconds"] == pytest.approx(
        payload["train_step_seconds"]
    )


def test_update_time_never_goes_negative() -> None:
    """Clock jitter must not produce a negative stage duration."""
    assert StageTimings(train_step_seconds=1.0, rollout_seconds=1.5).update_seconds == 0.0


def test_retention_defaults_to_one_before_any_rollout() -> None:
    """An empty run reports full retention rather than dividing by zero."""
    assert StageTimings().as_dict()["completion_retention"] == 1.0


def test_mixin_counts_generated_and_updated_completions() -> None:
    """Rollout counts what was sampled; ``compute_loss`` counts what trained."""
    trainer = _ProfiledFake()
    trainer._generate_and_score_completions(None)  # pylint: disable=protected-access
    trainer.training_step(None, {}, None)
    trainer.compute_loss(None, {"advantages": _Column(2)})

    stages = trainer.profiling_report()["stages"]
    assert stages["completions_generated"] == 8
    assert stages["completions_updated"] == 2
    assert stages["completion_retention"] == pytest.approx(0.25)
    assert stages["rollout_calls"] == 1
    assert stages["train_step_calls"] == 1
    assert stages["loss_calls"] == 1
    assert trainer.calls == ["rollout", "training_step", "compute_loss"]


def test_mixin_ignores_batches_without_advantages() -> None:
    """Evaluation batches carry no advantages and must not be counted."""
    trainer = _ProfiledFake()
    trainer.compute_loss(None, {"input_ids": _Column(4)})
    assert trainer.profiling_report()["stages"]["loss_calls"] == 0


def test_report_records_where_the_memory_figure_came_from() -> None:
    """The report names its memory backend so the number can be interpreted."""
    report = _ProfiledFake().profiling_report({"run_name": "x"})
    assert report["peak_memory_source"] in {"cuda", "mps", "host_rss"}
    assert report["run_name"] == "x"
    assert "environment" in report and "torch" in report["environment"]


def test_saving_the_report_writes_valid_json(tmp_path: Path) -> None:
    """The report round-trips through JSON on disk."""
    import json  # pylint: disable=import-outside-toplevel

    destination = _ProfiledFake().save_profiling_report(
        tmp_path / "nested" / "profile.json", {"run_name": "smoke"}
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["run_name"] == "smoke"
    assert "stages" in payload


def test_host_rss_is_positive_and_used_as_the_cpu_fallback() -> None:
    """A CPU-only run still reports a meaningful peak-memory number."""
    assert peak_host_rss_bytes() > 0.0
    memory = peak_memory_bytes()
    assert memory["allocated"] >= 0.0 and memory["reserved"] >= 0.0


def test_environment_is_json_serialisable() -> None:
    """The environment block must survive ``json.dumps``."""
    import json  # pylint: disable=import-outside-toplevel

    environment = collect_environment()
    assert json.loads(json.dumps(environment))["torch"] == environment["torch"]
