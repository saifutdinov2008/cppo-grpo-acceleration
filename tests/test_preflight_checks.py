"""Unit tests for the preflight's completion-length gate."""

from __future__ import annotations

import pytest

from cppo.preflight_checks import check_log


def _log(ratio: str) -> str:
    """Build a training log line containing a clipped ratio.

    Args:
        ratio: The value to embed.

    Returns:
        A one-line log resembling TRL's metric output.
    """
    return (
        "{'loss': '0', 'completions/mean_length': '1024', "
        f"'completions/clipped_ratio': '{ratio}', 'reward': '0.06'}}"
    )


def test_normal_termination_passes() -> None:
    """A policy that mostly emits EOS is fine."""
    assert check_log(_log("0.08")) == pytest.approx(0.08)


def test_everything_truncated_fails() -> None:
    """The exact failure that cost an aborted sweep: nothing terminates."""
    with pytest.raises(ValueError, match="100% of completions hit the length cap"):
        check_log(_log("1"))


def test_threshold_is_inclusive_at_the_boundary() -> None:
    """Exactly at the threshold is tolerated; above it is not."""
    assert check_log(_log("0.5")) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        check_log(_log("0.51"))


def test_missing_metric_is_an_error_not_a_pass() -> None:
    """A log without the metric must fail loudly rather than silently pass."""
    with pytest.raises(ValueError, match="no completions/clipped_ratio"):
        check_log("Training completed successfully.")


def test_scientific_notation_is_parsed() -> None:
    """TRL formats small numbers in scientific notation."""
    assert check_log(_log("1.5e-02")) == pytest.approx(0.015)
