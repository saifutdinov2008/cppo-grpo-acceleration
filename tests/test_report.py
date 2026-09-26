"""Unit tests for the result-table renderer."""

from __future__ import annotations

from typing import Any

from cppo.report import render_tables, summarize_run


def _profile(name: str, pruning_rate: float, wall: float, reward: float) -> dict[str, Any]:
    """Build a minimal profiling payload for the renderer.

    Args:
        name: Run name.
        pruning_rate: Pruning rate to record in the geometry block.
        wall: Wall-clock seconds.
        reward: Final logged reward.

    Returns:
        A dict shaped like ``profile.json``.
    """
    retained = max(1, int(8 * (1 - pruning_rate)))
    return {
        "run_name": name,
        "algorithm": "CPPO" if pruning_rate else "GRPO",
        "wall_clock_seconds": wall,
        "peak_memory_allocated_gib": 12.5,
        "global_step": 64,
        "geometry": {
            "pruning_rate": pruning_rate,
            "num_retained": retained,
            "allocation_multiplier": max(1, 8 // retained),
        },
        "stages": {
            "rollout_seconds": wall * 0.6,
            "update_seconds": wall * 0.4,
            "completions_generated": 1024,
            "completions_updated": 1024 * retained // 8,
            "completion_retention": retained / 8,
        },
        "log_history": [{"reward": reward - 0.1}, {"reward": reward}, {"loss": 0.0}],
    }


def test_summarize_run_extracts_the_row() -> None:
    """A profiling payload reduces to the fields the tables need."""
    summary = summarize_run(_profile("cppo-p75", 0.75, 100.0, 1.5), {"gsm8k": 42.0})
    assert summary.algorithm == "CPPO"
    assert summary.num_retained == 2
    assert summary.allocation_multiplier == 4
    assert summary.retention == 0.25
    assert summary.final_reward == 1.5
    assert summary.accuracies == {"gsm8k": 42.0}
    assert summary.update_share == 0.4


def test_summarize_run_tolerates_a_missing_reward_history() -> None:
    """A run that logged no reward yields ``None`` rather than raising."""
    profile = _profile("grpo", 0.0, 100.0, 1.0)
    profile["log_history"] = [{"loss": 0.1}]
    assert summarize_run(profile).final_reward is None


def test_markdown_tables_report_speedup_against_the_first_row() -> None:
    """The first summary is the baseline for the speedup column."""
    summaries = [
        summarize_run(_profile("grpo", 0.0, 200.0, 1.0), {"gsm8k": 40.0}),
        summarize_run(_profile("cppo-p75", 0.75, 100.0, 1.2), {"gsm8k": 41.0}),
    ]
    rendered = render_tables(summaries)
    assert "| grpo |" in rendered
    assert "1.00x" in rendered and "2.00x" in rendered
    assert "gsm8k" in rendered
    assert "_No update-stage benchmark available._" in rendered


def test_latex_tables_use_booktabs_and_escape_percent() -> None:
    """LaTeX output is a ``tabular`` whose content escapes percent signs."""
    summaries = [summarize_run(_profile("grpo", 0.0, 200.0, 1.0))]
    rendered = render_tables(summaries, latex=True)
    assert "\\begin{tabular}" in rendered and "\\toprule" in rendered
    assert "\\resizebox" in rendered  # wide tables are shrunk to the text block
    assert "0.00\\%" in rendered

    # A bare `%` would comment out the rest of the line. The only legitimate
    # one is a trailing `%` used to swallow a line break.
    for line in rendered.splitlines():
        body = line[:-1] if line.endswith("%") and not line.endswith("\\%") else line
        assert "%" not in body.replace("\\%", ""), line


def test_benchmark_table_reports_both_speedup_columns() -> None:
    """Measured pruning-only speedup sits next to the allocation-derived one."""
    benchmark = {
        "num_generations": 8,
        "results": [
            {
                "pruning_rate": 0.0,
                "num_retained": 8,
                "completions_per_step": 8,
                "tokens_per_step": 2048,
                "mean_step_seconds": 4.0,
                "stdev_step_seconds": 0.01,
                "peak_memory_gib": 9.0,
                "peak_reserved_gib": 9.5,
                "speedup": 1.0,
            },
            {
                "pruning_rate": 0.75,
                "num_retained": 2,
                "completions_per_step": 2,
                "tokens_per_step": 512,
                "mean_step_seconds": 1.25,
                "stdev_step_seconds": 0.01,
                "peak_memory_gib": 9.0,
                "peak_reserved_gib": 9.5,
                "speedup": 3.2,
            },
        ],
    }
    rendered = render_tables([summarize_run(_profile("grpo", 0.0, 10.0, 1.0))], benchmark)
    assert "Tokens/step" in rendered and "2048" in rendered
    # Pruning alone falls short of G/k = 4 because of the fixed per-step cost.
    assert "3.20x" in rendered
    # Allocation refills the batch to m*k = 8 completions, i.e. the baseline
    # width, so the step costs the baseline time while covering m = 4 times
    # more questions: a 4x question throughput gain.
    assert "4.00x" in rendered


def test_allocation_column_accounts_for_partly_filled_batches() -> None:
    """When k does not divide G the refilled batch is narrower than G."""
    # Cost model behind the numbers: T(c) = 1 + c, so T(8) = 9.
    benchmark = {
        "num_generations": 8,
        "results": [
            {
                "pruning_rate": 0.0,
                "num_retained": 8,
                "completions_per_step": 8,
                "tokens_per_step": 8,
                "mean_step_seconds": 9.0,
                "stdev_step_seconds": 0.0,
                "peak_memory_gib": 1.0,
                "speedup": 1.0,
            },
            {
                "pruning_rate": 0.625,
                "num_retained": 3,
                "completions_per_step": 3,
                "tokens_per_step": 3,
                "mean_step_seconds": 4.0,
                "stdev_step_seconds": 0.0,
                "peak_memory_gib": 1.0,
                "speedup": 2.25,
            },
        ],
    }
    rendered = render_tables([summarize_run(_profile("grpo", 0.0, 10.0, 1.0))], benchmark)
    # m = 8 // 3 = 2, so the step holds 6 completions: T(6) = 7, and the
    # throughput gain is 2 * 9 / 7 = 2.57x -- not the naive 2.00x.
    assert "2.57x" in rendered
