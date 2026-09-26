"""Aggregate run artefacts into the tables used by the report.

Three kinds of JSON artefact are produced by the pipeline:

* ``profile.json`` -- written by :mod:`cppo.train`, one per training run.
* ``eval.json`` -- written by :mod:`cppo.evaluate`, one per checkpoint.
* ``update_stage_benchmark.json`` -- written by the update-stage benchmark.

This module renders them as Markdown (for ``README.md``) or LaTeX (for
``report/report.tex``) so that both documents are generated from the same
numbers and cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

LOGGER = logging.getLogger(__name__)

__all__ = ["RunSummary", "main", "render_tables", "summarize_run"]


@dataclass(frozen=True)
class RunSummary:
    """One row of the training-performance table."""

    run_name: str
    algorithm: str
    pruning_rate: float
    num_retained: int
    allocation_multiplier: int
    optimizer_steps: int
    wall_clock_seconds: float
    rollout_seconds: float
    update_seconds: float
    peak_memory_gib: float
    completions_generated: int
    completions_updated: int
    retention: float
    final_reward: float | None
    accuracies: dict[str, float]

    @property
    def update_share(self) -> float:
        """Fraction of the measured training step spent in the update stage."""
        total = self.rollout_seconds + self.update_seconds
        return self.update_seconds / total if total else 0.0


def _final_reward(log_history: Iterable[dict[str, Any]]) -> float | None:
    """Return the last logged mean reward, if the run recorded one.

    Args:
        log_history: The trainer's ``state.log_history``.

    Returns:
        The final ``reward`` entry, or ``None`` when absent.
    """
    rewards = [
        entry["reward"]
        for entry in log_history
        if isinstance(entry.get("reward"), (int, float))
    ]
    return float(rewards[-1]) if rewards else None


def summarize_run(
    profile: dict[str, Any], accuracies: dict[str, float] | None = None
) -> RunSummary:
    """Reduce one profiling report to a table row.

    Args:
        profile: The parsed ``profile.json`` payload.
        accuracies: Optional per-task accuracies for the run's checkpoint.

    Returns:
        The populated :class:`RunSummary`.
    """
    stages = profile.get("stages", {})
    geometry = profile.get("geometry", {})
    return RunSummary(
        run_name=str(profile.get("run_name", "unknown")),
        algorithm=str(profile.get("algorithm", "GRPO")),
        pruning_rate=float(geometry.get("pruning_rate", 0.0)),
        num_retained=int(geometry.get("num_retained", 0)),
        allocation_multiplier=int(geometry.get("allocation_multiplier", 1)),
        optimizer_steps=int(profile.get("global_step", 0)),
        wall_clock_seconds=float(profile.get("wall_clock_seconds", 0.0)),
        rollout_seconds=float(stages.get("rollout_seconds", 0.0)),
        update_seconds=float(stages.get("update_seconds", 0.0)),
        peak_memory_gib=float(profile.get("peak_memory_allocated_gib", 0.0)),
        completions_generated=int(stages.get("completions_generated", 0)),
        completions_updated=int(stages.get("completions_updated", 0)),
        retention=float(stages.get("completion_retention", 1.0)),
        final_reward=_final_reward(profile.get("log_history", [])),
        accuracies=dict(accuracies or {}),
    )


def _fmt(value: float | None, digits: int = 2, dash: str = "--") -> str:
    """Format an optional float for a table cell.

    Args:
        value: The number to render, or ``None``.
        digits: Decimal places.
        dash: Placeholder used for ``None``.

    Returns:
        The rendered cell.
    """
    return dash if value is None else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], latex: bool) -> str:
    """Render a table in Markdown or LaTeX.

    Args:
        headers: Column headers.
        rows: Row cells, already formatted as strings.
        latex: ``True`` for a LaTeX ``tabular``, ``False`` for Markdown.

    Returns:
        The rendered table.
    """
    if not latex:
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    spec = "l" + "r" * (len(headers) - 1)
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(header.replace("%", r"\%") for header in headers) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(cell.replace("%", r"\%") for cell in row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _performance_table(summaries: Sequence[RunSummary], latex: bool) -> str:
    """Render the training-performance comparison table.

    Args:
        summaries: Run summaries, baseline first.
        latex: Whether to emit LaTeX.

    Returns:
        The rendered table.
    """
    baseline = summaries[0].wall_clock_seconds if summaries else 0.0
    headers = [
        "Method",
        "P",
        "k",
        "m",
        "Steps",
        "Wall clock (s)",
        "Rollout (s)",
        "Update (s)",
        "Peak mem (GiB)",
        "Speedup",
    ]
    rows = [
        [
            summary.run_name,
            f"{100 * summary.pruning_rate:.2f}%",
            str(summary.num_retained),
            f"{summary.allocation_multiplier}x",
            str(summary.optimizer_steps),
            _fmt(summary.wall_clock_seconds, 1),
            _fmt(summary.rollout_seconds, 1),
            _fmt(summary.update_seconds, 1),
            _fmt(summary.peak_memory_gib, 2),
            _fmt(baseline / summary.wall_clock_seconds if summary.wall_clock_seconds else None)
            + "x",
        ]
        for summary in summaries
    ]
    return _table(headers, rows, latex)


def _accuracy_table(summaries: Sequence[RunSummary], latex: bool) -> str:
    """Render the evaluation-accuracy table.

    Args:
        summaries: Run summaries, baseline first.
        latex: Whether to emit LaTeX.

    Returns:
        The rendered table, or a placeholder when no accuracies are present.
    """
    tasks = sorted({task for summary in summaries for task in summary.accuracies})
    if not tasks:
        if latex:
            return r"\emph{No evaluation results available.}"
        return "_No evaluation results available._"
    headers = ["Method", "P", *tasks, "Final reward"]
    rows = [
        [
            summary.run_name,
            f"{100 * summary.pruning_rate:.2f}%",
            *[_fmt(summary.accuracies.get(task)) for task in tasks],
            _fmt(summary.final_reward, 3),
        ]
        for summary in summaries
    ]
    return _table(headers, rows, latex)


def _benchmark_table(benchmark: dict[str, Any] | None, latex: bool) -> str:
    """Render the update-stage micro-benchmark table.

    Two speedup columns are reported.  *Pruning only* is measured directly: the
    update batch shrinks from ``G`` to ``k`` completions, so the step gets
    cheaper but the device is left under-occupied, and the fixed per-step cost
    (the optimiser update over all parameters) stops the gain from reaching
    ``G/k``.  *With allocation* is derived from the same measurements: dynamic
    completion allocation refills the batch back to ``G`` completions drawn
    from ``m = G // k`` questions, so the step time returns to the baseline row
    while covering ``m`` times more questions -- a question throughput gain of
    exactly ``m``, with no overhead penalty.

    Args:
        benchmark: The parsed benchmark payload, or ``None``.
        latex: Whether to emit LaTeX.

    Returns:
        The rendered table, or a placeholder when no benchmark is present.
    """
    if not benchmark:
        if latex:
            return r"\emph{No update-stage benchmark available.}"
        return "_No update-stage benchmark available._"
    group_size = int(benchmark.get("num_generations", 0))
    headers = [
        "P",
        "k",
        "Completions/step",
        "Tokens/step",
        "Step time (s)",
        "Speedup (pruning only)",
        "Speedup (with allocation)",
    ]
    rows = []
    for result in benchmark.get("results", []):
        retained = int(result["num_retained"])
        multiplier = max(1, group_size // retained) if group_size else 1
        rows.append(
            [
                f"{100 * result['pruning_rate']:.2f}%",
                str(retained),
                str(result["completions_per_step"]),
                str(result["tokens_per_step"]),
                f"{result['mean_step_seconds']:.4f} +/- {result['stdev_step_seconds']:.4f}",
                f"{result['speedup']:.2f}x",
                f"{multiplier:.2f}x",
            ]
        )
    return _table(headers, rows, latex)


def render_tables(
    summaries: Sequence[RunSummary],
    benchmark: dict[str, Any] | None = None,
    *,
    latex: bool = False,
) -> str:
    """Render every report table as one document fragment.

    Args:
        summaries: Run summaries, baseline first.
        benchmark: Optional update-stage benchmark payload.
        latex: Whether to emit LaTeX instead of Markdown.

    Returns:
        The concatenated tables, separated by headings.
    """
    def heading(text: str) -> str:
        # A bare `%` starts a comment in LaTeX and would swallow the rest of
        # the line, so headings are escaped exactly like table cells are.
        return f"\\subsection*{{{text.replace('%', r'\%')}}}" if latex else f"### {text}"

    blocks = [
        heading("Training performance"),
        _performance_table(summaries, latex),
        heading("Downstream accuracy (lm-evaluation-harness, % exact match)"),
        _accuracy_table(summaries, latex),
        heading("Update-stage micro-benchmark"),
        _benchmark_table(benchmark, latex),
    ]
    return "\n\n".join(blocks) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file.

    Args:
        path: File to read.

    Returns:
        The parsed payload.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the report generator."""
    parser = argparse.ArgumentParser(description="Render result tables from run artefacts.")
    parser.add_argument(
        "--profiles", nargs="+", required=True, help="profile.json paths, baseline first"
    )
    parser.add_argument(
        "--evals", nargs="*", default=[], help="eval.json paths, aligned with --profiles"
    )
    parser.add_argument("--benchmark", default=None, help="update-stage benchmark JSON")
    parser.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    parser.add_argument("--output", default=None, help="write here instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for ``cppo-report``.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    accuracies: list[dict[str, float]] = []
    for index in range(len(args.profiles)):
        if index < len(args.evals) and args.evals[index] not in {"", "none"}:
            payload = _load_json(Path(args.evals[index]))
            accuracies.append({k: float(v) for k, v in payload.get("cppo_summary", {}).items()})
        else:
            accuracies.append({})

    summaries = [
        summarize_run(_load_json(Path(path)), accuracy)
        for path, accuracy in zip(args.profiles, accuracies)
    ]
    benchmark = _load_json(Path(args.benchmark)) if args.benchmark else None
    rendered = render_tables(summaries, benchmark, latex=args.format == "latex")

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        LOGGER.info("Wrote %s", destination)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
