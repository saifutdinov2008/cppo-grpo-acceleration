"""Render the report's figures from the measured benchmark artefacts.

Three figures are produced, as vector PDF for the LaTeX report and PNG for the
README:

``update_stage_scaling``
    Measured update-stage step time against the number of retained
    completions, with the least-squares fit that separates the
    batch-independent cost from the per-completion cost.
``throughput_gain``
    Measured pruning-only speedup against the allocation-derived question
    throughput gain, showing where each falls short of the ideal ``G/k``.
``amdahl_ceiling``
    The end-to-end speedup ceiling implied by the update stage's share of
    training time -- the reason published CPPO speedups vary so widely.

Example:
    python benchmarks/plot_results.py \\
        --benchmark results/update_stage_mps_qwen3_0.6b.json --outdir report/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

LOGGER = logging.getLogger(__name__)

# Palette roles. Categorical slots 1-2 and an ordinal blue ramp, both validated
# for colour-vision deficiency and contrast against the light chart surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e8e7e4"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
ORDINAL = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

LINE_WIDTH = 2.0
MARKER_SIZE = 7.0

__all__ = ["amdahl_speedup", "linear_fit", "main", "plot_all"]


def _style_axes(axes: Any, xlabel: str, ylabel: str, title: str) -> None:
    """Apply the shared recessive-chrome style to one axes object.

    Args:
        axes: The matplotlib axes to style.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Axes title.
    """
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
    axes.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    axes.grid(True, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
        axes.spines[side].set_linewidth(0.8)


def _legend(axes: Any, loc: str = "best") -> None:
    """Attach a legend in the shared style.

    Args:
        axes: The matplotlib axes to attach the legend to.
        loc: Matplotlib legend placement.
    """
    legend = axes.legend(frameon=False, fontsize=9, loc=loc)
    for text in legend.get_texts():
        text.set_color(INK_MUTED)


def linear_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares fit of ``y = a + b*x``.

    Args:
        points: The ``(x, y)`` pairs to fit.

    Returns:
        ``(intercept, slope, r_squared)``.
    """
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    variance = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    return intercept, slope, (1.0 - residual / total if total else 1.0)


def amdahl_speedup(pruning_rate: float, update_share: float) -> float:
    """Return the end-to-end speedup implied by Amdahl's law.

    Args:
        pruning_rate: CPPO pruning rate ``P``; update work scales by ``1 - P``.
        update_share: Fraction ``f`` of baseline training time in the update
            stage.

    Returns:
        The achievable whole-run speedup.
    """
    return 1.0 / ((1.0 - update_share) + update_share * (1.0 - pruning_rate))


def _plot_update_scaling(benchmark: dict[str, Any], outdir: Path, stem: str) -> None:
    """Draw update-stage step time against retained completions.

    Args:
        benchmark: The parsed benchmark payload.
        outdir: Directory to write the figures into.
        stem: File stem for the output figures.
    """
    results = benchmark["results"]
    retained = [float(r["num_retained"]) for r in results]
    seconds = [float(r["mean_step_seconds"]) for r in results]
    intercept, slope, r_squared = linear_fit(list(zip(retained, seconds)))

    figure, axes = plt.subplots(figsize=(6.2, 3.6), facecolor=SURFACE)
    span = [0.0, max(retained) * 1.04]
    axes.plot(
        span,
        [intercept + slope * x for x in span],
        color=INK_MUTED,
        linewidth=1.0,
        zorder=2,
        label=f"fit: $T(k) = {intercept:.3f} + {slope:.3f}\\,k$   ($R^2 = {r_squared:.4f}$)",
    )
    axes.plot(
        retained,
        seconds,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=2.0,
        linestyle="none",
        color=SERIES_1,
        zorder=3,
        label="measured",
    )
    # The intercept is the headline: it is the cost pruning cannot remove.
    # Marked with a dot and labelled beside it -- a leader line drawn across
    # the plot would read as a second series.
    axes.plot([0.0], [intercept], marker="o", markersize=5.0, color=INK_MUTED, zorder=4)
    axes.annotate(
        f"batch-independent cost {intercept:.2f} s "
        f"({100 * intercept / seconds[0]:.0f}% of the baseline step)",
        xy=(0.0, intercept),
        xytext=(10, -9),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8.5,
        ha="left",
        va="top",
    )
    axes.set_xlim(0.0, max(retained) * 1.04)
    axes.set_ylim(0.0, max(seconds) * 1.12)
    _style_axes(
        axes,
        "retained completions per question, $k$",
        "update-stage step time (s)",
        f"Update stage is linear in $k$ — {benchmark.get('model', 'model')}, "
        f"$G = {benchmark.get('num_generations', '?')}$",
    )
    _legend(axes, loc="lower right")
    _save(figure, outdir, stem)


def _plot_throughput_gain(benchmark: dict[str, Any], outdir: Path, stem: str) -> None:
    """Draw pruning-only speedup against the allocation-derived throughput gain.

    Args:
        benchmark: The parsed benchmark payload.
        outdir: Directory to write the figures into.
        stem: File stem for the output figures.
    """
    results = benchmark["results"]
    group_size = int(benchmark.get("num_generations", 8))
    retained = [float(r["num_retained"]) for r in results]
    seconds = [float(r["mean_step_seconds"]) for r in results]
    intercept, slope, _ = linear_fit(list(zip(retained, seconds)))
    baseline = seconds[0]

    rates = [100.0 * float(r["pruning_rate"]) for r in results]
    pruning_only = [float(r["speedup"]) for r in results]
    allocated = []
    for value in retained:
        multiplier = max(1, group_size // int(value))
        allocated.append(multiplier * baseline / (intercept + slope * multiplier * value))
    ideal = [group_size / value for value in retained]

    figure, axes = plt.subplots(figsize=(6.2, 3.6), facecolor=SURFACE)
    axes.plot(
        rates,
        pruning_only,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=2.0,
        linewidth=LINE_WIDTH,
        color=SERIES_1,
        zorder=2,
        label="pruning only (measured)",
    )
    axes.plot(
        rates,
        allocated,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=SURFACE,
        markeredgewidth=2.0,
        linewidth=LINE_WIDTH,
        color=SERIES_2,
        zorder=3,
        label="with dynamic allocation",
    )
    # The reference is drawn last, thin and dashed, so it stays visible where
    # the allocated curve sits on top of it instead of being buried by it.
    axes.plot(
        rates,
        ideal,
        color=INK_MUTED,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=4,
        label="ideal $G/k$",
    )
    # Label only the endpoints: the story is the gap at high pruning rates.
    for value, colour in ((allocated[-1], SERIES_2), (pruning_only[-1], SERIES_1)):
        axes.annotate(
            f"{value:.2f}×",
            xy=(rates[-1], value),
            xytext=(10, -3),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=9,
            ha="left",
        )
        axes.plot([rates[-1]], [value], marker="o", markersize=0.1, color=colour)
    axes.set_xlim(-3.0, max(rates) * 1.16)
    axes.set_ylim(0.0, max(ideal) * 1.12)
    _style_axes(
        axes,
        "pruning rate $P$ (%)",
        "update-stage speedup (×)",
        "Allocation recovers what the fixed per-step cost takes away",
    )
    axes.text(
        0.0,
        -0.30,
        "Below $P = 50$% the two curves coincide: $m = \\lfloor G/k \\rfloor = 1$, "
        "so there is nothing to reallocate.",
        transform=axes.transAxes,
        color=INK_MUTED,
        fontsize=8,
    )
    _legend(axes, loc="upper left")
    _save(figure, outdir, stem)


def _plot_amdahl(update_shares: Sequence[float], outdir: Path, stem: str) -> None:
    """Draw the end-to-end speedup ceiling as a function of the update share.

    Args:
        update_shares: Values of ``f`` to draw one curve for each.
        outdir: Directory to write the figures into.
        stem: File stem for the output figures.
    """
    rates = [index / 200.0 for index in range(0, 199)]
    figure, axes = plt.subplots(figsize=(6.2, 3.6), facecolor=SURFACE)
    for index, share in enumerate(update_shares):
        curve = [amdahl_speedup(rate, share) for rate in rates]
        colour = ORDINAL[min(index, len(ORDINAL) - 1)]
        axes.plot(
            [100.0 * rate for rate in rates],
            curve,
            color=colour,
            linewidth=LINE_WIDTH,
            zorder=2 + index,
            label=f"$f = {share:.2f}$  (ceiling {1 / (1 - share):.1f}×)",
        )
        axes.annotate(
            f"$f = {share:.2f}$",
            xy=(100.0 * rates[-1], curve[-1]),
            xytext=(7, -3),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8.5,
            ha="left",
        )
    axes.set_yscale("log")
    # Room on the right for the direct labels, clear of the curves.
    axes.set_xlim(-2.0, 116.0)
    axes.set_ylim(1.0, 60.0)
    axes.set_yticks([1, 2, 5, 10, 20, 50])
    axes.set_yticklabels(["1×", "2×", "5×", "10×", "20×", "50×"])
    _style_axes(
        axes,
        "pruning rate $P$ (%)",
        "end-to-end speedup",
        "Completion pruning cannot outrun the rollout: $S(P) = 1 / ((1-f) + f(1-P))$",
    )
    _legend(axes)
    _save(figure, outdir, stem)


def _save(figure: Any, outdir: Path, stem: str) -> None:
    """Write a figure as both PDF and PNG.

    Args:
        figure: The matplotlib figure.
        outdir: Directory to write into; created if missing.
        stem: File stem, without extension.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix, dpi in (("pdf", 300), ("png", 200)):
        destination = outdir / f"{stem}.{suffix}"
        figure.savefig(destination, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
        LOGGER.info("Wrote %s", destination)
    plt.close(figure)


def plot_all(benchmark: dict[str, Any], outdir: Path) -> None:
    """Render every report figure.

    Args:
        benchmark: The parsed update-stage benchmark payload.
        outdir: Directory to write the figures into.
    """
    _plot_update_scaling(benchmark, outdir, "update_stage_scaling")
    _plot_throughput_gain(benchmark, outdir, "throughput_gain")
    _plot_amdahl((0.50, 0.70, 0.88, 0.97), outdir, "amdahl_ceiling")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="update-stage benchmark JSON")
    parser.add_argument("--outdir", default="report/figures")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    plot_all(payload, Path(args.outdir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
