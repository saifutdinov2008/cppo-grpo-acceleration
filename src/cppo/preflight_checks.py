"""Fail a preflight when completions never terminate.

A policy that never emits an end-of-sequence token has every completion
truncated at the length cap.  With ``mask_truncated_completions`` enabled --
the default, and the DAPO recommendation -- every one of those completions is
masked out of the loss, so the gradient is exactly zero and a multi-hour sweep
trains nothing at all.

The failure is silent: training proceeds, the loss prints as ``0``, and only
the completion-length metrics give it away.  This check reads them and exits
non-zero, so a preflight catches in minutes what would otherwise cost hours.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Above this fraction of truncated completions the gradient signal is mostly
#: masked away and the run is not worth starting.
MAX_CLIPPED_RATIO = 0.5

__all__ = ["check_log", "main"]


def check_log(text: str, threshold: float = MAX_CLIPPED_RATIO) -> float:
    """Return the first logged clipped ratio, raising if it is too high.

    Args:
        text: Contents of a training log.
        threshold: Highest tolerable fraction of truncated completions.

    Returns:
        The clipped ratio found in the log.

    Raises:
        ValueError: If no clipped ratio is present, or it exceeds ``threshold``.
    """
    match = re.search(r"'completions/clipped_ratio': '([0-9.eE+-]+)'", text)
    if match is None:
        raise ValueError("no completions/clipped_ratio found in the training log")

    ratio = float(match.group(1))
    if ratio > threshold:
        raise ValueError(
            f"{ratio:.0%} of completions hit the length cap. With truncation "
            "masking enabled most of the batch contributes no gradient, so the "
            "run would train on almost nothing. Either disable Qwen3 thinking "
            "(enable_thinking: false) or raise --max-completion-length."
        )
    return ratio


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` when completions terminate normally.
    """
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: check_clipped_ratio.py <training.log>", file=sys.stderr)
        return 2

    text = Path(args[0]).read_text(encoding="utf-8", errors="ignore")
    try:
        ratio = check_log(text)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"clipped_ratio = {ratio:.3f} -- completions terminate normally")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
