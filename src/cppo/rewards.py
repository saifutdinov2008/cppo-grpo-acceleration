"""Rule-based reward functions for GRPO/CPPO training on math problems.

Following the CPPO paper (arXiv:2503.22342, Appendix A) the total reward is
the sum of a *format* term and an *accuracy* term::

    r_i = R_format(o_i) + R_accuracy(o_i)

The accuracy term is evaluated with `math_verify`, which performs symbolic
equivalence checking rather than string matching, so ``\\frac{1}{2}`` and
``0.5`` are both accepted for a gold answer of ``1/2``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Sequence

LOGGER = logging.getLogger(__name__)

RewardFunc = Callable[..., list[float]]

#: Matches the ``<think>...</think><answer>...</answer>`` layout of the paper.
THINK_ANSWER_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL
)
#: Matches a single ``\boxed{...}`` occurrence.
BOXED_RE = re.compile(r"\\boxed\{")
#: Fallback numeric extractor used when `math_verify` cannot parse a string.
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")

ACCURACY_REWARD_WEIGHT = 2.0
FORMAT_REWARD_WEIGHT = 1.0

__all__ = [
    "accuracy_reward",
    "build_reward_functions",
    "extract_boxed",
    "format_reward_boxed",
    "format_reward_think_answer",
    "reward_function_names",
]


def _completion_to_text(completion: Any) -> str:
    """Normalise a TRL completion into plain text.

    Args:
        completion: Either a raw string (standard dataset format) or a list of
            chat messages (conversational format).

    Returns:
        The assistant text of the completion.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def extract_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` in ``text``.

    Brace matching is performed explicitly so that nested braces such as
    ``\\boxed{\\frac{1}{2}}`` are handled correctly.

    Args:
        text: Model output to scan.

    Returns:
        The inner content of the final ``\\boxed{}`` group, or ``None`` when
        no balanced group is present.
    """
    starts = [m.end() for m in BOXED_RE.finditer(text)]
    for start in reversed(starts):
        depth = 1
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index]
    return None


def _normalise(value: str) -> str:
    """Strip formatting noise so two answer strings can be compared literally."""
    cleaned = value.strip().strip("$").replace(" ", "").replace(",", "")
    cleaned = cleaned.replace("\\!", "").replace("\\,", "").replace("\\%", "")
    cleaned = cleaned.rstrip(".")
    if cleaned.startswith("\\text{") and cleaned.endswith("}"):
        cleaned = cleaned[len("\\text{") : -1]
    return cleaned


def _fallback_equal(predicted: str, gold: str) -> bool:
    """Compare two answers without `math_verify`.

    Args:
        predicted: The model's answer string.
        gold: The ground-truth answer string.

    Returns:
        ``True`` when the strings agree literally or numerically.
    """
    left, right = _normalise(predicted), _normalise(gold)
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) < 1e-6
    except (TypeError, ValueError):
        return False


def _symbolic_equal(predicted: str, gold: str) -> bool:
    """Check mathematical equivalence, preferring `math_verify`.

    Args:
        predicted: The model's answer string.
        gold: The ground-truth answer string.

    Returns:
        ``True`` when the two expressions are mathematically equivalent.
    """
    try:
        # Imported lazily: `math_verify` pulls in sympy, which is slow to load
        # and is not needed by callers that only use the format rewards.
        from math_verify import parse, verify  # pylint: disable=import-outside-toplevel

        gold_parsed = parse(f"${gold}$")
        pred_parsed = parse(predicted)
        if gold_parsed and pred_parsed and bool(verify(gold_parsed, pred_parsed)):
            return True
    except ImportError:
        LOGGER.debug("math_verify unavailable; falling back to string comparison")
    except Exception as exc:  # pylint: disable=broad-except
        # `math_verify` raises a wide variety of sympy errors on malformed
        # model output; a parse failure must never abort a training step.
        LOGGER.debug("math_verify failed on %r: %s", predicted[:80], exc)
    return _fallback_equal(predicted, gold)


def accuracy_reward(completions: list[Any], solution: list[str], **_: Any) -> list[float]:
    """Reward completions whose final answer matches the ground truth.

    Args:
        completions: Batch of completions in TRL's standard or conversational
            format.
        solution: Ground-truth answer strings, one per completion.
        **_: Remaining dataset columns forwarded by TRL; unused.

    Returns:
        ``ACCURACY_REWARD_WEIGHT`` for a correct answer and ``0.0`` otherwise.
    """
    rewards: list[float] = []
    for completion, gold in zip(completions, solution):
        text = _completion_to_text(completion)
        candidate = extract_boxed(text)
        if candidate is None:
            numbers = NUMBER_RE.findall(text)
            candidate = numbers[-1] if numbers else None
        correct = candidate is not None and _symbolic_equal(candidate, gold)
        rewards.append(ACCURACY_REWARD_WEIGHT if correct else 0.0)
    return rewards


def format_reward_boxed(completions: list[Any], **_: Any) -> list[float]:
    """Reward completions that state exactly one ``\\boxed{}`` final answer.

    Args:
        completions: Batch of completions in TRL's standard or conversational
            format.
        **_: Remaining dataset columns forwarded by TRL; unused.

    Returns:
        ``FORMAT_REWARD_WEIGHT`` when the layout is respected, else ``0.0``.
    """
    rewards: list[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        occurrences = len(BOXED_RE.findall(text))
        valid = occurrences == 1 and extract_boxed(text) is not None
        rewards.append(FORMAT_REWARD_WEIGHT if valid else 0.0)
    return rewards


def format_reward_think_answer(completions: list[Any], **_: Any) -> list[float]:
    """Reward the ``<think>...</think><answer>...</answer>`` layout.

    This reproduces the format reward of the CPPO paper verbatim and is used
    by the ``think_answer`` prompt style.

    Args:
        completions: Batch of completions in TRL's standard or conversational
            format.
        **_: Remaining dataset columns forwarded by TRL; unused.

    Returns:
        ``FORMAT_REWARD_WEIGHT`` when the layout is respected, else ``0.0``.
    """
    return [
        FORMAT_REWARD_WEIGHT
        if THINK_ANSWER_RE.match(_completion_to_text(completion))
        else 0.0
        for completion in completions
    ]


def build_reward_functions(prompt_style: str) -> list[RewardFunc]:
    """Return the reward functions matching a prompt style.

    Args:
        prompt_style: Either ``"boxed"`` or ``"think_answer"``.

    Returns:
        ``[accuracy_reward, format_reward]`` in the order TRL will log them.

    Raises:
        ValueError: If ``prompt_style`` is unknown.
    """
    if prompt_style == "boxed":
        return [accuracy_reward, format_reward_boxed]
    if prompt_style == "think_answer":
        return [accuracy_reward, format_reward_think_answer]
    raise ValueError(f"unknown prompt style {prompt_style!r}")


def reward_function_names(prompt_style: str) -> list[str]:
    """Return the logged names of the reward functions for ``prompt_style``.

    Args:
        prompt_style: Either ``"boxed"`` or ``"think_answer"``.

    Returns:
        The ``__name__`` of each reward function, in order.
    """
    return [func.__name__ for func in build_reward_functions(prompt_style)]
