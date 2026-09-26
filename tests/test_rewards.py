"""Unit tests for the rule-based reward functions."""

from __future__ import annotations

import pytest

from cppo.rewards import (
    ACCURACY_REWARD_WEIGHT,
    FORMAT_REWARD_WEIGHT,
    accuracy_reward,
    build_reward_functions,
    extract_boxed,
    format_reward_boxed,
    format_reward_think_answer,
    reward_function_names,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"the answer is \boxed{42}", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{a} then \boxed{b}", "b"),  # the last group wins
        (r"\boxed{\text{x}^{2}}", r"\text{x}^{2}"),
        ("no box here", None),
        (r"\boxed{unbalanced", None),
    ],
)
def test_extract_boxed(text: str, expected: str | None) -> None:
    """Brace matching recovers nested and repeated boxed groups."""
    assert extract_boxed(text) == expected


def test_accuracy_reward_accepts_exact_match() -> None:
    """A correct boxed answer earns the full accuracy reward."""
    completions = [r"Reasoning... \boxed{34}"]
    assert accuracy_reward(completions, solution=["34"]) == [ACCURACY_REWARD_WEIGHT]


def test_accuracy_reward_rejects_wrong_answer() -> None:
    """A wrong answer earns nothing."""
    completions = [r"Reasoning... \boxed{35}"]
    assert accuracy_reward(completions, solution=["34"]) == [0.0]


def test_accuracy_reward_accepts_equivalent_forms() -> None:
    """Symbolically equal answers written differently are accepted."""
    completions = [r"\boxed{\frac{1}{2}}", r"\boxed{0.5}"]
    rewards = accuracy_reward(completions, solution=["0.5", "1/2"])
    assert rewards == [ACCURACY_REWARD_WEIGHT, ACCURACY_REWARD_WEIGHT]


def test_accuracy_reward_falls_back_to_last_number() -> None:
    """Without a boxed group the trailing number is used as the answer."""
    completions = ["step one, step two, so the total is 34"]
    assert accuracy_reward(completions, solution=["34"]) == [ACCURACY_REWARD_WEIGHT]


def test_accuracy_reward_handles_conversational_format() -> None:
    """Chat-style completions are unwrapped before scoring."""
    completions = [[{"role": "assistant", "content": r"\boxed{7}"}]]
    assert accuracy_reward(completions, solution=["7"]) == [ACCURACY_REWARD_WEIGHT]


def test_accuracy_reward_survives_unparseable_output() -> None:
    """Garbage output scores zero instead of raising."""
    completions = [r"\boxed{\frac{{{}}}", ""]
    assert accuracy_reward(completions, solution=["1", "2"]) == [0.0, 0.0]


def test_format_reward_boxed_requires_exactly_one_box() -> None:
    """The boxed format reward penalises zero or multiple boxed groups."""
    completions = [r"\boxed{1}", r"\boxed{1} and \boxed{2}", "none"]
    assert format_reward_boxed(completions) == [FORMAT_REWARD_WEIGHT, 0.0, 0.0]


def test_format_reward_think_answer() -> None:
    """The paper's tag layout is matched exactly, including multi-line bodies."""
    good = "<think>\nreasoning\n</think>\n<answer> 42 </answer>"
    bad_order = "<answer> 42 </answer><think> reasoning </think>"
    missing = "<think> reasoning </think>"
    rewards = format_reward_think_answer([good, bad_order, missing])
    assert rewards == [FORMAT_REWARD_WEIGHT, 0.0, 0.0]


def test_build_reward_functions_pairs_accuracy_and_format() -> None:
    """Each prompt style maps to its accuracy and format reward pair."""
    assert reward_function_names("boxed") == ["accuracy_reward", "format_reward_boxed"]
    assert reward_function_names("think_answer") == [
        "accuracy_reward",
        "format_reward_think_answer",
    ]
    assert len(build_reward_functions("boxed")) == 2


def test_build_reward_functions_rejects_unknown_style() -> None:
    """An unknown prompt style is a configuration error."""
    with pytest.raises(ValueError):
        build_reward_functions("nonexistent")
