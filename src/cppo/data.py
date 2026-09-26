"""Loading and prompt formatting for the DAPO-Math-17k training corpus.

`open-r1/DAPO-Math-17k-Processed` exposes one row per problem with a raw
``prompt`` string and a ``solution`` string holding the gold answer.  This
module converts those rows into TRL's *conversational* format, which keeps the
training prompt identical to the prompt `lm_eval` builds when it is invoked
with ``--apply_chat_template --system_instruction``.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from datasets import Dataset, load_dataset

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET: Final[str] = "open-r1/DAPO-Math-17k-Processed"
DEFAULT_CONFIG: Final[str] = "en"
DEFAULT_SPLIT: Final[str] = "train"

#: Asks for free-form reasoning terminated by a single ``\boxed{}`` answer.
#: This is the default because `lm_eval`'s ``minerva_math`` and ``aime24``
#: tasks both extract the answer from ``\boxed{}``, so the training and
#: evaluation answer formats agree.
BOXED_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant that solves mathematics problems. "
    "Reason step by step, and put your final answer within \\boxed{}."
)

#: Verbatim template from the CPPO paper (Appendix B, MATH variant).
THINK_ANSWER_SYSTEM_PROMPT: Final[str] = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. And the "
    "answer should be of the following format: \"Therefore, the final answer "
    "is: \\boxed{ANSWER}. I hope it is correct.\" (without quotes) where "
    "ANSWER is just the final number or expression that solves the problem. "
    "The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., <think> reasoning "
    "process here </think>\n<answer> answer here </answer>."
)

PROMPT_STYLES: Final[dict[str, str]] = {
    "boxed": BOXED_SYSTEM_PROMPT,
    "think_answer": THINK_ANSWER_SYSTEM_PROMPT,
}

__all__ = [
    "BOXED_SYSTEM_PROMPT",
    "PROMPT_STYLES",
    "THINK_ANSWER_SYSTEM_PROMPT",
    "filter_by_prompt_length",
    "load_training_dataset",
    "system_prompt_for",
]


def system_prompt_for(prompt_style: str) -> str:
    """Return the system prompt associated with ``prompt_style``.

    Args:
        prompt_style: Either ``"boxed"`` or ``"think_answer"``.

    Returns:
        The system prompt string.

    Raises:
        ValueError: If ``prompt_style`` is unknown.
    """
    try:
        return PROMPT_STYLES[prompt_style]
    except KeyError as exc:
        raise ValueError(
            f"unknown prompt style {prompt_style!r}; "
            f"expected one of {sorted(PROMPT_STYLES)}"
        ) from exc


def _to_conversation(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    """Map one dataset row to TRL's conversational schema.

    Args:
        row: A raw dataset row with ``prompt`` and ``solution`` fields.
        system_prompt: System message prepended to every conversation.

    Returns:
        A dict with ``prompt`` (chat messages) and ``solution`` (gold answer).
    """
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["prompt"]},
        ],
        "solution": str(row["solution"]),
    }


def load_training_dataset(
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str | None = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    *,
    prompt_style: str = "boxed",
    max_samples: int | None = None,
    shuffle_seed: int | None = 42,
) -> Dataset:
    """Load and format the GRPO/CPPO training corpus.

    Args:
        dataset_name: Hub id of the dataset to load.
        dataset_config: Dataset configuration name, or ``None``.
        split: Split to load.
        prompt_style: Prompt template to apply; see :data:`PROMPT_STYLES`.
        max_samples: Optional cap on the number of problems, applied after
            shuffling.  Useful for smoke tests and short benchmark runs.
        shuffle_seed: Seed used to shuffle before truncating; ``None`` keeps
            the original order.

    Returns:
        A dataset with the ``prompt`` and ``solution`` columns TRL expects.
    """
    system_prompt = system_prompt_for(prompt_style)
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if not isinstance(dataset, Dataset):  # pragma: no cover - defensive
        raise TypeError(f"expected a single split, got {type(dataset).__name__}")

    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    keep = {"prompt", "solution"}
    dataset = dataset.map(
        _to_conversation,
        fn_kwargs={"system_prompt": system_prompt},
        remove_columns=[name for name in dataset.column_names if name not in keep],
        desc="Formatting prompts",
    )
    LOGGER.info("Loaded %d problems from %s", len(dataset), dataset_name)
    return dataset


def filter_by_prompt_length(
    dataset: Dataset, tokenizer: Any, max_prompt_tokens: int, *, num_proc: int | None = None
) -> Dataset:
    """Drop problems whose templated prompt exceeds a token budget.

    TRL 1.14 no longer truncates prompts, so over-long problems would silently
    inflate rollout memory.  Filtering is preferable to truncation because a
    truncated maths problem is unanswerable and would only add reward noise.

    Args:
        dataset: A dataset in TRL's conversational format.
        tokenizer: Tokeniser providing ``apply_chat_template``.
        max_prompt_tokens: Inclusive upper bound on templated prompt length.
        num_proc: Worker processes for the filter.

    Returns:
        The filtered dataset.
    """

    def is_short_enough(batch: dict[str, Any]) -> list[bool]:
        rendered = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch["prompt"]
        ]
        lengths = [len(ids) for ids in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
        return [length <= max_prompt_tokens for length in lengths]

    before = len(dataset)
    dataset = dataset.filter(
        is_short_enough, batched=True, num_proc=num_proc, desc="Filtering long prompts"
    )
    LOGGER.info(
        "Kept %d/%d problems with prompts of at most %d tokens",
        len(dataset),
        before,
        max_prompt_tokens,
    )
    return dataset
