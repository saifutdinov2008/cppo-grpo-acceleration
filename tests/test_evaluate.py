"""Unit tests for the lm-evaluation-harness wrapper.

The harness itself is not invoked here -- that needs a model and many minutes.
What is tested is everything around it: the argument string handed to
`lm_eval`, and the reduction of its result payload to one headline number per
task.
"""

from __future__ import annotations

from typing import Any

import pytest

from cppo.evaluate import DEFAULT_TASKS, build_model_args, summarize_results


def test_default_tasks_are_the_three_requested_benchmarks() -> None:
    """The study's benchmarks are the harness defaults for this pipeline."""
    assert DEFAULT_TASKS == ("gsm8k", "minerva_math", "aime24")


def test_hf_model_args_are_minimal() -> None:
    """The Transformers backend takes only what it needs."""
    args = build_model_args("outputs/cppo-p75", backend="hf", dtype="bfloat16")
    assert args == "pretrained=outputs/cppo-p75,dtype=bfloat16,trust_remote_code=True"


def test_hf_model_args_can_pin_a_device() -> None:
    """A device can be pinned, which is what makes a CPU-only check possible."""
    args = build_model_args("m", backend="hf", device="cpu")
    assert args.endswith("device=cpu")


def test_vllm_model_args_carry_the_serving_knobs() -> None:
    """The vLLM backend needs its memory and parallelism settings."""
    args = build_model_args(
        "m", backend="vllm", max_model_len=8192, gpu_memory_utilization=0.9,
        tensor_parallel_size=4,
    )
    assert "max_model_len=8192" in args
    assert "gpu_memory_utilization=0.9" in args
    assert "tensor_parallel_size=4" in args


def test_unknown_backend_is_rejected() -> None:
    """A typo in the backend name is a configuration error."""
    with pytest.raises(ValueError, match="unknown backend"):
        build_model_args("m", backend="tensorrt")


def test_summary_prefers_the_documented_metric_per_task() -> None:
    """Each task has a preferred filter; the summary reports percentages."""
    results: dict[str, Any] = {
        "results": {
            "gsm8k": {
                "exact_match,flexible-extract": 0.41,
                "exact_match,strict-match": 0.39,
            },
            "minerva_math": {"math_verify,none": 0.125, "exact_match,none": 0.10},
            "aime24": {"exact_match,none": 0.0333},
        }
    }
    summary = summarize_results(results)
    assert summary["gsm8k"] == pytest.approx(41.0)
    assert summary["minerva_math"] == pytest.approx(12.5)
    assert summary["aime24"] == pytest.approx(3.33)


def test_summary_falls_back_when_the_preferred_metric_is_absent() -> None:
    """A task reporting only strict-match still yields a number."""
    results = {"results": {"gsm8k": {"exact_match,strict-match": 0.5}}}
    assert summarize_results(results)["gsm8k"] == pytest.approx(50.0)


def test_summary_handles_unknown_tasks_and_empty_payloads() -> None:
    """Tasks outside the study still summarise; an empty payload is empty."""
    assert summarize_results({"results": {"custom": {"acc,none": 0.25}}}) == {"custom": 25.0}
    assert not summarize_results({})
    # A task exposing no accuracy-shaped metric contributes no row at all.
    assert not summarize_results({"results": {"weird": {"stderr,none": 0.01}}})
