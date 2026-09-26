"""Unit tests for run-configuration loading and precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from cppo.config import RunSettings, load_config_file, load_settings, settings_from_mapping


def _write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` and return the path.

    Args:
        path: Destination file.
        text: File contents.

    Returns:
        The path that was written.
    """
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_describe_the_grpo_baseline() -> None:
    """The dataclass defaults are a runnable GRPO configuration."""
    settings = RunSettings()
    assert settings.pruning_rate == 0.0
    assert settings.beta == 0.0
    assert settings.num_generations == 8
    assert settings.model_name_or_path == "Qwen/Qwen3-0.6B"


def test_profile_path_defaults_into_the_output_dir(tmp_path: Path) -> None:
    """Without an explicit path the report lands next to the checkpoint."""
    settings = RunSettings(output_dir=str(tmp_path / "run"))
    assert settings.resolved_profile_path() == tmp_path / "run" / "profile.json"
    explicit = RunSettings(output_dir="x", profile_path=str(tmp_path / "p.json"))
    assert explicit.resolved_profile_path() == tmp_path / "p.json"


def test_yaml_overrides_dataclass_defaults(tmp_path: Path) -> None:
    """Values from a config file win over the dataclass defaults."""
    config = _write(tmp_path / "run.yaml", "pruning_rate: 0.75\nnum_generations: 16\n")
    settings = load_settings(["--config", str(config)])
    assert settings.pruning_rate == 0.75
    assert settings.num_generations == 16


def test_cli_overrides_yaml(tmp_path: Path) -> None:
    """Command-line flags win over the config file."""
    config = _write(tmp_path / "run.yaml", "pruning_rate: 0.75\nseed: 1\n")
    settings = load_settings(["--config", str(config), "--pruning-rate", "0.5"])
    assert settings.pruning_rate == 0.5
    assert settings.seed == 1  # untouched by the CLI


def test_boolean_flags_have_negations(tmp_path: Path) -> None:
    """Every boolean field exposes both ``--flag`` and ``--no-flag``."""
    config = _write(tmp_path / "run.yaml", "dynamic_allocation: true\n")
    assert load_settings(["--config", str(config)]).dynamic_allocation is True
    assert (
        load_settings(["--config", str(config), "--no-dynamic-allocation"]).dynamic_allocation
        is False
    )


def test_optional_fields_accept_the_literal_none() -> None:
    """``--max-samples none`` clears an optional integer field."""
    assert load_settings(["--max-samples", "none"]).max_samples is None
    assert load_settings(["--max-samples", "512"]).max_samples == 512


def test_extends_merges_parent_and_child(tmp_path: Path) -> None:
    """A child config inherits the parent's values and overrides its own."""
    _write(tmp_path / "base.yaml", "pruning_rate: 0.0\nseed: 42\nnum_generations: 8\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\npruning_rate: 0.75\n")
    merged = load_config_file(child)
    assert merged == {"pruning_rate": 0.75, "seed": 42, "num_generations": 8}


def test_extends_follows_a_chain(tmp_path: Path) -> None:
    """Inheritance chains resolve deepest-first."""
    _write(tmp_path / "a.yaml", "seed: 1\nnum_generations: 4\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\nnum_generations: 8\n")
    child = _write(tmp_path / "c.yaml", "extends: b.yaml\npruning_rate: 0.5\n")
    assert load_config_file(child) == {"seed": 1, "num_generations": 8, "pruning_rate": 0.5}


def test_extends_rejects_cycles(tmp_path: Path) -> None:
    """A cyclic inheritance chain is reported rather than looping forever."""
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="cyclic"):
        load_config_file(tmp_path / "a.yaml")


def test_missing_config_is_reported(tmp_path: Path) -> None:
    """A missing config file raises rather than silently using defaults."""
    with pytest.raises(FileNotFoundError):
        load_settings(["--config", str(tmp_path / "absent.yaml")])


def test_unknown_keys_are_rejected() -> None:
    """A typo in a config key is a hard error, not a silently ignored field."""
    with pytest.raises(ValueError, match="unknown configuration keys: typo_field"):
        settings_from_mapping({"typo_field": 1})


def test_shipped_configs_load_and_agree_with_their_names() -> None:
    """Every checked-in experiment config parses and sets the expected rate."""
    expected = {
        "grpo": 0.0,
        "cppo_p50": 0.5,
        "cppo_p75": 0.75,
        "cppo_p875": 0.875,
        "cppo_p75_no_allocation": 0.75,
    }
    root = Path(__file__).resolve().parents[1] / "configs"
    for name, rate in expected.items():
        settings = load_settings(["--config", str(root / f"{name}.yaml")])
        assert settings.pruning_rate == rate, name
        assert settings.run_name
    assert not load_settings(
        ["--config", str(root / "cppo_p75_no_allocation.yaml")]
    ).dynamic_allocation
