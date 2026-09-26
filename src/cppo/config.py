"""Run configuration: a single dataclass, a YAML loader and a CLI generator.

Every knob of the GRPO and CPPO pipelines lives in :class:`RunSettings`.  The
argument parser is generated from the dataclass fields, so a new option needs
to be declared exactly once and YAML files, ``--flags`` and the code all stay
in sync.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Sequence, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "RunSettings",
    "build_parser",
    "load_config_file",
    "load_settings",
    "settings_from_mapping",
]

#: Key allowing one config file to inherit from another, resolved relative to
#: the inheriting file.  Chains are followed; cycles raise.
EXTENDS_KEY = "extends"

_SIMPLE_TYPES: dict[Any, Any] = {int: int, float: float, str: str}


@dataclass
class RunSettings:  # pylint: disable=too-many-instance-attributes
    """Everything needed to reproduce one GRPO or CPPO training run.

    Field groups map onto the pipeline stages: model, data, rollout, CPPO,
    optimisation, runtime and output.
    """

    # -- run identity ------------------------------------------------------
    run_name: str = "grpo-baseline"
    output_dir: str = "outputs/grpo-baseline"
    seed: int = 42

    # -- model -------------------------------------------------------------
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = True
    use_cpu: bool = False
    use_peft: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # -- data --------------------------------------------------------------
    dataset_name: str = "open-r1/DAPO-Math-17k-Processed"
    dataset_config: str | None = "en"
    dataset_split: str = "train"
    prompt_style: str = "boxed"
    max_samples: int | None = None
    max_prompt_length: int = 640
    max_completion_length: int = 1024

    # -- rollout -----------------------------------------------------------
    num_generations: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_tensor_parallel_size: int = 1
    # Frees the vLLM KV cache and weights between rollouts, so the sampler and
    # the trainer do not hold GPU memory at the same time.
    vllm_enable_sleep_mode: bool = True
    vllm_importance_sampling_correction: bool = False

    # -- CPPO --------------------------------------------------------------
    pruning_rate: float = 0.0
    dynamic_allocation: bool = True
    drop_zero_advantage: bool = False

    # -- optimisation ------------------------------------------------------
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 0.2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float | None = 0.28
    loss_type: str = "dapo"
    mask_truncated_completions: bool = True
    scale_rewards: str = "group"
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # -- runtime -----------------------------------------------------------
    logging_steps: int = 1
    save_steps: int = 0
    save_final_model: bool = True
    report_to: list[str] = field(default_factory=list)
    log_completions: bool = False
    num_completions_to_print: int = 2
    profile_path: str | None = None

    def resolved_profile_path(self) -> Path:
        """Return the destination for the profiling report.

        Returns:
            ``profile_path`` when set, else ``<output_dir>/profile.json``.
        """
        if self.profile_path:
            return Path(self.profile_path)
        return Path(self.output_dir) / "profile.json"


def _field_type(annotation: Any) -> tuple[Any, bool]:
    """Reduce a field annotation to a concrete type and an optionality flag.

    Args:
        annotation: The resolved dataclass field annotation.

    Returns:
        A ``(base_type, is_optional)`` pair.
    """
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [arg for arg in get_args(annotation) if arg is not NoneType]
        return (args[0] if args else str), True
    return annotation, False


def _optional_type(base: Any) -> Any:
    """Wrap a converter so that the literal ``none`` maps to ``None``.

    Args:
        base: The underlying converter, e.g. ``int``.

    Returns:
        A converter accepting ``"none"``/``"null"``/``""`` as ``None``.
    """

    def convert(value: str) -> Any:
        if value.lower() in {"none", "null", ""}:
            return None
        return base(value)

    convert.__name__ = f"optional_{getattr(base, '__name__', 'value')}"
    return convert


def build_parser(description: str) -> argparse.ArgumentParser:
    """Generate an argument parser from :class:`RunSettings`.

    Boolean fields get a matching ``--flag`` / ``--no-flag`` pair, list fields
    accept ``nargs="*"``, and optional fields accept the literal ``none``.
    Every default is ``None`` so that an unset flag is distinguishable from a
    flag set to the dataclass default; YAML values therefore win over dataclass
    defaults, and command-line values win over both.

    Args:
        description: Text shown in ``--help``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="YAML file with RunSettings overrides")
    hints = get_type_hints(RunSettings)
    for spec in fields(RunSettings):
        flag = "--" + spec.name.replace("_", "-")
        base, optional = _field_type(hints[spec.name])
        if base is bool:
            parser.add_argument(flag, dest=spec.name, action="store_true", default=None)
            parser.add_argument(
                "--no-" + spec.name.replace("_", "-"),
                dest=spec.name,
                action="store_false",
                default=None,
            )
        elif get_origin(base) is list:
            parser.add_argument(flag, dest=spec.name, nargs="*", default=None)
        elif base in _SIMPLE_TYPES:
            parser.add_argument(
                flag,
                dest=spec.name,
                type=_optional_type(base) if optional else base,
                default=None,
            )
        else:  # pragma: no cover - defensive; no such fields exist today
            parser.add_argument(flag, dest=spec.name, default=None)
    return parser


def load_config_file(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read a YAML config, resolving its ``extends`` chain.

    Values in the inheriting file override those of the file it extends, so a
    per-experiment config only has to state its differences from the shared
    baseline.

    Args:
        path: Path to the YAML file.
        _seen: Files already visited on this chain; used to detect cycles.

    Returns:
        The merged mapping, with the ``extends`` key removed.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not hold a mapping, or the chain cycles.
    """
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(str(item) for item in (*_seen, path))
        raise ValueError(f"cyclic config inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path} must contain a mapping")

    parent_name = loaded.pop(EXTENDS_KEY, None)
    if parent_name is None:
        return loaded
    parent = load_config_file(path.parent / str(parent_name), (*_seen, path))
    parent.update(loaded)
    return parent


def settings_from_mapping(mapping: dict[str, Any]) -> RunSettings:
    """Build :class:`RunSettings` from a plain dict, rejecting unknown keys.

    Args:
        mapping: Field name to value.

    Returns:
        The populated settings object.

    Raises:
        ValueError: If the mapping contains a key that is not a field.
    """
    known = {spec.name for spec in fields(RunSettings)}
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
    return RunSettings(**mapping)


def load_settings(
    argv: Sequence[str] | None = None, *, description: str = "Train with GRPO or CPPO."
) -> RunSettings:
    """Resolve settings from dataclass defaults, a YAML file and the CLI.

    Precedence is dataclass defaults < YAML < command line.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.
        description: Text shown in ``--help``.

    Returns:
        The fully resolved settings.

    Raises:
        FileNotFoundError: If ``--config`` points at a missing file.
    """
    namespace = build_parser(description).parse_args(argv)
    values: dict[str, Any] = {}

    if namespace.config is not None:
        values.update(load_config_file(Path(namespace.config)))

    for spec in fields(RunSettings):
        supplied = getattr(namespace, spec.name, None)
        if supplied is not None:
            values[spec.name] = supplied
    return settings_from_mapping(values)
