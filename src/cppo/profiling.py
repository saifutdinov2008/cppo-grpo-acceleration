"""Wall-clock and memory instrumentation shared by the GRPO and CPPO runs.

Both trainers mix in :class:`ProfilingMixin`, so the baseline and the
accelerated run are measured by exactly the same code path.  The mixin splits
a training step into the two stages the CPPO paper targets:

``rollout``
    Autoregressive sampling of the completion group, reward computation and
    advantage estimation -- everything performed by TRL inside
    ``_generate_and_score_completions``.

``update``
    Forward and backward passes of the policy (and, when enabled, reference
    and old-policy) models plus the optimiser step.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "ProfilingMixin",
    "StageTimings",
    "collect_environment",
    "peak_host_rss_bytes",
    "peak_memory_bytes",
    "reset_peak_memory",
    "resolve_memory_backend",
]

_BYTES_PER_GIB = 1024.0**3


def _mps_available() -> bool:
    """Return whether the Apple MPS backend is usable.

    Returns:
        ``True`` when torch was built with MPS and a device is present.
    """
    backend = getattr(torch.backends, "mps", None)
    return backend is not None and bool(backend.is_available())


def resolve_memory_backend(device: torch.device | None = None) -> str:
    """Name the backend whose memory counters apply to a run.

    The machine's *available* backends are not the right thing to look at: a
    run pinned to CPU on a Mac would otherwise report Apple MPS counters that
    stay at zero. When the caller knows which device the trainer is actually
    using, that device decides.

    Args:
        device: The device the work runs on, or ``None`` to probe the machine.

    Returns:
        ``"cuda"``, ``"mps"`` or ``"cpu"``.
    """
    if device is not None:
        return "cuda" if device.type == "cuda" else "mps" if device.type == "mps" else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if _mps_available() else "cpu"


def reset_peak_memory(device: torch.device | None = None) -> None:
    """Reset the accelerator's peak-memory counter, if it has one.

    Args:
        device: The device to reset, or ``None`` to probe the machine.
    """
    backend = resolve_memory_backend(device)
    if backend == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif backend == "mps":
        # `torch.mps` exposes no reset for the driver allocator, so emptying
        # the cache is the closest available equivalent.
        torch.mps.empty_cache()


def peak_host_rss_bytes() -> float:
    """Return the process's peak resident set size in bytes.

    This is the only memory figure available for a CPU-only run, and it is a
    true high-water mark rather than an instantaneous reading.

    Returns:
        Peak RSS in bytes.
    """
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # `ru_maxrss` is bytes on macOS and kilobytes on Linux.
    return usage if sys.platform == "darwin" else usage * 1024.0


def peak_memory_bytes(device: torch.device | None = None) -> dict[str, float]:
    """Return peak memory in bytes for the current process.

    Args:
        device: The device the work runs on, or ``None`` to probe the machine.

    Returns:
        A mapping with ``allocated`` and ``reserved`` figures. On CUDA both are
        true peaks. On Apple MPS the backend exposes only instantaneous
        readings, so a caller that needs a peak must sample repeatedly. On CPU
        the process's peak resident set size is reported, which is a genuine
        high-water mark.
    """
    backend = resolve_memory_backend(device)
    if backend == "cuda":
        return {
            "allocated": float(torch.cuda.max_memory_allocated()),
            "reserved": float(torch.cuda.max_memory_reserved()),
        }
    if backend == "mps":
        return {
            "allocated": float(torch.mps.current_allocated_memory()),
            "reserved": float(torch.mps.driver_allocated_memory()),
        }
    rss = peak_host_rss_bytes()
    return {"allocated": rss, "reserved": rss}


def collect_environment() -> dict[str, Any]:
    """Describe the machine the measurement was taken on.

    Returns:
        A JSON-serialisable description of the Python, torch and device stack.
    """
    devices: list[str] = []
    if torch.cuda.is_available():
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": devices,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


@dataclass
class StageTimings:
    """Accumulated per-stage wall-clock time and completion counts.

    ``train_step_seconds`` covers everything ``Trainer.training_step`` does,
    which includes the rollout: TRL generates from inside ``_prepare_inputs``,
    which the training step calls.  The update stage is therefore derived by
    subtraction rather than measured directly, which keeps the two stages
    exactly additive.
    """

    train_step_seconds: float = 0.0
    rollout_seconds: float = 0.0
    train_step_calls: int = 0
    rollout_calls: int = 0
    loss_calls: int = 0
    completions_generated: int = 0
    completions_updated: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def update_seconds(self) -> float:
        """Wall-clock time spent outside the rollout: forward, backward, step."""
        return max(0.0, self.train_step_seconds - self.rollout_seconds)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view including derived averages."""
        payload: dict[str, Any] = {
            "train_step_seconds": self.train_step_seconds,
            "rollout_seconds": self.rollout_seconds,
            "update_seconds": self.update_seconds,
            "train_step_calls": self.train_step_calls,
            "rollout_calls": self.rollout_calls,
            "loss_calls": self.loss_calls,
            "completions_generated": self.completions_generated,
            "completions_updated": self.completions_updated,
            "mean_rollout_seconds": (
                self.rollout_seconds / self.rollout_calls if self.rollout_calls else 0.0
            ),
            "mean_update_seconds": (
                self.update_seconds / self.loss_calls if self.loss_calls else 0.0
            ),
            "seconds_per_updated_completion": (
                self.update_seconds / self.completions_updated
                if self.completions_updated
                else 0.0
            ),
            "completion_retention": (
                self.completions_updated / self.completions_generated
                if self.completions_generated
                else 1.0
            ),
        }
        payload.update(self.extra)
        return payload


class ProfilingMixin:
    """Adds stage timing and peak-memory tracking to a TRL trainer.

    The mixin must appear *before* the trainer class in the MRO so that its
    overrides of ``_generate_and_score_completions`` and ``training_step``
    wrap the trainer implementations.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the wrapped trainer and the timing accumulators."""
        super().__init__(*args, **kwargs)
        self.stage_timings = StageTimings()
        self._wall_clock_start: float | None = None
        reset_peak_memory(self._memory_device())

    def _memory_device(self) -> torch.device | None:
        """Return the device the wrapped trainer actually runs on.

        Returns:
            The accelerator's device, or ``None`` if it is not available yet.
        """
        accelerator = getattr(self, "accelerator", None)
        device = getattr(accelerator, "device", None)
        return device if isinstance(device, torch.device) else None

    # -- TRL hooks ---------------------------------------------------------
    def _generate_and_score_completions(self, inputs: Any) -> Any:
        """Time the rollout stage and forward to the wrapped trainer.

        Args:
            inputs: The prompt batch handed over by TRL.

        Returns:
            Whatever the wrapped trainer's implementation returns.
        """
        start = time.perf_counter()
        outputs = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        self.stage_timings.rollout_seconds += time.perf_counter() - start
        self.stage_timings.rollout_calls += 1
        advantages = outputs.get("advantages") if isinstance(outputs, dict) else None
        if advantages is not None:
            self.stage_timings.completions_generated += int(advantages.shape[0])
        return outputs

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """Time one full training step and forward to the wrapped trainer.

        Args:
            *args: Positional arguments of ``Trainer.training_step``.
            **kwargs: Keyword arguments of ``Trainer.training_step``.

        Returns:
            The loss tensor returned by the wrapped trainer.
        """
        if self._wall_clock_start is None:
            self._wall_clock_start = time.perf_counter()
        start = time.perf_counter()
        loss = super().training_step(*args, **kwargs)  # type: ignore[misc]
        self.stage_timings.train_step_seconds += time.perf_counter() - start
        self.stage_timings.train_step_calls += 1
        return loss

    def compute_loss(self, *args: Any, **kwargs: Any) -> Any:
        """Count the completions that actually reach the policy gradient.

        ``_prepare_inputs`` runs inside ``training_step``, so this is the first
        hook that sees the post-rollout -- and, for CPPO, post-pruning -- batch.

        Args:
            *args: Positional arguments of ``Trainer.compute_loss``.
            **kwargs: Keyword arguments of ``Trainer.compute_loss``.

        Returns:
            The loss returned by the wrapped trainer.
        """
        inputs = kwargs.get("inputs", args[1] if len(args) > 1 else None)
        if isinstance(inputs, dict) and "advantages" in inputs:
            self.stage_timings.completions_updated += int(inputs["advantages"].shape[0])
            self.stage_timings.loss_calls += 1
        return super().compute_loss(*args, **kwargs)  # type: ignore[misc]

    # -- reporting ---------------------------------------------------------
    def profiling_report(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Assemble the measurement record for this run.

        Args:
            extra: Additional fields (run name, hyper-parameters, accuracy)
                merged into the report.

        Returns:
            A JSON-serialisable dict describing timings, memory and machine.
        """
        device = self._memory_device()
        memory = peak_memory_bytes(device)
        elapsed = (
            time.perf_counter() - self._wall_clock_start
            if self._wall_clock_start is not None
            else 0.0
        )
        report: dict[str, Any] = {
            "wall_clock_seconds": elapsed,
            "peak_memory_source": resolve_memory_backend(device),
            "peak_memory_allocated_gib": memory["allocated"] / _BYTES_PER_GIB,
            "peak_memory_reserved_gib": memory["reserved"] / _BYTES_PER_GIB,
            "stages": self.stage_timings.as_dict(),
            "environment": collect_environment(),
        }
        if extra:
            report.update(extra)
        return report

    def save_profiling_report(
        self, path: str | Path, extra: dict[str, Any] | None = None
    ) -> Path:
        """Write :meth:`profiling_report` to ``path`` as JSON.

        Args:
            path: Destination file; parent directories are created.
            extra: Additional fields merged into the report.

        Returns:
            The path that was written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.profiling_report(extra), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination
