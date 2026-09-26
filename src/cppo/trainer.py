"""GRPO and CPPO trainers built on top of `trl.GRPOTrainer`.

Two trainers are exported:

:class:`ProfiledGRPOTrainer`
    Unmodified TRL GRPO with stage timing and peak-memory tracking bolted on.
    This is the baseline; no optimisation logic is changed.

:class:`CPPOTrainer`
    The same trainer plus CPPO completion pruning.  After TRL has sampled a
    group of ``G`` completions per question and turned the rewards into
    group-relative advantages, only the ``k = floor(G * (1 - P))`` completions
    with the largest ``|A_i|`` are kept for the policy forward and backward
    pass.

Where the pruning happens
-------------------------
The CPPO paper prunes before the policy, reference and old-policy forward
passes.  TRL computes rewards -- and therefore advantages -- *after* the
reference and old-policy log-probabilities, so a faithful "prune first"
ordering is only reachable by reimplementing a 600-line method against
private TRL internals.

Instead this implementation prunes at the end of
``_generate_and_score_completions``, i.e. after the rollout and before the
policy forward/backward pass, and relies on a configuration in which the
auxiliary forward passes do not exist at all:

* ``beta = 0.0`` -- no reference model, so no reference forward pass.  The
  CPPO paper derives its pruning criterion from exactly this KL-free
  approximation of the gradient (its Eq. 6), and TRL defaults to it.
* ``num_iterations = 1`` and ``steps_per_generation == gradient_accumulation_steps``
  -- the samples are on-policy, so TRL skips the old-policy forward pass.
* ``vllm_importance_sampling_correction = False`` -- otherwise TRL recomputes
  old-policy log-probabilities to correct the vLLM/training mismatch.

Under those settings the only forward pass over completions is the policy one,
and pruning before it reproduces Eq. (9) of the paper exactly.  When they are
violated the trainer warns: pruning is still correct, but part of the update
stage runs on unpruned completions, so the measured speedup is a lower bound.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from trl import GRPOTrainer

from .profiling import ProfilingMixin
from .pruning import (
    compute_retained_completions,
    select_by_absolute_advantage,
    summarize_pruning,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["CPPOMixin", "CPPOTrainer", "ProfiledGRPOTrainer"]


class ProfiledGRPOTrainer(ProfilingMixin, GRPOTrainer):
    """Stock TRL GRPO with stage timing and peak-memory instrumentation."""


class CPPOMixin:
    """Adds CPPO completion pruning to a TRL GRPO trainer.

    The mixin must precede :class:`~cppo.profiling.ProfilingMixin` in the MRO
    so that the profiler records the number of completions *generated* rather
    than the number that survived pruning.
    """

    if TYPE_CHECKING:
        # Supplied by the ``GRPOTrainer`` this mixin is combined with. Declared
        # here (annotation only, so nothing is created at runtime) to document
        # the contract and to let a type checker verify the mixin in isolation.
        args: Any
        accelerator: Any
        beta: float
        model: Any
        num_generations: int | None
        num_iterations: int
        _metrics: dict[str, Any]

    def __init__(
        self,
        *args: Any,
        pruning_rate: float = 0.0,
        drop_zero_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure pruning and validate it against the trainer's batching.

        Args:
            *args: Positional arguments forwarded to ``GRPOTrainer``.
            pruning_rate: CPPO pruning rate ``P``.  ``0.0`` disables pruning
                and makes the trainer behave exactly like GRPO.
            drop_zero_advantage: Also discard completions whose advantage is
                numerically zero.  Exact only when ``beta == 0``, and only
                safe on a single process because it makes the retained count
                data-dependent.
            **kwargs: Keyword arguments forwarded to ``GRPOTrainer``.

        Raises:
            ValueError: If the local generation batch cannot be split into
                whole groups of ``num_generations`` completions.
        """
        super().__init__(*args, **kwargs)

        # TRL types `num_generations` as optional because it is resolved from
        # the config; by the time the trainer exists it is always an int.
        self.group_size: int = int(self.num_generations or 1)
        self.pruning_rate = float(pruning_rate)
        self.num_retained = compute_retained_completions(self.group_size, self.pruning_rate)
        self.pruning_enabled = self.num_retained < self.group_size
        self.drop_zero_advantage = bool(drop_zero_advantage)
        self._pruning_warned = False

        if self.drop_zero_advantage and self.accelerator.num_processes > 1:
            LOGGER.warning(
                "drop_zero_advantage makes the retained batch size data dependent, which "
                "deadlocks collective operations across %d processes; disabling it.",
                self.accelerator.num_processes,
            )
            self.drop_zero_advantage = False

        if self.pruning_enabled:
            self._validate_pruned_geometry()
            self._warn_about_unpruned_forward_passes()
            LOGGER.info(
                "CPPO enabled: G=%d, P=%.4f, k=%d (update stage sees %.1f%% of completions)",
                self.group_size,
                self.pruning_rate,
                self.num_retained,
                100.0 * self.num_retained / self.group_size,
            )

    # -- validation --------------------------------------------------------
    def _validate_pruned_geometry(self) -> None:
        """Check that pruning leaves TRL's micro-batch split lossless.

        TRL splits the generation batch into ``steps_per_generation`` chunks
        with integer division, so a retained count that is not a multiple of
        that number would silently drop the remainder.  On a single process
        TRL's own ``generation_batch_size % num_generations`` check already
        implies the group-alignment condition; the check below is what catches
        a multi-process layout that would split a group across devices.

        Raises:
            ValueError: If the local generation batch is not a whole number of
                groups.
        """
        local_generation_batch = (
            self.args.per_device_train_batch_size * self.args.steps_per_generation
        )
        if local_generation_batch % self.group_size != 0:
            raise ValueError(
                f"per_device_train_batch_size * steps_per_generation = "
                f"{local_generation_batch} is not a multiple of num_generations="
                f"{self.group_size}. CPPO prunes per question, which requires every "
                f"completion group to live entirely on one process."
            )
        retained = local_generation_batch * self.num_retained // self.group_size
        if retained % self.args.steps_per_generation != 0:
            LOGGER.warning(
                "After pruning, %d completions do not divide evenly into %d micro-batches; "
                "TRL will drop the %d-completion remainder each round. Choose a "
                "per-device batch size that is a multiple of %d to avoid this.",
                retained,
                self.args.steps_per_generation,
                retained % self.args.steps_per_generation,
                self.group_size // self.num_retained,
            )

    def _warn_about_unpruned_forward_passes(self) -> None:
        """Warn when auxiliary forward passes run before the pruning point."""
        reasons: list[str] = []
        if self.beta != 0.0:
            reasons.append("beta != 0 keeps a reference-model forward pass")
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self.args.gradient_accumulation_steps % generate_every != 0:
            reasons.append("misaligned generation adds an old-policy forward pass")
        if getattr(self, "use_vllm", False) and getattr(
            self.args, "vllm_importance_sampling_correction", False
        ):
            reasons.append("vLLM importance-sampling correction adds an old-policy forward pass")
        if reasons:
            LOGGER.warning(
                "CPPO prunes after the rollout, so these unpruned forward passes remain: %s. "
                "Training is still correct, but the measured speedup understates CPPO.",
                "; ".join(reasons),
            )

    # -- pruning -----------------------------------------------------------
    def _generate_and_score_completions(self, inputs: Any) -> Any:
        """Sample a completion group, then keep only its high-advantage part.

        Args:
            inputs: The prompt batch handed over by TRL.

        Returns:
            The batch dict produced by TRL, restricted to the retained
            completions.
        """
        outputs = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        if not self.pruning_enabled or not self.model.training:
            return outputs
        return self._prune_batch(outputs)

    def _prune_batch(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Apply the CPPO retention mask to every per-completion tensor.

        Args:
            outputs: The batch dict returned by TRL's rollout stage.

        Returns:
            A new dict holding only the retained rows.  Scalars and any value
            whose leading dimension does not match the batch are passed
            through untouched.
        """
        advantages = outputs["advantages"]
        batch_size = int(advantages.shape[0])
        if batch_size % self.group_size != 0:
            if not self._pruning_warned:
                LOGGER.error(
                    "Local batch of %d completions is not a multiple of num_generations=%d; "
                    "skipping pruning for this batch.",
                    batch_size,
                    self.group_size,
                )
                self._pruning_warned = True
            return outputs

        mask = select_by_absolute_advantage(
            advantages,
            num_generations=self.group_size,
            num_retained=self.num_retained,
            drop_zero_advantage=self.drop_zero_advantage,
        )
        if not bool(mask.any()):
            # Every advantage was zero. Keep the plain top-k mask so the
            # update stage still receives well-formed tensors; the resulting
            # gradient is zero either way.
            mask = select_by_absolute_advantage(
                advantages,
                num_generations=self.group_size,
                num_retained=self.num_retained,
            )

        stats = summarize_pruning(advantages, mask, self.group_size)
        pruned = self._index_batch(outputs, mask, batch_size)
        self._recompute_num_items_in_batch(pruned)
        self._log_pruning_stats(stats)
        return pruned

    @staticmethod
    def _index_batch(
        outputs: dict[str, Any], mask: torch.Tensor, batch_size: int
    ) -> dict[str, Any]:
        """Select ``mask`` rows from every batch-aligned entry of ``outputs``.

        Args:
            outputs: The batch dict returned by TRL's rollout stage.
            mask: Boolean retention mask of length ``batch_size``.
            batch_size: Number of completions before pruning.

        Returns:
            A new dict with the retained rows.
        """
        keep_indices = [index for index, flag in enumerate(mask.tolist()) if flag]
        pruned: dict[str, Any] = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
                pruned[key] = value[mask.to(value.device)]
            elif isinstance(value, list) and len(value) == batch_size:
                pruned[key] = [value[index] for index in keep_indices]
            else:
                pruned[key] = value
        return pruned

    def _recompute_num_items_in_batch(self, pruned: dict[str, Any]) -> None:
        """Rescale the token normaliser to the retained completions.

        TRL's ``dapo``/``cispo``/``vespo`` losses divide by
        ``num_items_in_batch``, the number of loss-carrying tokens in the whole
        generation batch.  Leaving the pre-pruning value in place would shrink
        the loss -- and therefore the effective learning rate -- by ``k / G``.

        Args:
            pruned: The pruned batch dict, modified in place.
        """
        if "num_items_in_batch" not in pruned:
            return
        loss_mask = pruned["completion_mask"]
        if pruned.get("tool_mask") is not None:
            loss_mask = loss_mask * pruned["tool_mask"]
        pruned["num_items_in_batch"] = self.accelerator.gather(loss_mask.sum()).sum()

    def _log_pruning_stats(self, stats: Any) -> None:
        """Record pruning diagnostics on the trainer's metric buffer.

        Args:
            stats: The :class:`~cppo.pruning.PruningStats` for this batch.
        """
        mode = "train" if self.model.training else "eval"
        metrics = self._metrics[mode]
        metrics["cppo/retention"].append(stats.num_kept / max(stats.num_total, 1))
        metrics["cppo/retained_signal_fraction"].append(stats.retained_signal_fraction)
        metrics["cppo/abs_advantage_kept"].append(stats.mean_abs_advantage_kept)
        metrics["cppo/abs_advantage_dropped"].append(stats.mean_abs_advantage_dropped)
        metrics["cppo/frac_degenerate_groups"].append(
            stats.num_degenerate_groups / max(stats.num_groups, 1)
        )


class CPPOTrainer(CPPOMixin, ProfilingMixin, GRPOTrainer):
    """TRL GRPO with CPPO completion pruning and stage instrumentation.

    The MRO is ``CPPOMixin -> ProfilingMixin -> GRPOTrainer`` so that the
    profiler observes the full, unpruned rollout while the update stage sees
    only the retained completions.  ``pruning_rate=0.0`` makes the trainer a
    drop-in replacement for :class:`ProfiledGRPOTrainer`.
    """
