"""GRPO and CPPO post-training pipelines for reasoning LLMs.

The package is organised around the two stages the CPPO paper accelerates:

``cppo.pruning``
    Completion selection by absolute advantage -- the mathematical core.
``cppo.geometry``
    Translation of the pruning rate into TRL's batching parameters, including
    dynamic completion allocation.
``cppo.trainer``
    ``ProfiledGRPOTrainer`` (baseline) and ``CPPOTrainer`` (accelerated).
``cppo.train`` / ``cppo.evaluate``
    Command-line entry points for training and `lm_eval` evaluation.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
