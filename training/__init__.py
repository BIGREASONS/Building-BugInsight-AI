"""BugInsight AI — Training pipelines package.

Exposes the unified :class:`Trainer` and :class:`TrainingHistory` for both
the BiLSTM and CodeBERT training loops.
"""

from training.trainer import Trainer, TrainingHistory

__all__ = [
    "Trainer",
    "TrainingHistory",
]
