"""BugInsight AI — Evaluation and metrics package.

Exposes shared metric computation, visualisation helpers, and statistical
validation utilities used across all model experiments.
"""

from evaluation.metrics import (
    compute_metrics,
    generate_class_distribution,
    generate_confusion_matrix,
)
from evaluation.statistics import (
    aggregate_seed_results,
    compare_models,
    generate_table1,
)

__all__ = [
    "compute_metrics",
    "generate_class_distribution",
    "generate_confusion_matrix",
    "aggregate_seed_results",
    "compare_models",
    "generate_table1",
]
