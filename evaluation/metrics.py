"""BugInsight AI — Shared Evaluation Metrics.

Provides functions to compute classification metrics, generate
publication-quality confusion-matrix heat-maps, and plot class-distribution
bar charts.  Every model module (Random Forest, XGBoost, BiLSTM, CodeBERT)
delegates its ``evaluate()`` call to :func:`compute_metrics` so that the
metric dictionaries are structurally identical across all experiments.

Typical usage::

    from evaluation.metrics import compute_metrics, generate_confusion_matrix

    metrics = compute_metrics(y_true, y_pred, label_names)
    generate_confusion_matrix(y_true, y_pred, label_names, save_path="cm.png")
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Use a non-interactive backend so plots can be saved on headless machines
# (Kaggle kernels, CI runners, etc.) without an active display.
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global plot style — applied once on import
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    }
)


# =========================================================================
# Core metrics computation
# =========================================================================
def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        label_names: Ordered list of class names.  When provided the
            classification report and per-class F1 are restricted to
            these labels.

    Returns:
        Dictionary with keys:

        * **accuracy** – overall accuracy (float).
        * **precision** – macro-averaged precision (float).
        * **recall** – macro-averaged recall (float).
        * **f1** – macro-averaged F1 score (float).
        * **per_class_f1** – ``{label: f1}`` mapping for each class.
        * **classification_report** – formatted string from
          :func:`~sklearn.metrics.classification_report`.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy: float = float(accuracy_score(y_true_arr, y_pred_arr))
    precision: float = float(
        precision_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            labels=label_names,
            zero_division=0,
        )
    )
    recall: float = float(
        recall_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            labels=label_names,
            zero_division=0,
        )
    )
    f1: float = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            average="macro",
            labels=label_names,
            zero_division=0,
        )
    )

    # Per-class F1
    per_class_f1_values = f1_score(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=label_names,
        zero_division=0,
    )
    if label_names is not None:
        per_class_f1: Dict[str, float] = {
            label: float(score)
            for label, score in zip(label_names, per_class_f1_values)
        }
    else:
        unique_labels = sorted(set(y_true_arr) | set(y_pred_arr))
        per_class_f1 = {
            label: float(score)
            for label, score in zip(unique_labels, per_class_f1_values)
        }

    report: str = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=label_names,
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_f1": per_class_f1,
        "classification_report": report,
    }

    logger.info(
        "Metrics computed — Accuracy: %.4f | Precision: %.4f | "
        "Recall: %.4f | F1: %.4f",
        accuracy,
        precision,
        recall,
        f1,
    )
    return metrics


# =========================================================================
# Confusion matrix visualisation
# =========================================================================
def generate_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    label_names: List[str],
    save_path: Union[str, Path],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple = (8, 6),
) -> Path:
    """Generate and save a publication-quality confusion-matrix heat-map.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        label_names: Ordered class names (axis tick labels).
        save_path: File path for the saved PNG image.
        normalize: If ``True`` the matrix values are row-normalised to
            show recall per class.
        title: Plot title.
        cmap: Matplotlib colour-map name.
        figsize: Figure size in inches ``(width, height)``.

    Returns:
        Resolved path to the saved image.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=label_names)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes absent from y_true
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(np.float64) / row_sums
    else:
        cm_display = cm.astype(np.float64)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_classes = len(label_names)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Annotate cells — show count and (normalised percentage)
    fmt = ".2f" if normalize else "d"
    thresh = cm_display.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            value = cm_display[i, j]
            count = cm[i, j]
            text = f"{value:{fmt}}\n({count})" if normalize else f"{count}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved → %s", save_path)
    return save_path


# =========================================================================
# Class distribution plot
# =========================================================================
def generate_class_distribution(
    labels: Sequence[str],
    label_names: List[str],
    save_path: Union[str, Path],
    title: str = "Class Distribution",
    figsize: tuple = (8, 5),
    palette: Optional[List[str]] = None,
) -> Path:
    """Generate and save a publication-quality class-distribution bar chart.

    Args:
        labels: Flat sequence of all labels in the dataset split.
        label_names: Ordered class names (determines bar order).
        save_path: File path for the saved PNG image.
        title: Plot title.
        figsize: Figure size in inches ``(width, height)``.
        palette: Optional list of hex colour codes for the bars.

    Returns:
        Resolved path to the saved image.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels_arr = np.asarray(labels)
    counts = [int(np.sum(labels_arr == label)) for label in label_names]
    total = sum(counts)
    percentages = [
        (c / total * 100) if total > 0 else 0.0 for c in counts
    ]

    if palette is None:
        palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    colours = [palette[i % len(palette)] for i in range(len(label_names))]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(label_names))
    bars = ax.bar(x_pos, counts, color=colours, edgecolor="grey", linewidth=0.5)

    # Annotate bars with count and percentage
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(label_names, rotation=0)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    logger.info("Class distribution chart saved → %s", save_path)
    return save_path
