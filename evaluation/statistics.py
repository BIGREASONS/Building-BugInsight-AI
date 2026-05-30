"""BugInsight AI — Statistical Validation for Multi-Seed Experiments.

Provides utilities to aggregate metrics across seeds, compare models with
paired statistical tests, and generate the final Table 1 summary for the
research paper.

Typical usage::

    from evaluation.statistics import (
        aggregate_seed_results,
        compare_models,
        generate_table1,
    )

    agg = aggregate_seed_results("random_forest", metrics_dir)
    pvals = compare_models("random_forest", "xgboost", metrics_dir)
    generate_table1(metrics_dir, "outputs/figures/table1.csv")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Metrics that are scalar floats and can be aggregated across seeds.
_AGGREGATE_KEYS: List[str] = ["accuracy", "precision", "recall", "f1"]


# =========================================================================
# Internal helpers
# =========================================================================
def _load_seed_files(
    model_name: str,
    metrics_dir: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load all ``{model_name}_seed*.json`` files from *metrics_dir*.

    Args:
        model_name: Model identifier (e.g. ``"random_forest"``).
        metrics_dir: Directory containing the JSON metric files.

    Returns:
        List of parsed metric dictionaries, sorted by seed.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    metrics_dir = Path(metrics_dir)
    pattern = f"{model_name}_seed*.json"
    files = sorted(metrics_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No metric files matching '{pattern}' found in {metrics_dir}"
        )

    results: List[Dict[str, Any]] = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        results.append(data)
        logger.debug("Loaded %s", filepath.name)

    logger.info(
        "Loaded %d seed result file(s) for model '%s' from %s",
        len(results),
        model_name,
        metrics_dir,
    )
    return results


def _extract_metric_vector(
    results: List[Dict[str, Any]],
    key: str,
) -> np.ndarray:
    """Extract a 1-D array of a specific metric across seed runs.

    Args:
        results: List of per-seed metric dictionaries.
        key: Metric key to extract (e.g. ``"f1"``).

    Returns:
        NumPy float64 array of length ``len(results)``.

    Raises:
        KeyError: If the key is missing from any result file.
    """
    values: List[float] = []
    for r in results:
        if key not in r:
            raise KeyError(
                f"Metric '{key}' not found in result for seed={r.get('seed')}"
            )
        values.append(float(r[key]))
    return np.array(values, dtype=np.float64)


# =========================================================================
# Public API
# =========================================================================
def aggregate_seed_results(
    model_name: str,
    metrics_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Aggregate per-seed metric files for a single model.

    Computes the mean and standard deviation for each scalar metric
    (accuracy, precision, recall, F1).

    Args:
        model_name: Model identifier (e.g. ``"random_forest"``).
        metrics_dir: Directory containing the JSON metric files.

    Returns:
        Dictionary with structure::

            {
                "model": "random_forest",
                "n_seeds": 5,
                "seeds": [42, 123, 456, 789, 999],
                "accuracy_mean": 0.82, "accuracy_std": 0.01,
                "precision_mean": …, "precision_std": …,
                "recall_mean": …, "recall_std": …,
                "f1_mean": …, "f1_std": …,
                "per_class_f1_mean": {"Critical": …, …},
                "per_class_f1_std": {"Critical": …, …},
            }
    """
    results = _load_seed_files(model_name, metrics_dir)

    agg: Dict[str, Any] = {
        "model": model_name,
        "n_seeds": len(results),
        "seeds": [r.get("seed") for r in results],
    }

    # Scalar metrics
    for key in _AGGREGATE_KEYS:
        vec = _extract_metric_vector(results, key)
        agg[f"{key}_mean"] = float(np.mean(vec))
        agg[f"{key}_std"] = float(np.std(vec, ddof=1)) if len(vec) > 1 else 0.0

    # Per-class F1 aggregation
    per_class_runs: Dict[str, List[float]] = {}
    for r in results:
        per_class = r.get("per_class_f1", {})
        for label, score in per_class.items():
            per_class_runs.setdefault(label, []).append(float(score))

    per_class_f1_mean: Dict[str, float] = {}
    per_class_f1_std: Dict[str, float] = {}
    for label, scores in per_class_runs.items():
        arr = np.array(scores, dtype=np.float64)
        per_class_f1_mean[label] = float(np.mean(arr))
        per_class_f1_std[label] = (
            float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        )

    agg["per_class_f1_mean"] = per_class_f1_mean
    agg["per_class_f1_std"] = per_class_f1_std

    logger.info(
        "Aggregated %d seeds for '%s': Accuracy=%.4f±%.4f, F1=%.4f±%.4f",
        agg["n_seeds"],
        model_name,
        agg["accuracy_mean"],
        agg["accuracy_std"],
        agg["f1_mean"],
        agg["f1_std"],
    )
    return agg


def compare_models(
    model_a: str,
    model_b: str,
    metrics_dir: Union[str, Path],
    metric_key: str = "f1",
) -> Dict[str, Any]:
    """Run paired statistical tests comparing two models across seeds.

    Performs a **paired two-sided t-test** and a **Wilcoxon signed-rank
    test** on the per-seed values of the chosen metric.

    Args:
        model_a: First model identifier.
        model_b: Second model identifier.
        metrics_dir: Directory containing the JSON metric files.
        metric_key: Scalar metric to compare (default ``"f1"``).

    Returns:
        Dictionary with keys:

        * **model_a** / **model_b** – model names.
        * **metric** – the metric being compared.
        * **n_seeds** – number of paired observations.
        * **model_a_mean** / **model_b_mean** – mean values.
        * **ttest_statistic** / **ttest_pvalue** – paired *t*-test results.
        * **wilcoxon_statistic** / **wilcoxon_pvalue** – Wilcoxon results
          (``None`` when the sample is too small or all differences are
          zero).

    Raises:
        ValueError: If the two models have a different number of seed runs.
    """
    results_a = _load_seed_files(model_a, metrics_dir)
    results_b = _load_seed_files(model_b, metrics_dir)

    if len(results_a) != len(results_b):
        raise ValueError(
            f"Seed count mismatch: '{model_a}' has {len(results_a)} runs "
            f"but '{model_b}' has {len(results_b)} runs."
        )

    vec_a = _extract_metric_vector(results_a, metric_key)
    vec_b = _extract_metric_vector(results_b, metric_key)
    n_seeds = len(vec_a)

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(vec_a, vec_b)

    # Wilcoxon signed-rank test — requires ≥ 6 non-zero differences for
    # a meaningful result; gracefully degrade otherwise.
    w_stat: Optional[float] = None
    w_pval: Optional[float] = None
    diffs = vec_a - vec_b
    if np.count_nonzero(diffs) >= 1 and n_seeds >= 6:
        try:
            w_result = stats.wilcoxon(vec_a, vec_b)
            w_stat = float(w_result.statistic)
            w_pval = float(w_result.pvalue)
        except ValueError as exc:
            logger.warning(
                "Wilcoxon test skipped for '%s' vs '%s': %s",
                model_a,
                model_b,
                exc,
            )
    else:
        logger.info(
            "Wilcoxon test skipped — need ≥ 6 seeds with non-zero "
            "differences (got %d seeds, %d non-zero diffs).",
            n_seeds,
            int(np.count_nonzero(diffs)),
        )

    comparison: Dict[str, Any] = {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric_key,
        "n_seeds": n_seeds,
        "model_a_mean": float(np.mean(vec_a)),
        "model_b_mean": float(np.mean(vec_b)),
        "ttest_statistic": float(t_stat),
        "ttest_pvalue": float(t_pval),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pval,
    }

    logger.info(
        "Model comparison (%s) '%s' vs '%s': "
        "means=%.4f vs %.4f | t-test p=%.4g | Wilcoxon p=%s",
        metric_key,
        model_a,
        model_b,
        comparison["model_a_mean"],
        comparison["model_b_mean"],
        comparison["ttest_pvalue"],
        f"{w_pval:.4g}" if w_pval is not None else "N/A",
    )
    return comparison


def generate_table1(
    metrics_dir: Union[str, Path],
    output_path: Union[str, Path],
    model_names: Optional[List[str]] = None,
) -> str:
    """Generate the final Table 1 (Model | Accuracy | Precision | Recall | F1).

    The table is written as **CSV** (at *output_path*) and also returned as
    a human-readable formatted string suitable for logging or inclusion in
    a paper draft.

    If a companion ``.txt`` file is desired alongside the CSV, replace the
    ``.csv`` extension in *output_path* with ``.txt`` to get the pretty
    table.

    Args:
        metrics_dir: Directory containing per-seed JSON metric files.
        output_path: Destination path for the CSV output.
        model_names: Explicit list of model identifiers.  When ``None``,
            all unique model prefixes found via
            ``{model}_seed*.json`` patterns in *metrics_dir* are
            collected automatically.

    Returns:
        A formatted string representation of Table 1.
    """
    metrics_dir = Path(metrics_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-discover model names if not provided
    if model_names is None:
        model_names = _discover_model_names(metrics_dir)
        if not model_names:
            raise FileNotFoundError(
                f"No *_seed*.json files found in {metrics_dir}"
            )

    rows: List[Dict[str, str]] = []
    for model_name in model_names:
        try:
            agg = aggregate_seed_results(model_name, metrics_dir)
        except FileNotFoundError:
            logger.warning(
                "Skipping '%s' — no seed files found.", model_name,
            )
            continue

        rows.append(
            {
                "Model": model_name,
                "Accuracy": _fmt_mean_std(
                    agg["accuracy_mean"], agg["accuracy_std"],
                ),
                "Precision": _fmt_mean_std(
                    agg["precision_mean"], agg["precision_std"],
                ),
                "Recall": _fmt_mean_std(
                    agg["recall_mean"], agg["recall_std"],
                ),
                "F1": _fmt_mean_std(agg["f1_mean"], agg["f1_std"]),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No aggregatable seed results found in {metrics_dir}"
        )

    # ---- Write CSV ----
    header = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    csv_lines: List[str] = [",".join(header)]
    for row in rows:
        csv_lines.append(",".join(row[h] for h in header))
    csv_text = "\n".join(csv_lines) + "\n"

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(csv_text)
    logger.info("Table 1 CSV saved → %s", output_path)

    # ---- Write formatted TXT alongside ----
    txt_path = output_path.with_suffix(".txt")
    formatted = _format_table(header, rows)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(formatted)
    logger.info("Table 1 TXT saved → %s", txt_path)

    logger.info("Table 1:\n%s", formatted)
    return formatted


# =========================================================================
# Internal formatting helpers
# =========================================================================
def _fmt_mean_std(mean: float, std: float) -> str:
    """Format a ``mean±std`` string to 4 decimal places."""
    return f"{mean:.4f}±{std:.4f}"


def _format_table(
    header: List[str],
    rows: List[Dict[str, str]],
) -> str:
    """Build a nicely aligned ASCII table from header and row dicts.

    Args:
        header: Column names.
        rows: List of dicts mapping column name → display value.

    Returns:
        Multi-line string with aligned columns and separator lines.
    """
    # Compute column widths
    col_widths: Dict[str, int] = {}
    for h in header:
        col_widths[h] = max(
            len(h), *(len(row.get(h, "")) for row in rows),
        )

    sep = "+-" + "-+-".join("-" * col_widths[h] for h in header) + "-+"
    hdr_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in header) + " |"

    lines: List[str] = [sep, hdr_line, sep]
    for row in rows:
        line = "| " + " | ".join(
            row.get(h, "").ljust(col_widths[h]) for h in header
        ) + " |"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def _discover_model_names(metrics_dir: Path) -> List[str]:
    """Discover unique model prefixes from ``*_seed*.json`` files.

    Args:
        metrics_dir: Directory to scan.

    Returns:
        Sorted list of unique model name strings.
    """
    names: set = set()
    for filepath in metrics_dir.glob("*_seed*.json"):
        # Filenames follow the pattern: {model_name}_seed{N}.json
        stem = filepath.stem  # e.g. "random_forest_seed42"
        # Split from the right on "_seed" to handle model names with underscores
        parts = stem.rsplit("_seed", 1)
        if len(parts) == 2:
            names.add(parts[0])
    return sorted(names)
