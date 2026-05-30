"""BugInsight AI — Run All Multi-Seed Experiments (Milestone 6).

Orchestrates the full E1 experiment: trains all 4 baseline models across
5 random seeds, then generates statistical comparisons and Table 1.

Usage::

    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --seeds 42 123 456
    python scripts/run_all_experiments.py --models codebert bilstm
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config
from utils import ensure_output_dirs, set_seed, setup_logging

logger = logging.getLogger(__name__)

# =========================================================================
# Classical model training (inline — no separate script needed)
# =========================================================================


def train_classical_model(
    model_name: str,
    seed: int,
    config,
) -> dict:
    """Train a classical ML model and return metrics.

    Args:
        model_name: ``"random_forest"`` or ``"xgboost"``.
        seed: Random seed.
        config: Loaded Config instance.

    Returns:
        Dictionary of evaluation metrics.
    """
    from data.dataset import get_text_data
    from utils import save_metrics

    set_seed(seed)
    X_train, X_val, X_test, y_train, y_val, y_test = get_text_data(config)

    if model_name == "random_forest":
        from models.random_forest import RandomForestSeverityClassifier
        model = RandomForestSeverityClassifier(config)
    elif model_name == "xgboost":
        from models.xgboost_model import XGBoostSeverityClassifier
        model = XGBoostSeverityClassifier(config)
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    logger.info("Training %s with seed=%d ...", model_name, seed)
    model.train(X_train, y_train)

    logger.info("Evaluating %s on test set ...", model_name)
    metrics = model.evaluate(X_test, y_test)

    # Save model
    model_dir = config.get_path("outputs.models_dir")
    model.save_model(str(model_dir / f"{model_name}_seed{seed}.joblib"))

    # Save metrics
    save_metrics(metrics, model_name, seed, project_root=config.project_root)

    return metrics


# =========================================================================
# Deep learning model training (delegates to subprocess)
# =========================================================================


def train_dl_model(
    model_name: str,
    seed: int,
    config_path: Optional[str] = None,
) -> None:
    """Train a deep learning model by invoking its training script.

    Args:
        model_name: ``"bilstm"`` or ``"codebert"``.
        seed: Random seed.
        config_path: Optional path to config.yaml.
    """
    if model_name == "bilstm":
        script = str(_PROJECT_ROOT / "training" / "train_bilstm.py")
    elif model_name == "codebert":
        script = str(_PROJECT_ROOT / "training" / "train_codebert.py")
    else:
        raise ValueError(f"Unknown DL model: {model_name}")

    cmd = [sys.executable, script, "--seed", str(seed)]
    if config_path:
        cmd.extend(["--config", config_path])

    logger.info("Launching: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=False)
    if result.returncode != 0:
        logger.error("%s training failed (seed=%d, exit=%d)", model_name, seed, result.returncode)
    else:
        logger.info("%s training completed (seed=%d)", model_name, seed)


# =========================================================================
# Main orchestrator
# =========================================================================


def main() -> None:
    """Run the full multi-seed experiment suite."""
    parser = argparse.ArgumentParser(
        description="BugInsight AI — Run All Baseline Experiments (E1)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run (default: from config.yaml).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["random_forest", "xgboost", "bilstm", "codebert"],
        help="Models to train (default: all four).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, only generate Table 1 from existing metrics.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        log_dir=config.get("logging.log_dir", "outputs/logs"),
        level=config.get("logging.level", "INFO"),
    )
    ensure_output_dirs(config.project_root)

    seeds = args.seeds or config.seeds
    models = args.models or ["random_forest", "xgboost", "bilstm", "codebert"]

    logger.info("=" * 70)
    logger.info("BugInsight AI — Multi-Seed Experiment Suite (E1)")
    logger.info("Models : %s", models)
    logger.info("Seeds  : %s", seeds)
    logger.info("=" * 70)

    if not args.skip_training:
        total = len(models) * len(seeds)
        completed = 0

        for model_name in models:
            for seed in seeds:
                completed += 1
                logger.info(
                    "[%d/%d] Training %s (seed=%d) ...",
                    completed, total, model_name, seed,
                )

                start = time.time()
                if model_name in ("random_forest", "xgboost"):
                    metrics = train_classical_model(model_name, seed, config)
                    logger.info("  F1=%.4f (%.1fs)", metrics["f1"], time.time() - start)
                else:
                    train_dl_model(model_name, seed, args.config)
                    logger.info("  Completed in %.1fs", time.time() - start)

    # -----------------------------------------------------------------------
    # Generate Table 1
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Generating Table 1 ...")
    logger.info("=" * 70)

    from evaluation.statistics import generate_table1, compare_models

    metrics_dir = config.get_path("outputs.metrics_dir")
    table_path = config.get_path("outputs.figures_dir") / "table1"
    generate_table1(str(metrics_dir), str(table_path))

    # -----------------------------------------------------------------------
    # Statistical comparisons (CodeBERT vs each baseline)
    # -----------------------------------------------------------------------
    logger.info("Running statistical comparisons ...")
    for baseline in ["random_forest", "xgboost", "bilstm"]:
        if baseline in models and "codebert" in models:
            try:
                result = compare_models("codebert", baseline, str(metrics_dir))
                logger.info(
                    "CodeBERT vs %s: t-test p=%.4f, wilcoxon p=%s",
                    baseline,
                    result.get("t_test_p_value", float("nan")),
                    result.get("wilcoxon_p_value", "N/A"),
                )
            except Exception as e:
                logger.warning("Comparison CodeBERT vs %s failed: %s", baseline, e)

    logger.info("=" * 70)
    logger.info("All experiments complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
