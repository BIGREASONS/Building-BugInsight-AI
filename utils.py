"""BugInsight AI — Reproducibility and experiment utilities.

Provides helpers for seeding, output directory management, logging setup,
and experiment result persistence.
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("All random seeds set to %d", seed)


def setup_logging(
    log_dir: str = "outputs/logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    project_root: Optional[Path] = None,
) -> None:
    """Configure the root logger for the project.

    Args:
        log_dir: Directory for log files (relative to project root).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_to_file: Whether to write logs to a timestamped file.
        log_to_console: Whether to write logs to stdout.
        project_root: Project root for resolving relative paths.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_to_file:
        resolved_log_dir = (project_root / log_dir).resolve()
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = resolved_log_dir / f"buginsight_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info("Log file: %s", log_file)


def ensure_output_dirs(project_root: Optional[Path] = None) -> Dict[str, Path]:
    """Create and return all output directories.

    Args:
        project_root: Project root for resolving relative paths.

    Returns:
        Dictionary mapping directory names to resolved paths.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    dirs = {
        "metrics": project_root / "outputs" / "metrics",
        "models": project_root / "outputs" / "models",
        "logs": project_root / "outputs" / "logs",
        "figures": project_root / "outputs" / "figures",
    }

    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)

    logger.info("Output directories verified under %s/outputs/", project_root)
    return dirs


def save_metrics(
    metrics: Dict[str, Any],
    model_name: str,
    seed: int,
    output_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
) -> Path:
    """Save experiment metrics to a JSON file.

    The file is named ``{model_name}_seed{seed}.json`` and placed in the
    metrics output directory.

    Args:
        metrics: Dictionary of metric name → value.
        model_name: Name of the model (e.g., ``"codebert"``).
        seed: Random seed used for this run.
        output_dir: Explicit output directory.  When ``None``, defaults to
            ``outputs/metrics/``.
        project_root: Project root for resolving relative paths.

    Returns:
        Path to the saved JSON file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    if output_dir is None:
        output_dir = project_root / "outputs" / "metrics"

    output_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "model": model_name,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    filename = f"{model_name}_seed{seed}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)

    logger.info("Metrics saved → %s", filepath)
    return filepath


def get_device() -> str:
    """Return the best available device string ('cuda' or 'cpu').

    Returns:
        Device string suitable for ``torch.device()``.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info("Using GPU: %s (x%d)", gpu_name, gpu_count)
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
        return device
    except ImportError:
        logger.warning("PyTorch not installed, defaulting to CPU")
        return "cpu"
