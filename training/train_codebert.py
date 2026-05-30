"""BugInsight AI — CodeBERT Training Script.

End-to-end pipeline: load config → load preprocessed data → tokenise with
Hugging Face tokenizer → create DataLoaders → instantiate CodeBERTClassifier
→ train → evaluate → save metrics.

Supports Hugging Face Accelerate for multi-GPU / mixed-precision when
available.

Usage::

    python -m training.train_codebert --seed 42
    python -m training.train_codebert --seed 42 --config configs/config.yaml

Multi-GPU with Accelerate::

    accelerate launch -m training.train_codebert --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config  # noqa: E402
from models.codebert_classifier import CodeBERTClassifier  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from utils import (  # noqa: E402
    ensure_output_dirs,
    get_device,
    save_metrics,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Dataset
# =========================================================================


class CodeBERTBugDataset(Dataset):
    """PyTorch Dataset that tokenises texts for CodeBERT on the fly.

    Pre-tokenises all samples at construction time to avoid repeated work.

    Attributes:
        encodings: Dictionary of tokenizer outputs (``input_ids``,
            ``attention_mask``).
        labels: Integer severity labels.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        """Initialise the dataset.

        Args:
            texts: List of raw bug report texts.
            labels: Corresponding integer severity labels.
            tokenizer: Hugging Face tokenizer instance.
            max_length: Maximum sequence length (padding + truncation).
        """
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        logger.info(
            "Tokenised %d samples (max_length=%d)", len(texts), max_length
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a dict with ``input_ids``, ``attention_mask``, ``labels``.

        Args:
            idx: Sample index.

        Returns:
            Dictionary consumable by :class:`CodeBERTClassifier`.
        """
        item: Dict[str, torch.Tensor] = {
            key: val[idx] for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =========================================================================
# Data loading helpers
# =========================================================================


def _load_split(filepath: Path) -> Tuple[List[str], List[int]]:
    """Load a single CSV/Parquet split and return texts + labels.

    Tries ``.parquet`` first, then ``.csv``.

    Args:
        filepath: Path without extension — the function appends
            ``.parquet`` / ``.csv`` as needed.

    Returns:
        Tuple of ``(texts, labels)`` as plain Python lists.

    Raises:
        FileNotFoundError: If neither format exists.
    """
    parquet = filepath.with_suffix(".parquet")
    csv = filepath.with_suffix(".csv")

    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"No data file found at {parquet} or {csv}"
        )

    text_col = "text" if "text" in df.columns else "description"
    label_col = "label" if "label" in df.columns else "severity"

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"Expected columns '{text_col}' and '{label_col}' in {filepath}; "
            f"found {list(df.columns)}"
        )

    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


def load_data(
    config: object,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Load train / val / test splits from the processed data directory.

    Args:
        config: BugInsight ``Config`` instance.

    Returns:
        Six-tuple of ``(train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels)``.
    """
    processed_dir = config.get_path("dataset.processed_dir")
    logger.info("Loading data splits from %s", processed_dir)

    train_texts, train_labels = _load_split(processed_dir / "train")
    val_texts, val_labels = _load_split(processed_dir / "val")
    test_texts, test_labels = _load_split(processed_dir / "test")

    logger.info(
        "Data loaded — train=%d | val=%d | test=%d",
        len(train_texts),
        len(val_texts),
        len(test_texts),
    )
    return (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
    )


# =========================================================================
# Accelerate wrapper
# =========================================================================


def _try_get_accelerator() -> Optional[Any]:
    """Attempt to create a Hugging Face ``Accelerator``.

    Returns:
        ``Accelerator`` instance or ``None`` if the library is unavailable
        or initialisation fails.
    """
    try:
        from accelerate import Accelerator

        accelerator = Accelerator()
        logger.info(
            "Accelerate available — device=%s | num_processes=%d | "
            "mixed_precision=%s",
            accelerator.device,
            accelerator.num_processes,
            accelerator.mixed_precision,
        )
        return accelerator
    except ImportError:
        logger.info(
            "Accelerate not installed; falling back to single-device training."
        )
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("Accelerate init failed (%s); falling back.", exc)
        return None


# =========================================================================
# Main entry point
# =========================================================================


def main() -> None:
    """Run the full CodeBERT training pipeline."""
    # ------------------------------------------------------------------
    # CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train CodeBERT severity classifier"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    config = load_config(args.config)
    setup_logging(
        log_dir=config.get("logging.log_dir", "outputs/logs"),
        level=config.get("logging.level", "INFO"),
        log_to_file=config.get("logging.log_to_file", True),
        log_to_console=config.get("logging.log_to_console", True),
        project_root=config.project_root,
    )
    ensure_output_dirs(config.project_root)
    set_seed(args.seed)

    logger.info("=" * 60)
    logger.info("  BugInsight AI — CodeBERT Training (seed=%d)", args.seed)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        test_texts,
        test_labels,
    ) = load_data(config)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    model_name = config.get(
        "models.codebert.model_name", "microsoft/codebert-base"
    )
    max_length = config.get("models.codebert.max_length", 512)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(
        "Tokenizer loaded: %s (vocab_size=%d)", model_name, tokenizer.vocab_size
    )

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = CodeBERTBugDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    val_ds = CodeBERTBugDataset(
        val_texts, val_labels, tokenizer, max_length
    )
    test_ds = CodeBERTBugDataset(
        test_texts, test_labels, tokenizer, max_length
    )

    batch_size = config.get("models.codebert.batch_size", 16)
    num_workers = config.get("training.num_workers", 4)
    pin_memory = config.get("training.pin_memory", True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = CodeBERTClassifier.from_config(config)
    logger.info("Model architecture:\n%s", model)

    # ------------------------------------------------------------------
    # Optional: Accelerate wrapping for multi-GPU
    # ------------------------------------------------------------------
    accelerator = _try_get_accelerator()

    if accelerator is not None:
        # Accelerate will handle device placement, AMP, and DDP
        # We still use our Trainer but let Accelerate wrap the objects
        model, train_loader, val_loader, test_loader = (
            accelerator.prepare(model, train_loader, val_loader, test_loader)
        )
        logger.info("Model and DataLoaders wrapped with Accelerate.")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name="codebert",
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    history = trainer.train()

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    y_true, y_pred, metrics = trainer.evaluate(test_loader)

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    save_metrics(
        metrics=metrics,
        model_name="codebert",
        seed=args.seed,
        project_root=config.project_root,
    )

    logger.info("=" * 60)
    logger.info("  CodeBERT training complete (seed=%d)", args.seed)
    logger.info(
        "  Test accuracy=%.4f | macro_f1=%.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
