"""BugInsight AI — BiLSTM Training Script.

End-to-end pipeline: load config → load preprocessed data → build vocabulary
→ create DataLoaders → instantiate BiLSTMClassifier → train → evaluate →
save metrics.

Usage::

    python -m training.train_bilstm --seed 42
    python -m training.train_bilstm --seed 42 --config configs/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running as ``python -m``
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config  # noqa: E402
from models.bilstm import (  # noqa: E402
    BiLSTMClassifier,
    SimpleVocab,
    bilstm_collate_fn,
)
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


class BugSeverityDataset(Dataset):
    """PyTorch Dataset for bug severity classification (BiLSTM).

    Each sample is encoded as a 1-D tensor of token indices produced by
    :class:`SimpleVocab`.

    Attributes:
        texts: Raw text strings.
        labels: Integer severity labels.
        vocab: Vocabulary used for encoding.
        max_length: Maximum sequence length (truncation).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: SimpleVocab,
        max_length: int = 256,
    ) -> None:
        """Initialise the dataset.

        Args:
            texts: List of raw bug report texts.
            labels: Corresponding integer severity labels.
            vocab: Pre-built :class:`SimpleVocab` instance.
            max_length: Maximum token sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return ``(token_ids_tensor, label)`` for a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(LongTensor, int)``.
        """
        ids = self.vocab.encode(self.texts[idx], max_length=self.max_length)
        # Guarantee at least one token to avoid zero-length sequences
        if len(ids) == 0:
            ids = [1]  # <UNK>
        return torch.tensor(ids, dtype=torch.long), self.labels[idx]


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

    # Expect columns: 'text' (or 'description') and 'label' (or 'severity')
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
# Main entry point
# =========================================================================


def main() -> None:
    """Run the full BiLSTM training pipeline."""
    # ------------------------------------------------------------------
    # CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train BiLSTM severity classifier"
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
    logger.info("  BugInsight AI — BiLSTM Training (seed=%d)", args.seed)
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
    # Build vocabulary
    # ------------------------------------------------------------------
    max_vocab = config.get("models.bilstm.vocab_size", 30_000)
    vocab = SimpleVocab(max_vocab_size=max_vocab, min_freq=2)
    vocab.build(train_texts)

    max_length = config.get("models.bilstm.max_length", 256)

    # ------------------------------------------------------------------
    # Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = BugSeverityDataset(train_texts, train_labels, vocab, max_length)
    val_ds = BugSeverityDataset(val_texts, val_labels, vocab, max_length)
    test_ds = BugSeverityDataset(test_texts, test_labels, vocab, max_length)

    batch_size = config.get("models.bilstm.batch_size", 32)
    num_workers = config.get("training.num_workers", 4)
    pin_memory = config.get("training.pin_memory", True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=bilstm_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bilstm_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bilstm_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BiLSTMClassifier.from_config(config, vocab.vocab_size)
    logger.info("Model architecture:\n%s", model)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_name="bilstm",
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
        model_name="bilstm",
        seed=args.seed,
        project_root=config.project_root,
    )

    logger.info("=" * 60)
    logger.info("  BiLSTM training complete (seed=%d)", args.seed)
    logger.info(
        "  Test accuracy=%.4f | macro_f1=%.4f",
        metrics["accuracy"],
        metrics["macro_f1"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
