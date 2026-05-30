"""BugInsight AI — PyTorch Dataset and DataLoader utilities.

Provides :class:`BugReportDataset` for transformer-based training and
helper functions for both deep-learning and classical-ML data pipelines.

Usage (transformer)::

    from data.dataset import create_data_loaders
    train_dl, val_dl, test_dl = create_data_loaders(config)

Usage (classical ML)::

    from data.dataset import get_text_data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_text_data(config)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config_loader import Config, load_config

logger = logging.getLogger(__name__)


# ===================================================================
# PyTorch Dataset
# ===================================================================

class BugReportDataset(Dataset):
    """PyTorch Dataset for bug severity prediction.

    Each sample returns a dictionary with ``input_ids``,
    ``attention_mask``, and ``labels`` tensors, ready for a transformer
    encoder.

    Args:
        texts: List of text strings (title + description).
        labels: List of integer labels.
        tokenizer: HuggingFace-compatible tokenizer.
        max_length: Maximum number of tokens per sample.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must have "
                "the same length."
            )
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(
            "BugReportDataset created: %d samples, max_length=%d",
            len(self.texts),
            self.max_length,
        )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single tokenized sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with ``input_ids``, ``attention_mask``, and ``labels``.
        """
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ===================================================================
# Train / Val / Test Splitting
# ===================================================================

def _combine_text(row: pd.Series) -> str:
    """Combine title and description into a single input string.

    Args:
        row: DataFrame row with ``title`` and ``description`` columns.

    Returns:
        Combined text string.
    """
    title = str(row.get("title", "")).strip()
    desc = str(row.get("description", "")).strip()
    if title and desc:
        return f"{title} {desc}"
    return title or desc


def _encode_labels(
    series: pd.Series,
    label_order: List[str],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode string severity labels to integers.

    Args:
        series: Pandas Series of string severity labels.
        label_order: Ordered list of label names from config.

    Returns:
        Tuple of (integer-encoded array, label-to-int mapping).
    """
    label2id: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(label_order)}

    # Warn about unknown labels
    unknown = set(series.unique()) - set(label2id.keys())
    if unknown:
        logger.warning(
            "Found labels not in label_order — they will be dropped: %s", unknown
        )

    encoded = series.map(label2id)
    return encoded.values, label2id


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
    label_col: str = "severity",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train / val / test with stratification.

    Uses scikit-learn's ``train_test_split`` under the hood.

    Args:
        df: Input DataFrame.
        test_size: Fraction of data for the test set.
        val_size: Fraction of data for the validation set.
        seed: Random seed for reproducibility.
        label_col: Column to stratify on.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_col],
    )

    # Second split: train vs val (val_size is fraction of the *original* data)
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df[label_col],
    )

    logger.info(
        "Stratified split: train=%d, val=%d, test=%d (seed=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ===================================================================
# DataLoader factory (transformer pipeline)
# ===================================================================

def create_data_loaders(
    config: Optional[Config] = None,
    tokenizer: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for transformer training.

    Args:
        config: Configuration object.  Loads default if ``None``.
        tokenizer: HuggingFace tokenizer.  If ``None``, the CodeBERT
            tokenizer is loaded via :func:`data.tokenizer.get_tokenizer`.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if config is None:
        config = load_config()

    # Load processed CSV
    processed_path = config.get_path("dataset.processed_file")
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Run data.preprocess.preprocess_all() first."
        )
    df = pd.read_csv(processed_path)
    logger.info("Loaded processed data: %d rows from %s", len(df), processed_path)

    # Parameters
    label_order: List[str] = config.label_order
    seed: int = config.seed
    test_size: float = config.get("dataset.test_size", 0.15)
    val_size: float = config.get("dataset.val_size", 0.15)
    max_length: int = config.get("dataset.max_text_length", 512)
    batch_size: int = config.get("models.codebert.batch_size", 16)
    num_workers: int = config.get("training.num_workers", 4)
    pin_memory: bool = config.get("training.pin_memory", True)

    # Filter to known labels only
    df = df[df["severity"].isin(label_order)].reset_index(drop=True)
    logger.info("After filtering to known labels: %d rows", len(df))

    # Encode labels
    labels_array, label2id = _encode_labels(df["severity"], label_order)
    logger.info("Label mapping: %s", label2id)

    # Combine text fields
    texts = df.apply(_combine_text, axis=1).tolist()

    # Stratified split — we split indices and then slice
    train_df, val_df, test_df = stratified_split(
        df, test_size, val_size, seed
    )

    train_texts = train_df.apply(_combine_text, axis=1).tolist()
    val_texts = val_df.apply(_combine_text, axis=1).tolist()
    test_texts = test_df.apply(_combine_text, axis=1).tolist()

    train_labels = _encode_labels(train_df["severity"], label_order)[0].tolist()
    val_labels = _encode_labels(val_df["severity"], label_order)[0].tolist()
    test_labels = _encode_labels(test_df["severity"], label_order)[0].tolist()

    # Tokenizer
    if tokenizer is None:
        from data.tokenizer import get_tokenizer

        model_name = config.get("models.codebert.model_name", "microsoft/codebert-base")
        tokenizer = get_tokenizer(model_name)

    # Datasets
    train_ds = BugReportDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = BugReportDataset(val_texts, val_labels, tokenizer, max_length)
    test_ds = BugReportDataset(test_texts, test_labels, tokenizer, max_length)

    # DataLoaders
    from data.tokenizer import dynamic_padding_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dynamic_padding_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dynamic_padding_collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dynamic_padding_collate_fn,
        drop_last=False,
    )

    logger.info(
        "DataLoaders ready: train=%d batches, val=%d batches, test=%d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader


# ===================================================================
# Classical ML data accessor
# ===================================================================

def get_text_data(
    config: Optional[Config] = None,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """Return (texts, labels) tuples for classical ML (TF-IDF) pipelines.

    Args:
        config: Configuration object.  Loads default if ``None``.

    Returns:
        Tuple of three (texts, labels) pairs for train, val, and test.
        Labels are integer-encoded according to ``label_order`` in config.
    """
    if config is None:
        config = load_config()

    processed_path = config.get_path("dataset.processed_file")
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Run data.preprocess.preprocess_all() first."
        )
    df = pd.read_csv(processed_path)
    logger.info("Loaded processed data: %d rows from %s", len(df), processed_path)

    label_order: List[str] = config.label_order
    seed: int = config.seed
    test_size: float = config.get("dataset.test_size", 0.15)
    val_size: float = config.get("dataset.val_size", 0.15)

    # Filter to known labels
    df = df[df["severity"].isin(label_order)].reset_index(drop=True)

    # Stratified split
    train_df, val_df, test_df = stratified_split(
        df, test_size, val_size, seed
    )

    # Build (texts, labels) for each split
    def _extract(
        split_df: pd.DataFrame,
    ) -> Tuple[List[str], List[int]]:
        texts = split_df.apply(_combine_text, axis=1).tolist()
        labels_arr, _ = _encode_labels(split_df["severity"], label_order)
        return texts, labels_arr.tolist()

    train_data = _extract(train_df)
    val_data = _extract(val_df)
    test_data = _extract(test_df)

    logger.info(
        "Text data prepared: train=%d, val=%d, test=%d",
        len(train_data[0]),
        len(val_data[0]),
        len(test_data[0]),
    )
    return train_data, val_data, test_data
