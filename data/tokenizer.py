"""BugInsight AI — Tokenizer Utilities for Transformer Models.

Wraps HuggingFace ``AutoTokenizer`` for CodeBERT and provides a
dynamic-padding collate function that reduces wasted computation
by padding each batch only to its longest sequence.

Usage::

    from data.tokenizer import get_tokenizer, dynamic_padding_collate_fn
    tokenizer = get_tokenizer("microsoft/codebert-base")
"""

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def get_tokenizer(
    model_name: str = "microsoft/codebert-base",
    cache_dir: Optional[str] = None,
) -> Any:
    """Load and return a HuggingFace tokenizer.

    The tokenizer is downloaded on first use and cached locally.  This
    function is thin wrapper that standardises tokenizer acquisition
    across the project.

    Args:
        model_name: HuggingFace model identifier or local path.
        cache_dir: Optional directory to cache the tokenizer files.

    Returns:
        A ``transformers.PreTrainedTokenizer`` instance.

    Raises:
        ImportError: If the ``transformers`` package is not installed.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for tokenizer loading. "
            "Install it with: pip install transformers"
        ) from exc

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=True,
    )
    logger.info(
        "Tokenizer loaded: vocab_size=%d, model_max_length=%d",
        tokenizer.vocab_size,
        tokenizer.model_max_length,
    )
    return tokenizer


def dynamic_padding_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function that pads each batch to its longest sequence.

    Instead of padding every sample to ``max_length`` (which wastes
    memory and FLOPS), this function truncates padding to the length of
    the longest sample *within the batch*.

    Expected input format per sample::

        {
            "input_ids":      Tensor[seq_len],
            "attention_mask": Tensor[seq_len],
            "labels":         Tensor[],          # scalar
        }

    Args:
        batch: List of sample dictionaries from
            :class:`data.dataset.BugReportDataset`.

    Returns:
        Batched dictionary with padded ``input_ids`` and
        ``attention_mask`` and stacked ``labels``.
    """
    # Determine the actual max length in this batch
    max_len = max(sample["input_ids"].size(0) for sample in batch)

    input_ids_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for sample in batch:
        seq_len = sample["input_ids"].size(0)
        pad_len = max_len - seq_len

        if pad_len > 0:
            # Pad with zeros (standard pad token id for most tokenizers)
            input_ids = torch.cat(
                [sample["input_ids"], torch.zeros(pad_len, dtype=torch.long)]
            )
            attention_mask = torch.cat(
                [sample["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
        else:
            input_ids = sample["input_ids"][:max_len]
            attention_mask = sample["attention_mask"][:max_len]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(sample["labels"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }
