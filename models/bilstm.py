"""BugInsight AI — BiLSTM Severity Classifier.

Implements a bidirectional LSTM with attention-based pooling for bug severity
prediction.  Includes a lightweight :class:`SimpleVocab` for building a
word-level vocabulary from training texts, and a :func:`bilstm_collate_fn`
for padding variable-length sequences within a batch.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

logger = logging.getLogger(__name__)

# Special token indices
PAD_IDX: int = 0
UNK_IDX: int = 1


# =========================================================================
# Vocabulary
# =========================================================================


class SimpleVocab:
    """Word-level vocabulary built from training texts.

    Attributes:
        word2idx: Mapping from word string to integer index.
        idx2word: Mapping from integer index to word string.
        word_freq: Raw word frequency counts from the training corpus.
    """

    _TOKENIZE_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^\s\w]")

    def __init__(
        self,
        max_vocab_size: int = 30_000,
        min_freq: int = 2,
    ) -> None:
        """Initialise a vocabulary builder.

        Args:
            max_vocab_size: Maximum number of tokens to retain (most frequent
                first, excluding special tokens).
            min_freq: Minimum corpus frequency for a token to be included.
        """
        self.max_vocab_size: int = max_vocab_size
        self.min_freq: int = min_freq

        # Special tokens always occupy the first two positions.
        self.word2idx: Dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}
        self.word_freq: Counter = Counter()

    # -----------------------------------------------------------------
    # Building
    # -----------------------------------------------------------------

    def build(self, texts: List[str]) -> "SimpleVocab":
        """Build the vocabulary from a list of raw text strings.

        Args:
            texts: Training-set texts (one string per sample).

        Returns:
            ``self`` for fluent chaining.
        """
        for text in texts:
            tokens = self._tokenize(text)
            self.word_freq.update(tokens)

        # Keep only tokens meeting the minimum frequency threshold
        qualified = [
            (word, freq)
            for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ]

        # Truncate to max_vocab_size (special tokens are additional)
        for word, _ in qualified[: self.max_vocab_size]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        logger.info(
            "Vocabulary built: %d tokens (from %d unique, min_freq=%d)",
            len(self.word2idx),
            len(self.word_freq),
            self.min_freq,
        )
        return self

    # -----------------------------------------------------------------
    # Encoding
    # -----------------------------------------------------------------

    def encode(
        self, text: str, max_length: Optional[int] = None
    ) -> List[int]:
        """Convert a raw text string to a list of integer indices.

        Args:
            text: Raw text string.
            max_length: If provided, truncate the output to this length.

        Returns:
            List of token indices.
        """
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Return the total number of tokens (including special tokens)."""
        return len(self.word2idx)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple regex tokenizer that splits on words, numbers, and punctuation.

        Args:
            text: Raw text string.

        Returns:
            List of lowercase token strings.
        """
        return [t.lower() for t in SimpleVocab._TOKENIZE_RE.findall(text)]


# =========================================================================
# Collate function
# =========================================================================


def bilstm_collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences into a padded batch.

    Each element in *batch* is a ``(token_ids_tensor, label)`` tuple.

    Args:
        batch: List of ``(sequence_tensor, label)`` pairs.

    Returns:
        Tuple of ``(padded_sequences, lengths, labels)`` where
        ``padded_sequences`` has shape ``(batch, max_seq_len)``.
    """
    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(s) for s in sequences], dtype=torch.long
    )

    # pad_sequence expects (T,) tensors; result is (max_T, B)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return padded, lengths, labels_tensor


# =========================================================================
# Attention layer
# =========================================================================


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM hidden states.

    Computes a weighted sum of encoder outputs using a learnable
    context vector.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Initialise the attention layer.

        Args:
            hidden_dim: Dimensionality of each LSTM output vector.
        """
        super().__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        lstm_output: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention-weighted representation.

        Args:
            lstm_output: LSTM outputs of shape ``(batch, seq_len, hidden_dim)``.
            lengths: Original (unpadded) sequence lengths of shape ``(batch,)``.

        Returns:
            Weighted representation of shape ``(batch, hidden_dim)``.
        """
        # Score each timestep: (B, T, H) -> (B, T, 1)
        energy = torch.tanh(self.attention(lstm_output))
        scores = self.context(energy).squeeze(-1)  # (B, T)

        # Mask padded positions so they receive zero attention weight
        max_len = lstm_output.size(1)
        mask = torch.arange(max_len, device=lstm_output.device).unsqueeze(0)
        mask = mask >= lengths.unsqueeze(1)  # True where padded
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=1)  # (B, T)

        # Weighted sum over timesteps: (B, 1, T) @ (B, T, H) -> (B, H)
        weighted = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return weighted


# =========================================================================
# BiLSTM classifier
# =========================================================================


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with attention pooling for severity classification.

    Architecture::

        Embedding → BiLSTM (N layers) → Attention Pooling → Dropout → FC → logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = PAD_IDX,
    ) -> None:
        """Initialise the BiLSTM classifier.

        Args:
            vocab_size: Size of the vocabulary (including special tokens).
            embedding_dim: Dimensionality of token embeddings.
            hidden_dim: Number of features in each LSTM direction.
            num_layers: Number of stacked LSTM layers.
            num_classes: Number of output severity classes.
            dropout: Dropout probability applied between layers.
            bidirectional: Whether to use a bidirectional LSTM.
            padding_idx: Index used for padding (zeroed embeddings).
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        direction_factor = 2 if bidirectional else 1
        lstm_output_dim = hidden_dim * direction_factor

        self.attention = Attention(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "BiLSTMClassifier | params: %s (trainable: %s)",
            f"{total_params:,}",
            f"{trainable:,}",
        )

    # -----------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Xavier/Kaiming initialisation to linear and LSTM weights."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name and "lstm" in name:
                # Forget-gate bias trick: set to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)

    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            input_ids: Padded token indices of shape ``(batch, seq_len)``.
            lengths: Unpadded sequence lengths of shape ``(batch,)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        embedded = self.embedding(input_ids)  # (B, T, E)

        # Pack, run LSTM, unpack
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            lstm_out, batch_first=True
        )  # (B, T, H*dir)

        # Attention pooling over timesteps
        attn_out = self.attention(lstm_out, lengths)  # (B, H*dir)

        out = self.dropout(attn_out)
        logits = self.fc(out)  # (B, num_classes)
        return logits

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Any,
        vocab_size: int,
    ) -> "BiLSTMClassifier":
        """Create a :class:`BiLSTMClassifier` from a :class:`Config` object.

        Args:
            config: BugInsight ``Config`` instance.
            vocab_size: Actual vocabulary size (from :class:`SimpleVocab`).

        Returns:
            Configured :class:`BiLSTMClassifier`.
        """
        return cls(
            vocab_size=vocab_size,
            embedding_dim=config.get("models.bilstm.embedding_dim", 128),
            hidden_dim=config.get("models.bilstm.hidden_dim", 256),
            num_layers=config.get("models.bilstm.num_layers", 2),
            num_classes=config.num_classes,
            dropout=config.get("models.bilstm.dropout", 0.3),
            bidirectional=config.get("models.bilstm.bidirectional", True),
        )
