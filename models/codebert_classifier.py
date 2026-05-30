"""BugInsight AI — CodeBERT Severity Classifier.

Fine-tunes ``microsoft/codebert-base`` (or any compatible Hugging Face
encoder) for bug severity prediction.  Uses the ``[CLS]`` token
representation followed by dropout and a linear classification head.
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


class CodeBERTClassifier(nn.Module):
    """CodeBERT-based severity classifier.

    Architecture::

        CodeBERT encoder  →  [CLS] representation  →  Dropout  →  Linear  →  logits

    The encoder layers can be selectively frozen for transfer learning.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        num_classes: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_layers: Optional[int] = None,
    ) -> None:
        """Initialise the CodeBERT classifier.

        Args:
            model_name: Hugging Face model identifier for the pretrained
                encoder (e.g. ``"microsoft/codebert-base"``).
            num_classes: Number of output severity classes.
            dropout: Dropout probability for the classification head.
            freeze_encoder: If ``True``, freeze **all** encoder parameters.
                Mutually exclusive with *freeze_layers*.
            freeze_layers: If set, freeze the embedding layer and the first
                *freeze_layers* transformer blocks (0-indexed), leaving the
                remaining blocks and the classification head trainable.
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained encoder
        self.encoder_config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name, config=self.encoder_config
        )

        hidden_size: int = self.encoder_config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # Initialise classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # Apply freezing strategy
        if freeze_encoder:
            self._freeze_all_encoder()
        elif freeze_layers is not None:
            self._freeze_n_layers(freeze_layers)

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "CodeBERTClassifier (%s) | params: %s (trainable: %s)",
            model_name,
            f"{total_params:,}",
            f"{trainable:,}",
        )

    # -----------------------------------------------------------------
    # Freezing / unfreezing
    # -----------------------------------------------------------------

    def _freeze_all_encoder(self) -> None:
        """Freeze every parameter in the encoder (embeddings + all layers)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Froze ALL encoder parameters.")

    def _freeze_n_layers(self, n: int) -> None:
        """Freeze embeddings and the first *n* transformer layers.

        Args:
            n: Number of initial transformer layers to freeze (0-indexed).
        """
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze the first n encoder layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

        logger.info("Froze encoder embeddings + first %d transformer layers.", n)

    def unfreeze_all(self) -> None:
        """Unfreeze every parameter in the model (encoder + head)."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model parameters.")

    def unfreeze_encoder_from(self, layer_idx: int) -> None:
        """Unfreeze transformer layers from *layer_idx* onwards.

        Args:
            layer_idx: Index of the first layer to unfreeze (0-indexed).
        """
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i >= layer_idx:
                for param in layer.parameters():
                    param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Unfroze encoder layers %d+ | trainable params: %s",
            layer_idx,
            f"{trainable:,}",
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning raw logits.

        Args:
            input_ids: Token indices of shape ``(batch, seq_len)``.
            attention_mask: Binary mask of shape ``(batch, seq_len)``
                indicating real tokens (1) vs. padding (0).
            token_type_ids: Optional segment IDs; defaults to ``None``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        encoder_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        # [CLS] token representation is the first token in the sequence
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, H)

        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (B, num_classes)
        return logits

    # -----------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "CodeBERTClassifier":
        """Create a :class:`CodeBERTClassifier` from a :class:`Config` object.

        Args:
            config: BugInsight ``Config`` instance.

        Returns:
            Configured :class:`CodeBERTClassifier`.
        """
        model_name = config.get("models.codebert.model_name", "microsoft/codebert-base")
        num_classes = config.num_classes
        # CodeBERT default dropout comes from the pretrained config;
        # we allow overriding via our YAML.
        dropout = config.get("models.codebert.dropout", 0.1)

        return cls(
            model_name=model_name,
            num_classes=num_classes,
            dropout=dropout,
        )
