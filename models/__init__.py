"""BugInsight AI — Model architectures package.

Exposes the two deep-learning classifiers used for bug severity prediction:

* :class:`BiLSTMClassifier` — Bidirectional LSTM with attention pooling.
* :class:`CodeBERTClassifier` — Fine-tuned CodeBERT with linear head.

Supporting utilities:

* :class:`SimpleVocab` — Word-level vocabulary builder for BiLSTM.
* :func:`bilstm_collate_fn` — Collate function for variable-length sequences.
"""

from models.bilstm import BiLSTMClassifier, SimpleVocab, bilstm_collate_fn
from models.codebert_classifier import CodeBERTClassifier

__all__ = [
    "BiLSTMClassifier",
    "CodeBERTClassifier",
    "SimpleVocab",
    "bilstm_collate_fn",
]
