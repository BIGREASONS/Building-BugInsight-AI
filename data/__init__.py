"""BugInsight AI — Data loading, preprocessing, and dataset package.

Exposes key classes and functions for the data pipeline:

- **Preprocessing**: :func:`preprocess.preprocess_all`
- **Dataset**: :class:`dataset.BugReportDataset`,
  :func:`dataset.create_data_loaders`, :func:`dataset.get_text_data`
- **Tokenizer**: :func:`tokenizer.get_tokenizer`,
  :func:`tokenizer.dynamic_padding_collate_fn`
"""

from data.dataset import BugReportDataset, create_data_loaders, get_text_data
from data.preprocess import preprocess_all
from data.tokenizer import dynamic_padding_collate_fn, get_tokenizer

__all__ = [
    "BugReportDataset",
    "create_data_loaders",
    "dynamic_padding_collate_fn",
    "get_text_data",
    "get_tokenizer",
    "preprocess_all",
]
