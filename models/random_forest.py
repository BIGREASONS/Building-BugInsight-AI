"""BugInsight AI — Random Forest Severity Classifier.

Implements a TF-IDF → Random Forest pipeline for multi-class bug severity
prediction.  All hyperparameters are read from ``configs/config.yaml`` so that
experiments are fully reproducible and configurable without code changes.

Typical usage::

    from configs.config_loader import load_config
    from models.random_forest import RandomForestSeverityClassifier

    cfg = load_config()
    clf = RandomForestSeverityClassifier(cfg)
    clf.train(X_train, y_train)
    metrics = clf.evaluate(X_test, y_test)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from configs.config_loader import Config
from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class RandomForestSeverityClassifier:
    """TF-IDF + Random Forest pipeline for bug severity classification.

    Attributes:
        config: Global :class:`~configs.config_loader.Config` instance.
        pipeline: Fitted scikit-learn :class:`~sklearn.pipeline.Pipeline`
            containing the TF-IDF vectorizer and Random Forest estimator.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the classifier from the project configuration.

        Args:
            config: Loaded :class:`~configs.config_loader.Config` object.
        """
        self.config = config

        # ---- TF-IDF parameters ----
        tfidf_max_features: int = config.get(
            "models.random_forest.tfidf_max_features", 10_000,
        )
        tfidf_ngram_range: List[int] = config.get(
            "models.random_forest.tfidf_ngram_range", [1, 2],
        )

        # ---- Random Forest parameters ----
        n_estimators: int = config.get(
            "models.random_forest.n_estimators", 300,
        )
        max_depth: Optional[int] = config.get(
            "models.random_forest.max_depth", None,
        )
        min_samples_split: int = config.get(
            "models.random_forest.min_samples_split", 5,
        )

        seed: int = config.seed

        self.pipeline: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=tfidf_max_features,
                        ngram_range=tuple(tfidf_ngram_range),
                        sublinear_tf=True,
                        strip_accents="unicode",
                        dtype=np.float32,
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=seed,
                        n_jobs=-1,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

        logger.info(
            "RandomForestSeverityClassifier initialised — "
            "tfidf_max_features=%d, ngram_range=%s, n_estimators=%d, "
            "max_depth=%s, min_samples_split=%d, seed=%d",
            tfidf_max_features,
            tuple(tfidf_ngram_range),
            n_estimators,
            max_depth,
            min_samples_split,
            seed,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X_train: Sequence[str], y_train: Sequence[str]) -> None:
        """Fit the TF-IDF + Random Forest pipeline on training data.

        Args:
            X_train: Iterable of raw text documents (bug reports).
            y_train: Corresponding severity labels.
        """
        logger.info(
            "Training Random Forest on %d samples …", len(X_train),
        )
        self.pipeline.fit(X_train, y_train)
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Return predicted severity labels for the given texts.

        Args:
            X: Iterable of raw text documents.

        Returns:
            NumPy array of predicted label strings.
        """
        predictions: np.ndarray = self.pipeline.predict(X)
        logger.debug("Predicted %d samples.", len(predictions))
        return predictions

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Return class-probability estimates for the given texts.

        Args:
            X: Iterable of raw text documents.

        Returns:
            NumPy array of shape ``(n_samples, n_classes)``.
        """
        return self.pipeline.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X_test: Sequence[str],
        y_test: Sequence[str],
        label_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on a held-out test set.

        Args:
            X_test: Raw text documents for evaluation.
            y_test: Ground-truth severity labels.
            label_names: Optional ordered list of class names for the
                classification report.  Falls back to
                ``config.label_order``.

        Returns:
            Dictionary containing *accuracy*, *precision*, *recall*,
            *f1*, *per_class_f1*, and *classification_report*.
        """
        if label_names is None:
            label_names = self.config.label_order

        y_pred = self.predict(X_test)
        metrics = compute_metrics(
            y_true=list(y_test),
            y_pred=list(y_pred),
            label_names=label_names,
        )

        logger.info(
            "Random Forest evaluation — Accuracy: %.4f | F1 (macro): %.4f",
            metrics["accuracy"],
            metrics["f1"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Serialise the fitted pipeline to disk with :mod:`joblib`.

        Args:
            path: Destination file path.  When ``None`` the model is
                saved to ``<outputs.models_dir>/random_forest.joblib``.

        Returns:
            Resolved :class:`~pathlib.Path` to the saved file.
        """
        if path is None:
            models_dir = self.config.get_path("outputs.models_dir")
            models_dir.mkdir(parents=True, exist_ok=True)
            path = models_dir / "random_forest.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, path)
        logger.info("Model saved → %s", path)
        return path

    @classmethod
    def load_model(
        cls,
        path: Union[str, Path],
        config: Config,
    ) -> "RandomForestSeverityClassifier":
        """Load a previously saved pipeline from disk.

        Args:
            path: Path to the ``.joblib`` file.
            config: Loaded :class:`~configs.config_loader.Config` object
                (needed so the returned instance has access to project
                settings).

        Returns:
            A :class:`RandomForestSeverityClassifier` with the
            deserialised pipeline attached.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        instance = cls.__new__(cls)
        instance.config = config
        instance.pipeline = joblib.load(path)
        logger.info("Model loaded ← %s", path)
        return instance
