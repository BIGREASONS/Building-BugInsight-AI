"""BugInsight AI — XGBoost Severity Classifier.

Implements a TF-IDF → XGBoost pipeline for multi-class bug severity
prediction.  All hyperparameters are read from ``configs/config.yaml`` so that
experiments are fully reproducible and configurable without code changes.

Typical usage::

    from configs.config_loader import load_config
    from models.xgboost_model import XGBoostSeverityClassifier

    cfg = load_config()
    clf = XGBoostSeverityClassifier(cfg)
    clf.train(X_train, y_train)
    metrics = clf.evaluate(X_test, y_test)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from configs.config_loader import Config
from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class XGBoostSeverityClassifier:
    """TF-IDF + XGBoost pipeline for bug severity classification.

    Because :class:`~xgboost.XGBClassifier` requires integer-encoded
    labels, this class maintains an internal :class:`LabelEncoder` that
    maps severity strings to integers and back.

    Attributes:
        config: Global :class:`~configs.config_loader.Config` instance.
        pipeline: Fitted scikit-learn :class:`~sklearn.pipeline.Pipeline`
            containing the TF-IDF vectorizer and XGBoost estimator.
        label_encoder: Encoder that maps severity labels ↔ integers.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the classifier from the project configuration.

        Args:
            config: Loaded :class:`~configs.config_loader.Config` object.
        """
        self.config = config

        # ---- TF-IDF parameters ----
        tfidf_max_features: int = config.get(
            "models.xgboost.tfidf_max_features", 10_000,
        )
        tfidf_ngram_range: List[int] = config.get(
            "models.xgboost.tfidf_ngram_range", [1, 2],
        )

        # ---- XGBoost parameters ----
        n_estimators: int = config.get(
            "models.xgboost.n_estimators", 300,
        )
        max_depth: int = config.get("models.xgboost.max_depth", 6)
        learning_rate: float = config.get(
            "models.xgboost.learning_rate", 0.1,
        )
        subsample: float = config.get("models.xgboost.subsample", 0.8)
        colsample_bytree: float = config.get(
            "models.xgboost.colsample_bytree", 0.8,
        )

        seed: int = config.seed

        self.label_encoder: LabelEncoder = LabelEncoder()

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
                    XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=seed,
                        use_label_encoder=False,
                        eval_metric="mlogloss",
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        )

        logger.info(
            "XGBoostSeverityClassifier initialised — "
            "tfidf_max_features=%d, ngram_range=%s, n_estimators=%d, "
            "max_depth=%d, learning_rate=%.4f, subsample=%.2f, "
            "colsample_bytree=%.2f, seed=%d",
            tfidf_max_features,
            tuple(tfidf_ngram_range),
            n_estimators,
            max_depth,
            learning_rate,
            subsample,
            colsample_bytree,
            seed,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X_train: Sequence[str], y_train: Sequence[str]) -> None:
        """Fit the TF-IDF + XGBoost pipeline on training data.

        String labels are automatically integer-encoded via the internal
        :class:`LabelEncoder`.

        Args:
            X_train: Iterable of raw text documents (bug reports).
            y_train: Corresponding severity labels (strings).
        """
        logger.info(
            "Training XGBoost on %d samples …", len(X_train),
        )
        y_encoded = self.label_encoder.fit_transform(y_train)
        self.pipeline.fit(X_train, y_encoded)
        logger.info(
            "Training complete. Classes: %s",
            list(self.label_encoder.classes_),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: Sequence[str]) -> np.ndarray:
        """Return predicted severity labels (strings) for the given texts.

        Args:
            X: Iterable of raw text documents.

        Returns:
            NumPy array of predicted label strings.
        """
        y_encoded: np.ndarray = self.pipeline.predict(X)
        predictions: np.ndarray = self.label_encoder.inverse_transform(
            y_encoded.astype(int),
        )
        logger.debug("Predicted %d samples.", len(predictions))
        return predictions

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        """Return class-probability estimates for the given texts.

        The columns correspond to the classes in
        ``self.label_encoder.classes_``.

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
            y_test: Ground-truth severity labels (strings).
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
            "XGBoost evaluation — Accuracy: %.4f | F1 (macro): %.4f",
            metrics["accuracy"],
            metrics["f1"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Serialise the fitted pipeline and label encoder with :mod:`joblib`.

        Args:
            path: Destination file path.  When ``None`` the model is
                saved to ``<outputs.models_dir>/xgboost.joblib``.

        Returns:
            Resolved :class:`~pathlib.Path` to the saved file.
        """
        if path is None:
            models_dir = self.config.get_path("outputs.models_dir")
            models_dir.mkdir(parents=True, exist_ok=True)
            path = models_dir / "xgboost.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        artefact = {
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
        }
        joblib.dump(artefact, path)
        logger.info("Model saved → %s", path)
        return path

    @classmethod
    def load_model(
        cls,
        path: Union[str, Path],
        config: Config,
    ) -> "XGBoostSeverityClassifier":
        """Load a previously saved pipeline from disk.

        Args:
            path: Path to the ``.joblib`` file.
            config: Loaded :class:`~configs.config_loader.Config` object
                (needed so the returned instance has access to project
                settings).

        Returns:
            An :class:`XGBoostSeverityClassifier` with the deserialised
            pipeline and label encoder attached.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        artefact = joblib.load(path)
        instance = cls.__new__(cls)
        instance.config = config
        instance.pipeline = artefact["pipeline"]
        instance.label_encoder = artefact["label_encoder"]
        logger.info("Model loaded ← %s", path)
        return instance
