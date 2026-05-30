"""BugInsight AI — Unified Evaluation Entry Point.

Runs inference on a trained model checkpoint and generates metrics, confusion
matrices, and classification reports.  Supports all four baseline models.

Usage::

    python evaluate.py --model codebert --seed 42
    python evaluate.py --model random_forest --seed 42
    python evaluate.py --model xgboost --seed 42
    python evaluate.py --model bilstm --seed 42
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from configs.config_loader import load_config
from evaluation.metrics import (
    compute_metrics,
    generate_class_distribution,
    generate_confusion_matrix,
)
from utils import ensure_output_dirs, set_seed, setup_logging

logger = logging.getLogger(__name__)


# =========================================================================
# Classical model evaluation
# =========================================================================

def _evaluate_classical(model_name: str, config) -> dict:
    """Evaluate a classical ML model (RF or XGBoost).

    Args:
        model_name: ``"random_forest"`` or ``"xgboost"``.
        config: Loaded :class:`Config` instance.

    Returns:
        Dictionary of computed metrics.
    """
    from data.dataset import get_text_data

    (_, _), (_, _), (X_test, y_test) = get_text_data(config)

    model_path = config.get_path("outputs.models_dir") / f"{model_name}_best.joblib"

    if model_name == "random_forest":
        from models.random_forest import RandomForestSeverityClassifier
        model = RandomForestSeverityClassifier.load_model(str(model_path), config)
    elif model_name == "xgboost":
        from models.xgboost_model import XGBoostSeverityClassifier
        model = XGBoostSeverityClassifier.load_model(str(model_path), config)
    else:
        raise ValueError(f"Unknown classical model: {model_name}")

    metrics = model.evaluate(X_test, y_test)
    return metrics


# =========================================================================
# Deep learning model evaluation
# =========================================================================

def _evaluate_deep_learning(model_name: str, seed: int, config) -> dict:
    """Evaluate a deep learning model (BiLSTM or CodeBERT).

    Args:
        model_name: ``"bilstm"`` or ``"codebert"``.
        seed: Random seed identifying the checkpoint.
        config: Loaded :class:`Config` instance.

    Returns:
        Dictionary of computed metrics.
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = config.get_path("outputs.models_dir")

    if model_name == "bilstm":
        from models.bilstm import BiLSTMClassifier, SimpleVocab, bilstm_collate_fn
        from data.dataset import get_text_data

        (_, _), (_, _), (X_test, y_test) = get_text_data(config)
        label_order = config.label_order

        # Load checkpoint
        ckpt_path = checkpoint_dir / f"bilstm_seed{seed}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = checkpoint_dir / f"bilstm_seed{seed}_last.pt"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Rebuild vocab from checkpoint if saved, otherwise from data
        vocab = SimpleVocab(
            max_vocab_size=config.get("models.bilstm.vocab_size", 30000)
        )
        if "vocab" in checkpoint:
            vocab.word2idx = checkpoint["vocab"]["word2idx"]
            vocab.idx2word = checkpoint["vocab"]["idx2word"]
        else:
            from data.dataset import get_text_data as _gtd
            (X_train, _), (_, _), (_, _) = _gtd(config)
            vocab.build(X_train)

        # Prepare test data
        label2idx = {label: idx for idx, label in enumerate(label_order)}
        max_len = config.get("models.bilstm.max_length", 256)

        test_ids = []
        test_labels = []
        for text, label in zip(X_test, y_test):
            tokens = vocab.tokenize(text)[:max_len]
            ids = vocab.encode(tokens)
            test_ids.append(ids)
            test_labels.append(label2idx[label])

        from torch.utils.data import TensorDataset

        # Manual padding for evaluation
        padded = torch.zeros(len(test_ids), max_len, dtype=torch.long)
        lengths = []
        for i, ids in enumerate(test_ids):
            length = min(len(ids), max_len)
            padded[i, :length] = torch.tensor(ids[:length], dtype=torch.long)
            lengths.append(length)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        labels_tensor = torch.tensor(test_labels, dtype=torch.long)

        test_dataset = torch.utils.data.TensorDataset(
            padded, lengths_tensor, labels_tensor
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Load model
        model = BiLSTMClassifier.from_config(config, vocab_size=len(vocab))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                ids_batch, lens_batch, labs_batch = batch
                ids_batch = ids_batch.to(device)
                lens_batch = lens_batch.to(device)
                logits = model(ids_batch, lens_batch)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labs_batch.numpy().tolist())

        y_true = [label_order[i] for i in all_labels]
        y_pred = [label_order[i] for i in all_preds]
        metrics = compute_metrics(y_true, y_pred, label_order)

    elif model_name == "codebert":
        from models.codebert_classifier import CodeBERTClassifier
        from data.dataset import create_data_loaders
        from data.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            config.get("models.codebert.model_name", "microsoft/codebert-base")
        )
        _, _, test_loader = create_data_loaders(config, tokenizer)

        # Load checkpoint
        ckpt_path = checkpoint_dir / f"codebert_seed{seed}_best.pt"
        if not ckpt_path.exists():
            ckpt_path = checkpoint_dir / f"codebert_seed{seed}_last.pt"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = CodeBERTClassifier.from_config(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        label_order = config.label_order
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())

        y_true = [label_order[i] for i in all_labels]
        y_pred = [label_order[i] for i in all_preds]
        metrics = compute_metrics(y_true, y_pred, label_order)

    else:
        raise ValueError(f"Unknown deep learning model: {model_name}")

    return metrics, y_true, y_pred


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="BugInsight AI — Model Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["random_forest", "xgboost", "bilstm", "codebert"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed identifying the checkpoint (for DL models).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        log_dir=config.get("logging.log_dir", "outputs/logs"),
        level=config.get("logging.level", "INFO"),
    )
    set_seed(args.seed)
    output_dirs = ensure_output_dirs(config.project_root)

    logger.info("=" * 60)
    logger.info("BugInsight AI — Evaluating: %s (seed=%d)", args.model, args.seed)
    logger.info("=" * 60)

    label_order = config.label_order

    if args.model in ("random_forest", "xgboost"):
        metrics = _evaluate_classical(args.model, config)
        # For classical models, re-run prediction to get y_true/y_pred for plots
        from data.dataset import get_text_data
        (_, _), (_, _), (X_test, y_test) = get_text_data(config)
        if args.model == "random_forest":
            from models.random_forest import RandomForestSeverityClassifier
            model_path = config.get_path("outputs.models_dir") / f"{args.model}_best.joblib"
            model = RandomForestSeverityClassifier.load_model(str(model_path), config)
        else:
            from models.xgboost_model import XGBoostSeverityClassifier
            model_path = config.get_path("outputs.models_dir") / f"{args.model}_best.joblib"
            model = XGBoostSeverityClassifier.load_model(str(model_path), config)
        y_pred = model.predict(X_test)
        y_true = y_test
    else:
        metrics, y_true, y_pred = _evaluate_deep_learning(
            args.model, args.seed, config
        )

    # Save metrics
    from utils import save_metrics
    save_metrics(metrics, args.model, args.seed, project_root=config.project_root)

    # Generate confusion matrix
    cm_path = output_dirs["figures"] / f"confusion_matrix_{args.model}_seed{args.seed}.png"
    generate_confusion_matrix(y_true, y_pred, label_order, str(cm_path))

    # Log results
    logger.info("Results for %s (seed=%d):", args.model, args.seed)
    logger.info("  Accuracy:  %.4f", metrics["accuracy"])
    logger.info("  Precision: %.4f", metrics["precision"])
    logger.info("  Recall:    %.4f", metrics["recall"])
    logger.info("  F1:        %.4f", metrics["f1"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
