"""BugInsight AI — Unified PyTorch Trainer.

Provides a single :class:`Trainer` class that handles the training loop for
both the BiLSTM and CodeBERT models.  Supports:

* Mixed-precision training (``torch.cuda.amp``) when a CUDA device is present.
* Gradient clipping.
* Learning-rate scheduling (linear warmup + decay for transformer models,
  ``ReduceLROnPlateau`` for LSTM models).
* Early stopping with configurable patience.
* Checkpoint saving / resuming.
* Training-history tracking for downstream plotting.
"""

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =========================================================================
# History container
# =========================================================================


class TrainingHistory:
    """Accumulates per-epoch metrics for later plotting / analysis."""

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []

    def record(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Append one epoch of metrics.

        Args:
            train_loss: Average training loss for the epoch.
            train_acc: Training accuracy for the epoch.
            val_loss: Average validation loss for the epoch.
            val_acc: Validation accuracy for the epoch.
            lr: Learning rate at end of epoch.
        """
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)

    def to_dict(self) -> Dict[str, List[float]]:
        """Serialise history to a plain dictionary.

        Returns:
            Dictionary with lists of per-epoch values.
        """
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
        }


# =========================================================================
# LR schedule helpers
# =========================================================================


def _get_linear_warmup_decay_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a schedule with linear warmup then linear decay to 0.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Steps in the warmup phase.
        num_training_steps: Total training steps.

    Returns:
        ``LambdaLR`` scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


# =========================================================================
# Trainer
# =========================================================================


class Trainer:
    """Unified training loop for BiLSTM and CodeBERT models.

    Args:
        model: PyTorch model instance.
        train_loader: Training :class:`~torch.utils.data.DataLoader`.
        val_loader: Validation :class:`~torch.utils.data.DataLoader`.
        config: BugInsight ``Config`` instance.
        model_name: ``"bilstm"`` or ``"codebert"`` — selects optimiser,
            scheduler, and mixed-precision defaults.
        seed: Random seed used for this run.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        model_name: str,
        seed: int,
    ) -> None:
        self.config = config
        self.model_name = model_name
        self.seed = seed

        # Device
        device_cfg = config.get("training.device", "auto")
        if device_cfg == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device_cfg)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Model-specific hyper-parameters
        model_cfg_key = f"models.{model_name}"
        self.epochs: int = config.get(f"{model_cfg_key}.epochs", 10)
        self.patience: int = config.get(f"{model_cfg_key}.patience", 5)
        self.lr: float = config.get(f"{model_cfg_key}.learning_rate", 1e-3)
        self.grad_accum_steps: int = config.get(
            f"{model_cfg_key}.gradient_accumulation_steps", 1
        )

        # Training config
        self.clip_norm: float = config.get(
            "training.gradient_clip_max_norm", 1.0
        )
        self.checkpoint_dir = config.get_path("training.checkpoint_dir")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision — only on CUDA and when fp16 is requested
        self.use_amp: bool = (
            self.device.type == "cuda"
            and config.get(f"{model_cfg_key}.fp16", False)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimiser & scheduler (model-specific)
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any  # LambdaLR or ReduceLROnPlateau
        self._setup_optimizer_and_scheduler()

        # Early stopping state
        self.best_val_loss: float = float("inf")
        self.best_val_acc: float = 0.0
        self.epochs_without_improvement: int = 0
        self.best_model_state: Optional[Dict[str, Any]] = None

        # History
        self.history = TrainingHistory()

        # Resume support
        self.start_epoch: int = 0
        resume: bool = config.get("training.resume", False)
        resume_ckpt: Optional[str] = config.get("training.resume_checkpoint")
        if resume and resume_ckpt is not None:
            self._load_checkpoint(Path(resume_ckpt))

        logger.info(
            "Trainer initialised | model=%s | device=%s | epochs=%d | "
            "lr=%.2e | amp=%s | patience=%d",
            model_name,
            self.device,
            self.epochs,
            self.lr,
            self.use_amp,
            self.patience,
        )

    # -----------------------------------------------------------------
    # Optimiser / scheduler setup
    # -----------------------------------------------------------------

    def _setup_optimizer_and_scheduler(self) -> None:
        """Configure optimiser and LR scheduler based on model type."""
        if self.model_name == "codebert":
            weight_decay = self.config.get(
                "models.codebert.weight_decay", 0.01
            )
            # AdamW with decoupled weight decay for transformer
            no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
            param_groups = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if p.requires_grad
                        and not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if p.requires_grad
                        and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(param_groups, lr=self.lr)

            total_steps = (
                len(self.train_loader)
                // self.grad_accum_steps
                * self.epochs
            )
            warmup_ratio = self.config.get(
                "models.codebert.warmup_ratio", 0.1
            )
            warmup_steps = int(total_steps * warmup_ratio)
            self.scheduler = _get_linear_warmup_decay_schedule(
                self.optimizer, warmup_steps, total_steps
            )
            self._scheduler_type = "step"
            logger.info(
                "CodeBERT optimiser: AdamW | total_steps=%d | "
                "warmup_steps=%d",
                total_steps,
                warmup_steps,
            )
        else:
            # BiLSTM: plain Adam + ReduceLROnPlateau
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.lr,
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                verbose=False,
            )
            self._scheduler_type = "epoch"
            logger.info("BiLSTM optimiser: Adam + ReduceLROnPlateau")

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    def train(self) -> TrainingHistory:
        """Run the full training loop.

        Returns:
            :class:`TrainingHistory` with per-epoch metrics.
        """
        logger.info(
            "Starting training for %d epochs (from epoch %d)",
            self.epochs,
            self.start_epoch,
        )

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.record(train_loss, train_acc, val_loss, val_acc, current_lr)

            # LR scheduling
            if self._scheduler_type == "epoch":
                self.scheduler.step(val_loss)

            elapsed = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d [%.1fs] — "
                "train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f | lr=%.2e",
                epoch + 1,
                self.epochs,
                elapsed,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
            )

            # Check for improvement
            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )
                self._save_checkpoint(epoch, is_best=True)
                logger.info(
                    "  ↳ New best model (val_loss=%.4f, val_acc=%.4f)",
                    val_loss,
                    val_acc,
                )
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    "  ↳ No improvement for %d epoch(s)",
                    self.epochs_without_improvement,
                )

            # Save latest checkpoint every epoch
            self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without "
                    "improvement.",
                    self.patience,
                )
                break

        # Restore best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                "Restored best model weights (val_loss=%.4f, val_acc=%.4f)",
                self.best_val_loss,
                self.best_val_acc,
            )

        return self.history

    # -----------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Tuple of ``(avg_loss, accuracy)`` for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            loss, batch_correct, batch_total = self._forward_batch(batch)

            # Scale loss for gradient accumulation
            loss = loss / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps
            correct += batch_correct
            total += batch_total

            # Gradient accumulation step
            if (step + 1) % self.grad_accum_steps == 0 or (
                step + 1
            ) == len(self.train_loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Per-step scheduling for transformers
                if self._scheduler_type == "step":
                    self.scheduler.step()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # -----------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """Run a validation pass over the entire validation set.

        Returns:
            Tuple of ``(avg_loss, accuracy)``.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            loss, batch_correct, batch_total = self._forward_batch(batch)
            total_loss += loss.item()
            correct += batch_correct
            total += batch_total

        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    # -----------------------------------------------------------------

    def _forward_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, int, int]:
        """Compute loss and accuracy for a single batch.

        Automatically detects batch format (BiLSTM vs. CodeBERT).

        Args:
            batch: Tuple from the DataLoader.

        Returns:
            Tuple of ``(loss_tensor, num_correct, batch_size)``.
        """
        if self.model_name == "bilstm":
            # Batch: (padded_ids, lengths, labels)
            input_ids, lengths, labels = batch
            input_ids = input_ids.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(
                "cuda", enabled=self.use_amp
            ):
                logits = self.model(input_ids, lengths)
                loss = self.criterion(logits, labels)
        else:
            # Batch: dict-like with input_ids, attention_mask, labels
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast(
                "cuda", enabled=self.use_amp
            ):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        num_correct = (preds == labels).sum().item()
        batch_size = labels.size(0)

        return loss, num_correct, batch_size

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Evaluate the model on a test set and return predictions + metrics.

        Args:
            test_loader: Test :class:`~torch.utils.data.DataLoader`.

        Returns:
            Tuple of ``(y_true, y_pred, metrics_dict)`` where *metrics_dict*
            contains accuracy, per-class precision/recall/F1, and macro F1.
        """
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss = 0.0

        for batch in test_loader:
            loss, _, _ = self._forward_batch(batch)
            total_loss += loss.item()

            if self.model_name == "bilstm":
                _, _, labels = batch
                input_ids, lengths, _ = batch
                input_ids = input_ids.to(self.device)
                lengths = lengths.to(self.device)
                with torch.amp.autocast(
                    "cuda", enabled=self.use_amp
                ):
                    logits = self.model(input_ids, lengths)
            else:
                labels = batch["labels"]
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with torch.amp.autocast(
                    "cuda", enabled=self.use_amp
                ):
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        # Compute metrics using sklearn
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )

        label_names = self.config.label_order
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        macro_precision = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        report = classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        avg_loss = total_loss / max(len(test_loader), 1)

        metrics: Dict[str, Any] = {
            "test_loss": avg_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "classification_report": report,
            "training_history": self.history.to_dict(),
        }

        logger.info(
            "Evaluation complete — accuracy=%.4f | macro_f1=%.4f | "
            "weighted_f1=%.4f",
            accuracy,
            macro_f1,
            weighted_f1,
        )

        return y_true, y_pred, metrics

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch index (0-based).
            is_best: If ``True``, save with a ``_best`` suffix.
        """
        tag = "best" if is_best else "last"
        filename = f"{self.model_name}_seed{self.seed}_{tag}.pt"
        filepath = self.checkpoint_dir / filename

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "epochs_without_improvement": self.epochs_without_improvement,
            "scaler_state_dict": (
                self.scaler.state_dict() if self.use_amp else None
            ),
            "model_name": self.model_name,
            "seed": self.seed,
            "history": self.history.to_dict(),
        }

        torch.save(checkpoint, filepath)
        logger.debug("Checkpoint saved → %s", filepath)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Resume training from a previously saved checkpoint.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        self.start_epoch = checkpoint.get("epoch", 0) + 1

        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Restore history
        history_dict = checkpoint.get("history")
        if history_dict is not None:
            self.history.train_loss = history_dict.get("train_loss", [])
            self.history.train_acc = history_dict.get("train_acc", [])
            self.history.val_loss = history_dict.get("val_loss", [])
            self.history.val_acc = history_dict.get("val_acc", [])
            self.history.learning_rates = history_dict.get(
                "learning_rates", []
            )

        logger.info(
            "Resumed from checkpoint %s (epoch %d, best_val_loss=%.4f)",
            checkpoint_path.name,
            self.start_epoch - 1,
            self.best_val_loss,
        )
