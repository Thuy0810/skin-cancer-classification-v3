from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from skin_cancer.evaluation.metrics import compute_classification_metrics
from skin_cancer.core.utils import save_checkpoint


@dataclass
class EpochOutput:
    loss: float
    metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    mixed_precision: bool = True,
    label_names: list[str] | None = None,
) -> EpochOutput:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []
    y_prob_batches: list[np.ndarray] = []

    progress = tqdm(dataloader, desc="train" if is_train else "valid", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            use_amp = mixed_precision and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        probabilities = torch.softmax(logits.detach(), dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        y_true_batches.append(labels.detach().cpu().numpy())
        y_pred_batches.append(predictions.cpu().numpy())
        y_prob_batches.append(probabilities.cpu().numpy())
        progress.set_postfix(loss=float(loss.item()))

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    y_prob = np.concatenate(y_prob_batches)
    epoch_loss = running_loss / len(dataloader.dataset)

    if label_names is None:
        label_names = [str(i) for i in range(y_prob.shape[1])]
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, label_names)
    return EpochOutput(loss=epoch_loss, metrics=metrics, y_true=y_true, y_pred=y_pred, y_prob=y_prob)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epochs: int,
    label_names: list[str],
    label_to_id: dict[str, int],
    checkpoint_path: str | Path,
    config_dict: dict[str, Any],
    monitor_metric: str = "macro_f1",
    mixed_precision: bool = True,
    early_stopping_patience: int = 5,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    best_metric = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and device.type == "cuda")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_output = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            label_names=label_names,
        )
        val_output = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            mixed_precision=mixed_precision,
            label_names=label_names,
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_output.loss),
            "val_loss": float(val_output.loss),
            "train_macro_f1": float(train_output.metrics["macro_f1"]),
            "val_macro_f1": float(val_output.metrics["macro_f1"]),
            "val_accuracy": float(val_output.metrics["accuracy"]),
            "val_recall_mel": float(val_output.metrics.get("recall_mel", 0.0)),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_output.loss:.4f} | "
            f"val_loss={val_output.loss:.4f} | "
            f"val_macro_f1={val_output.metrics['macro_f1']:.4f} | "
            f"val_recall_mel={val_output.metrics.get('recall_mel', 0.0):.4f}"
        )

        if mlflow_module is not None:
            mlflow_module.log_metrics({f"train/{k}": v for k, v in train_output.metrics.items() if isinstance(v, (int, float))}, step=epoch)
            mlflow_module.log_metrics({f"val/{k}": v for k, v in val_output.metrics.items() if isinstance(v, (int, float))}, step=epoch)
            mlflow_module.log_metric("train/loss", train_output.loss, step=epoch)
            mlflow_module.log_metric("val/loss", val_output.loss, step=epoch)

        current_metric = float(val_output.metrics.get(monitor_metric, val_output.metrics["macro_f1"]))
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_output.metrics,
                label_to_id=label_to_id,
                config=config_dict,
            )
            print(f"Saved new best model to {checkpoint_path} ({monitor_metric}={best_metric:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    return {"best_metric": best_metric, "best_epoch": best_epoch, "history": history}
