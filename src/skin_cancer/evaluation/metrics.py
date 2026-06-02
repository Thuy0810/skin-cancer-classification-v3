from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    label_names: list[str],
) -> dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    cm_normalized = cm.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm_normalized,
        row_sums,
        out=np.zeros_like(cm_normalized),
        where=row_sums != 0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
    }

    if y_prob is not None:
        try:
            metrics["macro_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["macro_auc_ovr"] = None

    for label in label_names:
        if label in report:
            metrics[f"recall_{label}"] = float(report[label]["recall"])
            metrics[f"f1_{label}"] = float(report[label]["f1-score"])

    return metrics


def save_confusion_matrix_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_dir: str | Path,
    base_name: str = "confusion_matrix",
) -> None:
    output_dir = Path(output_dir)
    plot_confusion_matrix(
        y_true,
        y_pred,
        label_names,
        output_dir / f"{base_name}.png",
        normalize=False,
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        label_names,
        output_dir / f"{base_name}_normalized.png",
        normalize=True,
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    output_path: str | Path,
    normalize: bool = False,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)

    fmt = ".2f" if normalize else "d"
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
