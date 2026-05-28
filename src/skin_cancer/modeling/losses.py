from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        alpha: optional class weights tensor with shape [num_classes].
        gamma: focusing parameter. Larger values focus more on hard examples.
        reduction: mean, sum, or none.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(labels: list[int] | torch.Tensor, num_classes: int) -> torch.Tensor:
    """Balanced class weights: N / (C * count_c)."""
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights


def build_criterion(
    loss_name: str,
    num_classes: int,
    train_labels: list[int] | torch.Tensor,
    device: torch.device,
    alpha: str | None = "balanced",
    gamma: float = 2.0,
) -> nn.Module:
    class_weights = None
    if alpha == "balanced":
        class_weights = compute_class_weights(train_labels, num_classes).to(device)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal_loss":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    raise ValueError(f"Unsupported loss: {loss_name}")
