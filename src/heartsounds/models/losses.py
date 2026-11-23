"""Composite training loss jointly optimizing calibration, recall, and ROC-AUC."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 1.5,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Class-weighted binary focal loss operating directly on raw logits."""
    pw = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    modulating = (1.0 - p_t).clamp(min=1e-6).pow(gamma)
    return (modulating * bce).mean()


def pairwise_auc_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float = 0.8,
) -> torch.Tensor:
    """Smooth pairwise surrogate for ROC-AUC (jointly pushes up sensitivity & specificity).

    Supports soft Mixup targets by treating `targets > 0.5` as positive and `< 0.5` as negative.
    """
    pos_mask = targets > 0.5
    neg_mask = targets < 0.5
    if not bool(pos_mask.any()) or not bool(neg_mask.any()):
        return logits.new_zeros(())

    pos_logits = logits[pos_mask].unsqueeze(1)  # (N_pos, 1)
    neg_logits = logits[neg_mask].unsqueeze(0)  # (1, N_neg)
    diffs = pos_logits - neg_logits - margin
    return F.softplus(-diffs).mean()


class CompositeDopplerLoss(nn.Module):
    """Combines weighted Focal BCE, pairwise ROC-AUC loss, and auxiliary subtype CE."""

    def __init__(
        self,
        *,
        focal_gamma: float = 1.5,
        pos_weight: float = 3.5,
        auc_weight: float = 0.35,
        subtype_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.focal_gamma = float(focal_gamma)
        self.pos_weight = float(pos_weight)
        self.auc_weight = float(auc_weight)
        self.subtype_weight = float(subtype_weight)

    def forward(
        self,
        binary_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        subtype_logits: torch.Tensor | None = None,
        subtype_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss_focal = focal_bce_with_logits(
            binary_logits,
            binary_targets,
            gamma=self.focal_gamma,
            pos_weight=self.pos_weight,
        )
        loss_auc = pairwise_auc_margin_loss(binary_logits, binary_targets)
        total = loss_focal + self.auc_weight * loss_auc

        if self.subtype_weight > 0.0 and subtype_logits is not None and subtype_targets is not None:
            loss_sub = F.cross_entropy(subtype_logits, subtype_targets, label_smoothing=0.05)
            total = total + self.subtype_weight * loss_sub
        return total
