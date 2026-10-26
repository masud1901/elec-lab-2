"""Loss functions used by the ablation study."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    ``alpha`` may be None, a scalar, or a per-class vector.  With gamma=0 and
    alpha=None this reduces to cross-entropy.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float | Sequence[float]] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None and not isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(list(alpha), dtype=torch.float32))
        else:
            self.alpha = None if alpha is None else float(alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp_min(0.0) ** self.gamma) * log_pt
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(logits.device)[target]
            else:
                alpha_t = self.alpha
            loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str = "ce", focal_gamma: float = 2.0, focal_alpha: float = -1.0):
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "focal":
        alpha = None if focal_alpha < 0 else float(focal_alpha)
        return FocalLoss(gamma=focal_gamma, alpha=alpha)
    raise ValueError(f"Unknown loss: {name}")
