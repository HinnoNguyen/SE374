"""Mixed MS-SSIM + L1 loss (Zhao et al. 2017), as also used by Rajaraman et al. 2021."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

# ms_ssim with 5 scales and an 11x11 window needs the smaller side > (11 - 1) * 2**4.
MIN_SIDE_MSSSIM = 161


def structural_similarity(pred: torch.Tensor, target: torch.Tensor, multiscale: bool = True,
                          size_average: bool = True) -> torch.Tensor:
    pred = pred.clamp(0, 1).float()
    target = target.float()
    if multiscale and min(pred.shape[-2:]) >= MIN_SIDE_MSSSIM:
        return ms_ssim(pred, target, data_range=1.0, size_average=size_average)
    return ssim(pred, target, data_range=1.0, size_average=size_average)


class MixedLoss(nn.Module):
    def __init__(self, alpha: float = 0.84):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred.float(), target.float())
        if self.alpha <= 0:
            return l1
        s = structural_similarity(pred, target)
        return self.alpha * (1 - s) + (1 - self.alpha) * l1
