"""Per-image image-quality metrics for images in [0, 1]."""
from __future__ import annotations

import torch

from src.bone_suppression.losses import MIN_SIDE_MSSSIM, structural_similarity


@torch.no_grad()
def image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
    """pred, target: (N, 1, H, W). Returns lists (one value per image)."""
    pred = pred.clamp(0, 1).float()
    target = target.float()
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    mae = (pred - target).abs().flatten(1).mean(1)
    psnr = 10 * torch.log10(1.0 / mse.clamp_min(1e-10))
    out = {
        "mae": mae.tolist(),
        "psnr": psnr.tolist(),
        "ssim": structural_similarity(pred, target, multiscale=False, size_average=False).tolist(),
    }
    if min(pred.shape[-2:]) >= MIN_SIDE_MSSSIM:
        out["ms_ssim"] = structural_similarity(pred, target, multiscale=True, size_average=False).tolist()
    return out
