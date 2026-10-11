"""Evaluate a bone-suppression checkpoint on the held-out split.

Reports MAE / PSNR / SSIM / MS-SSIM for
  - "identity": the untouched input compared with the target (lower bound),
  - the model prediction.
Also saves comparison figures: input | prediction | target | |prediction - target|.

Example
    python -m src.bone_suppression.evaluate --ckpt checkpoints/bone_suppression/unet/best.pt ^
        --data-root data/processed/bse_512 --split test --out-dir results/bone_suppression/unet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.bone_suppression.dataset import PairedXrayDataset
from src.bone_suppression.metrics import image_metrics
from src.bone_suppression.models import load_checkpoint
from src.common.io import imwrite
from src.common.utils import get_device, save_json


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-figures", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    ds = PairedXrayDataset(args.data_root, args.split, augment=False, cache=False)
    out = Path(args.out_dir)

    rows = []
    for i in range(len(ds)):
        x, y, pid = ds[i]
        x, y = x[None].to(device), y[None].to(device)
        pred = model(x).clamp(0, 1)
        for name, img in (("identity", x), ("model", pred)):
            m = {k: v[0] for k, v in image_metrics(img, y).items()}
            rows.append({"id": pid, "method": name, **m})
        if i < args.n_figures:
            xi, pi, yi = (t[0, 0].cpu().numpy() for t in (x, pred, y))
            err = np.abs(pi - yi)
            err = err / (err.max() + 1e-8)
            imwrite(out / "figures" / f"{pid}.png", np.concatenate([xi, pi, yi, err], axis=1))

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image.csv", index=False)
    summary = df.drop(columns="id").groupby("method").agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(out / "summary.csv")
    save_json({"ckpt": str(args.ckpt), "model": ckpt["model_name"], "epoch": ckpt.get("epoch"),
               "split": args.split, "n_images": len(ds),
               "summary": summary.reset_index().to_dict(orient="records")}, out / "summary.json")

    means = df.drop(columns="id").groupby("method").mean()
    print(f"{args.split} set: {len(ds)} images")
    print(means.round(4).to_string())
    print(f"Figures (input | prediction | target | error) in {out / 'figures'}")


if __name__ == "__main__":
    main()
