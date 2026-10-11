"""Train a bone-suppression model on prepared JSRT/BSE-JSRT pairs.

Examples
    # U-Net on full 512x512 images
    python -m src.bone_suppression.train --data-root data/processed/bse_512 --model unet --epochs 150

    # ResNet-BS (Rajaraman 2021) on random 256x256 crops
    python -m src.bone_suppression.train --data-root data/processed/bse_512 --model resnet_bs --crop 256 --batch-size 8

Outputs (in --out-dir, default checkpoints/bone_suppression/<model>):
    best.pt, last.pt, history.csv, config.json
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.bone_suppression.dataset import PairedXrayDataset
from src.bone_suppression.losses import MixedLoss
from src.bone_suppression.metrics import image_metrics
from src.bone_suppression.models import build_model
from src.common.utils import default_workers, get_device, save_json, set_seed


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model", default="unet", choices=["unet", "resnet_bs", "ae"])
    ap.add_argument("--base", type=int, default=32, help="U-Net / AE base channels")
    ap.add_argument("--n-blocks", type=int, default=16, help="ResNet-BS residual blocks")
    ap.add_argument("--no-residual", action="store_true", help="predict the image directly")
    ap.add_argument("--crop", type=int, default=0, help="random crop size for training (0 = full image)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.84, help="weight of (1 - MS-SSIM) in the loss")
    ap.add_argument("--patience", type=int, default=30, help="early stopping on val SSIM (0 = off)")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


@torch.no_grad()
def validate(model, loader, device) -> dict[str, float]:
    model.eval()
    agg: dict[str, list[float]] = {}
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        m = image_metrics(model(x), y)
        for k, v in m.items():
            agg.setdefault(k, []).extend(v)
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out = Path(args.out_dir or f"checkpoints/bone_suppression/{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    train_ds = PairedXrayDataset(args.data_root, "train", augment=True, crop=args.crop)
    val_ds = PairedXrayDataset(args.data_root, "val", augment=False)
    data_size = int(val_ds[0][0].shape[-1])
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                          num_workers=args.workers, pin_memory=pin, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    if args.model == "resnet_bs":
        model_kwargs = {"n_blocks": args.n_blocks}
    else:
        model_kwargs = {"base": args.base}
    model = build_model(args.model, residual=not args.no_residual, **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device}  model={args.model} ({n_params:.2f}M params)  "
          f"train={len(train_ds)} val={len(val_ds)}  image={data_size}px  crop={args.crop or 'full'}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_dl))
    total = args.epochs * steps_per_epoch
    warmup = min(500, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = MixedLoss(args.alpha)

    config = {**vars(args), "data_size": data_size, "model_kwargs": model_kwargs, "device": str(device)}
    save_json(config, out / "config.json")

    def make_ckpt(epoch, metrics):
        return {"model_name": args.model, "model_kwargs": model_kwargs, "residual": not args.no_residual,
                "data_size": data_size, "epoch": epoch, "val_metrics": metrics,
                "state_dict": model.state_dict(), "config": config}

    history, best, bad = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        val = validate(model, val_dl, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"],
               **{f"val_{k}": v for k, v in val.items()}, "time_s": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        score = val["ssim"]
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save(make_ckpt(epoch, val), out / "best.pt")
        else:
            bad += 1
        torch.save(make_ckpt(epoch, val), out / "last.pt")
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val PSNR {val['psnr']:.2f}  "
              f"SSIM {val['ssim']:.4f}{'  *' if improved else ''}  ({row['time_s']:.0f}s)")
        if args.patience and bad >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    print(f"Best val SSIM {best:.4f}. Checkpoints in {out}")


if __name__ == "__main__":
    main()
