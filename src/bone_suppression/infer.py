"""Apply a trained bone-suppression model to one image or a whole folder.

Works with our own checkpoints and with converted published ones
(scripts/convert_pretrained.py -> checkpoints/pretrained/*.pt).

Two modes:
- residual-upsample (default): run the model at its training resolution, take the
  predicted *bone layer* (input - output), upsample it to the original size and
  subtract it from the full-resolution input.  Soft-tissue detail (small nodules)
  stays at full resolution; only the smooth bone layer is upsampled.
- direct: resize -> model -> resize back (simpler, blurrier on big images).

Examples (process all VinDr-CXR PNGs):
    python -m src.bone_suppression.infer --ckpt checkpoints/pretrained/rajaraman_resnet_bs.pt ^
        --input data/raw/vindr_1024/train --output data/vindr_bs/rajaraman/train --skip-existing
    python -m src.bone_suppression.infer --ckpt checkpoints/bone_suppression/unet/best.pt ^
        --input data/raw/vindr_1024/train --output data/vindr_suppressed/train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.bone_suppression.models import load_checkpoint
from src.common.io import IMG_EXTS, imread_gray, imwrite, resize
from src.common.utils import get_device


@torch.no_grad()
def suppress(model, img: np.ndarray, model_size: int, device, mode: str = "residual-upsample",
             invert: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """img: float32 [0,1] HxW, standard polarity (bones bright).
    Returns (suppressed, bone_layer), both HxW at input size and in standard polarity.
    invert=True for models trained on negative images (e.g. the hmchuong AE)."""
    h, w = img.shape
    src = 1.0 - img if invert else img
    small = resize(src, model_size) if (h, w) != (model_size, model_size) else src
    x = torch.from_numpy(np.ascontiguousarray(small, dtype=np.float32))[None, None].to(device)
    pred_small = model(x)[0, 0].float().cpu().numpy()
    if mode == "direct":
        out = resize(pred_small, (w, h)) if (h, w) != pred_small.shape else pred_small
    else:
        delta = small - pred_small  # what the model removed, at model resolution
        delta = resize(delta, (w, h)) if (h, w) != delta.shape else delta
        out = src - delta
    out = np.clip(out, 0, 1)
    if invert:
        out = 1.0 - out
    return out, img - out


def select_shard(files: list, shard: str | None) -> list:
    """shard "i/n" (1-based) -> every n-th file starting at i, to split work across machines."""
    if not shard:
        return files
    i, n = (int(v) for v in shard.split("/"))
    if not 1 <= i <= n:
        raise SystemExit("--shard must look like 1/4 .. 4/4")
    return files[i - 1::n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--mode", default="residual-upsample", choices=["residual-upsample", "direct"])
    ap.add_argument("--model-size", type=int, default=None,
                    help="resolution fed to the model (default: the training image size stored in the checkpoint)")
    ap.add_argument("--save-bone", action="store_true", help="also save the bone layer to <output>_bone")
    ap.add_argument("--bit16", action="store_true", help="save 16-bit PNGs")
    ap.add_argument("--invert", default="auto", choices=["auto", "yes", "no"],
                    help="process negative images (auto: as stored in the checkpoint)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--shard", default=None, help="i/n: only process part i of n (split work between machines)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick test)")
    ap.add_argument("--ids", default=None,
                    help="CSV with an image_id column (e.g. data/vindr_splits.csv): only process those images; "
                         "rows with used=False are skipped")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    model_size = args.model_size or int(ckpt["data_size"])
    invert = ckpt.get("invert_input", False) if args.invert == "auto" else args.invert == "yes"

    inp = Path(args.input)
    dicom_exts = {".dcm", ".dicom"}
    files = [inp] if inp.is_file() else sorted(p for p in inp.iterdir() if p.suffix.lower() in IMG_EXTS | dicom_exts)
    files = [f for f in files if f.suffix.lower() in IMG_EXTS | dicom_exts]
    if args.ids:
        import pandas as pd

        ids_df = pd.read_csv(args.ids)
        if "used" in ids_df.columns:
            ids_df = ids_df[ids_df["used"].astype(bool)]
        wanted = set(ids_df["image_id"].astype(str))
        files = [f for f in files if f.stem in wanted]
    files = select_shard(files, args.shard)
    if args.limit:
        files = files[:args.limit]
    out_dir = Path(args.output)
    bone_dir = Path(str(out_dir) + "_bone")
    print(f"{len(files)} images | model {ckpt['model_name']} @ {model_size}px | mode {args.mode} | "
          f"invert {invert} | device {device}")

    for f in tqdm(files, desc="bone suppression"):
        dst = out_dir / f"{f.stem}.png"
        if args.skip_existing and dst.exists():
            continue
        img = imread_gray(f)
        out, bone = suppress(model, img, model_size, device, args.mode, invert)
        imwrite(dst, out, bit16=args.bit16)
        if args.save_bone:
            bone = np.clip(bone, 0, None)
            imwrite(bone_dir / f"{f.stem}.png", bone / (bone.max() + 1e-8))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
