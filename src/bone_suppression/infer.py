"""Apply a trained bone-suppression model to one image or a whole folder.

Two modes:
- residual-upsample (default): run the model at its training resolution, take the
  predicted *bone layer* (input - output), upsample it to the original size and
  subtract it from the full-resolution input.  Soft-tissue detail (small nodules)
  stays at full resolution; only the smooth bone layer is upsampled.
- direct: resize -> model -> resize back (simpler, blurrier on big images).

Example (process all VinDr-CXR PNGs):
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
from src.common.io import IMG_EXTS, imread_gray, imwrite, list_images, resize
from src.common.utils import get_device


@torch.no_grad()
def suppress(model, img: np.ndarray, model_size: int, device, mode: str = "residual-upsample") -> tuple[np.ndarray, np.ndarray]:
    """img: float32 [0,1] HxW. Returns (suppressed, bone_layer), both HxW at input size."""
    h, w = img.shape
    small = resize(img, model_size) if (h, w) != (model_size, model_size) else img
    x = torch.from_numpy(np.ascontiguousarray(small, dtype=np.float32))[None, None].to(device)
    pred_small = model(x)[0, 0].float().cpu().numpy()
    if mode == "direct":
        out = resize(pred_small, (w, h)) if (h, w) != pred_small.shape else pred_small
        out = np.clip(out, 0, 1)
        return out, np.clip(img - out, 0, None)
    bone_small = small - pred_small
    bone = resize(bone_small, (w, h)) if (h, w) != bone_small.shape else bone_small
    out = np.clip(img - bone, 0, 1)
    return out, bone


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
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    model_size = args.model_size or int(ckpt["data_size"])

    inp = Path(args.input)
    files = [inp] if inp.is_file() else list_images(inp) + sorted(inp.glob("*.dcm"))
    files = [f for f in files if f.suffix.lower() in IMG_EXTS | {".dcm"}]
    out_dir = Path(args.output)
    bone_dir = Path(str(out_dir) + "_bone")
    print(f"{len(files)} images | model {ckpt['model_name']} @ {model_size}px | mode {args.mode} | device {device}")

    for f in tqdm(files, desc="bone suppression"):
        dst = out_dir / f"{f.stem}.png"
        if args.skip_existing and dst.exists():
            continue
        img = imread_gray(f)
        out, bone = suppress(model, img, model_size, device, args.mode)
        imwrite(dst, out, bit16=args.bit16)
        if args.save_bone:
            imwrite(bone_dir / f"{f.stem}.png", bone / (bone.max() + 1e-8))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
