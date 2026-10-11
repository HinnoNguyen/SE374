"""Build 3-channel images for experiment C: [original, bone-suppressed, CLAHE(original)].

YOLO normally receives the gray X-ray copied into 3 identical channels; here each
channel carries different information so the detector can use both views.

Example
    python -m src.detection.make_fused --orig-dir data/raw/vindr_1024/train ^
        --supp-dir data/vindr_suppressed/train --out-dir data/vindr_fused/train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common.io import clahe, imread_gray, imwrite, list_images, resize


def fuse(orig: np.ndarray, supp: np.ndarray) -> np.ndarray:
    if supp.shape != orig.shape:
        supp = resize(supp, (orig.shape[1], orig.shape[0]))
    # cv2 writes BGR: channel 0 = original, 1 = suppressed, 2 = CLAHE
    return np.stack([orig, supp, clahe(orig)], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--supp-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    files = list_images(args.orig_dir)
    missing = 0
    for f in tqdm(files, desc="fuse"):
        dst = out / f"{f.stem}.png"
        if args.skip_existing and dst.exists():
            continue
        s = Path(args.supp_dir) / f"{f.stem}.png"
        if not s.exists():
            missing += 1
            continue
        imwrite(dst, fuse(imread_gray(f), imread_gray(s)))
    print(f"Saved {len(files) - missing} fused images to {out} (missing suppressed: {missing})")


if __name__ == "__main__":
    main()
