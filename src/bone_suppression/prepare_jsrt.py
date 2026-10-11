"""Prepare paired JSRT (source) / BSE-JSRT (target) images for training.

Steps
1. Pair source and target images by file name.
2. Resize both to a common square size (default 512).
3. (Optional) Register the target onto the source with OpenCV ECC (affine),
   because small misalignments make the network learn blurry outputs.
4. Save 16-bit PNGs to <out>/source and <out>/target, write splits.csv and a few previews.

Example
    python -m src.bone_suppression.prepare_jsrt ^
        --source-dir data/raw/bse/JSRT --target-dir data/raw/bse/BSE_JSRT ^
        --out-dir data/processed/bse_512 --size 512 --register
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.io import imread_gray, imwrite, list_images, resize


def pair_images(src_dir: Path, tgt_dir: Path) -> list[tuple[str, Path, Path]]:
    src = {p.stem.lower(): p for p in list_images(src_dir, recursive=True)}
    tgt = {p.stem.lower(): p for p in list_images(tgt_dir, recursive=True)}
    common = sorted(set(src) & set(tgt))
    if common:
        missing = len(src) - len(common), len(tgt) - len(common)
        if any(missing):
            print(f"[warn] unmatched files: {missing[0]} source, {missing[1]} target (ignored)")
        return [(src[k].stem, src[k], tgt[k]) for k in common]
    if len(src) == len(tgt) and src:
        print("[warn] no common file names; pairing by sorted order. CHECK the previews!")
        return [(src[s].stem, src[s], tgt[t]) for s, t in zip(sorted(src), sorted(tgt))]
    raise SystemExit(f"Could not pair images: {len(src)} source vs {len(tgt)} target files")


def register_ecc(fixed: np.ndarray, moving: np.ndarray, iters: int = 200) -> tuple[np.ndarray, bool]:
    """Align `moving` onto `fixed` using an affine ECC transform."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    try:
        _, warp = cv2.findTransformECC(
            fixed.astype(np.float32), moving.astype(np.float32), warp,
            cv2.MOTION_AFFINE, criteria, None, 5,
        )
    except cv2.error:
        return moving, False
    h, w = fixed.shape
    aligned = cv2.warpAffine(
        moving, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, True


def make_splits(ids: list[str], val: float, test: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test)))
    n_val = max(1, int(round(n * val)))
    split = ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
    return pd.DataFrame({"id": ids, "split": split}).sort_values("id").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True, help="Original chest X-rays (JSRT)")
    ap.add_argument("--target-dir", required=True, help="Bone-suppressed counterparts (BSE-JSRT)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--register", action="store_true", help="ECC-align target to source")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-previews", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out_dir)
    pairs = pair_images(Path(args.source_dir), Path(args.target_dir))
    print(f"Found {len(pairs)} image pairs")

    rows = []
    for i, (pid, sp, tp) in enumerate(tqdm(pairs, desc="prepare")):
        s = resize(imread_gray(sp), args.size)
        t = resize(imread_gray(tp), args.size)
        reg_ok = None
        if args.register:
            t, reg_ok = register_ecc(s, t)
        imwrite(out / "source" / f"{pid}.png", s, bit16=True)
        imwrite(out / "target" / f"{pid}.png", t, bit16=True)
        bone = np.clip(s - t, 0, None)
        rows.append({
            "id": pid,
            "src_mean": float(s.mean()), "tgt_mean": float(t.mean()),
            "mae_src_tgt": float(np.abs(s - t).mean()), "registered": reg_ok,
        })
        if i < args.n_previews:
            bone_vis = bone / (bone.max() + 1e-8)
            imwrite(out / "previews" / f"{pid}.png", np.concatenate([s, t, bone_vis], axis=1))

    stats = pd.DataFrame(rows)
    splits = make_splits(stats["id"].tolist(), args.val_ratio, args.test_ratio, args.seed)
    splits.to_csv(out / "splits.csv", index=False)
    stats.to_csv(out / "stats.csv", index=False)

    print(splits["split"].value_counts().to_string())
    print(f"mean |source - target| = {stats['mae_src_tgt'].mean():.4f}")
    if args.register:
        print(f"registration succeeded on {int(stats['registered'].sum())}/{len(stats)} pairs")
    if (stats["tgt_mean"] > stats["src_mean"] + 0.05).mean() > 0.5:
        print("[warn] targets are brighter than sources on average - check that the "
              "folders are not swapped or inverted (see previews/)")
    print(f"Saved to {out}  (previews: source | target | bone layer)")


if __name__ == "__main__":
    main()
