"""Convert VinDr-CXR (Kaggle VinBigData version) annotations to YOLO format.

* Each training image was annotated by 3 radiologists -> boxes of the same class
  are merged with Weighted Boxes Fusion (WBF, Solovyev et al. 2021).
* The image-level split (train/val/test) is created ONCE and saved to --splits-csv.
  Re-running with another --images-dir (e.g. bone-suppressed images) reuses the
  same split, so experiments A / B / C are directly comparable.
* Boxes in train.csv are in ORIGINAL DICOM pixels. Resized PNG datasets ship a
  meta file with the original sizes (columns width/height or dim0/dim1).

Example
    # A: original images
    python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv ^
        --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/raw/vindr_1024/train ^
        --out-root data/yolo/original
    # B: bone-suppressed images (same labels, same split)
    python -m src.detection.prepare_vindr --csv data/raw/vindr/train.csv ^
        --meta-csv data/raw/vindr_1024/train_meta.csv --images-dir data/vindr_suppressed/train ^
        --out-root data/yolo/suppressed
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

NO_FINDING_ID = 14
# Class ids of the Kaggle VinBigData version (14 abnormalities + 14 = "No finding").
VINDR_CLASSES = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation",
                 "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion",
                 "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"]


def load_sizes(df: pd.DataFrame, meta_csv: str | None) -> pd.DataFrame:
    """Return a DataFrame image_id -> (orig_w, orig_h)."""
    sources = []
    if meta_csv:
        sources.append(pd.read_csv(meta_csv))
    sources.append(df)
    for src in sources:
        cols = set(src.columns)
        if {"image_id", "width", "height"} <= cols:
            m = src[["image_id", "width", "height"]]
        elif {"image_id", "dim0", "dim1"} <= cols:  # dim0 = rows (height), dim1 = cols (width)
            m = src[["image_id", "dim1", "dim0"]].rename(columns={"dim1": "width", "dim0": "height"})
        else:
            continue
        return m.drop_duplicates("image_id").rename(columns={"width": "orig_w", "height": "orig_h"})
    raise SystemExit("Original image sizes not found. Pass --meta-csv with columns "
                     "image_id + (width, height) or (dim0, dim1).")


def make_splits(df: pd.DataFrame, val: float, test: float, keep_normal: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    abnormal = df.loc[df["class_id"] != NO_FINDING_ID, "image_id"].unique()
    images = df["image_id"].unique()
    normal = np.setdiff1d(images, abnormal)
    rows = []
    for group, ids, is_abn in (("abnormal", np.sort(abnormal), True), ("normal", np.sort(normal), False)):
        ids = ids.copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = int(round(n * test)), int(round(n * val))
        splits = ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
        used = np.ones(n, dtype=bool) if is_abn else (rng.random(n) < keep_normal)
        # no label information is stored, so the file can be shared (git) between machines
        rows += [{"image_id": i, "split": s, "used": bool(u)}
                 for i, s, u in zip(ids, splits, used)]
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def fuse_boxes(group: pd.DataFrame, w: float, h: float, iou_thr: float, skip_thr: float):
    """WBF over radiologists for one image. Returns (boxes_xyxy_norm, labels)."""
    from ensemble_boxes import weighted_boxes_fusion

    boxes_list, scores_list, labels_list = [], [], []
    for _, g in group.groupby("rad_id"):
        b = g[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=np.float64)
        b = b / np.array([w, h, w, h])
        b = np.clip(b, 0.0, 1.0)
        ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        if not ok.any():
            continue
        boxes_list.append(b[ok].tolist())
        scores_list.append([1.0] * int(ok.sum()))
        labels_list.append(g["class_id"].to_numpy()[ok].astype(int).tolist())
    if not boxes_list:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    boxes, _, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                             iou_thr=iou_thr, skip_box_thr=skip_thr)
    return np.asarray(boxes), np.asarray(labels, dtype=int)


def place_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different drive / unsupported -> copy
    elif mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="Kaggle train.csv")
    ap.add_argument("--meta-csv", default=None, help="CSV with original image sizes")
    ap.add_argument("--images-dir", required=True, help="folder with <image_id>.png")
    ap.add_argument("--img-ext", default=".png")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--splits-csv", default="data/vindr_splits.csv")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--keep-normal", type=float, default=0.2,
                    help="fraction of 'No finding' images kept as background images")
    ap.add_argument("--iou-thr", type=float, default=0.6, help="WBF IoU threshold")
    ap.add_argument("--skip-box-thr", type=float, default=0.0)
    ap.add_argument("--link", default="hardlink", choices=["hardlink", "copy", "symlink"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    names = (df.loc[df["class_id"] != NO_FINDING_ID, ["class_id", "class_name"]]
             .drop_duplicates().sort_values("class_id"))
    class_names = dict(enumerate(VINDR_CLASSES))
    for r in names.itertuples():
        if class_names.get(int(r.class_id)) != str(r.class_name):
            print(f"[warn] class {r.class_id}: csv says '{r.class_name}', expected '{class_names.get(int(r.class_id))}'")
        class_names[int(r.class_id)] = str(r.class_name)
    nc = max(class_names) + 1
    sizes = load_sizes(df, args.meta_csv).set_index("image_id")

    splits_path = Path(args.splits_csv)
    if splits_path.exists():
        splits = pd.read_csv(splits_path)
        print(f"Reusing split from {splits_path}")
    else:
        splits = make_splits(df, args.val, args.test, args.keep_normal, args.seed)
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(splits_path, index=False)
        print(f"Created split -> {splits_path}")

    out = Path(args.out_root)
    images_dir = Path(args.images_dir)
    ann = df[df["class_id"] != NO_FINDING_ID].groupby("image_id")
    used = splits[splits["used"]]
    counts = np.zeros((3, nc), dtype=int)
    split_idx = {"train": 0, "val": 1, "test": 2}
    missing = 0
    for r in tqdm(used.itertuples(), total=len(used), desc="convert"):
        src = images_dir / f"{r.image_id}{args.img_ext}"
        if not src.exists():
            missing += 1
            continue
        lines = []
        if r.image_id in ann.groups:  # images with at least one abnormality box
            w, h = sizes.loc[r.image_id, ["orig_w", "orig_h"]]
            boxes, labels = fuse_boxes(ann.get_group(r.image_id), float(w), float(h),
                                       args.iou_thr, args.skip_box_thr)
            for (x1, y1, x2, y2), c in zip(boxes, labels):
                lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
                counts[split_idx[r.split], c] += 1
        label_path = out / "labels" / r.split / f"{r.image_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        place_image(src, out / "images" / r.split / f"{r.image_id}{args.img_ext}", args.link)

    data_yaml = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                 "test": "images/test", "names": {i: class_names.get(i, f"class_{i}") for i in range(nc)}}
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    stats = pd.DataFrame(counts.T, columns=["train", "val", "test"])
    stats.insert(0, "class", [class_names.get(i, f"class_{i}") for i in range(nc)])
    stats.to_csv(out / "box_counts.csv", index=False)
    n_img = used.groupby("split").size().to_dict()
    print(f"Images per split (used): {n_img}   missing image files: {missing}")
    print(stats.to_string(index=False))
    print(f"YOLO dataset written to {out} (data.yaml)")


if __name__ == "__main__":
    main()
