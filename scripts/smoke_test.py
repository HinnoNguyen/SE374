"""End-to-end smoke test on tiny SYNTHETIC data (runs on CPU in a few minutes).

Checks that the whole pipeline works on your machine before downloading the
real datasets:
  prepare_jsrt -> train (unet / resnet_bs / ae) -> evaluate -> infer
  -> prepare_vindr (original / suppressed / fused) -> [YOLO train -> evaluate -> compare]

Usage
  python scripts/smoke_test.py               # bone suppression + data preparation
  python scripts/smoke_test.py --with-yolo   # also a 1-epoch YOLO run (no download needed)
  python scripts/smoke_test.py --keep        # keep the temporary folder for inspection
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.io import imwrite  # noqa: E402


def synthetic_xray(size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Soft-tissue image + the same image with bright rib-like stripes."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    body = np.exp(-(((xx - 0.5) / 0.38) ** 2 + ((yy - 0.55) / 0.45) ** 2) ** 2)
    lungs = sum(np.exp(-(((xx - cx) / 0.13) ** 2 + ((yy - 0.5) / 0.25) ** 2)) for cx in (0.33, 0.67))
    soft = np.clip(0.15 + 0.55 * body - 0.3 * lungs + 0.02 * rng.standard_normal((size, size)), 0, 1)
    ribs = np.zeros_like(soft)
    for k in range(7):
        y0 = 0.25 + 0.075 * k + rng.uniform(-0.01, 0.01)
        curve = y0 + 0.12 * (xx - 0.5) ** 2
        ribs += np.exp(-((yy - curve) / 0.012) ** 2)
    ribs *= (body > 0.3) * 0.25
    return np.clip(soft + ribs, 0, 1).astype(np.float32), soft.astype(np.float32)


def make_bse(root: Path, n: int, size: int, rng) -> None:
    for i in range(n):
        src, tgt = synthetic_xray(size, rng)
        imwrite(root / "JSRT" / f"JPCLN{i:03d}.png", src)
        imwrite(root / "BSE_JSRT" / f"JPCLN{i:03d}.png", tgt)


def make_vindr(root: Path, n: int, size: int, rng) -> None:
    classes = ["Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "ILD",
               "Infiltration", "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion",
               "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis"]
    rows, meta = [], []
    orig = size * 2  # pretend the DICOMs were 2x larger
    for i in range(n):
        iid = f"img{i:04d}"
        img, _ = synthetic_xray(size, rng)
        meta.append({"image_id": iid, "dim0": orig, "dim1": orig})
        if i % 3 == 0:
            for r in ("R1", "R2", "R3"):
                rows.append({"image_id": iid, "class_name": "No finding", "class_id": 14, "rad_id": r,
                             "x_min": np.nan, "y_min": np.nan, "x_max": np.nan, "y_max": np.nan})
        else:
            c = int(rng.integers(0, 3)) * 5  # a few classes only
            x0, y0 = rng.uniform(0.2, 0.6, size=2) * orig
            w = rng.uniform(0.1, 0.25) * orig
            cv2.circle(img, (int((x0 + w / 2) / 2), int((y0 + w / 2) / 2)), int(w / 4), 1.0, -1)
            for r in ("R1", "R2", "R3"):
                j = rng.uniform(-0.02, 0.02, size=4) * orig
                rows.append({"image_id": iid, "class_name": classes[c], "class_id": c, "rad_id": r,
                             "x_min": x0 + j[0], "y_min": y0 + j[1], "x_max": x0 + w + j[2], "y_max": y0 + w + j[3]})
        imwrite(root / "png" / f"{iid}.png", img)
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(meta).to_csv(root / "train_meta.csv", index=False)


def step(name: str, *args: str) -> None:
    print(f"\n=== {name} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-yolo", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="se374_smoke_"))
    rng = np.random.default_rng(0)
    try:
        make_bse(tmp / "bse", n=20, size=192, rng=rng)
        make_vindr(tmp / "vindr", n=30, size=256, rng=rng)
        bse, ck = tmp / "bse_proc", tmp / "ckpt"

        step("prepare_jsrt", "-m", "src.bone_suppression.prepare_jsrt", "--source-dir", str(tmp / "bse/JSRT"),
             "--target-dir", str(tmp / "bse/BSE_JSRT"), "--out-dir", str(bse), "--size", "192", "--register",
             "--val-ratio", "0.15", "--test-ratio", "0.15")
        step("train unet", "-m", "src.bone_suppression.train", "--data-root", str(bse), "--model", "unet",
             "--base", "8", "--epochs", "3", "--batch-size", "4", "--workers", "0", "--out-dir", str(ck / "unet"))
        step("train resnet_bs (crop)", "-m", "src.bone_suppression.train", "--data-root", str(bse),
             "--model", "resnet_bs", "--n-blocks", "2", "--crop", "96", "--epochs", "1", "--workers", "0",
             "--out-dir", str(ck / "resnet_bs"))
        step("train ae", "-m", "src.bone_suppression.train", "--data-root", str(bse), "--model", "ae",
             "--base", "8", "--epochs", "1", "--workers", "0", "--out-dir", str(ck / "ae"))
        step("evaluate", "-m", "src.bone_suppression.evaluate", "--ckpt", str(ck / "unet/best.pt"),
             "--data-root", str(bse), "--out-dir", str(tmp / "results/bse"))
        step("infer (VinDr images)", "-m", "src.bone_suppression.infer", "--ckpt", str(ck / "unet/best.pt"),
             "--input", str(tmp / "vindr/png"), "--output", str(tmp / "vindr/supp"), "--save-bone")
        step("make_fused", "-m", "src.detection.make_fused", "--orig-dir", str(tmp / "vindr/png"),
             "--supp-dir", str(tmp / "vindr/supp"), "--out-dir", str(tmp / "vindr/fused"))
        common = ["--csv", str(tmp / "vindr/train.csv"), "--meta-csv", str(tmp / "vindr/train_meta.csv"),
                  "--splits-csv", str(tmp / "vindr/splits.csv"), "--keep-normal", "0.5"]
        for variant, folder in (("original", "png"), ("suppressed", "supp"), ("fused", "fused")):
            step(f"prepare_vindr {variant}", "-m", "src.detection.prepare_vindr", *common,
                 "--images-dir", str(tmp / "vindr" / folder), "--out-root", str(tmp / "yolo" / variant))

        labels_a = sorted((tmp / "yolo/original/labels").rglob("*.txt"))
        labels_b = sorted((tmp / "yolo/suppressed/labels").rglob("*.txt"))
        assert [p.name for p in labels_a] == [p.name for p in labels_b], "splits differ between variants!"

        if args.with_yolo:
            res = tmp / "results/det"
            for variant in ("original", "suppressed"):
                step(f"YOLO train {variant}", "-m", "src.detection.train_yolo",
                     "--data", str(tmp / f"yolo/{variant}/data.yaml"), "--model", "yolo11n.yaml",
                     "--name", variant, "--project", str(tmp / "runs"), "--imgsz", "128", "--epochs", "1",
                     "--batch", "4", "--workers", "0", "--device", "cpu", "--mosaic", "0")
                step(f"YOLO evaluate {variant}", "-m", "src.detection.evaluate_yolo",
                     "--weights", str(tmp / f"runs/{variant}/weights/last.pt"),
                     "--data", str(tmp / f"yolo/{variant}/data.yaml"), "--imgsz", "128", "--batch", "4",
                     "--device", "cpu", "--out", str(res / f"{variant}.csv"))
            step("compare", "-m", "src.detection.compare", f"A={res / 'original.csv'}",
                 f"B={res / 'suppressed.csv'}", "--out", str(res / "compare"))
        print(f"\nSMOKE TEST PASSED  (workdir: {tmp})")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
