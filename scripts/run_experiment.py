"""Run ONE experiment end-to-end with the same steps on any machine (Kaggle, Colab, Windows, Mac).

  1. original image sizes (meta CSV, or built from DICOM headers)       -> data/vindr_train_sizes.csv
  2. shared train/val/test split + YOLO data for A_original              -> data/vindr_splits.csv
  3. (B experiments) bone suppression of the images in the split         -> data/vindr_bs/<exp>/train
  4. YOLO training (or --resume)                                         -> runs/vindr/<exp>
  5. evaluation on the test split + copy of the run's small files        -> results/detection/

Commit data/vindr_splits.csv, data/vindr_train_sizes.csv and results/ to git so everybody uses the
same split. Weights (runs/.../best.pt) go to GitHub Releases / Drive, never into git.

Examples
  # Kaggle: find the attached inputs automatically
  python scripts/run_experiment.py --experiment A_original --auto-find /kaggle/input
  python scripts/run_experiment.py --experiment B1_rajaraman --auto-find /kaggle/input
  # local paths
  python scripts/run_experiment.py --experiment B2_hmchuong --csv data/raw/vindr/train.csv ^
      --images-dir data/raw/vindr_1024/train --meta-csv data/raw/vindr_1024/train_meta.csv
  # own bone-suppression model
  python scripts/run_experiment.py --experiment B4_unet --bs-ckpt checkpoints/bone_suppression/unet/best.pt --auto-find /kaggle/input
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES_CSV = ROOT / "data" / "vindr_train_sizes.csv"
SPLITS_CSV = ROOT / "data" / "vindr_splits.csv"

PRETRAINED = {  # experiment name -> (checkpoint, convert_pretrained --which)
    "B1_rajaraman": ("checkpoints/pretrained/rajaraman_resnet_bs.pt", "rajaraman"),
    "B2_hmchuong": ("checkpoints/pretrained/hmchuong_ae.pt", "hmchuong"),
}


def run(*args: str) -> None:
    print("\n$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def find_inputs(root: Path) -> dict[str, Path | None]:
    """Locate train.csv, the resized PNG folder, a meta CSV and the DICOM folder under `root`."""
    import pandas as pd

    found: dict[str, Path | None] = {"csv": None, "images_dir": None, "meta_csv": None, "dicom_dir": None}
    png_dirs: list[tuple[int, Path]] = []
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        n_png = sum(f.lower().endswith(".png") for f in filenames)
        if n_png:
            png_dirs.append((n_png, d))
        if found["dicom_dir"] is None and d.name == "train" and any(f.lower().endswith((".dicom", ".dcm")) for f in filenames[:50]):
            found["dicom_dir"] = d
        for f in filenames:
            if not f.lower().endswith(".csv"):
                continue
            try:
                cols = set(pd.read_csv(d / f, nrows=2).columns)
            except Exception:
                continue
            if found["csv"] is None and {"image_id", "class_id", "rad_id", "x_min"} <= cols:
                found["csv"] = d / f
            elif found["meta_csv"] is None and "image_id" in cols and ({"width", "height"} <= cols or {"dim0", "dim1"} <= cols) \
                    and "test" not in f.lower():
                found["meta_csv"] = d / f
    if png_dirs:  # prefer a folder called "train", then the biggest one
        png_dirs.sort(key=lambda t: (t[1].name == "train", t[0]), reverse=True)
        found["images_dir"] = png_dirs[0][1]
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="A_original | B1_rajaraman | B2_hmchuong | B<k>_<name> with --bs-ckpt")
    ap.add_argument("--bs-ckpt", default=None, help="bone-suppression checkpoint for custom B experiments")
    ap.add_argument("--auto-find", default=None, help="folder to search for the inputs (e.g. /kaggle/input)")
    ap.add_argument("--csv", default=None, help="Kaggle train.csv")
    ap.add_argument("--images-dir", default=None, help="resized PNGs <image_id>.png")
    ap.add_argument("--meta-csv", default=None, help="original sizes (width/height or dim0/dim1)")
    ap.add_argument("--dicom-dir", default=None, help="competition DICOMs, used to build sizes if no meta CSV")
    # YOLO settings: keep IDENTICAL for every experiment
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--link", default="hardlink" if os.name == "nt" else "symlink",
                    choices=["hardlink", "copy", "symlink"],
                    help="how images are placed in the YOLO folders (symlink on Kaggle/Colab, hardlink on Windows)")
    ap.add_argument("--resume", action="store_true", help="continue runs/vindr/<exp>/weights/last.pt")
    ap.add_argument("--prepare-only", action="store_true", help="stop before training")
    ap.add_argument("--skip-train", action="store_true", help="only evaluate an existing best.pt")
    args = ap.parse_args()
    exp = args.experiment

    # ---- inputs ---------------------------------------------------------------------------
    found = find_inputs(Path(args.auto_find)) if args.auto_find else {}
    csv = args.csv or found.get("csv")
    images_dir = args.images_dir or found.get("images_dir")
    meta_csv = args.meta_csv or found.get("meta_csv")
    dicom_dir = args.dicom_dir or found.get("dicom_dir")
    print(f"train.csv : {csv}\nimages    : {images_dir}\nmeta csv  : {meta_csv}\ndicom dir : {dicom_dir}")
    if not csv or not images_dir:
        raise SystemExit("train.csv or the PNG folder not found: pass --csv / --images-dir")

    # ---- 1. original sizes ----------------------------------------------------------------
    if meta_csv is None:
        if not SIZES_CSV.exists():
            if not dicom_dir:
                raise SystemExit("No meta CSV, no data/vindr_train_sizes.csv and no DICOM folder to build one")
            run("scripts/dicom_sizes.py", "--dicom-dir", str(dicom_dir), "--out", str(SIZES_CSV))
            print(f"!! New {SIZES_CSV.name}: download it and commit it to git")
        meta_csv = SIZES_CSV

    common = ["--csv", str(csv), "--meta-csv", str(meta_csv), "--splits-csv", str(SPLITS_CSV), "--link", args.link]

    # ---- 2. split + A data (cheap, also creates the shared split) ----------------------------
    new_split = not SPLITS_CSV.exists()
    run("-m", "src.detection.prepare_vindr", *common, "--images-dir", str(images_dir),
        "--out-root", str(ROOT / "data" / "yolo" / "A_original"))
    if new_split:
        print(f"!! New {SPLITS_CSV.name}: download it and commit it to git BEFORE anyone runs another experiment")

    # ---- 3. bone suppression for B experiments -----------------------------------------------
    if exp != "A_original":
        if args.bs_ckpt:
            ckpt = Path(args.bs_ckpt)
        elif exp in PRETRAINED:
            ckpt = ROOT / PRETRAINED[exp][0]
            if not ckpt.exists():
                run("scripts/convert_pretrained.py", "--which", PRETRAINED[exp][1])
        else:
            raise SystemExit(f"Unknown experiment {exp}: pass --bs-ckpt")
        bs_dir = ROOT / "data" / "vindr_bs" / exp / "train"
        run("-m", "src.bone_suppression.infer", "--ckpt", str(ckpt), "--input", str(images_dir),
            "--output", str(bs_dir), "--ids", str(SPLITS_CSV), "--skip-existing")
        run("-m", "src.detection.prepare_vindr", *common, "--images-dir", str(bs_dir),
            "--out-root", str(ROOT / "data" / "yolo" / exp))

    if args.prepare_only:
        print("Prepared. Stop (--prepare-only).")
        return

    # ---- 4. training ------------------------------------------------------------------------
    data_yaml = ROOT / "data" / "yolo" / exp / "data.yaml"
    yolo = ["--data", str(data_yaml), "--name", exp, "--project", str(ROOT / "runs" / "vindr"),
            "--workers", str(args.workers)] + (["--device", args.device] if args.device else [])
    if not args.skip_train:
        if args.resume:
            run("-m", "src.detection.train_yolo", *yolo, "--resume")
        else:
            run("-m", "src.detection.train_yolo", *yolo, "--model", args.model, "--imgsz", str(args.imgsz),
                "--epochs", str(args.epochs), "--batch", str(args.batch))

    # ---- 5. evaluation + keep the small files ------------------------------------------------
    run_dir = ROOT / "runs" / "vindr" / exp
    results = ROOT / "results" / "detection"
    run("-m", "src.detection.evaluate_yolo", "--weights", str(run_dir / "weights" / "best.pt"),
        "--data", str(data_yaml), "--imgsz", str(args.imgsz), "--out", str(results / f"{exp}.csv"),
        *(["--device", args.device] if args.device else []))
    keep = results / "runs" / exp
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("results.csv", "args.yaml", "run_info.json", "results.png", "confusion_matrix.png"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, keep / name)
    print(f"\nDone: {results / (exp + '.csv')}  |  weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
