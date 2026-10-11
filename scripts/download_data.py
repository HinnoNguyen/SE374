"""Download the datasets with the Kaggle CLI.

Requirements
  1. pip install kaggle
  2. Kaggle account -> Settings -> "Create New Token" -> put kaggle.json in
     %USERPROFILE%\\.kaggle\\ (Windows) or ~/.kaggle/ (Linux/macOS).
     NEVER commit kaggle.json.
  3. For VinDr-CXR: open
     https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection
     and click "Join / I understand and accept" the competition rules once.

Usage
  python scripts/download_data.py --what bse      # JSRT + BSE-JSRT (~small)
  python scripts/download_data.py --what vindr    # train.csv + resized 1024px PNGs (several GB)
  python scripts/download_data.py --what all
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BSE_DATASET = "hmchuong/xray-bone-shadow-supression"
VINDR_COMPETITION = "vinbigdata-chest-xray-abnormalities-detection"
# Community re-upload of the competition images as 1024x1024 PNGs (verified to exist).
VINDR_PNG_DATASETS = {
    1024: "xhlulu/vinbigdata-chest-xray-resized-png-1024x1024",
}


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    if shutil.which(cmd[0]) is None:
        sys.exit("kaggle CLI not found. Run: pip install kaggle")
    subprocess.run(cmd, check=True)


def unzip_all(folder: Path) -> None:
    for z in folder.glob("*.zip"):
        print(f"unzip {z.name}")
        with zipfile.ZipFile(z) as f:
            f.extractall(folder)
        z.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="all", choices=["bse", "vindr", "all"])
    ap.add_argument("--vindr-size", type=int, default=1024, choices=sorted(VINDR_PNG_DATASETS))
    args = ap.parse_args()

    if args.what in ("bse", "all"):
        out = RAW / "bse"
        out.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "datasets", "download", "-d", BSE_DATASET, "-p", str(out), "--unzip"])
        print(f"BSE data in {out}. Look for the JSRT and BSE_JSRT folders and pass them to prepare_jsrt.")

    if args.what in ("vindr", "all"):
        csv_dir = RAW / "vindr"
        csv_dir.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "competitions", "download", "-c", VINDR_COMPETITION, "-f", "train.csv", "-p", str(csv_dir)])
        unzip_all(csv_dir)
        png_dir = RAW / f"vindr_{args.vindr_size}"
        png_dir.mkdir(parents=True, exist_ok=True)
        run(["kaggle", "datasets", "download", "-d", VINDR_PNG_DATASETS[args.vindr_size], "-p", str(png_dir), "--unzip"])
        print(f"VinDr annotations in {csv_dir}, images in {png_dir}.")
        print("Check which meta CSV (original sizes) came with the PNGs, e.g. train_meta.csv.")


if __name__ == "__main__":
    main()
