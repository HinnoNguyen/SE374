"""Create the "original image size" meta CSV that prepare_vindr needs, from DICOM headers.

Boxes in VinDr-CXR train.csv are in original DICOM pixels. If the resized PNG dataset you use
does not ship a meta file (width/height or dim0/dim1), build one from the competition DICOMs.
Only the headers are read (no pixel data), so it is fast. On Kaggle the DICOMs are already
attached as competition input, so nothing has to be downloaded.

Example (Kaggle)
    python scripts/dicom_sizes.py --dicom-dir /kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train ^
        --out data/vindr_train_sizes.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import pydicom

    files = sorted(p for p in Path(args.dicom_dir).iterdir() if p.suffix.lower() in {".dicom", ".dcm"})
    if not files:
        raise SystemExit(f"No .dicom/.dcm files in {args.dicom_dir}")
    rows = []
    for f in tqdm(files, desc="read headers"):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        rows.append({"image_id": f.stem, "width": int(ds.Columns), "height": int(ds.Rows)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} sizes -> {out}")


if __name__ == "__main__":
    main()
