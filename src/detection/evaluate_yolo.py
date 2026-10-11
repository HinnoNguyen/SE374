"""Evaluate a trained YOLO model on the held-out test split, per class.

Writes a CSV with precision, recall, AP@0.5 and AP@0.5:0.95 for every class,
plus an 'ALL' row.  Feed several of these CSVs to compare.py.

Example
    python -m src.detection.evaluate_yolo --weights runs/vindr/A_original/weights/best.pt ^
        --data data/yolo/original/data.yaml --out results/detection/A_original.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True, help="output CSV")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    kwargs = dict(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                  conf=args.conf, iou=args.iou, plots=True, project=str(Path(args.out).parent / "val_plots"),
                  name=Path(args.out).stem, exist_ok=True)
    if args.device is not None:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)
    box = metrics.box
    names = metrics.names

    rows = []
    for i, c in enumerate(box.ap_class_index):
        c = int(c)
        rows.append({"class_id": c, "class": names[c], "precision": float(box.p[i]),
                     "recall": float(box.r[i]), "AP50": float(box.ap50[i]), "AP50_95": float(box.ap[i])})
    rows.append({"class_id": -1, "class": "ALL", "precision": float(box.mp), "recall": float(box.mr),
                 "AP50": float(box.map50), "AP50_95": float(box.map)})
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
