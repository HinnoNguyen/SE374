"""Train an Ultralytics YOLO detector on a prepared VinDr-CXR dataset.

Use the SAME hyper-parameters for experiments A (original), B (suppressed)
and C (fused) so that only the input images differ.

Example
    python -m src.detection.train_yolo --data data/yolo/original/data.yaml --name A_original
    python -m src.detection.train_yolo --data data/yolo/suppressed/data.yaml --name B_suppressed
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml from prepare_vindr")
    ap.add_argument("--model", default="yolo11s.pt", help="pretrained weights or a model .yaml")
    ap.add_argument("--name", required=True, help="run name, e.g. A_original")
    ap.add_argument("--project", default="runs/vindr")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--cache", default=False, help="'ram' or 'disk' to cache images")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    kwargs = dict(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                  patience=args.patience, workers=args.workers, seed=args.seed, deterministic=True,
                  project=args.project, name=args.name, exist_ok=True, fliplr=args.fliplr,
                  mosaic=args.mosaic, cache=args.cache, plots=True)
    if args.device is not None:
        kwargs["device"] = args.device
    model.train(**kwargs)
    print(f"Done. Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
