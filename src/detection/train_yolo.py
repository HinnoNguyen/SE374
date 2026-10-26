"""Train an Ultralytics YOLO detector on a prepared VinDr-CXR dataset.

Use the SAME hyper-parameters AND the same kind of machine (e.g. all on Kaggle T4)
for every experiment (A, B1, B2, ...) so that only the input images differ.

Examples
    python -m src.detection.train_yolo --data data/yolo/A_original/data.yaml --name A_original
    python -m src.detection.train_yolo --data data/yolo/B1_rajaraman/data.yaml --name B1_rajaraman
    # the session died (Colab/Kaggle time limit)? continue from runs/vindr/<name>/weights/last.pt
    python -m src.detection.train_yolo --data data/yolo/A_original/data.yaml --name A_original --resume
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def save_run_info(run_dir: Path, args: argparse.Namespace) -> None:
    """Record library versions + hardware so results from different machines can be told apart."""
    import torch
    import ultralytics

    info = {
        "args": vars(args),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


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
    ap.add_argument("--device", default=None, help="e.g. 0, mps or cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--cache", default="", choices=["", "ram", "disk"], help="cache images in RAM / on disk")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from <project>/<name>/weights/last.pt (same settings)")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = Path(args.project) / args.name
    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Nothing to resume: {last} not found")
        print(f"Resuming from {last}")
        kwargs = {"resume": True, "workers": args.workers}
        if args.device is not None:
            kwargs["device"] = args.device
        YOLO(str(last)).train(**kwargs)
    else:
        save_run_info(run_dir, args)
        kwargs = dict(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                      patience=args.patience, workers=args.workers, seed=args.seed, deterministic=True,
                      project=args.project, name=args.name, exist_ok=True, fliplr=args.fliplr,
                      mosaic=args.mosaic, cache=args.cache or False, plots=True)
        if args.device is not None:
            kwargs["device"] = args.device
        YOLO(args.model).train(**kwargs)
    print(f"Done. Best weights: {run_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
