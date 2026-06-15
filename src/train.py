"""
Train a YOLO detector for sensitive-text regions on radiographs.

Augmentation is tuned for overlaid text:
  - no flips / rotation / shear (mirrored or rotated glyphs never occur),
  - no mosaic (GT boxes are ~9 px tall at 512 px; mosaic halves them),
  - no random erasing (it would delete the very objects we detect),
  - mild translate/scale + brightness only.

Usage:
  python train.py --variant hybrid --epochs 100
  python train.py --variant raw --epochs 15 --name ab_raw   # quick A/B run
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ is one level down)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="hybrid", choices=["raw", "hybrid"])
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "datasets" / f"{args.variant}.yaml"),
        project=str(ROOT / "runs" / "detect"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device="cpu",
        name=args.name or f"deid_{args.variant}",
        exist_ok=True,
        # text-specific augmentation
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        erasing=0.0,
        translate=0.10,
        scale=0.20,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        plots=True,
    )


if __name__ == "__main__":
    main()
