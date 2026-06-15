"""
Evaluate a trained detector on the validation split: per-class AP plus the
de-identification recall (boxes found / boxes annotated) that matters most
for privacy.

Usage:
  python evaluate.py --weights runs/detect/deid_hybrid/weights/best.pt --variant hybrid
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ is one level down)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", default="hybrid", choices=["raw", "hybrid"])
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    data_yaml = ROOT / "datasets" / f"{args.variant}.yaml"
    model = YOLO(args.weights)
    metrics = model.val(
        data=str(data_yaml),
        project=str(ROOT / "runs" / "detect"),
        imgsz=args.imgsz,
        conf=args.conf,
        device="cpu",
        plots=True,
    )

    print(f"\n=== {args.weights} on {data_yaml} ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"recall:   {metrics.box.mr:.4f}   <- privacy-critical: missed boxes leak PII")
    print(f"precision:{metrics.box.mp:.4f}")
    print("\nPer class (AP50):")
    for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {model.names[int(idx)]:<6} {ap:.4f}")


if __name__ == "__main__":
    main()
