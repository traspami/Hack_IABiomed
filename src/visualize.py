"""
Overlay ground-truth boxes (green) and model detections (red) on val images
to inspect localization quality.

Usage:
  python visualize_gt_vs_pred.py --weights runs/detect/deid_hybrid/weights/last.pt \
      [--variant hybrid] [--n 8] [--out gt_vs_pred]
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ is one level down)
CLASS_NAMES = {0: "name", 1: "id", 2: "age", 3: "date", 4: "time"}
GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--variant", default="hybrid", choices=["raw", "hybrid"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=8, help="Number of images (0 = all)")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "gt_vs_pred"))
    args = parser.parse_args()

    root = ROOT / "datasets" / args.variant
    images = sorted((root / "images" / args.split).glob("*.png"))
    if args.n:
        images = images[: args.n]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    for img_path in images:
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]
        # display on the raw channel (hybrid stores raw in channel 0)
        canvas = cv2.cvtColor(im[:, :, 0], cv2.COLOR_GRAY2BGR)

        label_path = root / "labels" / args.split / f"{img_path.stem}.txt"
        for line in label_path.read_text().splitlines():
            c, xc, yc, bw, bh = line.split()
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
            x2, y2 = int(xc + bw / 2), int(yc + bh / 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), GT_COLOR, 1)
            cv2.putText(canvas, CLASS_NAMES[int(c)], (x1, max(10, y1 - 3)),
                        0, 0.38, GT_COLOR)

        result = model.predict(im, conf=args.conf, imgsz=512,
                               device="cpu", verbose=False)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(canvas, (x1, y1), (x2, y2), PRED_COLOR, 1)
            cv2.putText(canvas, f"{model.names[int(box.cls)]} {float(box.conf):.2f}",
                        (x1, min(h - 2, y2 + 12)), 0, 0.38, PRED_COLOR)

        cv2.imwrite(str(out_dir / img_path.name), canvas)

    print(f"Wrote {len(images)} overlays to {out_dir}/ "
          f"(green = ground truth, red = prediction)")


if __name__ == "__main__":
    main()
