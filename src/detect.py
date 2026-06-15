"""
Detect sensitive text on a new image (or folder) and save:
  - the image with labeled boxes drawn on it  -> outputs/detections/<name>.png
  - the detected labels (class, confidence, box) -> outputs/detections/<name>.txt
  - a combined report.json for the whole run

Unlike anonymize.py this does NOT mask anything — it just shows what the model
found. Applies the same hybrid preprocessing the model was trained on.

Usage:
  python src/detect.py --source path/to/new_xray.png
  python src/detect.py --source path/to/folder --weights models/deid_hybrid/weights/best.pt
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from prepare_dataset import hybrid_channels

ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ is one level down)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
BOX_COLOR = (0, 0, 255)  # red


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--weights",
                        default=str(ROOT / "models" / "deid_hybrid" / "weights" / "best.pt"))
    parser.add_argument("--out", default=str(ROOT / "outputs" / "detections"))
    parser.add_argument("--variant", default="hybrid", choices=["raw", "hybrid"],
                        help="Must match the variant the model was trained on")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    source = Path(args.source)
    images = ([source] if source.is_file()
              else sorted(p for p in source.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    report = []
    for img_path in images:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"WARNING: could not read {img_path}")
            continue
        net_input = hybrid_channels(gray) if args.variant == "hybrid" \
            else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result = model.predict(net_input, conf=args.conf, imgsz=args.imgsz,
                               device="cpu", verbose=False)[0]

        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        detections = []
        lines = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 1)
            cv2.putText(canvas, f"{cls} {conf:.2f}", (x1, max(10, y1 - 3)),
                        0, 0.4, BOX_COLOR, 1)
            detections.append({"class": cls, "conf": conf, "box": [x1, y1, x2, y2]})
            lines.append(f"{cls}\t{conf:.3f}\t{x1} {y1} {x2} {y2}")

        cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), canvas)
        (out_dir / f"{img_path.stem}.txt").write_text(
            "class\tconf\tx1 y1 x2 y2\n" + "\n".join(lines) + "\n")
        report.append({"file": img_path.name, "detections": detections})
        print(f"{img_path.name}: {len(detections)} detections")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote annotated images + labels to {out_dir}/")


if __name__ == "__main__":
    main()
