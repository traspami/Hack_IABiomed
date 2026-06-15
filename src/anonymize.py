"""
De-identify radiographs end to end: detect sensitive text (name, id, age,
date, time) with the trained YOLO model and mask it on the original image.

Detection runs at a low confidence threshold and boxes are padded, because for
privacy a false positive (masking a few extra pixels) is far cheaper than a
false negative (leaking a patient name). An optional OCR pass on the masked
output flags any text that survived.

Usage:
  python anonymize.py --weights runs/detect/deid_hybrid/weights/best.pt \
      --source data/images/val --out anonymized --variant hybrid \
      --mode blackbox --ocr-check
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from prepare_dataset import hybrid_channels

ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ is one level down)
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def mask_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, mode: str):
    if mode == "blackbox":
        img[y1:y2, x1:x2] = 0
    elif mode == "blur":
        roi = img[y1:y2, x1:x2]
        if roi.size:
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
    elif mode == "pixelate":
        roi = img[y1:y2, x1:x2]
        if roi.size:
            small = cv2.resize(roi, (max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)))
            img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                           interpolation=cv2.INTER_NEAREST)


def ocr_leak_check(reader, img: np.ndarray, min_score: float = 0.5,
                   min_len: int = 3) -> list:
    """Return OCR hits on the masked image — text that escaped the detector."""
    leaks = []
    for box, text, score in reader.readtext(img):
        if score >= min_score and len(text.strip()) >= min_len:
            leaks.append({"text": text, "score": float(score),
                          "box": [[float(x), float(y)] for x, y in box]})
    return leaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--out", default=str(ROOT / "outputs" / "anonymized"))
    parser.add_argument("--variant", default="hybrid", choices=["raw", "hybrid"],
                        help="Must match the variant the model was trained on")
    parser.add_argument("--mode", default="blackbox",
                        choices=["blackbox", "blur", "pixelate"])
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Low on purpose: missing a box leaks PII")
    parser.add_argument("--pad", type=int, default=3, help="Box padding in px")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--ocr-check", action="store_true",
                        help="Run EasyOCR on masked output to flag leftover text")
    args = parser.parse_args()

    source = Path(args.source)
    images = ([source] if source.is_file()
              else sorted(p for p in source.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    reader = None
    if args.ocr_check:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

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
        h, w = gray.shape
        masked = gray.copy()
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, int(x1) - args.pad)
            y1 = max(0, int(y1) - args.pad)
            x2 = min(w, int(x2) + args.pad)
            y2 = min(h, int(y2) + args.pad)
            mask_region(masked, x1, y1, x2, y2, args.mode)
            detections.append({"class": model.names[int(box.cls)],
                               "conf": float(box.conf),
                               "box": [x1, y1, x2, y2]})

        cv2.imwrite(str(out_dir / img_path.name), masked)
        entry = {"file": img_path.name, "detections": detections}
        if reader is not None:
            entry["ocr_leaks"] = ocr_leak_check(reader, masked)
        report.append(entry)

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    n_det = sum(len(e["detections"]) for e in report)
    print(f"Masked {n_det} regions in {len(report)} images -> {out_dir}/")
    if reader is not None:
        leaky = [e["file"] for e in report if e.get("ocr_leaks")]
        if leaky:
            print(f"WARNING: OCR still reads text in {len(leaky)} images "
                  f"(see {report_path}):")
            for f in leaky[:10]:
                print(f"  {f}")
        else:
            print("OCR leak check: no readable text left.")


if __name__ == "__main__":
    main()
