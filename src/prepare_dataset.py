"""
Build YOLO-ready datasets from data/.

Fixes the image/label name mismatch (images are named *_annotated.png while
labels are *.txt) and generates two preprocessing variants to compare:

  raw    — original grayscale images, untouched.
  hybrid — 3-channel image: [raw, top-hat, CLAHE(raw)].
           Top-hat alone suppresses text that sits on bright anatomy (low local
           contrast), so the raw and CLAHE channels are kept alongside it.

Output:
  datasets/raw/{images,labels}/{train,val}/   + datasets/raw.yaml
  datasets/hybrid/{images,labels}/{train,val}/ + datasets/hybrid.yaml

Usage:
  python prepare_dataset.py [--data data] [--out datasets]
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parent.parent  # repo root (src/ is one level down)
CLASS_NAMES = {0: "name", 1: "id", 2: "age", 3: "date", 4: "time"}


def hybrid_channels(gray: np.ndarray) -> np.ndarray:
    """[raw, top-hat, CLAHE] stacked as a 3-channel image (BGR order on disk)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    return cv2.merge([gray, tophat, clahe])


def base_name(image_path: Path) -> str:
    """Image stem without the _annotated suffix, to match label file names."""
    return image_path.stem.removesuffix("_annotated")


def build_variant(data_root: Path, out_root: Path, variant: str):
    for split in ("train", "val"):
        img_src = data_root / "images" / split
        lbl_src = data_root / "labels" / split
        img_dst = out_root / variant / "images" / split
        lbl_dst = out_root / variant / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        images = sorted(img_src.glob("*.png"))
        missing = 0
        for img_path in tqdm(images, desc=f"{variant}/{split}"):
            stem = base_name(img_path)
            label = lbl_src / f"{stem}.txt"
            if not label.exists():
                missing += 1
                continue
            if variant == "raw":
                dst = img_dst / f"{stem}.png"
                if not dst.exists():
                    dst.symlink_to(img_path.resolve())
            else:
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(str(img_dst / f"{stem}.png"), hybrid_channels(gray))
            shutil.copy(label, lbl_dst / f"{stem}.txt")
        print(f"{variant}/{split}: {len(images) - missing} pairs"
              + (f" ({missing} images without label, skipped)" if missing else ""))

    yaml_path = out_root / f"{variant}.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items())
    yaml_path.write_text(
        f"path: {(out_root / variant).resolve()}\n"
        f"train: images/train\nval: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n\nnames:\n{names}\n"
    )
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data"), help="Source dataset root")
    parser.add_argument("--out", default=str(ROOT / "datasets"), help="Output root")
    parser.add_argument("--variants", nargs="+", default=["raw", "hybrid"],
                        choices=["raw", "hybrid"])
    args = parser.parse_args()

    for variant in args.variants:
        build_variant(Path(args.data), Path(args.out), variant)
