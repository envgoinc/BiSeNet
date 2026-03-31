#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_LABELS_DIR    = Path("/app/birdseye_run_10/labels")
OUTPUT_LABELS_DIR   = Path("/app/birdseye_run_10/labels_cleaned")
BOAT_REFERENCE_MASK = Path("/app/BiSeNet/data_birdseye_long_selection/labels/bev_C_00094m.png")

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# How close a pixel must be to pure blue or pure green (L-inf distance) to be
# considered that colour rather than a sticker artifact.
# Pure pixels will be distance 0; compression smear on sticker edges may be
# 10-50; genuine sticker RGB will be much higher.
SNAP_THRESHOLD = 30

# BGR values
BLUE_BGR  = np.array([255,   0,   0], dtype=np.int16)
GREEN_BGR = np.array([  0, 255,   0], dtype=np.int16)
RED_BGR   = np.array([  0,   0, 255], dtype=np.uint8)


def process(img_path: Path, ref_mask_black: np.ndarray, out_dir: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [SKIP] Could not read: {img_path}")
        return

    img16 = img.astype(np.int16)

    # L-inf distance to each candidate colour
    dist_blue  = np.max(np.abs(img16 - BLUE_BGR),  axis=2)
    dist_green = np.max(np.abs(img16 - GREEN_BGR), axis=2)

    is_blue  = (dist_blue  <= SNAP_THRESHOLD) & (dist_blue  <= dist_green)
    is_green = (dist_green <= SNAP_THRESHOLD) & (dist_green <  dist_blue)

    # Build clean output: default everything to blue, then stamp green
    out = np.full_like(img, BLUE_BGR)
    out[is_green] = GREEN_BGR

    # Step 2: overlay reference mask black areas as red
    out[ref_mask_black] = RED_BGR

    out_path = out_dir / img_path.name
    ok = cv2.imwrite(str(out_path), out)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")


def main() -> None:
    ref = cv2.imread(str(BOAT_REFERENCE_MASK))
    if ref is None:
        raise FileNotFoundError(f"Cannot read reference mask: {BOAT_REFERENCE_MASK}")

    ref_mask_black = np.all(ref == 0, axis=2)
    print(f"Reference mask black pixels: {ref_mask_black.sum()}")

    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(p for p in INPUT_LABELS_DIR.iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    if not label_files:
        print(f"No images found in {INPUT_LABELS_DIR}")
        return

    # Debug: top 10 colours in first image so you can verify BLUE_BGR / GREEN_BGR are right
    first = cv2.imread(str(label_files[0]))
    flat  = first.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    top = np.argsort(-counts)[:10]
    print("Top 10 BGR colours in first label:")
    for i in top:
        print(f"  BGR {colours[i]} — {counts[i]} pixels")

    print(f"\nProcessing {len(label_files)} images...")
    for i, img_path in enumerate(label_files, 1):
        process(img_path, ref_mask_black, OUTPUT_LABELS_DIR)
        if i % 100 == 0:
            print(f"  {i}/{len(label_files)} done...")

    print(f"\nDone. Output written to: {OUTPUT_LABELS_DIR}")


if __name__ == "__main__":
    main()