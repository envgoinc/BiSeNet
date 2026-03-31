
"""
Mask Overlay Video Generator
=============================
Overlays RGB masks semi-transparently over source images and writes an MP4.

All parameters are hardcoded at the top — no CLI arguments.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS — edit only this section
# ═══════════════════════════════════════════════════════════════════

# Folder containing the original images
IMAGES_DIR = Path("/app/mar23rd_data/merged/images")

# Folder containing the RGB mask PNGs
MASKS_DIR  = Path("/app/mar23rd_data/merged/labels_png_coloured_snapped2")

# Output video path
OUTPUT_VIDEO = Path("/app/BiSeNet/tools/overlays/full_setm23/view_full_m23_set.mp4")

# Alpha of the mask overlay (0.0 = invisible, 1.0 = fully opaque mask)
OVERLAY_ALPHA = 0.45

# Output video framerate
VIDEO_FPS = 60

# Resize output frames to this size (width, height).
# Set to None to use the native image size.
OUTPUT_SIZE = None   # e.g. (1280, 720)

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Gather and sort both lists independently
    mask_paths  = sorted(p for p in MASKS_DIR.iterdir()  if p.suffix.lower() in exts)
    image_paths = sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts)

    if not mask_paths:
        print(f"No mask files found in {MASKS_DIR}")
        return
    if not image_paths:
        print(f"No image files found in {IMAGES_DIR}")
        return

    print(f"Found {len(mask_paths)} masks and {len(image_paths)} images.")

    if len(mask_paths) != len(image_paths):
        print(f"  WARNING: counts differ — will pair up to min({len(mask_paths)}, {len(image_paths)}) frames.")

    pairs = list(zip(image_paths, mask_paths))
    print(f"Pairing {len(pairs)} frames by sorted order.")
    print(f"  First pair: {pairs[0][0].name}  <->  {pairs[0][1].name}")
    print(f"  Last  pair: {pairs[-1][0].name}  <->  {pairs[-1][1].name}")

    # ── Determine output frame size ─────────────────────────────────
    sample_img = cv2.imread(str(pairs[0][0]))
    if sample_img is None:
        print(f"Could not read sample image: {pairs[0][0]}")
        return

    if OUTPUT_SIZE is not None:
        frame_w, frame_h = OUTPUT_SIZE
    else:
        frame_h, frame_w = sample_img.shape[:2]

    # ── Set up video writer ─────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc,
                              float(VIDEO_FPS), (frame_w, frame_h))

    print(f"\nWriting {frame_w}×{frame_h} @ {VIDEO_FPS}fps → {OUTPUT_VIDEO}\n")

    for img_path, mask_path in tqdm(pairs, desc="Rendering frames", unit="frame"):
        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path))

        if img is None:
            tqdm.write(f"  SKIP (unreadable image): {img_path.name}")
            continue
        if mask is None:
            tqdm.write(f"  SKIP (unreadable mask): {mask_path.name}")
            continue

        img  = cv2.resize(img,  (frame_w, frame_h))
        mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        blended = cv2.addWeighted(img,  1.0 - OVERLAY_ALPHA,
                                   mask, OVERLAY_ALPHA, 0)
        writer.write(blended)

    writer.release()
    print(f"\nDone! Video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()