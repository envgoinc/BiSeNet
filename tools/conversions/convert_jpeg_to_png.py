import os
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_DIR = "/app/birdseye_run_9/frames/output_mask"   # change this
OUTPUT_DIR = "/app/birdseye_run_9/frames/output_mask_png"  # change this
IS_MASK = True  # True = process as mask, False = image

# =========================
# SETUP
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

files = [
    f for f in os.listdir(INPUT_DIR)
    if os.path.splitext(f.lower())[1] in valid_exts
]

# =========================
# PROCESS
# =========================
for fname in tqdm(files):
    in_path = os.path.join(INPUT_DIR, fname)
    stem, _ = os.path.splitext(fname)
    out_path = os.path.join(OUTPUT_DIR, stem + ".png")

    if IS_MASK:
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Ensure 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # Convert to binary {0,255} using threshold 127
        # Apply per channel
        img = (img > 127).astype(np.uint8) * 255

        # Ensure all channels identical (strict mask)
        # Collapse then expand to enforce consistency
        gray = img[:, :, 0]
        gray = (gray > 127).astype(np.uint8) * 255
        img = np.stack([gray]*3, axis=-1)

    else:
        # Just convert format, preserve image
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

    cv2.imwrite(out_path, img)