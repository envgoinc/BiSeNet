#!/usr/bin/env python3
"""
Create a split preview video from image + binary label pairs.

Layout:
- Final video size: 1184 x 1792
- Top half: binary mask
- Bottom half: RGB image
- FPS: 10 by default
- Frames are emitted in the exact order they appear in the config split entries

Example:
    python make_split_video.py \
        --config /app/mar15th_data/m15/splits.json \
        --image-dir /app/mar15th_data/m15/images \
        --label-dir /app/mar15th_data/m15/labels \
        --split train \
        --output /app/mar15th_data/m15/train_preview.mp4

Notes:
- Assumes filenames follow patterns like:
    image_<experiment>_frame000001.jpg
    label_<experiment>_frame000001.png
- Handles experiment ids using either dashes or underscores.
- Label images can be grayscale or 3-channel binary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_W = 1184
VIDEO_H = 1792
PANEL_W = VIDEO_W
PANEL_H = VIDEO_H // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="Path to splits JSON")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory with source images")
    parser.add_argument("--label-dir", required=True, type=Path, help="Directory with label masks")
    parser.add_argument("--split", required=True, type=str, help="Split name: train / val / test")
    parser.add_argument("--output", required=True, type=Path, help="Output video path")
    parser.add_argument("--fps", type=float, default=10.0, help="Output FPS (default: 10)")
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec (default: mp4v). Alternatives: avc1, XVID",
    )
    return parser.parse_args()


def extract_image_prefix_and_ext(filename: str) -> Tuple[str, str]:
    """
    From:
        image_2025-09-03-14-02-06_frame004081.jpg
    return:
        ('image_2025-09-03-14-02-06_frame', '.jpg')
    """
    m = re.match(r"^(.*_frame)(\d+)(\.[^.]+)$", filename)
    if not m:
        raise ValueError(f"Could not parse image filename pattern: {filename}")
    return m.group(1), m.group(3)


def extract_label_prefix_and_ext(filename: str) -> Tuple[str, str]:
    m = re.match(r"^(.*_frame)(\d+)(\.[^.]+)$", filename)
    if not m:
        raise ValueError(f"Could not parse label filename pattern: {filename}")
    return m.group(1), m.group(3)


def iter_split_pairs(
    split_entries: List[dict],
    image_dir: Path,
    label_dir: Path,
) -> Iterable[Tuple[Path, Path, str]]:
    """
    Yield (image_path, label_path, label_name) in config order.
    """
    for entry in split_entries:
        start_frame = int(entry["start_frame_num"])
        end_frame = int(entry["end_frame_num"])

        img_prefix, img_ext = extract_image_prefix_and_ext(entry["start_image"])
        lbl_prefix, lbl_ext = extract_label_prefix_and_ext(entry["start_label"])

        for frame_num in range(start_frame, end_frame + 1):
            frame_str = f"{frame_num:06d}"
            image_name = f"{img_prefix}{frame_str}{img_ext}"
            label_name = f"{lbl_prefix}{frame_str}{lbl_ext}"

            yield image_dir / image_name, label_dir / label_name, label_name


def fit_with_padding(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Resize while preserving aspect ratio, then center-pad to target size.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image with zero dimension")

    scale = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_binary_mask_bgr(path: Path) -> np.ndarray:
    """
    Read mask fast, threshold to {0,255}, return 3-channel BGR for video.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        # fallback in case it was saved oddly
        tmp = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if tmp is None:
            raise FileNotFoundError(f"Could not read label: {path}")
        mask = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def add_small_text(img: np.ndarray, text: str) -> np.ndarray:
    """
    Draw small readable text with a dark backing strip.
    """
    out = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    margin = 10

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    box_x1 = margin - 4
    box_y1 = margin - 4
    box_x2 = min(out.shape[1] - 1, margin + text_w + 4)
    box_y2 = min(out.shape[0] - 1, margin + text_h + baseline + 4)

    overlay = out.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

    cv2.putText(
        out,
        text,
        (margin, margin + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return out


def make_frame(image_bgr: np.ndarray, mask_bgr: np.ndarray, label_name: str) -> np.ndarray:
    top = fit_with_padding(mask_bgr, PANEL_W, PANEL_H)
    bottom = fit_with_padding(image_bgr, PANEL_W, PANEL_H)

    top = add_small_text(top, label_name)

    frame = np.vstack([top, bottom])
    return frame


def count_expected_frames(split_entries: List[dict]) -> int:
    total = 0
    for entry in split_entries:
        if "count" in entry:
            total += int(entry["count"])
        else:
            total += int(entry["end_frame_num"]) - int(entry["start_frame_num"]) + 1
    return total


def main() -> None:
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)

    splits = config.get("splits", {})
    if args.split not in splits:
        raise KeyError(
            f"Split '{args.split}' not found in config. Available: {sorted(splits.keys())}"
        )

    split_entries = splits[args.split]
    total_frames = count_expected_frames(split_entries)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (VIDEO_W, VIDEO_H))
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {args.output}. Try --codec avc1 or --codec XVID"
        )

    missing = 0

    try:
        pairs = iter_split_pairs(split_entries, args.image_dir, args.label_dir)
        for image_path, label_path, label_name in tqdm(
            pairs,
            total=total_frames,
            desc=f"Writing {args.split}",
            unit="frame",
        ):
            if not image_path.exists():
                print(f"[WARN] Missing image: {image_path}")
                missing += 1
                continue
            if not label_path.exists():
                print(f"[WARN] Missing label: {label_path}")
                missing += 1
                continue

            image_bgr = load_image_bgr(image_path)
            mask_bgr = load_binary_mask_bgr(label_path)
            frame = make_frame(image_bgr, mask_bgr, label_name)
            writer.write(frame)

    finally:
        writer.release()

    print(f"Done: {args.output}")
    print(f"Missing pairs skipped: {missing}")


if __name__ == "__main__":
    main()