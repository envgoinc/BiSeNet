"""
stabilise_masks.py
==================
Temporal filter for RGB segmentation masks.

Each mask pixel is exactly one of:
    RED   (255, 0,   0  )
    GREEN (0,   255, 0  )
    BLUE  (0,   0,   255)

The filter removes frame-to-frame flicker without introducing lag.
It works by keeping a per-pixel running vote across a sliding window
of recent frames, warped into the current frame via optical flow so
moving objects don't get smeared.

Clip boundaries (date changes in filename, or large scene cuts) reset
the filter so temporal context never bleeds across unrelated footage.
"""

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# =============================================================================
# PATHS
# =============================================================================

IMAGE_DIR  = "/app/mar23rd_data/merged/images"
LABEL_DIR  = "/app/mar23rd_data/merged/labels_png_coloured_snapped2"
OUTPUT_DIR = "/app/mar23rd_data/merged/labels_stabilised"
DEBUG_DIR  = "/app/mar23rd_data/merged/debug_stabilised"

# =============================================================================
# MODE
# =============================================================================

DEBUG_MODE = True   # True = fast preview; False = process everything

# In debug mode, process this many clips of this many frames each
DEBUG_N_CLIPS  = 10
DEBUG_N_FRAMES = 50  # consecutive frames from the start of each clip

# =============================================================================
# FILTER PARAMETERS
# =============================================================================

# --- Temporal window ---
# How many past frames to keep in the vote (not counting the current frame).
# Higher = smoother but slower to respond to fast motion.
WINDOW_SIZE = 4

# --- Optical flow (Farneback) ---
# Used to warp past masks into the current frame before voting,
# so the filter tracks moving objects instead of lagging behind them.
FLOW_PYR_SCALE  = 0.5
FLOW_LEVELS     = 3
FLOW_WINSIZE    = 13
FLOW_ITERATIONS = 3
FLOW_POLY_N     = 5
FLOW_POLY_SIGMA = 1.1

# Weight of the current frame's own mask in the vote (past frames each get 1).
# Raise to trust the current frame more; lower to smooth more aggressively.
CURRENT_WEIGHT = 2.0

# --- Scene cut detection ---
# Mean absolute pixel difference between consecutive greyscale frames,
# normalised to [0, 1]. A hard cut typically scores > 0.15.
SCENE_CUT_THRESHOLD = 0.12

# --- Debug video ---
DEBUG_VIDEO_FPS    = 8
DEBUG_VIDEO_FOURCC = "mp4v"

# =============================================================================
# FILE PATTERNS
# =============================================================================

IMAGE_GLOB = "image_*"
LABEL_GLOB = "label_*.png"

# Both separator styles found in the dataset:
#   2025-09-03-14-02-06   hyphens
#   2026_03_04_11_13_36   underscores
_DATE_RE  = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})")
_FRAME_RE = re.compile(r"frame(\d+)", re.IGNORECASE)


# =============================================================================
# HELPERS
# =============================================================================

def sort_key(path):
    dm = _DATE_RE.search(path.stem)
    fm = _FRAME_RE.search(path.stem)
    if dm and fm:
        return ("-".join(dm.groups()), int(fm.group(1)))
    return (path.stem,)


def date_str(path):
    dm = _DATE_RE.search(path.stem)
    return "-".join(dm.groups()) if dm else None


def load_image(path):
    img = cv2.imread(str(path))
    assert img is not None, f"Cannot read image: {path}"
    return img  # BGR


def load_mask(path):
    """Load RGB mask PNG, return as (H, W, 3) uint8 in BGR order."""
    m = cv2.imread(str(path))
    assert m is not None, f"Cannot read mask: {path}"
    return m  # BGR, but values are pure 255/0 per channel


def mask_to_class(mask_bgr):
    """
    Convert BGR mask image to a uint8 class map:
        0 = BLACK / unknown
        1 = RED   (BGR: 0, 0, 255)
        2 = GREEN (BGR: 0, 255, 0)
        3 = BLUE  (BGR: 255, 0, 0)
    """
    b, g, r = mask_bgr[:,:,0], mask_bgr[:,:,1], mask_bgr[:,:,2]
    cls = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    cls[r > 127] = 1
    cls[g > 127] = 2
    cls[b > 127] = 3
    return cls


def class_to_mask_bgr(cls):
    """Convert class map back to BGR mask image."""
    h, w = cls.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[cls == 1] = (0,   0,   255)  # RED   in BGR
    out[cls == 2] = (0,   255, 0  )  # GREEN in BGR
    out[cls == 3] = (255, 0,   0  )  # BLUE  in BGR
    return out


def compute_flow(prev_gray, curr_gray):
    return cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE,
        FLOW_ITERATIONS, FLOW_POLY_N, FLOW_POLY_SIGMA, 0,
    )


def warp_class_map(cls, flow):
    """
    Warp a class map using the given optical flow field.
    Uses nearest-neighbour so class labels stay integer.
    """
    h, w = flow.shape[:2]
    map_x = np.arange(w, dtype=np.float32)[None, :] + flow[:, :, 0]
    map_y = np.arange(h, dtype=np.float32)[:, None] + flow[:, :, 1]
    return cv2.remap(
        cls.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.uint8)


def scene_cut(prev_gray, curr_gray):
    diff = np.mean(np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))) / 255.0
    return diff > SCENE_CUT_THRESHOLD


# =============================================================================
# TEMPORAL FILTER
# =============================================================================

class TemporalFilter:
    """
    Keeps a sliding window of past class maps (flow-warped into the current
    frame) and takes a per-pixel majority vote to produce the output label.

    Call reset() at every clip boundary.
    Call update(curr_gray, curr_cls) for each frame; returns the stabilised
    class map.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._window = []   # list of class maps, oldest first
        self._prev_gray = None

    def update(self, curr_gray, curr_cls):
        """
        curr_gray : (H, W) uint8 greyscale of the current image
        curr_cls  : (H, W) uint8 class map of the current mask
        Returns   : (H, W) uint8 stabilised class map
        """
        n_classes = 4  # 0=bg, 1=red, 2=green, 3=blue

        # --- warp all buffered past frames into current coordinates ---
        warped_window = []
        if self._prev_gray is not None and len(self._window) > 0:
            # Compute flow from previous frame to current frame once
            flow = compute_flow(self._prev_gray, curr_gray)
            for past_cls in self._window:
                warped_window.append(warp_class_map(past_cls, flow))

        # --- vote ---
        # Accumulate per-class score at each pixel
        H, W = curr_cls.shape
        scores = np.zeros((n_classes, H, W), dtype=np.float32)

        # Past frames each count for 1 vote
        for w_cls in warped_window:
            for c in range(n_classes):
                scores[c] += (w_cls == c).astype(np.float32)

        # Current frame counts for CURRENT_WEIGHT votes
        for c in range(n_classes):
            scores[c] += (curr_cls == c).astype(np.float32) * CURRENT_WEIGHT

        # Pixel-wise argmax = winning class
        stabilised = np.argmax(scores, axis=0).astype(np.uint8)

        # --- update window ---
        self._window.append(curr_cls.copy())
        if len(self._window) > WINDOW_SIZE:
            self._window.pop(0)
        self._prev_gray = curr_gray.copy()

        return stabilised


# =============================================================================
# PAIR DISCOVERY + CLIP SPLITTING
# =============================================================================

def discover_pairs(image_dir, label_dir):
    """Match images to labels by (date, frame) key. Returns sorted list of (img_path, lbl_path)."""
    labels = sorted(Path(label_dir).glob(LABEL_GLOB), key=sort_key)
    label_index = {sort_key(l): l for l in labels}

    images = sorted(Path(image_dir).glob(IMAGE_GLOB), key=sort_key)

    pairs = []
    missing = 0
    for img in images:
        k = sort_key(img)
        if k in label_index:
            pairs.append((img, label_index[k]))
        else:
            missing += 1

    print(f"Images: {len(images)}  Labels: {len(labels)}  Matched: {len(pairs)}  Unmatched: {missing}")
    return pairs


def split_into_clips(pairs, max_clips=None):
    """
    Walk pairs in order, splitting at:
      - date changes in filename (hard boundary)
      - scene cuts detected by pixel diff (soft boundary)

    Stops early once max_clips complete clips are found (saves scanning the
    whole dataset in debug mode).

    Returns list of clips, each clip = list of (img_path, lbl_path).
    """
    if not pairs:
        return []

    clips = []
    current = [pairs[0]]
    prev_gray = cv2.cvtColor(load_image(pairs[0][0]), cv2.COLOR_BGR2GRAY)
    prev_date = date_str(pairs[0][0])

    for img_path, lbl_path in tqdm(pairs[1:], desc="Finding clip boundaries", unit="frame"):
        curr_date = date_str(img_path)
        curr_img  = load_image(img_path)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        hard_cut = curr_date != prev_date
        soft_cut = (not hard_cut) and scene_cut(prev_gray, curr_gray)

        if hard_cut or soft_cut:
            clips.append(current)
            current = []
            reason = "date" if hard_cut else "scene"
            tqdm.write(f"  [{reason} cut] {img_path.name}")
            if max_clips is not None and len(clips) >= max_clips:
                tqdm.write(f"  Reached {max_clips} clips, stopping scan.")
                prev_gray = curr_gray
                prev_date = curr_date
                break

        current.append((img_path, lbl_path))
        prev_gray = curr_gray
        prev_date = curr_date

    if current:
        clips.append(current)

    print(f"Found {len(clips)} clips.")
    return clips


# =============================================================================
# DEBUG VIDEO
# =============================================================================

def make_debug_frame(image_bgr, orig_mask_bgr, stab_mask_bgr):
    """Three panels side by side: image | original mask | stabilised mask."""
    H, W = image_bgr.shape[:2]

    # Overlay masks onto image copies at 50% alpha
    def overlay(img, mask):
        out = img.copy()
        fg = np.any(mask > 0, axis=2)
        out[fg] = (img[fg].astype(np.float32) * 0.5 + mask[fg].astype(np.float32) * 0.5).astype(np.uint8)
        return out

    p1 = image_bgr.copy()
    p2 = overlay(image_bgr, orig_mask_bgr)
    p3 = overlay(image_bgr, stab_mask_bgr)

    def label(img, text):
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),     1, cv2.LINE_AA)

    label(p1, "image")
    label(p2, "grabcut")
    label(p3, "stabilised")

    return np.hstack([p1, p2, p3])


def write_video(frames, path):
    if not frames:
        return
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*DEBUG_VIDEO_FOURCC)
    writer = cv2.VideoWriter(str(path), fourcc, DEBUG_VIDEO_FPS, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Debug video → {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if DEBUG_DIR:
        Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)

    # --- discover pairs ---
    pairs = discover_pairs(IMAGE_DIR, LABEL_DIR)
    if not pairs:
        print("No matched pairs found.")
        sys.exit(1)

    # --- split into clips, stopping early in debug mode ---
    max_clips = DEBUG_N_CLIPS if DEBUG_MODE else None
    clips = split_into_clips(pairs, max_clips=max_clips)

    # --- process ---
    filt = TemporalFilter()
    debug_frames = []

    clip_bar = tqdm(clips, desc="Clips", unit="clip")
    for clip in clip_bar:
        filt.reset()

        # In debug mode, only process the first N frames of each clip
        frames_to_process = clip[:DEBUG_N_FRAMES] if DEBUG_MODE else clip

        frame_bar = tqdm(frames_to_process, desc="  Frames", unit="frame", leave=False)
        for img_path, lbl_path in frame_bar:
            img_bgr  = load_image(img_path)
            curr_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            orig_mask = load_mask(lbl_path)
            curr_cls  = mask_to_class(orig_mask)

            stab_cls  = filt.update(curr_gray, curr_cls)
            stab_mask = class_to_mask_bgr(stab_cls)

            # Save
            out_path = Path(OUTPUT_DIR) / lbl_path.name
            cv2.imwrite(str(out_path), stab_mask)

            # Collect debug frame
            if DEBUG_DIR:
                debug_frames.append(make_debug_frame(img_bgr, orig_mask, stab_mask))

    # --- write debug video ---
    if DEBUG_DIR and debug_frames:
        write_video(debug_frames, Path(DEBUG_DIR) / "debug.mp4")

    print("Done.")


if __name__ == "__main__":
    main()