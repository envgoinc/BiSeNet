import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import re

# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS — edit only this section
# ═══════════════════════════════════════════════════════════════════

# ── I/O ─────────────────────────────────────────────────────────────
BASE_DIR = Path("/app/mar23rd_data/merged/")
INPUT_DIR   = BASE_DIR / "labels_png"
OUTPUT_DIR  = BASE_DIR / "labels_png_coloured"

# Tight boat reference mask (black boat on white, centred)
BOAT_REFERENCE_MASK = Path("/app/BiSeNet/data_birdseye_long_selection/labels/bev_C_00094m.png")

# Optional: folder of original images for overlay output.
# Set to None to skip overlay generation.
OVERLAY_IMAGE_DIR   = None #Path("/app/mar15th_data/m15_copy/images")

# If image filenames use a different prefix than label filenames, specify the
# find→replace so the stem lookup works correctly.
# e.g. labels:  "label_2025-09-03_frame000001.jpg"
#      images:  "image_2025-09-03_frame000001.jpg"
# → OVERLAY_IMAGE_STEM_REPLACE = ("label_", "image_")
# Set to None if stems are already identical.
OVERLAY_IMAGE_STEM_REPLACE = ("label_", "image_")   # (old, new) or None

# Output for overlay images (only used when OVERLAY_IMAGE_DIR is set)
OVERLAY_OUTPUT_DIR  = BASE_DIR / "debug_overlays"

# Debug folder for step 1
SMALL_BLOB_DEBUG_DIR = BASE_DIR / "debug_small_blob_removal_debug"

# ── Try-only mode ───────────────────────────────────────────────────
# Set to an integer N to process N randomly chosen images,
# or None to process all images.
TRY_ONLY = None   # e.g. 20

# ── Step 1: temporal small-blob removal ─────────────────────────────
ENABLE_TEMPORAL_REMOVAL = False

# Blobs with area (px²) BELOW this threshold are candidates for removal
SMALL_BLOB_MIN_AREA = 600

# A candidate blob must have this fractional overlap (IoU-style: intersection/
# candidate-area) with its counterpart in a neighbouring frame to "count"
SMALL_BLOB_OVERLAP_THRESH = 0.80

# The blob must appear (overlap-confirmed) in at least this many of the
# 5 frames (current ± 2) to be kept; otherwise it is removed
SMALL_BLOB_MIN_FRAMES = 2

# ── Step 2: boat removal ─────────────────────────────────────────────
# Centre-search radius (px) to find a seed pixel for flood-fill
BOAT_SEED_RADIUS = 10

# Blobs newly orphaned after boat removal, with area < this, become water
ORPHAN_MAX_AREA = 15000

# ── Step 3: erosion / dilation of objects ───────────────────────────
# Erosion kernel size (px); set to 0 to skip erosion
ERODE_KERNEL_SIZE  = 10

# Optional dilation after erosion to partially restore; set to 0 to skip
DILATE_KERNEL_SIZE = 8

# ── Overlay transparency ─────────────────────────────────────────────
# Alpha of the colour mask when blended over the original image (0–1)
OVERLAY_ALPHA = 0.45

# ── Misc ─────────────────────────────────────────────────────────────
BINARY_THRESHOLD = 128   # pixels darker than this → "foreground" (black)


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

RGB_BOAT  = (255,   0,   0)   # RED
RGB_OBJ   = (  0, 255,   0)   # GREEN
RGB_WATER = (  0,   0, 255)   # BLUE


def load_binary(path: Path) -> np.ndarray:
    """Load image → strict binary uint8: 0 = foreground, 255 = background."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    _, binary = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def get_blobs(binary: np.ndarray):
    """
    Return connected-component info for all foreground (black=0) blobs.
    Returns: (n_labels, label_map, stats, centroids)
    Background label 0 is included but typically ignored.
    """
    fg = (binary == 0).astype(np.uint8)
    return cv2.connectedComponentsWithStats(fg, connectivity=8)


def extract_largest_blob_mask(binary: np.ndarray) -> np.ndarray:
    """Return uint8 mask of the largest black blob (used for reference)."""
    n, labels, stats, _ = get_blobs(binary)
    if n < 2:
        raise ValueError("No foreground blob found in reference image.")
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == best).astype(np.uint8)


def centre_stamp(ref_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Place ref_mask (uint8 0/1) dead-centre on an h×w canvas."""
    rh, rw = ref_mask.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    dst_y = (h - rh) // 2;  dst_x = (w - rw) // 2
    src_y0 = max(0, -dst_y); dst_y0 = max(0, dst_y)
    src_x0 = max(0, -dst_x); dst_x0 = max(0, dst_x)
    copy_h = min(rh - src_y0, h - dst_y0)
    copy_w = min(rw - src_x0, w - dst_x0)
    if copy_h > 0 and copy_w > 0:
        canvas[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
            ref_mask[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
    return canvas


def parse_frame_info(path: Path):
    """
    Extract (sub_experiment_id, frame_number) from a filename like:
        label_2025-09-03-14-02-06_frame003201.jpg
    Returns (None, None) if parsing fails.
    """
    m = re.search(r'(.+)_frame(\d+)', path.stem)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def build_frame_index(all_paths):
    """
    Return dict: {(sub_exp, frame_no): Path}
    """
    idx = {}
    for p in all_paths:
        sub, frm = parse_frame_info(p)
        if sub is not None:
            idx[(sub, frm)] = p
    return idx


def get_neighbour_paths(path: Path, frame_idx: dict, window: int = 2):
    """
    Return a list of 5 paths: the current frame + ±window neighbours,
    staying within the same sub-experiment.  If the sequence boundary
    is hit, extra frames are borrowed from the other direction.
    """
    sub, frm = parse_frame_info(path)
    if sub is None:
        return [path]

    # Collect all frame numbers for this sub-experiment
    sub_frames = sorted(fn for (s, fn) in frame_idx if s == sub)
    if frm not in sub_frames:
        return [path]

    pos = sub_frames.index(frm)
    total = len(sub_frames)

    offsets = []
    for d in range(-window, window + 1):
        idx_candidate = pos + d
        if 0 <= idx_candidate < total:
            offsets.append(sub_frames[idx_candidate])
        # If out of range on one side, grab from the other
        elif idx_candidate < 0:
            mirror = pos + (window - d)   # push further into future
            if 0 <= mirror < total:
                offsets.append(sub_frames[mirror])
        else:  # idx_candidate >= total
            mirror = pos - (window + (d - window))
            if 0 <= mirror < total:
                offsets.append(sub_frames[mirror])

    # Deduplicate while preserving order, keep up to 2*window+1 entries
    seen = set()
    result = []
    for fn in offsets:
        if fn not in seen:
            seen.add(fn)
            result.append(frame_idx[(sub, fn)])
    return result


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — temporal small-blob removal
# ═══════════════════════════════════════════════════════════════════

def temporal_blob_removal(binary: np.ndarray, path: Path,
                           frame_idx: dict, debug_dir: Path) -> np.ndarray:
    """
    For each small blob in `binary`, check its temporal consistency
    across the ±2 neighbourhood.  Remove inconsistent small blobs.
    Returns the (possibly modified) binary image.
    """
    result = binary.copy()
    h, w = binary.shape

    n, labels, stats, _ = get_blobs(binary)
    neighbour_paths = get_neighbour_paths(path, frame_idx, window=2)
    neighbour_binaries = []
    for np_ in neighbour_paths:
        if np_ == path:
            continue
        try:
            neighbour_binaries.append(load_binary(np_))
        except Exception:
            pass

    removed_any = False
    removed_mask = np.zeros((h, w), dtype=np.uint8)

    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= SMALL_BLOB_MIN_AREA:
            continue

        blob_mask = (labels == lbl).astype(np.uint8)
        consistent_frames = 0

        for nb in neighbour_binaries:
            if nb.shape != binary.shape:
                nb = cv2.resize(nb, (w, h), interpolation=cv2.INTER_NEAREST)
            nb_fg = (nb == 0).astype(np.uint8)
            intersection = int(np.sum(blob_mask & nb_fg))
            overlap_frac = intersection / area if area > 0 else 0
            if overlap_frac >= SMALL_BLOB_OVERLAP_THRESH:
                consistent_frames += 1

        # +1 for the current frame itself
        total_present = consistent_frames + 1
        if total_present < SMALL_BLOB_MIN_FRAMES:
            # Remove: set foreground pixels to background (255)
            result[blob_mask == 1] = 255
            removed_mask[blob_mask == 1] = 255
            removed_any = True

    if removed_any:
        # Save debug image: green = removed blobs, grey = original binary
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis_bg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        vis_bg[removed_mask == 255] = (0, 200, 0)
        debug_path = debug_dir / path.name
        cv2.imwrite(str(debug_path), vis_bg)

    return result


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — boat extraction
# ═══════════════════════════════════════════════════════════════════

def extract_boat(binary: np.ndarray, boat_ref_mask: np.ndarray):
    """
    Use the reference mask (stamped at centre) as the boat region.
    Returns:
        boat_mask   : uint8 0/1  — exact boat pixels
        binary_no_boat : binary with boat pixels set to 255 (water)
    """
    h, w = binary.shape
    stamped = centre_stamp(boat_ref_mask, h, w)   # 0/1

    # The boat region is everything under the reference stamp that is
    # also foreground in the binary (avoids eating into actual water)
    boat_mask = (stamped == 1).astype(np.uint8)

    binary_no_boat = binary.copy()
    binary_no_boat[boat_mask == 1] = 255   # treat as water for further steps

    return boat_mask, binary_no_boat


def remove_orphan_blobs(binary_no_boat: np.ndarray, boat_mask: np.ndarray):
    """
    After boat removal, some previously-connected blobs may become
    isolated.  Any such blob < ORPHAN_MAX_AREA becomes water.
    Returns cleaned binary (boat pixels remain 255).
    """
    result = binary_no_boat.copy()
    n, labels, stats, _ = get_blobs(result)
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < ORPHAN_MAX_AREA:
            result[labels == lbl] = 255
    return result


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — erode/dilate objects
# ═══════════════════════════════════════════════════════════════════

def erode_dilate_objects(binary_no_boat: np.ndarray):
    """
    Erode each foreground blob.  Pixels lost to erosion become water.
    Optionally dilate afterwards to partially restore.
    Returns the cleaned binary (foreground = objects, background = water).
    """
    if ERODE_KERNEL_SIZE <= 0 and DILATE_KERNEL_SIZE <= 0:
        return binary_no_boat

    fg = (binary_no_boat == 0).astype(np.uint8)

    if ERODE_KERNEL_SIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ERODE_KERNEL_SIZE * 2 + 1,) * 2)
        fg = cv2.erode(fg, k, iterations=1)

    if DILATE_KERNEL_SIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE * 2 + 1,) * 2)
        fg = cv2.dilate(fg, k, iterations=1)

    result = np.full_like(binary_no_boat, 255)
    result[fg == 1] = 0
    return result


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — assemble RGB mask
# ═══════════════════════════════════════════════════════════════════

def assemble_rgb(boat_mask: np.ndarray,
                 object_binary: np.ndarray,
                 h: int, w: int) -> np.ndarray:
    """
    Build a 3-channel BGR image (OpenCV convention):
        BLUE  channel → water  (background)
        GREEN channel → objects
        RED   channel → boat
    Everything not boat and not object → water.
    """
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Default everything to water (BLUE in BGR = channel 0)
    rgb[:, :] = (RGB_WATER[2], RGB_WATER[1], RGB_WATER[0])   # BGR

    # Objects (GREEN)
    obj_mask = (object_binary == 0)
    rgb[obj_mask] = (RGB_OBJ[2], RGB_OBJ[1], RGB_OBJ[0])     # BGR

    # Boat (RED) — painted last so it takes priority over objects
    boat_bool = (boat_mask == 1)
    rgb[boat_bool] = (RGB_BOAT[2], RGB_BOAT[1], RGB_BOAT[0])  # BGR

    return rgb


def save_overlay(rgb_mask: np.ndarray, image_path: Path,
                 out_path: Path):
    """Blend rgb_mask over the original image and save."""
    orig = cv2.imread(str(image_path))
    if orig is None:
        return
    h, w = rgb_mask.shape[:2]
    orig_r = cv2.resize(orig, (w, h))
    blended = cv2.addWeighted(orig_r, 1 - OVERLAY_ALPHA,
                               rgb_mask, OVERLAY_ALPHA, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load reference ──────────────────────────────────────────────
    print("Loading boat reference mask …")
    ref_bin      = load_binary(BOAT_REFERENCE_MASK)
    boat_ref_mask = extract_largest_blob_mask(ref_bin)   # 0/1 uint8
    print(f"  Boat reference area: {int(np.sum(boat_ref_mask))} px²")

    # ── Gather all frame paths ───────────────────────────────────────
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_paths = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in exts)
    # Exclude reference itself from processing if present
    # all_paths = [p for p in all_paths if p.resolve() != BOAT_REFERENCE_MASK.resolve()]

    if not all_paths:
        print(f"No images found in {INPUT_DIR}. Exiting.")
        return

    # ── Build frame index BEFORE sampling (so neighbours are always reachable)
    frame_idx = build_frame_index(all_paths)

    # ── TRY_ONLY sampling ────────────────────────────────────────────
    if TRY_ONLY is not None:
        all_paths = random.sample(all_paths, min(TRY_ONLY, len(all_paths)))
        all_paths = sorted(all_paths)
        print(f"TRY_ONLY={TRY_ONLY}: processing {len(all_paths)} randomly sampled frames.")
    else:
        print(f"Processing all {len(all_paths)} frames.")

    # ── Main loop ────────────────────────────────────────────────────
    print(f"\nOutput → {OUTPUT_DIR}\n")

    for img_path in tqdm(all_paths, desc="RGB mask pipeline", unit="frame"):
        try:
            binary = load_binary(img_path)
        except Exception as e:
            tqdm.write(f"  SKIP {img_path.name}: {e}")
            continue

        h, w = binary.shape

        # ── Step 1 ──────────────────────────────────────────────────
        if ENABLE_TEMPORAL_REMOVAL:
            binary = temporal_blob_removal(
                binary, img_path, frame_idx, SMALL_BLOB_DEBUG_DIR)

        # ── Step 2 ──────────────────────────────────────────────────
        boat_mask, binary_no_boat = extract_boat(binary, boat_ref_mask)
        binary_no_boat = remove_orphan_blobs(binary_no_boat, boat_mask)

        # ── Step 3 ──────────────────────────────────────────────────
        object_binary = erode_dilate_objects(binary_no_boat)

        # ── Step 4 ──────────────────────────────────────────────────
        rgb = assemble_rgb(boat_mask, object_binary, h, w)

        # Save RGB mask as PNG (lossless)
        out_stem = img_path.stem
        out_path = OUTPUT_DIR / (out_stem + ".png")
        cv2.imwrite(str(out_path), rgb)

        # Optional overlay
        if OVERLAY_IMAGE_DIR is not None:
            overlay_dir = Path(OVERLAY_IMAGE_DIR)
            # Build the candidate stem, optionally replacing a filename prefix
            candidate_stem = out_stem
            if OVERLAY_IMAGE_STEM_REPLACE is not None:
                old_pfx, new_pfx = OVERLAY_IMAGE_STEM_REPLACE
                if candidate_stem.startswith(old_pfx):
                    candidate_stem = new_pfx + candidate_stem[len(old_pfx):]

            orig_found = None
            # 1) Try exact stem match (with or without prefix replacement)
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = overlay_dir / (candidate_stem + ext)
                if candidate.exists():
                    orig_found = candidate
                    break

            # 2) Fallback: search for any file containing the frame token
            if orig_found is None:
                frame_token_match = re.search(r'_frame\d+', out_stem)
                if frame_token_match:
                    frame_token = frame_token_match.group(0)  # e.g. "_frame003201"
                    for f in overlay_dir.iterdir():
                        if frame_token in f.stem and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                            orig_found = f
                            break

            if orig_found is not None:
                ov_out = OVERLAY_OUTPUT_DIR / (out_stem + "_overlay.jpg")
                save_overlay(rgb, orig_found, ov_out)
            else:
                tqdm.write(f"  No source image found for overlay: {out_stem}")

    print(f"\nDone!  RGB masks saved to: {OUTPUT_DIR}")
    if ENABLE_TEMPORAL_REMOVAL:
        print(f"  Small-blob debug frames: {SMALL_BLOB_DEBUG_DIR}")
    if OVERLAY_IMAGE_DIR:
        print(f"  Overlay images: {OVERLAY_OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# """
# RGB Mask Pipeline
# =================
# Converts binary mask images (black=boat+objects, white=water) into
# 3-class RGB masks:
#     RED   (255,   0,   0) = boat
#     GREEN (  0, 255,   0) = land / objects
#     BLUE  (  0,   0, 255) = water / background

# Pipeline steps
# --------------
#   STEP 1 (toggleable): Temporal small-blob removal
#       Look ±2 frames within the same sub-experiment.  Any blob smaller
#       than SMALL_BLOB_MIN_AREA px² that does NOT have ≥ SMALL_BLOB_OVERLAP_THRESH
#       overlap across ≥ SMALL_BLOB_MIN_FRAMES out of the 5-frame window is
#       removed (set to water).  Debug images saved to
#       OUTPUT_DIR / "small_blob_removal_debug/".

#   STEP 2: Boat cut-out via reference mask
#       The reference mask stamped at centre defines the boat region → RED.
#       Isolated blobs newly orphaned after boat removal that are
#       < ORPHAN_MAX_AREA px² become water.

#   STEP 3: Erosion / dilation of remaining object blobs
#       Each remaining black blob is eroded by ERODE_KERNEL_SIZE px.
#       Pixels eroded away become water.  An optional dilation
#       (DILATE_KERNEL_SIZE, default 0) can partially restore the mask.

#   STEP 3b (toggleable): Polygon contour snap
#       Per-blob edge refinement to remove waviness and snap to real image
#       edges.  For each object blob:
#         a) Extract contour → simplify with Ramer-Douglas-Peucker (RDP).
#            RDP alone removes most waviness without needing any image data.
#         b) For each polygon segment, search a ±SNAP_SEARCH_RADIUS px band
#            in the source RGB image for Canny edges.  If a strong parallel
#            edge line is found within the band, translate that segment to it.
#         c) Re-rasterize the snapped polygon → new object mask.
#       Works even on blurry images: the polygon shape comes from the mask;
#       image edges only refine the boundary position.
#       Debug images (original contour in yellow, snapped polygon in cyan,
#       edge map in faint red) saved to POLYGON_SNAP_DEBUG_DIR if set.

#   STEP 4: Assemble RGB output
#       Boat → RED, surviving object blobs → GREEN, everything else → BLUE.
#       Optional: if OVERLAY_IMAGE_DIR is set, also save semi-transparent
#       mask overlays on the original images.

# All parameters live at the top of this file — no CLI arguments.
# """

# import cv2
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import random
# import re

# # ═══════════════════════════════════════════════════════════════════
# #  PARAMETERS — edit only this section
# # ═══════════════════════════════════════════════════════════════════

# # ── I/O ─────────────────────────────────────────────────────────────
# INPUT_DIR   = Path("/app/mar15th_data/m15_copy/labels")
# OUTPUT_DIR  = Path("/app/mar15th_data/m15_copy/rgb_masks15")

# # Tight boat reference mask (black boat on white, centred)
# BOAT_REFERENCE_MASK = Path("/app/BiSeNet/data_birdseye_long_selection/labels/bev_C_00094m.png")

# # Optional: folder of original images for overlay output AND polygon snap.
# # Set to None to skip overlay generation (snap will then use RDP only).
# OVERLAY_IMAGE_DIR = Path("/app/mar15th_data/m15_copy/images")

# # If image filenames use a different prefix than label filenames, specify the
# # find→replace so the stem lookup works correctly.
# # e.g. labels:  "label_2025-09-03_frame000001.jpg"
# #      images:  "image_2025-09-03_frame000001.jpg"
# # → OVERLAY_IMAGE_STEM_REPLACE = ("label_", "image_")
# # Set to None if stems are already identical.
# OVERLAY_IMAGE_STEM_REPLACE = ("label_", "image_")

# # Output for overlay images (only used when OVERLAY_IMAGE_DIR is set)
# OVERLAY_OUTPUT_DIR   = OUTPUT_DIR / "overlays"

# # Debug folder for step 1
# SMALL_BLOB_DEBUG_DIR = OUTPUT_DIR / "small_blob_removal_debug"

# # Debug folder for step 3b polygon snap.
# # Each debug image shows: yellow = original contour, cyan = snapped polygon,
# # faint red = Canny edge map used for snapping.
# # Set to None to disable debug output.
# POLYGON_SNAP_DEBUG_DIR = OUTPUT_DIR / "polygon_snap_debug"

# # ── Try-only mode ───────────────────────────────────────────────────
# # Set to an integer N to process N randomly chosen images,
# # or None to process all images.
# TRY_ONLY = 100

# # ── Step 1: temporal small-blob removal ─────────────────────────────
# ENABLE_TEMPORAL_REMOVAL = True

# # Blobs with area (px²) BELOW this threshold are candidates for removal
# SMALL_BLOB_MIN_AREA = 5000

# # A candidate blob must have this fractional overlap (intersection /
# # candidate-area) with its counterpart in a neighbouring frame to "count"
# SMALL_BLOB_OVERLAP_THRESH = 0.80

# # The blob must appear (overlap-confirmed) in at least this many of the
# # 5 frames (current ± 2) to be kept; otherwise it is removed
# SMALL_BLOB_MIN_FRAMES = 2

# # ── Step 2: boat removal ─────────────────────────────────────────────
# BOAT_SEED_RADIUS = 10

# # Blobs newly orphaned after boat removal, with area < this, become water
# ORPHAN_MAX_AREA = 15000

# # ── Step 3: erosion / dilation of objects ───────────────────────────
# # Erosion kernel size (px); set to 0 to skip erosion
# ERODE_KERNEL_SIZE  = 10

# # Optional dilation after erosion to partially restore; set to 0 to skip
# DILATE_KERNEL_SIZE = 8

# # ── Step 3b: polygon contour snap ───────────────────────────────────
# ENABLE_POLYGON_SNAP = True

# # RDP simplification epsilon (px).
# # Higher → fewer vertices, straighter/simpler edges.
# # Even without edge snapping, RDP alone removes most waviness.
# # Recommended range: 1.0 – 6.0.  Start at 2.0, increase if still wavy.
# RDP_EPSILON = 2.5

# # Minimum blob area (px²) to attempt polygon snap.
# # Very small blobs are left as-is.
# SNAP_MIN_BLOB_AREA = 500

# # How far (px) from each polygon segment to search for a real image edge.
# # The algorithm sweeps offsets 0 … SNAP_SEARCH_RADIUS in both directions
# # perpendicular to each segment.
# SNAP_SEARCH_RADIUS = 12

# # Fraction of segment sample-points that must land on a Canny edge pixel
# # for that offset to be accepted as a snap candidate.
# # 0.20 = 20 % of points along the segment must have an edge there.
# SNAP_MIN_EDGE_DENSITY = 0.20

# # Canny thresholds.  Lower values detect more edges on blurry images.
# CANNY_LOW  = 20
# CANNY_HIGH = 60

# # Gaussian blur kernel applied to the source image before Canny.
# # Use an odd number ≥ 3, or 0 to skip pre-blur.
# CANNY_BLUR_KERNEL = 3

# # ── Overlay transparency ─────────────────────────────────────────────
# OVERLAY_ALPHA = 0.45

# # ── Misc ─────────────────────────────────────────────────────────────
# BINARY_THRESHOLD = 128   # pixels darker than this → "foreground" (black)


# # ═══════════════════════════════════════════════════════════════════
# #  COLOUR CONSTANTS  (BGR — OpenCV convention)
# # ═══════════════════════════════════════════════════════════════════

# _BOAT_BGR  = (0,   0, 255)   # RED   in RGB
# _OBJ_BGR   = (0, 255,   0)   # GREEN in RGB
# _WATER_BGR = (255, 0,   0)   # BLUE  in RGB


# # ═══════════════════════════════════════════════════════════════════
# #  HELPERS
# # ═══════════════════════════════════════════════════════════════════

# def load_binary(path: Path) -> np.ndarray:
#     """Load image → strict binary uint8: 0 = foreground, 255 = background."""
#     img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read: {path}")
#     _, binary = cv2.threshold(img, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
#     return binary


# def load_colour(path: Path):
#     return cv2.imread(str(path))


# def get_blobs(binary: np.ndarray):
#     fg = (binary == 0).astype(np.uint8)
#     return cv2.connectedComponentsWithStats(fg, connectivity=8)


# def extract_largest_blob_mask(binary: np.ndarray) -> np.ndarray:
#     n, labels, stats, _ = get_blobs(binary)
#     if n < 2:
#         raise ValueError("No foreground blob found in reference image.")
#     best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
#     return (labels == best).astype(np.uint8)


# def centre_stamp(ref_mask: np.ndarray, h: int, w: int) -> np.ndarray:
#     rh, rw = ref_mask.shape
#     canvas = np.zeros((h, w), dtype=np.uint8)
#     dst_y = (h - rh) // 2;  dst_x = (w - rw) // 2
#     src_y0 = max(0, -dst_y); dst_y0 = max(0, dst_y)
#     src_x0 = max(0, -dst_x); dst_x0 = max(0, dst_x)
#     copy_h = min(rh - src_y0, h - dst_y0)
#     copy_w = min(rw - src_x0, w - dst_x0)
#     if copy_h > 0 and copy_w > 0:
#         canvas[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
#             ref_mask[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
#     return canvas


# def parse_frame_info(path: Path):
#     m = re.search(r'(.+)_frame(\d+)', path.stem)
#     if m:
#         return m.group(1), int(m.group(2))
#     return None, None


# def build_frame_index(all_paths):
#     idx = {}
#     for p in all_paths:
#         sub, frm = parse_frame_info(p)
#         if sub is not None:
#             idx[(sub, frm)] = p
#     return idx


# def get_neighbour_paths(path: Path, frame_idx: dict, window: int = 2):
#     sub, frm = parse_frame_info(path)
#     if sub is None:
#         return [path]
#     sub_frames = sorted(fn for (s, fn) in frame_idx if s == sub)
#     if frm not in sub_frames:
#         return [path]
#     pos   = sub_frames.index(frm)
#     total = len(sub_frames)
#     offsets = []
#     for d in range(-window, window + 1):
#         ic = pos + d
#         if 0 <= ic < total:
#             offsets.append(sub_frames[ic])
#         elif ic < 0:
#             mirror = pos + (window - d)
#             if 0 <= mirror < total:
#                 offsets.append(sub_frames[mirror])
#         else:
#             mirror = pos - (window + (d - window))
#             if 0 <= mirror < total:
#                 offsets.append(sub_frames[mirror])
#     seen, result = set(), []
#     for fn in offsets:
#         if fn not in seen:
#             seen.add(fn)
#             result.append(frame_idx[(sub, fn)])
#     return result


# def find_source_image(label_stem: str):
#     """Return the corresponding source image Path, or None."""
#     if OVERLAY_IMAGE_DIR is None:
#         return None
#     overlay_dir = Path(OVERLAY_IMAGE_DIR)
#     candidate_stem = label_stem
#     if OVERLAY_IMAGE_STEM_REPLACE is not None:
#         old_pfx, new_pfx = OVERLAY_IMAGE_STEM_REPLACE
#         if candidate_stem.startswith(old_pfx):
#             candidate_stem = new_pfx + candidate_stem[len(old_pfx):]
#     for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
#         c = overlay_dir / (candidate_stem + ext)
#         if c.exists():
#             return c
#     # fallback: match by frame token
#     ft = re.search(r'_frame\d+', label_stem)
#     if ft:
#         token = ft.group(0)
#         for f in overlay_dir.iterdir():
#             if token in f.stem and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
#                 return f
#     return None


# # ═══════════════════════════════════════════════════════════════════
# #  STEP 1 — temporal small-blob removal
# # ═══════════════════════════════════════════════════════════════════

# def temporal_blob_removal(binary, path, frame_idx, debug_dir):
#     result = binary.copy()
#     h, w   = binary.shape
#     n, labels, stats, _ = get_blobs(binary)

#     nb_paths = get_neighbour_paths(path, frame_idx, window=2)
#     nb_bins  = []
#     for np_ in nb_paths:
#         if np_ == path:
#             continue
#         try:
#             nb_bins.append(load_binary(np_))
#         except Exception:
#             pass

#     removed_any  = False
#     removed_mask = np.zeros((h, w), dtype=np.uint8)

#     for lbl in range(1, n):
#         area = int(stats[lbl, cv2.CC_STAT_AREA])
#         if area >= SMALL_BLOB_MIN_AREA:
#             continue
#         blob_mask = (labels == lbl).astype(np.uint8)
#         consistent = 0
#         for nb in nb_bins:
#             if nb.shape != binary.shape:
#                 nb = cv2.resize(nb, (w, h), interpolation=cv2.INTER_NEAREST)
#             nb_fg = (nb == 0).astype(np.uint8)
#             inter = int(np.sum(blob_mask & nb_fg))
#             if (inter / area if area else 0) >= SMALL_BLOB_OVERLAP_THRESH:
#                 consistent += 1
#         if consistent + 1 < SMALL_BLOB_MIN_FRAMES:
#             result[blob_mask == 1] = 255
#             removed_mask[blob_mask == 1] = 255
#             removed_any = True

#     if removed_any:
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#         vis[removed_mask == 255] = (0, 200, 0)
#         cv2.imwrite(str(debug_dir / path.name), vis)

#     return result


# # ═══════════════════════════════════════════════════════════════════
# #  STEP 2 — boat extraction
# # ═══════════════════════════════════════════════════════════════════

# def extract_boat(binary, boat_ref_mask):
#     h, w      = binary.shape
#     stamped   = centre_stamp(boat_ref_mask, h, w)
#     boat_mask = (stamped == 1).astype(np.uint8)
#     binary_nb = binary.copy()
#     binary_nb[boat_mask == 1] = 255
#     return boat_mask, binary_nb


# def remove_orphan_blobs(binary_no_boat, boat_mask):
#     result = binary_no_boat.copy()
#     n, labels, stats, _ = get_blobs(result)
#     for lbl in range(1, n):
#         if int(stats[lbl, cv2.CC_STAT_AREA]) < ORPHAN_MAX_AREA:
#             result[labels == lbl] = 255
#     return result


# # ═══════════════════════════════════════════════════════════════════
# #  STEP 3 — erode / dilate objects
# # ═══════════════════════════════════════════════════════════════════

# def erode_dilate_objects(binary_no_boat):
#     if ERODE_KERNEL_SIZE <= 0 and DILATE_KERNEL_SIZE <= 0:
#         return binary_no_boat
#     fg = (binary_no_boat == 0).astype(np.uint8)
#     if ERODE_KERNEL_SIZE > 0:
#         k  = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (ERODE_KERNEL_SIZE * 2 + 1,) * 2)
#         fg = cv2.erode(fg, k, iterations=1)
#     if DILATE_KERNEL_SIZE > 0:
#         k  = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE * 2 + 1,) * 2)
#         fg = cv2.dilate(fg, k, iterations=1)
#     result = np.full_like(binary_no_boat, 255)
#     result[fg == 1] = 0
#     return result


# # ═══════════════════════════════════════════════════════════════════
# #  STEP 3b — polygon contour snap
# # ═══════════════════════════════════════════════════════════════════

# def _build_edge_map(src_bgr: np.ndarray) -> np.ndarray:
#     """
#     Canny edge map from source image.  Retries with heavier blur if the
#     initial result is too sparse (handles blurry source images).
#     """
#     gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
#     if CANNY_BLUR_KERNEL > 1:
#         k    = CANNY_BLUR_KERNEL if CANNY_BLUR_KERNEL % 2 == 1 else CANNY_BLUR_KERNEL + 1
#         gray = cv2.GaussianBlur(gray, (k, k), 0)
#     edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
#     # Retry with heavier blur if too sparse
#     if np.count_nonzero(edges) < gray.size * 0.001:
#         gray2  = cv2.GaussianBlur(gray, (7, 7), 0)
#         edges2 = cv2.Canny(gray2, max(5, CANNY_LOW // 2), max(15, CANNY_HIGH // 2))
#         if np.count_nonzero(edges2) > np.count_nonzero(edges):
#             edges = edges2
#     return edges


# def _snap_polygon(poly: np.ndarray, edge_map: np.ndarray,
#                   img_h: int, img_w: int) -> np.ndarray:
#     """
#     Snap each edge of a simplified polygon to the nearest strong image edge.

#     Strategy per segment:
#       - Sample n_samples points evenly along the segment.
#       - Sweep offsets 0 … SNAP_SEARCH_RADIUS in both normal directions.
#       - Accept the offset where the fraction of sample-points that land
#         on an edge pixel exceeds SNAP_MIN_EDGE_DENSITY.
#       - Accumulate per-vertex offset contributions from both adjacent
#         segments; move each vertex by its averaged normal offset.

#     Returns a new polygon array (N×1×2 int32).
#     """
#     pts = poly[:, 0, :].astype(float)   # (N, 2) as (x, y)
#     n   = len(pts)
#     # Per-vertex: list of (normal_vector, offset_px) from adjacent segments
#     vertex_contributions: list = [[] for _ in range(n)]

#     for i in range(n):
#         p1      = pts[i]
#         p2      = pts[(i + 1) % n]
#         seg_vec = p2 - p1
#         seg_len = float(np.linalg.norm(seg_vec))
#         if seg_len < 3:
#             continue

#         tang  = seg_vec / seg_len
#         norml = np.array([-tang[1], tang[0]])   # 90° CCW

#         n_samples = max(6, int(seg_len / 2))
#         ts        = np.linspace(0.0, 1.0, n_samples)
#         base_pts  = p1 + np.outer(ts, seg_vec)  # (n_samples, 2)

#         best_offset = 0
#         best_score  = SNAP_MIN_EDGE_DENSITY - 1e-9

#         for sign in (1, -1):
#             for off in range(1, SNAP_SEARCH_RADIUS + 1):
#                 shifted = base_pts + (sign * off) * norml
#                 xs      = np.clip(np.round(shifted[:, 0]).astype(int), 0, img_w - 1)
#                 ys      = np.clip(np.round(shifted[:, 1]).astype(int), 0, img_h - 1)
#                 score   = np.count_nonzero(edge_map[ys, xs]) / n_samples
#                 if score > best_score:
#                     best_score  = score
#                     best_offset = sign * off

#         if best_offset != 0:
#             vertex_contributions[i].append((norml, best_offset))
#             vertex_contributions[(i + 1) % n].append((norml, best_offset))

#     # Move each vertex by the mean of its contributions
#     new_pts = pts.copy()
#     for vi in range(n):
#         contribs = vertex_contributions[vi]
#         if not contribs:
#             continue
#         delta = np.zeros(2)
#         for nv, off in contribs:
#             delta += nv * off
#         delta      /= len(contribs)
#         new_pts[vi] = pts[vi] + delta

#     new_pts = np.clip(new_pts, 0, [img_w - 1, img_h - 1])
#     return new_pts.astype(np.int32).reshape(-1, 1, 2)


# def polygon_snap_objects(object_binary: np.ndarray,
#                           src_bgr,
#                           debug_path=None) -> np.ndarray:
#     """
#     Refine each object blob's boundary:
#       1. Contour extraction.
#       2. RDP simplification (removes waviness even without an image).
#       3. Edge-snap each segment to the nearest strong image edge.
#       4. Re-rasterize.

#     Returns a new binary mask (same shape/dtype as object_binary).
#     """
#     edge_map = _build_edge_map(src_bgr) if src_bgr is not None else None

#     h, w   = object_binary.shape
#     fg     = (object_binary == 0).astype(np.uint8)
#     result = np.full((h, w), 255, dtype=np.uint8)

#     # Debug canvas
#     if debug_path is not None:
#         dbg = src_bgr.copy() if src_bgr is not None \
#               else cv2.cvtColor(object_binary, cv2.COLOR_GRAY2BGR)
#     else:
#         dbg = None

#     n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

#     for lbl in range(1, n_blobs):
#         area = int(stats[lbl, cv2.CC_STAT_AREA])
#         blob_mask = ((labels == lbl) * 255).astype(np.uint8)

#         if area < SNAP_MIN_BLOB_AREA:
#             result[labels == lbl] = 0   # too small — copy as-is
#             continue

#         contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
#                                         cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             result[labels == lbl] = 0
#             continue

#         contour = max(contours, key=cv2.contourArea)

#         # ── RDP simplification ──────────────────────────────────────
#         poly = cv2.approxPolyDP(contour, RDP_EPSILON, closed=True)
#         if len(poly) < 3:
#             poly = contour   # fallback to raw contour

#         # ── Edge snap ───────────────────────────────────────────────
#         if edge_map is not None and len(poly) >= 3:
#             poly = _snap_polygon(poly, edge_map, h, w)

#         # ── Re-rasterize ────────────────────────────────────────────
#         cv2.fillPoly(result, [poly], 0)

#         # Debug: original contour = yellow, snapped polygon = cyan
#         if dbg is not None:
#             cv2.drawContours(dbg, [contour], -1, (0, 220, 220), 1)
#             cv2.polylines(dbg, [poly], True, (255, 80, 0), 2)

#     # Overlay edge map in faint red on debug image
#     if dbg is not None and edge_map is not None:
#         edge_overlay            = np.zeros_like(dbg)
#         edge_overlay[:, :, 2][edge_map > 0] = 180
#         dbg = cv2.addWeighted(dbg, 1.0, edge_overlay, 0.5, 0)

#     if dbg is not None and debug_path is not None:
#         Path(debug_path).parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(debug_path), dbg)

#     return result


# # ═══════════════════════════════════════════════════════════════════
# #  STEP 4 — assemble RGB mask
# # ═══════════════════════════════════════════════════════════════════

# def assemble_rgb(boat_mask, object_binary, h, w):
#     rgb = np.full((h, w, 3), _WATER_BGR, dtype=np.uint8)
#     rgb[object_binary == 0] = _OBJ_BGR
#     rgb[boat_mask == 1]     = _BOAT_BGR
#     return rgb


# def save_overlay(rgb_mask, image_path, out_path):
#     orig = cv2.imread(str(image_path))
#     if orig is None:
#         return
#     h, w    = rgb_mask.shape[:2]
#     orig_r  = cv2.resize(orig, (w, h))
#     blended = cv2.addWeighted(orig_r, 1 - OVERLAY_ALPHA,
#                                rgb_mask, OVERLAY_ALPHA, 0)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(out_path), blended)


# # ═══════════════════════════════════════════════════════════════════
# #  MAIN
# # ═══════════════════════════════════════════════════════════════════

# def main():
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     print("Loading boat reference mask …")
#     ref_bin       = load_binary(BOAT_REFERENCE_MASK)
#     boat_ref_mask = extract_largest_blob_mask(ref_bin)
#     print(f"  Boat reference area: {int(np.sum(boat_ref_mask))} px²")

#     exts      = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
#     all_paths = sorted(p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in exts)
#     all_paths = [p for p in all_paths
#                  if p.resolve() != BOAT_REFERENCE_MASK.resolve()]
#     if not all_paths:
#         print(f"No images found in {INPUT_DIR}. Exiting.")
#         return

#     # Build frame index BEFORE sampling so temporal neighbours always exist
#     frame_idx = build_frame_index(all_paths)

#     if TRY_ONLY is not None:
#         all_paths = sorted(random.sample(all_paths, min(TRY_ONLY, len(all_paths))))
#         print(f"TRY_ONLY={TRY_ONLY}: processing {len(all_paths)} randomly sampled frames.")
#     else:
#         print(f"Processing all {len(all_paths)} frames.")

#     print(f"\nOutput → {OUTPUT_DIR}\n")

#     for img_path in tqdm(all_paths, desc="RGB mask pipeline", unit="frame"):
#         try:
#             binary = load_binary(img_path)
#         except Exception as e:
#             tqdm.write(f"  SKIP {img_path.name}: {e}")
#             continue

#         h, w = binary.shape

#         # Load source image (used by step 3b and overlays)
#         src_path = find_source_image(img_path.stem)
#         src_bgr  = load_colour(src_path) if src_path else None
#         if src_bgr is not None:
#             src_bgr = cv2.resize(src_bgr, (w, h))

#         # Step 1 ─────────────────────────────────────────────────────
#         if ENABLE_TEMPORAL_REMOVAL:
#             binary = temporal_blob_removal(
#                 binary, img_path, frame_idx, SMALL_BLOB_DEBUG_DIR)

#         # Step 2 ─────────────────────────────────────────────────────
#         boat_mask, binary_no_boat = extract_boat(binary, boat_ref_mask)
#         binary_no_boat = remove_orphan_blobs(binary_no_boat, boat_mask)

#         # Step 3 ─────────────────────────────────────────────────────
#         object_binary = erode_dilate_objects(binary_no_boat)

#         # Step 3b ────────────────────────────────────────────────────
#         if ENABLE_POLYGON_SNAP:
#             dbg_p = None
#             if POLYGON_SNAP_DEBUG_DIR is not None:
#                 dbg_p = Path(POLYGON_SNAP_DEBUG_DIR) / (img_path.stem + "_snap.jpg")
#             object_binary = polygon_snap_objects(object_binary, src_bgr, dbg_p)

#         # Step 4 ─────────────────────────────────────────────────────
#         rgb = assemble_rgb(boat_mask, object_binary, h, w)
#         cv2.imwrite(str(OUTPUT_DIR / (img_path.stem + ".png")), rgb)

#         if src_bgr is not None:
#             save_overlay(rgb, src_path,
#                          OVERLAY_OUTPUT_DIR / (img_path.stem + "_overlay.jpg"))
#         elif OVERLAY_IMAGE_DIR is not None:
#             tqdm.write(f"  No source image found for overlay: {img_path.stem}")

#     print(f"\nDone!  RGB masks saved to: {OUTPUT_DIR}")
#     if ENABLE_TEMPORAL_REMOVAL:
#         print(f"  Step 1 debug  : {SMALL_BLOB_DEBUG_DIR}")
#     if ENABLE_POLYGON_SNAP and POLYGON_SNAP_DEBUG_DIR:
#         print(f"  Step 3b debug : {POLYGON_SNAP_DEBUG_DIR}")
#     if OVERLAY_IMAGE_DIR:
#         print(f"  Overlays      : {OVERLAY_OUTPUT_DIR}")


# if __name__ == "__main__":
#     main()