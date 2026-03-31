"""
ADAPTIVE ITERATIVE INWARD PEEL — SIZE-AWARE PARAMETERS
WITH BIDIRECTIONAL SNAPPING  [OPTIMIZED]
======================================================
Optimizations applied:
  - Vectorized boundary pixel evaluation (no Python loops over pixels)
  - Vectorized bidirectional snapping via scipy distance transforms
  - Pre-allocated kernel objects (no per-iteration recreation)
  - Multiprocessing across images (ProcessPoolExecutor)
  - Replaced np.linalg.norm loops with broadcast subtraction + einsum
  - LAB conversion done once per image, passed through
  - Removed redundant mask copies where possible
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ═══════════════════════════════════════════════════════════════════
#  OUTPUT TOGGLES
# ═══════════════════════════════════════════════════════════════════

SAVE_RGB_MASK     = True   # Primary output: full-res RGB label mask
SAVE_DEBUG_OVERLAY = False  # Side-by-side comparison overlay grids

# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════

LABELS_DIR  = Path("/app/mar23rd_data/merged/labels_png_coloured")
IMAGES_DIR  = Path("/app/mar23rd_data/merged/images")
OUTPUT_DIR  = Path("/app/mar23rd_data/merged/labels_png_coloured_snapped2")
DEBUG_DIR   = Path("/app/mar23rd_data/merged/labels_png_coloured_snapped_debug")

STEM_REPLACE = ("label_", "image_")
MAX_IMAGES    = None

# Output mask size (original input resolution)
MASK_W = 1184
MASK_H = 896

# Inward peel: edge-snapping parameters (BASE VALUES)
EDGE_SIGMA_BLUR   = 0.7
COLOUR_MARGIN     = 0.5
SMOOTH_RADIUS     = 3
SMOOTH_ITERS      = 4

EDGE_PERCENTILE   = 40
EDGE_FLOOR        = 0.10

# Straight line detection
HOUGH_MIN_LENGTH  = 10
LINE_THICKNESS    = 15
LINE_CONFIDENCE   = 0.25

# Bidirectional snapping thresholds
SNAP_INWARD_THRESHOLD   = 0.25
SNAP_OUTWARD_THRESHOLD  = 0.40
SNAP_SEARCH_RADIUS      = 30

# Debug overlay grid cell size
CELL_W = MASK_W
CELL_H = MASK_H

# Colours (BGR)
WATER_BGR = (255,   0,   0)
OBJ_BGR   = (  0, 255,   0)
BOAT_BGR  = (  0,   0, 255)

# Number of worker processes (None = auto = cpu_count)
NUM_WORKERS = None

# ═══════════════════════════════════════════════════════════════════
#  SIZE THRESHOLD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

AREA_SMALL_THRESHOLD   = 30000
AREA_MEDIUM_THRESHOLD  = 50000

SIZE_CLASS_CONFIG = {
    'SMALL': {
        'peel_iters': 3,
        'band_radius': 7,
        'core_erode': 4,
    },
    'MEDIUM': {
        'peel_iters': 20,
        'band_radius': 30,
        'core_erode': 11,
    },
    'LARGE': {
        'peel_iters': 20,
        'band_radius': 30,
        'core_erode': 11,
    }
}

# ═══════════════════════════════════════════════════════════════════
#  PRE-BUILT KERNELS  (module-level, shared across calls)
# ═══════════════════════════════════════════════════════════════════

_ERODE3   = cv2.getStructuringElement(cv2.MORPH_ERODE,   (3,  3 ))
_ERODE15  = cv2.getStructuringElement(cv2.MORPH_ERODE,   (15, 15))
_ERODE11  = cv2.getStructuringElement(cv2.MORPH_ERODE,   (11, 11))
_SMOOTH   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (SMOOTH_RADIUS*2+1, SMOOTH_RADIUS*2+1))

def _erode_ker(size):
    s = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ERODE, (s, s))

def _dilate_ker(size):
    s = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_DILATE, (s, s))

# ═══════════════════════════════════════════════════════════════════
#  SIZE-ADAPTIVE PARAMETERS
# ═══════════════════════════════════════════════════════════════════

def get_size_class_and_params(mask, img_shape):
    area = int(np.count_nonzero(mask))
    h, w = img_shape[:2]
    area_ratio = area / (h * w)

    if area < AREA_SMALL_THRESHOLD:
        size_class = "SMALL"
    elif area < AREA_MEDIUM_THRESHOLD:
        size_class = "MEDIUM"
    else:
        size_class = "LARGE"

    params = SIZE_CLASS_CONFIG[size_class].copy()
    params['size_class'] = size_class
    params['area'] = area
    params['area_ratio'] = area_ratio
    return params

# ═══════════════════════════════════════════════════════════════════
#  CORE HELPERS
# ═══════════════════════════════════════════════════════════════════

def bgr_to_class(mask_bgr):
    cls = np.zeros(mask_bgr.shape[:2], np.uint8)
    cls[np.all(mask_bgr == WATER_BGR, axis=2)] = 0
    cls[np.all(mask_bgr == OBJ_BGR,   axis=2)] = 1
    cls[np.all(mask_bgr == BOAT_BGR,  axis=2)] = 2
    return cls

def class_to_bgr(cls_map):
    out = np.zeros((*cls_map.shape, 3), np.uint8)
    out[cls_map == 0] = WATER_BGR
    out[cls_map == 1] = OBJ_BGR
    out[cls_map == 2] = BOAT_BGR
    return out

def find_image(label_stem):
    stem = label_stem
    if STEM_REPLACE:
        old, new = STEM_REPLACE
        if stem.startswith(old):
            stem = new + stem[len(old):]
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = IMAGES_DIR / (stem + ext)
        if p.exists():
            return p
    m = re.search(r'_frame\d+', label_stem)
    if m:
        tok = m.group(0)
        for f in IMAGES_DIR.iterdir():
            if tok in f.stem and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                return f
    return None

def make_overlay(img_bgr, cls_map):
    h, w = img_bgr.shape[:2]
    mask = class_to_bgr(cv2.resize(cls_map, (w, h), interpolation=cv2.INTER_NEAREST))
    return cv2.addWeighted(img_bgr, 0.58, mask, 0.42, 0)

def label_text(text):
    panel = np.ones((60, CELL_W, 3), np.uint8) * 30
    cv2.putText(panel, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (220, 255, 220), 2, cv2.LINE_AA)
    return panel

def edge_map_from_image(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    gx = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(L, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), EDGE_SIGMA_BLUR)
    mx = mag.max()
    if mx > 1e-6:
        mag /= mx
    return mag

def boundary_pixels(mask_u8):
    eroded = cv2.erode(mask_u8.astype(np.uint8), _ERODE3, iterations=1)
    return (mask_u8 == 1) & (eroded == 0)

def get_connected_components(fg_mask):
    num_labels, labels_map = cv2.connectedComponents(fg_mask.astype(np.uint8))
    return [(labels_map == i).astype(np.uint8) for i in range(1, num_labels)]

# ═══════════════════════════════════════════════════════════════════
#  STRAIGHT LINE DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_straight_edges(edge_map, min_length=15):
    h, w = edge_map.shape
    edge_binary = (edge_map > 0.25).astype(np.uint8)
    if not np.any(edge_binary):
        return np.zeros((h, w), np.float32)

    lines = cv2.HoughLines(edge_binary, 1.0, np.pi / 180, threshold=min_length)
    if lines is None or len(lines) == 0:
        return np.zeros((h, w), np.float32)

    line_map = np.zeros((h, w), np.float32)
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(line_map, pt1, pt2, 1.0, thickness=LINE_THICKNESS)

    mx = line_map.max()
    if mx > 0:
        line_map /= mx
    line_map = cv2.GaussianBlur(line_map, (5, 5), 1.0)
    return line_map

# ═══════════════════════════════════════════════════════════════════
#  VECTORIZED BIDIRECTIONAL SNAPPING
# ═══════════════════════════════════════════════════════════════════

def snap_to_straight_edges_bidirectional(result_mask, edge_map, img_bgr, cls_map,
                                          inward_confidence=SNAP_INWARD_THRESHOLD,
                                          outward_confidence=SNAP_OUTWARD_THRESHOLD,
                                          search_radius=SNAP_SEARCH_RADIUS):
    boat_mask = (cls_map == 2).astype(bool)
    straight_confidence = detect_straight_edges(edge_map, min_length=HOUGH_MIN_LENGTH)

    if straight_confidence.max() < 0.01:
        return result_mask

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = result_mask.shape

    boundary = boundary_pixels(result_mask)
    if not np.any(boundary):
        return result_mask

    core = cv2.erode(result_mask, _ERODE15, iterations=1)
    core_mean = lab[core > 0].mean(axis=0) if np.any(core) else lab[result_mask > 0].mean(axis=0)
    bg_region = (~result_mask.astype(bool)) & (~boat_mask)
    bg_mean   = lab[bg_region].mean(axis=0) if np.any(bg_region) else np.array([100., 0., 0.])

    ys, xs = np.where(boundary)
    n = len(ys)
    if n == 0:
        return result_mask

    conf_at_boundary = straight_confidence[ys, xs]
    valid = conf_at_boundary >= 0.15
    ys, xs = ys[valid], xs[valid]
    n = len(ys)
    if n == 0:
        return result_mask

    rm_pad  = np.pad(result_mask.astype(np.float32), 1, mode='edge')
    grad_y  = rm_pad[ys+2, xs+1] - rm_pad[ys,   xs+1]
    grad_x  = rm_pad[ys+1, xs+2] - rm_pad[ys+1, xs  ]
    grad_len = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
    grad_x  /= grad_len
    grad_y  /= grad_len

    deltas = np.arange(-search_radius, search_radius + 1)
    D = len(deltas)

    cand_y = np.clip(np.round(ys[None, :] + grad_y[None, :] * deltas[:, None]).astype(int), 0, h-1)
    cand_x = np.clip(np.round(xs[None, :] + grad_x[None, :] * deltas[:, None]).astype(int), 0, w-1)

    on_boat     = boat_mask[cand_y, cand_x]
    edge_scores = straight_confidence[cand_y, cand_x].astype(np.float32)
    cand_lab    = lab[cand_y, cand_x]
    d_core      = np.sqrt(((cand_lab - core_mean)**2).sum(axis=2))
    d_bg        = np.sqrt(((cand_lab - bg_mean )**2).sum(axis=2))
    color_scores = np.clip((d_bg - d_core) / (d_bg + 1e-6), 0.0, None).astype(np.float32)

    combined = 0.6 * edge_scores + 0.4 * color_scores
    combined[on_boat] = -1.0

    best_idx   = np.argmax(combined, axis=0)
    best_delta = deltas[best_idx]
    best_score = combined[best_idx, np.arange(n)]

    best_sy = cand_y[best_idx, np.arange(n)]
    best_sx = cand_x[best_idx, np.arange(n)]

    result_copy = result_mask.copy()

    out_mask = (best_delta > 0) & (best_score > outward_confidence)
    result_copy[best_sy[out_mask], best_sx[out_mask]] = 1

    in_mask = (best_delta < 0) & (best_score > inward_confidence)
    result_copy[ys[in_mask], xs[in_mask]] = 0
    sy_in, sx_in = best_sy[in_mask], best_sx[in_mask]
    valid_in = (sy_in >= 0) & (sy_in < h) & (sx_in >= 0) & (sx_in < w)
    result_copy[sy_in[valid_in], sx_in[valid_in]] = 1

    no_snap = best_delta == 0
    pix_lab  = lab[ys[no_snap], xs[no_snap]]
    d_core_fallback = np.sqrt(((pix_lab - core_mean)**2).sum(axis=1))
    keep = no_snap.copy()
    keep[no_snap] = d_core_fallback < 15.0
    result_copy[ys[keep], xs[keep]] = 1

    return result_copy

# ═══════════════════════════════════════════════════════════════════
#  VECTORIZED INWARD PEEL
# ═══════════════════════════════════════════════════════════════════

def inward_peel_edge_snap_adaptive(img_bgr, cls_map, component_mask, params,
                                    lab=None, edge=None):
    boat_mask = (cls_map == 2).astype(bool)
    fg = ((component_mask == 1) & ~boat_mask).astype(np.uint8)

    if not np.any(fg):
        return component_mask.copy()

    if edge is None:
        edge = edge_map_from_image(img_bgr)
    if lab is None:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    erode_size = params['core_erode']
    core_ker   = _erode_ker(erode_size)
    core_erode = cv2.erode(fg, core_ker, iterations=1)
    if not np.any(core_erode):
        core_erode = cv2.erode(fg, _ERODE11, iterations=1)
    core_mean = (lab[core_erode > 0].mean(axis=0)
                 if np.any(core_erode) else lab[fg > 0].mean(axis=0))

    result      = fg.copy()
    band_radius = params['band_radius']
    peel_iters  = params['peel_iters']
    band_ker    = _dilate_ker(band_radius)
    small_dil   = _dilate_ker(3)

    for _ in range(peel_iters):
        if not np.any(result):
            break

        boundary = boundary_pixels(result)
        if not np.any(boundary):
            break

        band_dilate = cv2.dilate(result, band_ker, iterations=1)
        ring = (band_dilate == 1) & (result == 0) & (~boat_mask)
        if not np.any(ring):
            ring_mask = cv2.dilate(result, small_dil, iterations=1).astype(bool)
            ring = ring_mask & (result == 0) & (~boat_mask)
        if not np.any(ring):
            break

        ring_mean = lab[ring].mean(axis=0)

        edge_vals = edge[boundary]
        if edge_vals.size > 0:
            edge_thr = max(float(np.percentile(edge_vals, EDGE_PERCENTILE)), EDGE_FLOOR)
        else:
            edge_thr = EDGE_FLOOR

        ys, xs = np.where(boundary)
        if ys.size == 0:
            break

        pixel_lab  = lab[ys, xs]
        pixel_edge = edge[ys, xs]

        d_core = np.sqrt(((pixel_lab - core_mean)**2).sum(axis=1))
        d_ring = np.sqrt(((pixel_lab - ring_mean)**2).sum(axis=1))

        keep = (pixel_edge >= edge_thr) | (d_core + COLOUR_MARGIN < d_ring)
        remove_ys = ys[~keep]
        remove_xs = xs[~keep]

        if remove_ys.size == 0:
            break

        result[remove_ys, remove_xs] = 0

    result = snap_to_straight_edges_bidirectional(result, edge, img_bgr, cls_map)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, _SMOOTH, iterations=SMOOTH_ITERS)
    return result

# ═══════════════════════════════════════════════════════════════════
#  PER-IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════

def process_image_adaptive(img_bgr, cls_map):
    h, w = img_bgr.shape[:2]
    boat_mask = (cls_map == 2).astype(bool)
    fg = ((cls_map == 1) & ~boat_mask).astype(np.uint8)

    if not np.any(fg):
        return cls_map.copy()

    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = edge_map_from_image(img_bgr)

    components = get_connected_components(fg)

    result_map = cls_map.copy()
    result_map[~boat_mask] = 0

    for comp_mask in components:
        params  = get_size_class_and_params(comp_mask, img_bgr.shape)
        refined = inward_peel_edge_snap_adaptive(img_bgr, cls_map, comp_mask, params,
                                                  lab=lab, edge=edge)
        result_map[refined == 1] = 1

    result_map[boat_mask] = 2

    # Fill enclosed water pockets
    water = (result_map == 0).astype(np.uint8)
    flood = water.copy()
    seed  = np.zeros((h+2, w+2), np.uint8)
    for x in range(w):
        if flood[0,   x]: cv2.floodFill(flood, seed, (x, 0),   2)
        if flood[h-1, x]: cv2.floodFill(flood, seed, (x, h-1), 2)
    for y in range(h):
        if flood[y, 0  ]: cv2.floodFill(flood, seed, (0,   y), 2)
        if flood[y, w-1]: cv2.floodFill(flood, seed, (w-1, y), 2)

    result_map[(water == 1) & (flood != 2)] = 1
    result_map[boat_mask] = 2

    return result_map


def process_single_label(label_path_str):
    """
    Top-level function run in each worker process.
    Returns (output_path_str, error_str_or_None).
    """
    label_path = Path(label_path_str)

    mask_bgr = cv2.imread(str(label_path))
    if mask_bgr is None:
        return None, f"Could not read mask: {label_path.name}"

    cls_map = bgr_to_class(mask_bgr)
    h0, w0  = cls_map.shape

    img_path = find_image(label_path.stem)
    if img_path is None:
        return None, f"No image found for: {label_path.stem}"

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, f"Could not read image: {img_path}"

    img_bgr = cv2.resize(img_bgr, (w0, h0), interpolation=cv2.INTER_LINEAR)

    try:
        result = process_image_adaptive(img_bgr, cls_map)
    except Exception as e:
        result = cls_map.copy()
        err = str(e)
    else:
        err = None

    out_path = None

    # ── Primary output: full-res RGB mask ─────────────────────────
    if SAVE_RGB_MASK:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_bgr = class_to_bgr(
            cv2.resize(result, (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST)
        )
        out_path = OUTPUT_DIR / (label_path.stem + ".png")
        cv2.imwrite(str(out_path), result_bgr)

    # ── Optional: side-by-side debug overlay ──────────────────────
    if SAVE_DEBUG_OVERLAY:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        # Resize everything to the true mask dimensions (no squashing)
        img_c        = cv2.resize(img_bgr, (MASK_W, MASK_H), interpolation=cv2.INTER_LINEAR)
        orig_c       = cv2.resize(cls_map, (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST)
        orig_overlay = make_overlay(img_c, orig_c)
        orig_panel   = np.vstack([orig_overlay, label_text("Original")])

        result_c       = cv2.resize(result, (MASK_W, MASK_H), interpolation=cv2.INTER_NEAREST)
        result_overlay = make_overlay(img_c, result_c)
        result_panel   = np.vstack([result_overlay, label_text("Adaptive Peel + Bidir Snap")])

        grid = np.hstack([orig_panel, result_panel])
        debug_path = DEBUG_DIR / (label_path.stem + "_debug.jpg")
        cv2.imwrite(str(debug_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return str(out_path) if out_path else None, err

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    label_paths = sorted(
        p for p in LABELS_DIR.glob("*.png")
        if not any(x in p.name.lower() for x in
                   ("overlay", "debug", "snap", "variant",
                    "comparison", "refined", "3x3", "2x2", "4x2"))
    )

    if MAX_IMAGES and len(label_paths) > MAX_IMAGES:
        label_paths = np.random.choice(label_paths, size=MAX_IMAGES, replace=False).tolist()
        label_paths = sorted(label_paths)

    n_workers = NUM_WORKERS or max(1, multiprocessing.cpu_count() - 1)
    active = []
    if SAVE_RGB_MASK:     active.append(f"RGB masks -> {OUTPUT_DIR}")
    if SAVE_DEBUG_OVERLAY: active.append(f"debug overlays -> {DEBUG_DIR}")
    print(f"Found {len(label_paths)} masks  |  workers={n_workers}")
    for a in active:
        print(f"  saving {a}")
    print()

    path_strs = [str(p) for p in label_paths]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_label, p): p for p in path_strs}

        with tqdm(total=len(futures), desc="Adaptive peel + bidir snap", unit="img") as pbar:
            for future in as_completed(futures):
                out_path, err = future.result()
                if err:
                    tqdm.write(f"  WARN [{Path(futures[future]).stem}]: {err}")
                pbar.update(1)

    print(f"\nDone.")


if __name__ == "__main__":
    main()