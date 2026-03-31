#!/usr/bin/env python3
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing as mp
import numpy as np
try:
    import cv2
except ImportError:
    raise SystemExit("Missing dependency: opencv-python. Install: pip install opencv-python")
try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("Missing dependency: tqdm. Install: pip install tqdm")
# =========================
# CONFIG (edit these)
# =========================
LARGE_IMG_DIR = r"/app/mar15th_data/m15/images"
LARGE_LABEL_DIR = r"/app/mar15th_data/m15/labels"
CLIPPED_IMG_DIR = r"/app/BiSeNet/data_birdseye_long_selection/images"
CLIPPED_LABEL_DIR = r"/app/BiSeNet/data_birdseye_long_selection/labels"
MERGED_JSON_IN = r"/app/mar15th_data/m15/m15_splits.json"  # path to provided merged.json
# Outputs:
OUTPUT_TXT = r"/app/mar15th_data/mapped/best_mappings_with_chronology.txt"
OUTPUT_VIDEO = r"/app/mar15th_data/mapped/side_by_side_matches.mp4"
OUTPUT_MATCHED_JSON = r"/app/mar15th_data/mapped/matched.json"
OUTPUT_MERGED_TOTAL_JSON = r"/app/mar15th_data/mapped/merged_total.json"
TOP_K = 10
MAX_HAMMING = 20  # None to disable filtering
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLIPPED_LABEL_EXT = ".png"   # your clipped labels are png
LARGE_LABEL_EXT = ".jpg"     # your merged labels are jpg
NUM_WORKERS = max(1, mp.cpu_count() - 1)
READ_GRAYSCALE = True
# Chronology model knobs
CUT_PENALTY = 8.0
FORWARD_JUMP_WEIGHT = 0.004
BACKWARD_HARD_PENALTY = 1e6
DIST_WEIGHT = 1.0
# Video output knobs
OUT_FPS = 10
PAD = 8
LABEL_H = 36
MAX_SIDE = 720
FAIL_FRAME_COLOR = (0, 0, 255)  # BGR
# Only match large images whose filename contains this year string (set None to disable)
LARGE_YEAR_FILTER = "2025"
# =========================
# Hashing (64-bit dHash)
# =========================
def dhash64_from_path(p: str) -> Optional[np.uint64]:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE if READ_GRAYSCALE else cv2.IMREAD_COLOR)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    x = 0
    for b in diff.flatten():
        x = (x << 1) | int(b)
    return np.uint64(x)
# =========================
# File listing
# =========================
def list_images(folder: str, require_year: Optional[str] = None) -> List[str]:
    base = Path(folder)
    if not base.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = []
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if require_year is None or require_year in p.stem:
                files.append(str(p))
    files.sort()
    return files
# =========================
# Parallel hashing
# =========================
def _hash_worker(p: str) -> Tuple[str, Optional[np.uint64]]:
    return p, dhash64_from_path(p)
def compute_hashes(paths: List[str], workers: int) -> Tuple[List[str], np.ndarray]:
    kept_paths: List[str] = []
    hashes: List[np.uint64] = []
    if workers <= 1:
        for p in tqdm(paths, desc="Hashing LARGE", unit="img"):
            h = dhash64_from_path(p)
            if h is not None:
                kept_paths.append(p)
                hashes.append(h)
    else:
        with mp.Pool(processes=workers) as pool:
            for p, h in tqdm(
                pool.imap_unordered(_hash_worker, paths, chunksize=128),
                total=len(paths),
                desc="Hashing LARGE",
                unit="img",
            ):
                if h is not None:
                    kept_paths.append(p)
                    hashes.append(h)
        order = np.argsort(np.array(kept_paths, dtype=object))
        kept_paths = [kept_paths[i] for i in order]
        hashes = [hashes[i] for i in order]
    return kept_paths, np.array(hashes, dtype=np.uint64)
# =========================
# Fast Hamming distance
# =========================
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def hamming_distances_uint64(query_hash: np.uint64, db_hashes: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(db_hashes, query_hash)
    xb = x.view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_LUT[xb].sum(axis=1).astype(np.int16)
# =========================
# Candidate parsing
# =========================
_RE_LARGE = re.compile(r"^(?P<key>.+?)_frame(?P<frame>\d+)$", re.IGNORECASE)
_RE_CLIPPED_FRAME = re.compile(r"(?:^|[_-])frame[_-]?(?P<frame>\d+)(?:$|[^0-9])", re.IGNORECASE)
def parse_large_key_frame(path: str) -> Tuple[str, int]:
    stem = Path(path).stem
    m = _RE_LARGE.match(stem)
    if m:
        return m.group("key"), int(m.group("frame"))
    return str(Path(path).parent), 0
def parse_clipped_frame_num(path_or_name: str, fallback_index: int) -> int:
    stem = Path(path_or_name).stem
    m = _RE_CLIPPED_FRAME.search(stem)
    if m:
        return int(m.group("frame"))
    return fallback_index
# =========================
# Matching (top-K per frame)
# =========================
def match_all_topk(
    clipped_paths: List[str],
    large_paths: List[str],
    large_hashes: np.ndarray,
    top_k: int,
    max_hamming: Optional[int],
) -> List[Tuple[str, List[Tuple[int, str]]]]:
    results = []
    for cp in tqdm(clipped_paths, desc="Top-K search (CLIPPED)", unit="img"):
        qh = dhash64_from_path(cp)
        if qh is None:
            results.append((cp, []))
            continue
        dists = hamming_distances_uint64(qh, large_hashes)
        if max_hamming is not None:
            idx = np.where(dists <= max_hamming)[0]
            if idx.size == 0:
                results.append((cp, []))
                continue
            subd = dists[idx]
            k = min(top_k, idx.size)
            best_sub = np.argpartition(subd, k - 1)[:k]
            best_idx = idx[best_sub]
        else:
            k = min(top_k, dists.size)
            best_idx = np.argpartition(dists, k - 1)[:k]
        best_idx = best_idx[np.argsort(dists[best_idx], kind="stable")]
        cand = [(int(dists[i]), large_paths[i]) for i in best_idx]
        results.append((cp, cand))
    return results
# =========================
# Chronology enforcement (DP/Viterbi)
# =========================
def choose_chronological_path(
    topk_results: List[Tuple[str, List[Tuple[int, str]]]]
) -> List[Tuple[str, Optional[Tuple[int, str]]]]:
    clipped = [cp for cp, _ in topk_results]
    C = len(clipped)
    cands: List[List[Tuple[int, str, str, int]]] = []
    for _, cand in topk_results:
        row = []
        for dist, lp in cand:
            key, fr = parse_large_key_frame(lp)
            row.append((dist, lp, key, fr))
        cands.append(row)
    if C == 0:
        return []
    dp: List[np.ndarray] = []
    back: List[np.ndarray] = []
    if len(cands[0]) == 0:
        dp.append(np.array([], dtype=np.float64))
        back.append(np.array([], dtype=np.int32))
    else:
        dp.append(np.array([DIST_WEIGHT * c[0] for c in cands[0]], dtype=np.float64))
        back.append(np.full((len(cands[0]),), -1, dtype=np.int32))
    for i in tqdm(range(1, C), desc="Chronology DP", unit="frame"):
        if len(cands[i]) == 0:
            dp.append(np.array([], dtype=np.float64))
            back.append(np.array([], dtype=np.int32))
            continue
        cur_n = len(cands[i])
        prev_n = len(cands[i - 1])
        cur_costs = np.full((cur_n,), np.inf, dtype=np.float64)
        cur_back = np.full((cur_n,), -1, dtype=np.int32)
        if prev_n == 0 or dp[i - 1].size == 0:
            for j, (dist, _, _, _) in enumerate(cands[i]):
                cur_costs[j] = DIST_WEIGHT * dist
            dp.append(cur_costs)
            back.append(cur_back)
            continue
        for j, (dist_j, _, key_j, fr_j) in enumerate(cands[i]):
            base = DIST_WEIGHT * dist_j
            best_val = np.inf
            best_k = -1
            for k, (_, _, key_k, fr_k) in enumerate(cands[i - 1]):
                prev_cost = dp[i - 1][k]
                if not np.isfinite(prev_cost):
                    continue
                if key_j == key_k:
                    if fr_j < fr_k:
                        trans = BACKWARD_HARD_PENALTY
                    else:
                        trans = FORWARD_JUMP_WEIGHT * float(fr_j - fr_k)
                else:
                    trans = CUT_PENALTY
                val = prev_cost + trans + base
                if val < best_val:
                    best_val = val
                    best_k = k
            cur_costs[j] = best_val
            cur_back[j] = best_k
        dp.append(cur_costs)
        back.append(cur_back)
    last_i = None
    last_j = None
    for i in range(C - 1, -1, -1):
        if dp[i].size > 0 and np.isfinite(dp[i]).any():
            last_i = i
            last_j = int(np.nanargmin(dp[i]))
            break
    chosen: List[Optional[int]] = [None] * C
    if last_i is None:
        return [(clipped[i], None) for i in range(C)]
    i = last_i
    j = last_j
    while i >= 0 and j >= 0 and dp[i].size > 0:
        chosen[i] = j
        j = int(back[i][j]) if back[i].size > 0 else -1
        i -= 1
    out: List[Tuple[str, Optional[Tuple[int, str]]]] = []
    for i in range(C):
        if chosen[i] is None or len(cands[i]) == 0:
            out.append((clipped[i], None))
        else:
            dist, lp, _, _ = cands[i][chosen[i]]
            out.append((clipped[i], (dist, lp)))
    return out
# =========================
# Split lookup + JSON generation (FIXED)
# =========================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
def build_split_index(merged: Dict[str, Any]) -> Dict[Tuple[str, int], str]:
    """
    Lookup: (experiment_key, frame_num) -> split_name (train/val/test)
    Expands each range. Your ranges are typically small (count~80) so this is fast.
    """
    idx: Dict[Tuple[str, int], str] = {}
    splits = merged.get("splits", {})
    for split_name, ranges in splits.items():
        if not isinstance(ranges, list):
            continue
        for r in ranges:
            si = r.get("start_image", "")
            ei = r.get("end_image", "")
            if not si or not ei:
                continue
            m1 = _RE_LARGE.match(Path(si).stem)
            m2 = _RE_LARGE.match(Path(ei).stem)
            if not (m1 and m2):
                continue
            key1, a = m1.group("key"), int(m1.group("frame"))
            key2, b = m2.group("key"), int(m2.group("frame"))
            if key1 != key2:
                continue
            lo, hi = (a, b) if a <= b else (b, a)
            for fr in range(lo, hi + 1):
                idx[(key1, fr)] = split_name
    return idx
def clipped_label_name_from_image_name(img_name: str) -> str:
    """
    CLIPPED labels are same basename but with 'm' inserted before extension, and stored as .png.
      bev_back_00001.jpg -> bev_back_00001m.png
      frame_432.jpg -> frame_432m.png
    """
    stem = Path(img_name).stem
    return f"{stem}m{CLIPPED_LABEL_EXT}"
def large_label_name_from_large_image_name(large_img_name: str) -> str:
    """
    LARGE labels look like: label_2025-09-03-14-02-06_frame000001.jpg
    for image filename:     image_2025-09-03-14-02-06_frame000001.jpg
    """
    stem = Path(large_img_name).stem
    return f"label_{stem}{LARGE_LABEL_EXT}"
def group_runs_in_clipped_order(
    clipped_img_names: List[str],
    chosen_path: List[Tuple[str, Optional[Tuple[int, str]]]],
    split_index: Dict[Tuple[str, int], str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Produce ranges where start/end are ALWAYS from the same naming family (clipped images),
    so you never get mixed names like bev_* to frame_* in one range.
    Run boundaries when:
      - split changes, OR
      - "family key" changes (prefix before a frame number, or whole stem if no frame pattern), OR
      - matched key changes, OR
      - (matched) mapped frame decreases
    """
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    def family_key(img_name: str) -> str:
        stem = Path(img_name).stem
        m = _RE_CLIPPED_FRAME.search(stem)
        if m:
            return stem[: m.start()].rstrip("_-")
        return stem

    recs = []
    for i, (cp_path, choice) in enumerate(chosen_path):
        cp_name = Path(cp_path).name
        rec = {
            "cp_name": cp_name,
            "cp_frame": parse_clipped_frame_num(cp_name, i),
            "family": family_key(cp_name),
            "matched": False,
            "match_key": None,
            "match_frame": None,
            "split": "train",
            "dist": None,
            "match_img_name": None,
        }
        if choice is not None:
            dist, lp = choice
            mk, mf = parse_large_key_frame(lp)
            rec["matched"] = True
            rec["match_key"] = mk
            rec["match_frame"] = mf
            rec["dist"] = dist
            rec["match_img_name"] = Path(lp).name
            rec["split"] = split_index.get((mk, mf), "train")
        recs.append(rec)

    def flush(run: List[Dict[str, Any]]) -> None:
        if not run:
            return
        split = run[0]["split"]
        start_img = run[0]["cp_name"]
        end_img = run[-1]["cp_name"]
        start_label = clipped_label_name_from_image_name(start_img)
        end_label = clipped_label_name_from_image_name(end_img)
        out[split].append({
            "start_image": start_img,
            "end_image": end_img,
            "start_label": start_label,
            "end_label": end_label,
            "start_frame_num": int(run[0]["cp_frame"]),
            "end_frame_num": int(run[-1]["cp_frame"]),
            "count": int(len(run)),
            "matched": bool(run[0]["matched"]),
            "match_key": run[0]["match_key"],
        })

    run: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    for rec in recs:
        if prev is None:
            run = [rec]
            prev = rec
            continue
        same_split = rec["split"] == prev["split"]
        same_family = rec["family"] == prev["family"]
        if rec["matched"] and prev["matched"]:
            same_mk = rec["match_key"] == prev["match_key"]
            nondecreasing = rec["match_frame"] >= prev["match_frame"]
            cont = same_split and same_family and same_mk and nondecreasing
        elif (not rec["matched"]) and (not prev["matched"]):
            cont = same_split and same_family
        else:
            cont = False
        if cont:
            run.append(rec)
        else:
            flush(run)
            run = [rec]
        prev = rec
    flush(run)
    return out

def make_matched_json(merged: Dict[str, Any], matched_splits: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": "matched",
        "source_experiments": merged.get("source_experiments", []),
        "generated_from": {
            "merged_json": str(MERGED_JSON_IN),
            "large_img_dir": str(LARGE_IMG_DIR),
            "clipped_img_dir": str(CLIPPED_IMG_DIR),
            "large_year_filter": LARGE_YEAR_FILTER,
        },
        "splits": matched_splits,
    }
def make_merged_total_json(merged: Dict[str, Any], matched_splits: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(merged))
    out.setdefault("splits", {})
    for split_name, ranges in matched_splits.items():
        out["splits"].setdefault(split_name, [])
        out["splits"][split_name].extend(ranges)
    out["experiment"] = str(out.get("experiment", "merged")) + "_total"
    return out
# =========================
# Side-by-side video
# =========================
def _resize_to_max_h(img: np.ndarray, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    scale = max_h / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
def _pad_to_h(img: np.ndarray, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih == h:
        return img
    pad = h - ih
    top = pad // 2
    bot = pad - top
    return cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
def _load_or_placeholder(path: Optional[str], target_h: int, target_w: int) -> np.ndarray:
    if path is None:
        return np.full((target_h, target_w, 3), FAIL_FRAME_COLOR, dtype=np.uint8)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.full((target_h, target_w, 3), FAIL_FRAME_COLOR, dtype=np.uint8)
    return img
def make_side_by_side_video(
    chosen_path: List[Tuple[str, Optional[Tuple[int, str]]]],
    out_path: str,
) -> None:
    first_left = None
    for cp, _ in chosen_path:
        first_left = cv2.imread(cp, cv2.IMREAD_COLOR)
        if first_left is not None:
            break
    if first_left is None:
        raise RuntimeError("Could not read any clipped images for video output.")
    left0 = _resize_to_max_h(first_left, MAX_SIDE)
    panel_h = left0.shape[0]
    left_w = left0.shape[1]
    right_w = left_w
    out_w = left_w + PAD + right_w
    out_h = LABEL_H + panel_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, OUT_FPS, (out_w, out_h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at: {out_path}")
    for idx, (cp, choice) in enumerate(tqdm(chosen_path, desc="Writing video", unit="frame")):
        left = cv2.imread(cp, cv2.IMREAD_COLOR)
        if left is None:
            left = np.full((panel_h, left_w, 3), FAIL_FRAME_COLOR, dtype=np.uint8)
        left = _resize_to_max_h(left, MAX_SIDE)
        left = _pad_to_h(left, panel_h)
        left = cv2.resize(left, (left_w, panel_h), interpolation=cv2.INTER_AREA)
        if choice is None:
            dist = None
            right_path = None
        else:
            dist, right_path = choice
        right = _load_or_placeholder(right_path, panel_h, right_w)
        if right_path is not None:
            right = _resize_to_max_h(right, MAX_SIDE)
            right = _pad_to_h(right, panel_h)
            right = cv2.resize(right, (right_w, panel_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        band = np.zeros((LABEL_H, out_w, 3), dtype=np.uint8)
        left_name = Path(cp).name
        right_name = Path(right_path).name if right_path else "(no match)"
        label_left = f"CLIPPED: {left_name}"
        label_right = f"MATCH: {right_name}" + (f"  dist={dist}" if dist is not None else "")
        cv2.putText(band, label_left, (10, int(LABEL_H * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(band, label_right, (10 + out_w // 2, int(LABEL_H * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        canvas[0:LABEL_H, :, :] = band
        y0 = LABEL_H
        canvas[y0:y0 + panel_h, 0:left_w, :] = left
        canvas[y0:y0 + panel_h, left_w:left_w + PAD, :] = 0
        canvas[y0:y0 + panel_h, left_w + PAD:left_w + PAD + right_w, :] = right
        vw.write(canvas)
    vw.release()
# =========================
# Main
# =========================
def main() -> int:
    t0 = time.time()
    large_files = list_images(LARGE_IMG_DIR, require_year=LARGE_YEAR_FILTER)
    clipped_files = list_images(CLIPPED_IMG_DIR)
    print(f"LARGE:   {len(large_files)} images (year filter: {LARGE_YEAR_FILTER!r})")
    print(f"CLIPPED: {len(clipped_files)} images")
    print(f"Hashing LARGE with {NUM_WORKERS} workers...")
    large_paths, large_hashes = compute_hashes(large_files, NUM_WORKERS)
    print(f"  kept {len(large_paths)} readable images")
    print("Computing TOP-K candidates for CLIPPED...")
    topk = match_all_topk(clipped_files, large_paths, large_hashes, TOP_K, MAX_HAMMING)
    print("Enforcing chronology (DP/Viterbi) across candidates...")
    chosen = choose_chronological_path(topk)
    print(f"Writing side-by-side video: {OUTPUT_VIDEO}")
    make_side_by_side_video(chosen, OUTPUT_VIDEO)
    print(f"Loading merged json: {MERGED_JSON_IN}")
    merged = load_json(MERGED_JSON_IN)
    print("Indexing splits from merged.json...")
    split_index = build_split_index(merged)
    print("Building matched splits (clipped-derived) with safe grouping...")
    clipped_names = [Path(p).name for p in clipped_files]
    matched_splits = group_runs_in_clipped_order(clipped_names, chosen, split_index)
    matched_json = make_matched_json(merged, matched_splits)
    merged_total_json = make_merged_total_json(merged, matched_splits)
    print(f"Writing matched.json: {OUTPUT_MATCHED_JSON}")
    write_json(OUTPUT_MATCHED_JSON, matched_json)
    print(f"Writing merged_total.json: {OUTPUT_MERGED_TOTAL_JSON}")
    write_json(OUTPUT_MERGED_TOTAL_JSON, merged_total_json)
    dt = time.time() - t0
    print("Done.")
    print(f"  Video:        {OUTPUT_VIDEO}")
    print(f"  matched.json:  {OUTPUT_MATCHED_JSON}")
    print(f"  merged_total:  {OUTPUT_MERGED_TOTAL_JSON}")
    print(f"Elapsed: {dt:.2f}s")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())