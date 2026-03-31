# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# generate_splits_json.py

# Generate a splits JSON (train/val/test) from an images folder and a labels folder.

# Adds additional binning constraints:
#   1) Minimum number of bins per split (default: 10)
#   2) Maximum length of any continuous segment (default: 80)

# Definitions:
# - "Index" is the numeric sequencing key extracted from filenames:
#     * Prefer 'frameNNN' token; else last digit-run in the stem.
# - "Continuous segment" means consecutive indices, e.g. 10,11,12 => segment 10-12.
# - We first build natural continuous runs, then we split runs into chunks of at most MAX_BIN_LEN.
# - If a split ends up with fewer than MIN_BINS segments, we further split long segments
#   (still respecting MAX_BIN_LEN) until MIN_BINS is reached or no more splitting is possible.

# No CLI args: edit CONFIG variables at the top.

# Naming schemes supported (mixed within same dataset):
#   (A) image_..._frame004401.jpg
#   (B) bev_back_00001.jpg
#   (C) bev_back_00001m.png
#   (D) identical basename in labels dir
# """

# from __future__ import annotations

# import json
# import os
# import os.path as osp
# import re
# from typing import Dict, List, Optional, Tuple

# # =========================
# # CONFIG (edit these only)
# # =========================

# IMAGES_DIR = "/app/birdseye_run_12/images"
# LABELS_DIR = "/app/birdseye_run_12/labels"
# OUTPUT_JSON = "/app/birdseye_run_12/generated_splits.json"

# EXPERIMENT_NAME = "generated"
# SOURCE_EXPERIMENTS: List[str] = []

# TRAIN_FRAC = 1.0
# VAL_FRAC = 0.0
# TEST_FRAC = 0.0

# STRICT_LABELS = True

# IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# LABEL_EXTS = {".png", ".jpg", ".jpeg"}

# # New constraints
# MIN_BINS_PER_SPLIT = 10          # minimum number of segments (bins) per split
# MAX_BIN_LEN = 80                 # maximum contiguous frames per segment

# # =========================
# # Helpers
# # =========================

# def list_files(root: str, exts: set[str]) -> List[str]:
#     out = []
#     for name in os.listdir(root):
#         p = osp.join(root, name)
#         if not osp.isfile(p):
#             continue
#         ext = osp.splitext(name)[1].lower()
#         if ext in exts:
#             out.append(name)
#     return out


# def extract_index(filename: str) -> Optional[int]:
#     base = osp.basename(filename)
#     m = re.search(r"frame(\d+)", base)
#     if m:
#         return int(m.group(1))

#     stem, _ = osp.splitext(base)
#     m2 = re.search(r"(\d+)(\D*)$", stem)
#     if m2:
#         return int(m2.group(1))

#     return None


# def candidate_label_basenames(img_base: str) -> List[str]:
#     stem, ext = osp.splitext(img_base)
#     ext_l = ext.lower()

#     cands: List[str] = []
#     cands.append(img_base)            # identical basename
#     cands.append(f"{stem}m{ext}")     # insert 'm' same ext
#     cands.append(f"{stem}m.png")      # insert 'm' png

#     if stem.startswith("image_"):
#         cands.append(f"label_{stem[len('image_'):]}" + ext)
#         cands.append(f"label_{stem[len('image_'):]}.png")

#     if ext_l != ".png":
#         cands.append(f"{stem}.png")

#     seen = set()
#     uniq = []
#     for x in cands:
#         if x not in seen:
#             uniq.append(x)
#             seen.add(x)
#     return uniq


# def resolve_label_for_image(labels_dir: str, img_base: str) -> Optional[str]:
#     for cand in candidate_label_basenames(img_base):
#         ext = osp.splitext(cand)[1].lower()
#         if ext and (ext not in LABEL_EXTS):
#             continue
#         if osp.isfile(osp.join(labels_dir, cand)):
#             return cand
#     return None


# def split_indices(sorted_items: List[Tuple[int, str, str]]) -> Dict[str, List[Tuple[int, str, str]]]:
#     n = len(sorted_items)
#     n_train = int(round(n * TRAIN_FRAC))
#     n_val = int(round(n * VAL_FRAC))
#     if n_train + n_val > n:
#         n_val = max(0, n - n_train)
#     train = sorted_items[:n_train]
#     val = sorted_items[n_train:n_train + n_val]
#     test = sorted_items[n_train + n_val:]
#     assert len(train) + len(val) + len(test) == n
#     return {"train": train, "val": val, "test": test}


# def make_initial_runs(items: List[Tuple[int, str, str]]) -> List[List[Tuple[int, str, str]]]:
#     """
#     Convert sorted items into natural contiguous runs (no max length constraint).
#     Each run is a list of (idx, img_base, lb_base) where idx are consecutive.
#     """
#     if not items:
#         return []
#     items = sorted(items, key=lambda x: x[0])

#     runs: List[List[Tuple[int, str, str]]] = []
#     cur_run: List[Tuple[int, str, str]] = [items[0]]

#     for i in range(1, len(items)):
#         prev = items[i - 1]
#         cur = items[i]
#         if cur[0] == prev[0] + 1:
#             cur_run.append(cur)
#         else:
#             runs.append(cur_run)
#             cur_run = [cur]
#     runs.append(cur_run)
#     return runs


# def chunk_run(run: List[Tuple[int, str, str]], max_len: int) -> List[List[Tuple[int, str, str]]]:
#     """
#     Split a contiguous run into chunks of length <= max_len.
#     """
#     if len(run) <= max_len:
#         return [run]
#     out = []
#     for i in range(0, len(run), max_len):
#         out.append(run[i:i + max_len])
#     return out


# def enforce_min_bins(chunks: List[List[Tuple[int, str, str]]], min_bins: int, max_len: int) -> List[List[Tuple[int, str, str]]]:
#     """
#     If chunks < min_bins, further split the longest chunks until we reach min_bins
#     or we cannot split any further (all chunks length==1).
#     Splitting respects max_len automatically because we only split existing chunks.
#     """
#     if len(chunks) >= min_bins:
#         return chunks

#     # We can only increase bin count by splitting chunks with length > 1.
#     # Repeatedly split the longest chunk into two parts.
#     chunks = list(chunks)
#     while len(chunks) < min_bins:
#         # find splittable chunk
#         splittable_idx = None
#         best_len = 0
#         for i, c in enumerate(chunks):
#             if len(c) > 1 and len(c) > best_len:
#                 best_len = len(c)
#                 splittable_idx = i

#         if splittable_idx is None:
#             # cannot split any further
#             break

#         c = chunks.pop(splittable_idx)
#         mid = len(c) // 2
#         left = c[:mid]
#         right = c[mid:]
#         # left/right lengths are <= original length, and original was already <= max_len
#         chunks.append(left)
#         chunks.append(right)

#     return chunks


# def chunks_to_segments(chunks: List[List[Tuple[int, str, str]]]) -> List[Dict]:
#     """
#     Convert chunks (each chunk contiguous) into JSON segment dicts.
#     """
#     segs: List[Dict] = []
#     for c in sorted(chunks, key=lambda x: x[0][0]):
#         i0 = c[0]
#         i1 = c[-1]
#         segs.append({
#             "start_image": i0[1],
#             "end_image": i1[1],
#             "start_label": i0[2],
#             "end_label": i1[2],
#             "start_frame_num": int(i0[0]),
#             "end_frame_num": int(i1[0]),
#             "count": int(i1[0] - i0[0] + 1),
#             "matched": False,
#             "match_key": None,
#         })
#     return segs


# def make_binned_segments(items: List[Tuple[int, str, str]], min_bins: int, max_len: int) -> List[Dict]:
#     """
#     Full binning pipeline for one split:
#       items -> contiguous runs -> chunk each run to max_len -> enforce min_bins -> segments
#     """
#     runs = make_initial_runs(items)

#     # chunk to max_len
#     chunks: List[List[Tuple[int, str, str]]] = []
#     for r in runs:
#         chunks.extend(chunk_run(r, max_len=max_len))

#     # enforce minimum bins
#     chunks = enforce_min_bins(chunks, min_bins=min_bins, max_len=max_len)

#     return chunks_to_segments(chunks)


# def main():
#     if not osp.isdir(IMAGES_DIR):
#         raise FileNotFoundError(f"IMAGES_DIR not found: {IMAGES_DIR}")
#     if not osp.isdir(LABELS_DIR):
#         raise FileNotFoundError(f"LABELS_DIR not found: {LABELS_DIR}")

#     image_files = list_files(IMAGES_DIR, IMAGE_EXTS)

#     indexed: Dict[int, Tuple[str, str]] = {}
#     missing_index = 0
#     missing_label = 0
#     kept = 0

#     for img_base in sorted(image_files):
#         idx = extract_index(img_base)
#         if idx is None:
#             missing_index += 1
#             continue

#         lb_base = resolve_label_for_image(LABELS_DIR, img_base)
#         if lb_base is None:
#             missing_label += 1
#             if STRICT_LABELS:
#                 continue
#             lb_base = img_base

#         if idx not in indexed:
#             indexed[idx] = (img_base, lb_base)
#             kept += 1

#     items = [(k, v[0], v[1]) for k, v in indexed.items()]
#     items.sort(key=lambda x: x[0])

#     if not items:
#         raise RuntimeError(
#             "No valid (image,label) pairs found.\n"
#             f"missing_index={missing_index}, missing_label={missing_label}, total_images={len(image_files)}"
#         )

#     split_map = split_indices(items)

#     splits_out = {
#         "train": make_binned_segments(split_map["train"], min_bins=MIN_BINS_PER_SPLIT, max_len=MAX_BIN_LEN),
#         "val": make_binned_segments(split_map["val"], min_bins=MIN_BINS_PER_SPLIT, max_len=MAX_BIN_LEN),
#         "test": make_binned_segments(split_map["test"], min_bins=MIN_BINS_PER_SPLIT, max_len=MAX_BIN_LEN),
#     }

#     out = {
#         "experiment": EXPERIMENT_NAME,
#         "source_experiments": SOURCE_EXPERIMENTS,
#         "generated_from": {
#             "images_dir": IMAGES_DIR,
#             "labels_dir": LABELS_DIR,
#         },
#         "stats": {
#             "total_images_seen": len(image_files),
#             "missing_index": missing_index,
#             "missing_label": missing_label,
#             "pairs_kept": kept,
#             "pairs_total_after_dedupe": len(items),
#             "train_frames": len(split_map["train"]),
#             "val_frames": len(split_map["val"]),
#             "test_frames": len(split_map["test"]),
#             "train_bins": len(splits_out["train"]),
#             "val_bins": len(splits_out["val"]),
#             "test_bins": len(splits_out["test"]),
#             "min_bins_per_split": MIN_BINS_PER_SPLIT,
#             "max_bin_len": MAX_BIN_LEN,
#             "strict_labels": STRICT_LABELS,
#         },
#         "splits": splits_out,
#     }

#     os.makedirs(osp.dirname(OUTPUT_JSON), exist_ok=True)
#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(out, f, indent=2)

#     print(f"Wrote: {OUTPUT_JSON}")
#     print(json.dumps(out["stats"], indent=2))


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_splits_json.py

Generate a splits JSON (train/val/test) from an images folder and a labels folder.

Adds additional binning constraints:
  1) Minimum number of bins per split (default: 10)
  2) Maximum length of any continuous segment (default: 80)

Definitions:
- "Index" is the numeric sequencing key extracted from filenames:
    * Prefer 'frameNNN' token; else 'run_NNN_image_NNN' pattern; else last digit-run in the stem.
- "Continuous segment" means consecutive indices, e.g. 10,11,12 => segment 10-12.
- We first build natural continuous runs, then we split runs into chunks of at most MAX_BIN_LEN.
- If a split ends up with fewer than MIN_BINS segments, we further split long segments
  (still respecting MAX_BIN_LEN) until MIN_BINS is reached or no more splitting is possible.

No CLI args: edit CONFIG variables at the top.

Naming schemes supported (mixed within same dataset):
  (A) image_..._frame004401.jpg
  (B) bev_back_00001.jpg
  (C) bev_back_00001m.png
  (D) identical basename in labels dir
  (E) run_015_image_000098.png  ->  run_015_label_000098.png
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from typing import Dict, List, Optional, Tuple

# =========================
# CONFIG (edit these only)
# =========================

IMAGES_DIR = "/app/birdseye_run_12/images"
LABELS_DIR = "/app/birdseye_run_12/labels"
OUTPUT_JSON = "/app/birdseye_run_12/generated_splits.json"

EXPERIMENT_NAME = "generated"
SOURCE_EXPERIMENTS: List[str] = []

TRAIN_FRAC = 1.0
VAL_FRAC = 0.0
TEST_FRAC = 0.0

STRICT_LABELS = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_EXTS = {".png", ".jpg", ".jpeg"}

# New constraints
MIN_BINS_PER_SPLIT = 10          # minimum number of segments (bins) per split
MAX_BIN_LEN = 80                 # maximum contiguous frames per segment

# Pattern: run_NNN_image_NNNNNN  (scheme E)
# Group 1 = run prefix (e.g. "run_015"), Group 2 = frame number
_RUN_IMAGE_RE = re.compile(r"^(run_\d+)_image_(\d+)$")

# =========================
# Helpers
# =========================

def list_files(root: str, exts: set[str]) -> List[str]:
    out = []
    for name in os.listdir(root):
        p = osp.join(root, name)
        if not osp.isfile(p):
            continue
        ext = osp.splitext(name)[1].lower()
        if ext in exts:
            out.append(name)
    return out


def extract_index(filename: str) -> Optional[int]:
    base = osp.basename(filename)
    stem, _ = osp.splitext(base)

    # Scheme A: frameNNN anywhere in the name
    m = re.search(r"frame(\d+)", base)
    if m:
        return int(m.group(1))

    # Scheme E: run_NNN_image_NNNNNN
    m = _RUN_IMAGE_RE.match(stem)
    if m:
        return int(m.group(2))

    # Fallback: last digit-run in stem (schemes B, C, D)
    m2 = re.search(r"(\d+)(\D*)$", stem)
    if m2:
        return int(m2.group(1))

    return None


def candidate_label_basenames(img_base: str) -> List[str]:
    stem, ext = osp.splitext(img_base)
    ext_l = ext.lower()

    cands: List[str] = []

    # Scheme E: run_NNN_image_NNNNNN -> run_NNN_label_NNNNNN
    m = _RUN_IMAGE_RE.match(stem)
    if m:
        run_prefix = m.group(1)
        frame_num  = m.group(2)
        cands.append(f"{run_prefix}_label_{frame_num}{ext}")
        if ext_l != ".png":
            cands.append(f"{run_prefix}_label_{frame_num}.png")

    cands.append(img_base)            # identical basename (schemes A/B/C/D)
    cands.append(f"{stem}m{ext}")     # insert 'm' same ext
    cands.append(f"{stem}m.png")      # insert 'm' png

    if stem.startswith("image_"):
        cands.append(f"label_{stem[len('image_'):]}" + ext)
        cands.append(f"label_{stem[len('image_'):]}.png")

    if ext_l != ".png":
        cands.append(f"{stem}.png")

    seen = set()
    uniq = []
    for x in cands:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def resolve_label_for_image(labels_dir: str, img_base: str) -> Optional[str]:
    for cand in candidate_label_basenames(img_base):
        ext = osp.splitext(cand)[1].lower()
        if ext and (ext not in LABEL_EXTS):
            continue
        if osp.isfile(osp.join(labels_dir, cand)):
            return cand
    return None


def split_indices(sorted_items: List[Tuple[int, str, str]]) -> Dict[str, List[Tuple[int, str, str]]]:
    n = len(sorted_items)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    train = sorted_items[:n_train]
    val = sorted_items[n_train:n_train + n_val]
    test = sorted_items[n_train + n_val:]
    assert len(train) + len(val) + len(test) == n
    return {"train": train, "val": val, "test": test}


def make_initial_runs(items: List[Tuple[int, str, str]]) -> List[List[Tuple[int, str, str]]]:
    if not items:
        return []
    items = sorted(items, key=lambda x: x[0])

    runs: List[List[Tuple[int, str, str]]] = []
    cur_run: List[Tuple[int, str, str]] = [items[0]]

    for i in range(1, len(items)):
        prev = items[i - 1]
        cur = items[i]
        if cur[0] == prev[0] + 1:
            cur_run.append(cur)
        else:
            runs.append(cur_run)
            cur_run = [cur]
    runs.append(cur_run)
    return runs


def chunk_run(run: List[Tuple[int, str, str]], max_len: int) -> List[List[Tuple[int, str, str]]]:
    if len(run) <= max_len:
        return [run]
    out = []
    for i in range(0, len(run), max_len):
        out.append(run[i:i + max_len])
    return out


def enforce_min_bins(chunks: List[List[Tuple[int, str, str]]], min_bins: int, max_len: int) -> List[List[Tuple[int, str, str]]]:
    if len(chunks) >= min_bins:
        return chunks

    chunks = list(chunks)
    while len(chunks) < min_bins:
        splittable_idx = None
        best_len = 0
        for i, c in enumerate(chunks):
            if len(c) > 1 and len(c) > best_len:
                best_len = len(c)
                splittable_idx = i

        if splittable_idx is None:
            break

        c = chunks.pop(splittable_idx)
        mid = len(c) // 2
        chunks.append(c[:mid])
        chunks.append(c[mid:])

    return chunks


def chunks_to_segments(chunks: List[List[Tuple[int, str, str]]]) -> List[Dict]:
    segs: List[Dict] = []
    for c in sorted(chunks, key=lambda x: x[0][0]):
        i0 = c[0]
        i1 = c[-1]
        segs.append({
            "start_image": i0[1],
            "end_image": i1[1],
            "start_label": i0[2],
            "end_label": i1[2],
            "start_frame_num": int(i0[0]),
            "end_frame_num": int(i1[0]),
            "count": int(i1[0] - i0[0] + 1),
            "matched": False,
            "match_key": None,
        })
    return segs


def make_binned_segments(items: List[Tuple[int, str, str]], min_bins: int, max_len: int) -> List[Dict]:
    runs = make_initial_runs(items)
    chunks: List[List[Tuple[int, str, str]]] = []
    for r in runs:
        chunks.extend(chunk_run(r, max_len=max_len))
    chunks = enforce_min_bins(chunks, min_bins=min_bins, max_len=max_len)
    return chunks_to_segments(chunks)


def main():
    if not osp.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"IMAGES_DIR not found: {IMAGES_DIR}")
    if not osp.isdir(LABELS_DIR):
        raise FileNotFoundError(f"LABELS_DIR not found: {LABELS_DIR}")

    image_files = list_files(IMAGES_DIR, IMAGE_EXTS)

    indexed: Dict[int, Tuple[str, str]] = {}
    missing_index = 0
    missing_label = 0
    kept = 0

    for img_base in sorted(image_files):
        idx = extract_index(img_base)
        if idx is None:
            missing_index += 1
            continue

        lb_base = resolve_label_for_image(LABELS_DIR, img_base)
        if lb_base is None:
            missing_label += 1
            if STRICT_LABELS:
                continue
            lb_base = img_base

        if idx not in indexed:
            indexed[idx] = (img_base, lb_base)
            kept += 1

    items = [(k, v[0], v[1]) for k, v in indexed.items()]
    items.sort(key=lambda x: x[0])

    if not items:
        raise RuntimeError(
            "No valid (image,label) pairs found.\n"
            f"missing_index={missing_index}, missing_label={missing_label}, total_images={len(image_files)}"
        )

    split_map = split_indices(items)

    splits_out = {
        "train": make_binned_segments(split_map["train"], min_bins=MIN_BINS_PER_SPLIT, max_len=MAX_BIN_LEN),
        "val":   make_binned_segments(split_map["val"],   min_bins=MIN_BINS_PER_SPLIT, max_len=MAX_BIN_LEN),
        "test":  make_binned_segments(split_map["test"],  min_bins=MIN_BINS_PER_SPLIT, max_len=MAX_BIN_LEN),
    }

    out = {
        "experiment": EXPERIMENT_NAME,
        "source_experiments": SOURCE_EXPERIMENTS,
        "generated_from": {
            "images_dir": IMAGES_DIR,
            "labels_dir": LABELS_DIR,
        },
        "stats": {
            "total_images_seen": len(image_files),
            "missing_index": missing_index,
            "missing_label": missing_label,
            "pairs_kept": kept,
            "pairs_total_after_dedupe": len(items),
            "train_frames": len(split_map["train"]),
            "val_frames": len(split_map["val"]),
            "test_frames": len(split_map["test"]),
            "train_bins": len(splits_out["train"]),
            "val_bins": len(splits_out["val"]),
            "test_bins": len(splits_out["test"]),
            "min_bins_per_split": MIN_BINS_PER_SPLIT,
            "max_bin_len": MAX_BIN_LEN,
            "strict_labels": STRICT_LABELS,
        },
        "splits": splits_out,
    }

    os.makedirs(osp.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {OUTPUT_JSON}")
    print(json.dumps(out["stats"], indent=2))


if __name__ == "__main__":
    main()