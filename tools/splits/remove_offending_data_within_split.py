"""
clean_splits.py

Takes a "bad frames" JSON and a "splits" JSON, then removes bad frame ranges
from all train/val/test buckets. Buckets are split into multiple sub-buckets
as needed to punch holes around bad frames. Buckets are never moved between sets.

Usage:
    python clean_splits.py \
        --bad bad_frames.json \
        --splits splits.json \
        --output cleaned_splits.json

Bad frames JSON format (list of objects):
    [
        {
            "experiment": "2026_03_04_14_54_33",
            "start_frame": 400,
            "end_frame": 550,
            ...   # other fields ignored
        },
        ...
    ]

Splits JSON format:
    {
        "experiment": "merged",
        "source_experiments": [...],
        "splits": {
            "train": [
                {
                    "start_image": "image_<exp>_frame<N>.jpg",
                    "end_image":   "image_<exp>_frame<N>.jpg",
                    "start_label": "label_<exp>_frame<N>.jpg",
                    "end_label":   "label_<exp>_frame<N>.jpg",
                    "start_frame_num": N,
                    "end_frame_num":   N,
                    "count": N
                },
                ...
            ],
            "val": [...],
            "test": [...]
        }
    }
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from copy import deepcopy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_experiment_from_filename(filename: str) -> str:
    """
    Extract the experiment ID from a filename like:
        image_2026_03_04_14_54_33_frame000327.jpg
        label_2025-09-03-14-02-06_frame006961.jpg
    Returns the middle portion between the first underscore/dash group and '_frame'.
    """
    # Strip prefix (image_ / label_)
    name = re.sub(r'^(image|label)_', '', filename)
    # Strip suffix (_frame...)
    name = re.sub(r'_frame\d+\.jpg$', '', name)
    return name


def make_filename(prefix: str, experiment: str, frame: int, ext: str = "jpg") -> str:
    """Reconstruct a filename: <prefix>_<experiment>_frame<NNNNNN>.<ext>"""
    return f"{prefix}_{experiment}_frame{frame:06d}.{ext}"


def build_bucket(experiment: str, start: int, end: int) -> dict:
    """Create a bucket dict from components."""
    return {
        "start_image":     make_filename("image", experiment, start),
        "end_image":       make_filename("image", experiment, end),
        "start_label":     make_filename("label", experiment, start),
        "end_label":       make_filename("label", experiment, end),
        "start_frame_num": start,
        "end_frame_num":   end,
        "count":           end - start + 1,
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_bad_index(bad_frames: list) -> dict:
    """
    Returns a dict:  experiment -> sorted list of (start, end) bad ranges (inclusive).
    Overlapping / adjacent ranges are merged.
    """
    raw = defaultdict(list)
    for entry in bad_frames:
        exp = entry["experiment"]
        s   = entry["start_frame"]
        e   = entry["end_frame"]
        if s > e:
            s, e = e, s          # defensive swap
        raw[exp].append((s, e))

    merged = {}
    for exp, ranges in raw.items():
        ranges.sort()
        result = [ranges[0]]
        for s, e in ranges[1:]:
            if s <= result[-1][1] + 1:   # overlapping or adjacent → merge
                result[-1] = (result[-1][0], max(result[-1][1], e))
            else:
                result.append((s, e))
        merged[exp] = result

    return merged


def punch_holes(bucket: dict, experiment: str, bad_ranges: list) -> list:
    """
    Given one bucket and the sorted bad ranges for its experiment,
    return a list of replacement buckets with bad frames removed.
    Bad ranges that don't touch this bucket are ignored.
    """
    b_start = bucket["start_frame_num"]
    b_end   = bucket["end_frame_num"]

    # Collect bad ranges that overlap this bucket
    overlapping = [
        (max(s, b_start), min(e, b_end))
        for s, e in bad_ranges
        if s <= b_end and e >= b_start
    ]

    if not overlapping:
        return [bucket]   # nothing to do

    # Walk through the bucket frame range, emitting good sub-ranges
    new_buckets = []
    cursor = b_start

    for bad_s, bad_e in overlapping:
        if cursor < bad_s:
            # Good frames before this bad range
            new_buckets.append(build_bucket(experiment, cursor, bad_s - 1))
        # Skip over the bad range
        cursor = bad_e + 1

    if cursor <= b_end:
        # Good frames after all bad ranges
        new_buckets.append(build_bucket(experiment, cursor, b_end))

    return new_buckets


def clean_splits(splits_data: dict, bad_index: dict) -> dict:
    """
    Process all train/val/test buckets, punching holes for bad frames.
    Returns a modified copy of splits_data.
    """
    result = deepcopy(splits_data)

    for split_name, buckets in result["splits"].items():
        new_buckets = []
        original_count = len(buckets)
        removed = 0
        fragmented = 0

        for bucket in buckets:
            # Determine which experiment this bucket belongs to
            experiment = parse_experiment_from_filename(bucket["start_image"])

            bad_ranges = bad_index.get(experiment, [])
            replacements = punch_holes(bucket, experiment, bad_ranges)

            if len(replacements) == 0:
                removed += 1
            elif len(replacements) > 1:
                fragmented += 1

            new_buckets.extend(replacements)

        result["splits"][split_name] = new_buckets

        added = len(new_buckets) - original_count
        print(
            f"  [{split_name:>5}]  {original_count} buckets → {len(new_buckets)} buckets  "
            f"(+{added} from splits, {removed} fully removed)"
        )

    return result


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def count_frames(splits_data: dict) -> dict:
    totals = {}
    for split_name, buckets in splits_data["splits"].items():
        totals[split_name] = sum(b["count"] for b in buckets)
    return totals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Remove bad frames from split buckets.")
    parser.add_argument("--bad",     required=True, help="Path to bad_frames.json")
    parser.add_argument("--splits",  required=True, help="Path to splits.json")
    parser.add_argument("--output",  required=True, help="Path for cleaned output JSON")
    args = parser.parse_args()

    # Load inputs
    with open(args.bad, "r") as f:
        bad_frames = json.load(f)
    print(f"Loaded {len(bad_frames)} bad frame ranges.")

    with open(args.splits, "r") as f:
        splits_data = json.load(f)

    split_names = list(splits_data["splits"].keys())
    total_buckets = sum(len(splits_data["splits"][s]) for s in split_names)
    print(f"Loaded splits with {total_buckets} total buckets across {split_names}.")

    # Build bad frame index
    bad_index = build_bad_index(bad_frames)
    total_bad_frames = sum(e - s + 1 for ranges in bad_index.values() for s, e in ranges)
    print(f"Bad frame index covers {len(bad_index)} experiments, {total_bad_frames} total bad frames.\n")

    # Frame counts before
    before = count_frames(splits_data)

    # Process
    print("Processing splits...")
    cleaned = clean_splits(splits_data, bad_index)

    # Frame counts after
    after = count_frames(cleaned)

    print("\nFrame count summary:")
    for split_name in split_names:
        diff = after[split_name] - before[split_name]
        print(f"  [{split_name:>5}]  {before[split_name]:>7} frames → {after[split_name]:>7} frames  ({diff:+d})")

    # Save output
    with open(args.output, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"\nCleaned splits saved to: {args.output}")


if __name__ == "__main__":
    main()