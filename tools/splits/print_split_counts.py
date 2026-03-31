# Trivial script for printing split descriptions. Useful.
#!/usr/bin/env python3
"""
Print total frame counts per split (train/val/test) from a splits JSON.

Usage:
  python count_splits.py /path/to/merged.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def sum_counts(ranges: List[Dict[str, Any]]) -> int:
    total = 0
    for r in ranges:
        # Prefer explicit count if present
        if isinstance(r, dict) and isinstance(r.get("count"), int):
            total += int(r["count"])
            continue

        # Fallback: compute from frame nums if present
        a = r.get("start_frame_num")
        b = r.get("end_frame_num")
        if isinstance(a, int) and isinstance(b, int):
            total += abs(b - a) + 1
            continue

        # Last resort: ignore malformed entries
    return total


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python count_splits.py /path/to/splits.json", file=sys.stderr)
        return 2

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    data = load_json(path)
    splits = data.get("splits", {})

    for split_name in ["train", "val", "test"]:
        ranges = splits.get(split_name, [])
        if not isinstance(ranges, list):
            ranges = []
        total = sum_counts(ranges)
        print(f"{split_name}: {total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
