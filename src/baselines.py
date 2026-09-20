"""Named trivial baselines. Copy-last is never a model feature."""

from __future__ import annotations

import random
from collections import defaultdict


def random_preds(
    examples: list[dict],
    col_ranges: dict[str, tuple[float, float]],
    rng: random.Random,
) -> dict[tuple[int, str], str]:
    out = {}
    for ex in examples:
        lo, hi = col_ranges.get(ex["col"], (0.0, 1.0))
        out[(int(ex["pid"]), ex["col"])] = str(int(round(rng.uniform(lo, hi))))
    return out


def majority_map(train_examples: list[dict]) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex in train_examples:
        counts[ex["col"]][str(ex["target"])] += 1
    return {col: max(vals, key=vals.get) for col, vals in counts.items()}


def apply_majority(examples: list[dict], majority: dict[str, str]) -> dict[tuple[int, str], str]:
    out = {}
    for ex in examples:
        key = (int(ex["pid"]), ex["col"])
        if ex["col"] not in majority:
            continue
        out[key] = majority[ex["col"]]
    return out


def copy_last_preds(
    examples: list[dict],
    wave1_3_lookup: dict[tuple[int, str], str],
) -> dict[tuple[int, str], str]:
    """ŷ = wave 1–3 value of the same column. Missing lookup → skip, never use gold."""
    out = {}
    missing = 0
    for ex in examples:
        key = (int(ex["pid"]), ex["col"])
        if key not in wave1_3_lookup:
            missing += 1
            continue
        out[key] = wave1_3_lookup[key]
    if missing:
        print(f"[copy-last] skipped {missing} rows with no wave1-3 diagnostic (not filled with gold).")
    return out
