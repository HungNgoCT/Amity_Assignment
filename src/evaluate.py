"""Score random / majority / copy-last / optional LoRA on the POC JSONL.

    python -m src.evaluate \\
        --train_jsonl data/poc/examples_train.jsonl \\
        --test_jsonl data/poc/examples_val.jsonl \\
        --diag_jsonl data/poc/diag_val.jsonl \\
        --leak_jsonl data/poc/leak_fixture.jsonl \\
        --out_dir results/poc

Slice-mean MAD uses train-only ranges. Not the official 17-task script → 'not verified'.
On the scored set, copy-last == ceiling (same pairs). Missing diag rows are skipped, never filled with gold.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.baselines import (
    answer_code_map,
    apply_majority,
    copy_last_preds,
    majority_map,
    random_preds,
)
from src.leak_test import assert_prompts_leak_free, load_jsonl

SEED = 42


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def train_ranges(train_examples: list[dict]) -> dict[str, tuple[float, float]]:
    by_col: dict[str, list[float]] = defaultdict(list)
    for ex in train_examples:
        v = to_float(ex["target"])
        if v is not None:
            by_col[ex["col"]].append(v)
    out = {}
    for col, vals in by_col.items():
        lo, hi = min(vals), max(vals)
        out[col] = (lo, hi if hi > lo else lo + 1.0)
    return out


def score(
    examples: list[dict],
    preds: dict[tuple[int, str], str],
    col_ranges: dict[str, tuple[float, float]],
    valid_codes: dict[str, list[str]] | None = None,
) -> dict:
    per_person: dict[int, list[float]] = defaultdict(list)
    per_type: dict[str, list[float]] = defaultdict(list)
    per_type_ex: dict[str, list[float]] = defaultdict(list)
    n_total = 0
    n_parsed = 0
    for ex in examples:
        key = (int(ex["pid"]), ex["col"])
        pred = preds.get(key)
        y_true = to_float(ex["target"])
        y_pred = to_float(pred) if pred is not None else None
        if y_true is None or not math.isfinite(y_true):
            continue
        n_total += 1
        lo, hi = col_ranges.get(ex["col"], (0.0, 1.0))
        allowed = {
            code
            for code in (to_float(v) for v in (valid_codes or {}).get(ex["col"], []))
            if code is not None and math.isfinite(code)
        }
        valid_pred = (
            y_pred is not None
            and math.isfinite(y_pred)
            and (valid_codes is None or y_pred in allowed)
        )
        if valid_pred:
            n_parsed += 1
            acc = max(0.0, 1.0 - abs(y_pred - y_true) / (hi - lo))
            exact = 1.0 if y_pred == y_true else 0.0
        else:
            # Keep invalid/missing outputs in the fixed denominator.
            acc = 0.0
            exact = 0.0
        per_person[int(ex["pid"])].append(acc)
        qtype = ex.get("qtype") or "Unknown"
        per_type[qtype].append(acc)
        per_type_ex[qtype].append(exact)
    means = [float(np.mean(v)) for v in per_person.values() if v]
    return {
        "slice_mean_mad": float(np.mean(means)) if means else float("nan"),
        "n_people": len(means),
        "parse_rate": n_parsed / n_total if n_total else 0.0,
        "n_total": n_total,
        "by_type_mad": {t: float(np.mean(v)) for t, v in per_type.items()},
        "by_type_exact": {t: float(np.mean(v)) for t, v in per_type_ex.items()},
    }


def model_preds(examples: list[dict], adapter_dir: str) -> dict[tuple[int, str], str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = Path(adapter_dir).parent / "bundle_manifest.json"
    base = "Qwen/Qwen2.5-0.5B-Instruct"
    if manifest.exists():
        base = json.loads(manifest.read_text(encoding="utf-8"))["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype),
        adapter_dir,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(
        f"Generating on {len(examples)} items, greedy, one-by-one "
        f"({device}). First step is CUDA warmup — it can sit quiet for 1–2 min.",
        flush=True,
    )
    try:
        from tqdm import tqdm

        iterator = tqdm(examples, desc="lora gen")
    except ImportError:
        iterator = examples
    out = {}
    for i, ex in enumerate(iterator, start=1):
        inputs = tokenizer(
            ex["prompt"] + "\n", return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        out[(int(ex["pid"]), ex["col"])] = text.strip().split("\n")[0].strip()
        if not hasattr(iterator, "update") and i % 50 == 0:
            print(f"  {i}/{len(examples)}", flush=True)
    return out


def print_row(name: str, s: dict, ceiling: float | None) -> None:
    ratio = f"{s['slice_mean_mad'] / ceiling:.3f}" if ceiling and ceiling == ceiling else "n/a"
    print(
        f"{name:16s} | slice_MAD={s['slice_mean_mad']:.3f} | MAD/ceiling={ratio} "
        f"| parse={s['parse_rate']:.1%} | n={s['n_people']}"
    )


def load_diag(path: Path) -> dict[tuple[int, str], str]:
    rows = load_jsonl(path)
    return {(int(d["pid"]), d["col"]): str(d["wave1_3_answer"]) for d in rows if d.get("wave1_3_answer") not in ("", "nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--test_jsonl", required=True, help="Usually examples_val.jsonl for the POC.")
    ap.add_argument("--diag_jsonl", required=True)
    ap.add_argument("--leak_jsonl", default="data/poc/leak_fixture.jsonl")
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--out_dir", default="results/poc")
    args = ap.parse_args()

    train = load_jsonl(args.train_jsonl)
    test = load_jsonl(args.test_jsonl)
    leak_path = Path(args.leak_jsonl)
    if not leak_path.exists():
        raise SystemExit(f"Missing {leak_path}. Run: python -m src.data.build_jsonl")
    assert_prompts_leak_free(load_jsonl(leak_path), source=str(leak_path))
    assert_prompts_leak_free(test, source=args.test_jsonl, require_hit=False)

    rng = random.Random(SEED)
    ranges = train_ranges(train)
    valid_codes = answer_code_map(train)
    print("[range] train-only empirical max-min. Not official mad_accuracy_evaluation.py → not verified.\n")

    diag = load_diag(Path(args.diag_jsonl))
    ceiling_preds = copy_last_preds(test, diag)
    ceiling = score(test, ceiling_preds, ranges, valid_codes)
    ceiling_mad = ceiling["slice_mean_mad"]
    print("copy-last and ceiling are the same pairs on this slice (D3 §2.2).\n")

    print("=== Baselines ===")
    rand_scores = score(test, random_preds(test, valid_codes, rng), ranges, valid_codes)
    maj = majority_map(train)
    maj_scores = score(test, apply_majority(test, maj), ranges, valid_codes)
    print_row("random", rand_scores, ceiling_mad)
    print_row("train_majority", maj_scores, ceiling_mad)
    print_row("copy_last", ceiling, ceiling_mad)

    report: dict = {
        "n_test": len(test),
        "range_source": "train-only empirical — not the official MAD script",
        "baselines": {
            "random": rand_scores,
            "train_majority": maj_scores,
            "copy_last": ceiling,
        },
        "ceiling_copy_last": ceiling,
    }

    if args.adapter_dir:
        print("\n=== LoRA ===")
        mp = model_preds(test, args.adapter_dir)
        ms = score(test, mp, ranges, valid_codes)
        print_row("lora_sft", ms, ceiling_mad)
        if ms["slice_mean_mad"] > ceiling_mad:
            print("WARNING: slice MAD > copy-last/ceiling. Leak alarm — do not report as SOTA.")
        print("Per-type (not pooled into one exact-match):")
        for t, v in sorted(ms["by_type_mad"].items()):
            print(f"  {t:8s} MAD={v:.3f} exact={ms['by_type_exact'].get(t, float('nan')):.3f}")
        report["model"] = ms

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out / "metrics.json")


if __name__ == "__main__":
    main()
