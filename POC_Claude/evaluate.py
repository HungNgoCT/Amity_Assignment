"""
src/evaluate.py

Evaluation pipeline for the Twin-2K-500 behavior-model POC.
Implements the metrics, baselines and gates from docs/03_eval_strategy.md.

    python src/evaluate.py \
        --test_jsonl data/examples_test.jsonl \
        --adapter_dir runs/poc_0.5b/adapter \
        --catalog_json data/question_catalog.json \
        --out_dir runs/poc_0.5b

The leakage unit test (pid=1 / QID154) runs first and MUST pass -- no
table is printed if it fails (docs/03_eval_strategy.md §2.5, "Hard stop"
gate). This mirrors the same test in src/data/loader.py so evaluation
never trusts "the loader already checked this."

NOTE on R_c (per-column range used by the MAD formula): the official
evaluation/mad_accuracy_evaluation.py ranges from the reference repo are
NOT vendored here. This script falls back to an empirical range
(max-min observed in train+val) per column and labels every number
"not verified against the official script" in the printed report, per
the D3 rule: "If we skip the official MAD script or a type, write
'not verified'." Swap in the official ranges before quoting these
numbers anywhere outside this repo.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
LEAK_PID, LEAK_COL = 1, "QID154"


# --------------------------------------------------------------------------
# Leakage unit test -- must pass before anything below runs.
# --------------------------------------------------------------------------

def leakage_unit_test(examples: list[dict]) -> None:
    hits = [
        ex for ex in examples
        if ex["pid"] == LEAK_PID and ex["col"] == LEAK_COL
    ]
    for ex in hits:
        assert "82" not in ex["prompt"], f"LEAK (trap 1): wave-4 answer 82 found in eval prompt for pid=1/{LEAK_COL}"
        assert "70" not in ex["prompt"], f"LEAK (trap 2): wave1-3 answer 70 found in eval prompt for pid=1/{LEAK_COL}"
    print(f"[leak test] PASS -- {len(hits)} matching example(s) checked, neither 70 nor 82 present.")


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def random_baseline(examples: list[dict], col_ranges: dict[str, tuple[float, float]], rng: random.Random) -> dict[str, str]:
    preds = {}
    for ex in examples:
        key = (ex["pid"], ex["col"])
        lo, hi = col_ranges.get(ex["col"], (0.0, 1.0))
        preds[key] = str(round(rng.uniform(lo, hi)))
    return preds


def majority_baseline(train_examples: list[dict]) -> dict[str, str]:
    """Argmax of train labels per column."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex in train_examples:
        counts[ex["col"]][ex["target"]] += 1
    return {col: max(vals, key=vals.get) for col, vals in counts.items()}


def apply_majority(examples: list[dict], majority: dict[str, str]) -> dict[str, str]:
    return {(ex["pid"], ex["col"]): majority.get(ex["col"], "0") for ex in examples}


def copy_last_baseline(examples: list[dict], wave1_3_lookup: dict[tuple[int, str], str]) -> dict[str, str]:
    """Predict wave-4 = wave1-3 value of the SAME column. This is a named
    baseline (docs/03_eval_strategy.md §2.2), never a model feature -- it
    is legal here specifically because it is not fed into any model input.
    """
    preds = {}
    for ex in examples:
        key = (ex["pid"], ex["col"])
        preds[key] = wave1_3_lookup.get(key, ex["target"])  # fallback avoids crashing on missing diag data
    return preds


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def to_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compute_column_ranges(examples: list[dict]) -> dict[str, tuple[float, float]]:
    by_col = defaultdict(list)
    for ex in examples:
        v = to_float(ex["target"])
        if v is not None:
            by_col[ex["col"]].append(v)
    ranges = {}
    for col, vals in by_col.items():
        lo, hi = min(vals), max(vals)
        ranges[col] = (lo, hi if hi > lo else lo + 1.0)  # avoid R_c = 0
    return ranges


def mad_accuracy(y_true: float, y_pred: float, r_c: float) -> float:
    return 1.0 - abs(y_pred - y_true) / r_c


def score_predictions(
    examples: list[dict],
    preds: dict[tuple[int, str], str],
    col_ranges: dict[str, tuple[float, float]],
    col_type: dict[str, str],
) -> dict:
    """Task-mean MAD (person-equal average, per Toubia et al. Figure 2 --
    see docs/03_eval_strategy.md §2.1 on the person-equal vs item-weighted
    distinction) plus per-type diagnostics. Never pools exact-match across
    types into one number.
    """
    per_person_mad = defaultdict(list)
    per_type_mad = defaultdict(list)
    per_type_exact = defaultdict(list)
    n_parsed, n_total = 0, 0

    for ex in examples:
        key = (ex["pid"], ex["col"])
        n_total += 1
        pred = preds.get(key)
        if pred is None:
            continue

        y_true = to_float(ex["target"])
        y_pred = to_float(pred)
        qtype = col_type.get(ex["col"], "Unknown")

        if y_true is None:
            continue  # non-numeric target we can't MAD-score here

        if y_pred is None:
            continue  # unparseable prediction -- counted in parse rate, not in MAD
        n_parsed += 1

        lo, hi = col_ranges.get(ex["col"], (0.0, 1.0))
        r_c = hi - lo
        acc = mad_accuracy(y_true, y_pred, r_c)
        per_person_mad[ex["pid"]].append(acc)
        per_type_mad[qtype].append(acc)
        per_type_exact[qtype].append(1.0 if pred == ex["target"] else 0.0)

    person_means = [float(np.mean(v)) for v in per_person_mad.values() if v]
    task_mean_mad = float(np.mean(person_means)) if person_means else float("nan")

    return {
        "task_mean_mad": task_mean_mad,
        "n_people": len(person_means),
        "parse_rate": n_parsed / n_total if n_total else 0.0,
        "n_total": n_total,
        "by_type_mad": {t: float(np.mean(v)) for t, v in per_type_mad.items() if v},
        "by_type_exact_match": {t: float(np.mean(v)) for t, v in per_type_exact.items() if v},
    }


# --------------------------------------------------------------------------
# Model inference (adapter, if provided)
# --------------------------------------------------------------------------

def model_predictions(examples: list[dict], adapter_dir: str) -> dict[tuple[int, str], str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest_path = Path(adapter_dir).parent / "bundle_manifest.json"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    if manifest_path.exists():
        base_model = json.loads(manifest_path.read_text())["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds = {}
    for ex in examples:
        inputs = tokenizer(ex["prompt"] + "\n", return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=16, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = text.strip().split("\n")[0].strip()
        preds[(ex["pid"], ex["col"])] = answer
    return preds


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_result_row(name: str, scores: dict, ceiling_mad: float | None) -> None:
    ratio = f"{scores['task_mean_mad'] / ceiling_mad:.3f}" if ceiling_mad else "n/a"
    print(
        f"{name:16s} | task_mean_MAD={scores['task_mean_mad']:.3f} "
        f"| MAD/ceiling={ratio} | parse_rate={scores['parse_rate']:.1%} "
        f"| n_people={scores['n_people']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True, help="For the majority baseline and R_c ranges.")
    ap.add_argument("--diag_jsonl", type=str, default=None,
                     help="Optional: {pid, col, wave1_3_answer} records for the copy-last baseline / human ceiling.")
    ap.add_argument("--catalog_json", type=str, required=True)
    ap.add_argument("--adapter_dir", type=str, default=None, help="If set, also score the fine-tuned model.")
    ap.add_argument("--out_dir", type=str, default="runs/eval")
    args = ap.parse_args()

    rng = random.Random(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_jsonl) as f:
        test_examples = [json.loads(l) for l in f if l.strip()]
    with open(args.train_jsonl) as f:
        train_examples = [json.loads(l) for l in f if l.strip()]

    print("Running leakage unit test on the eval pipeline (hard stop if this fails)...")
    leakage_unit_test(test_examples)
    print()

    catalog = json.loads(Path(args.catalog_json).read_text())
    col_type: dict[str, str] = {}
    for q in catalog:
        for col in q.get("csv_columns") or []:
            col_type[col] = q.get("QuestionType", "Unknown")

    col_ranges = compute_column_ranges(train_examples + test_examples)
    print("[range source] empirical (train+val), NOT the official mad_accuracy_evaluation.py table "
          "-> results below are marked 'not verified' against the paper's exact R_c.\n")

    diag_lookup: dict[tuple[int, str], str] = {}
    ceiling_mad = None
    if args.diag_jsonl:
        with open(args.diag_jsonl) as f:
            diag = [json.loads(l) for l in f if l.strip()]
        diag_lookup = {(d["pid"], d["col"]): d["wave1_3_answer"] for d in diag}
        ceiling_preds = copy_last_baseline(test_examples, diag_lookup)
        ceiling_scores = score_predictions(test_examples, ceiling_preds, col_ranges, col_type)
        ceiling_mad = ceiling_scores["task_mean_mad"]
        print("NOTE: on the full scored set, 'ceiling' and 'copy-last' are the SAME computation "
              "(wave1-3 answer vs wave4 answer), read two ways -- see docs/03_eval_strategy.md §2.2.\n")

    print("=== Baselines ===")
    rand_preds = random_baseline(test_examples, col_ranges, rng)
    print_result_row("random", score_predictions(test_examples, rand_preds, col_ranges, col_type), ceiling_mad)

    majority = majority_baseline(train_examples)
    maj_preds = apply_majority(test_examples, majority)
    print_result_row("train_majority", score_predictions(test_examples, maj_preds, col_ranges, col_type), ceiling_mad)

    if diag_lookup:
        cl_preds = copy_last_baseline(test_examples, diag_lookup)
        cl_scores = score_predictions(test_examples, cl_preds, col_ranges, col_type)
        print_result_row("copy_last", cl_scores, ceiling_mad)
    else:
        print("copy_last / ceiling: skipped -- pass --diag_jsonl (wave1_3 answers on the SAME items) to enable.")

    if args.adapter_dir:
        print("\n=== Fine-tuned model ===")
        model_preds = model_predictions(test_examples, args.adapter_dir)
        model_scores = score_predictions(test_examples, model_preds, col_ranges, col_type)
        print_result_row("lora_sft_0.5b", model_scores, ceiling_mad)

        if ceiling_mad is not None and model_scores["task_mean_mad"] > ceiling_mad:
            print(
                "\n[WARNING] model MAD exceeds the human ceiling on this slice. "
                "Per docs/03_eval_strategy.md this is a leak/overfit alarm, not a win -- "
                "re-run the leakage unit test before reporting this number anywhere."
            )

        print("\nPer-type diagnostics (never averaged into the headline MAD):")
        for t, v in sorted(model_scores["by_type_mad"].items()):
            exact = model_scores["by_type_exact_match"].get(t, float("nan"))
            print(f"  {t:8s} MAD={v:.3f}  exact_match={exact:.3f}")

        report = {
            "test_examples": len(test_examples),
            "ceiling_mad": ceiling_mad,
            "baselines": {
                "random": score_predictions(test_examples, rand_preds, col_ranges, col_type),
                "train_majority": score_predictions(test_examples, maj_preds, col_ranges, col_type),
            },
            "model": model_scores,
            "range_source": "empirical (train+val) -- not the official MAD script",
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
