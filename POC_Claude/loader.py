"""
src/data/loader.py

Leakage-safe data pipeline for the Twin-2K-500 behavior-model POC.
Implements the pipeline described in docs/02_model_plan.md §2.2.

    HF wave_split + catalog + CSVs
            |
            +-- persona: wave1_3_persona_json
            |     drop any question whose csv_columns intersect wave-4 columns
            +-- ask: wave4_Q_wave4_A -> deep-copy, delete Answers / Values / Selected*
            +-- label: wave4_response.csv[pid, col]
            +-- diagnostics (never in X):
                  wave4_Q_wave1_3_A, overlap columns in wave1_3_response.csv

Run directly to build train/val/test JSONL + run the leakage unit test:

    python -m src.data.loader --out_dir data/

Output:
    data/splits/pids_{train,val,test}.json
    data/overlap_columns.txt
    data/examples_{train,val,test}.jsonl   -- {pid, col, prompt, target}
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
SEED = 20250319
SPLIT = (0.70, 0.15, 0.15)  # train / val / test, by person

SYSTEM_PROMPT = "You are simulating one survey respondent."


# --------------------------------------------------------------------------
# 1. Load raw HF configs + tabular files
# --------------------------------------------------------------------------

def load_raw() -> dict[str, Any]:
    """Load wave_split (the ONLY persona source we use) + catalog + CSVs.

    full_persona is intentionally never loaded here: its persona_text /
    persona_json already contain the wave-4 answer on repeated items
    (trap 1, see docs/02_model_plan.md §2.1).
    """
    wave_split = load_dataset(REPO_ID, "wave_split")["data"].to_pandas()
    wave_split["pid"] = pd.to_numeric(wave_split["pid"], errors="raise").astype("int64")

    catalog_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename="question_catalog_and_human_response_csv/question_catalog.json",
    )
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    w13_csv = pd.read_csv(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename="question_catalog_and_human_response_csv/wave1_3_response.csv"))
    w4_csv = pd.read_csv(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename="question_catalog_and_human_response_csv/wave4_response.csv"))

    return {
        "wave_split": wave_split,
        "catalog": catalog,
        "w13_csv": w13_csv,
        "w4_csv": w4_csv,
    }


# --------------------------------------------------------------------------
# 2. Column <-> type/QID mapping (never col.startswith(QuestionID) --
#    QID290_5 would incorrectly match QID2 / QID29)
# --------------------------------------------------------------------------

def build_column_index(catalog: list[dict]) -> dict[str, dict]:
    col_to_question: dict[str, dict] = {}
    for q in catalog:
        for col in q.get("csv_columns") or []:
            col_to_question[col] = q
    return col_to_question


def overlap_columns(w13_csv: pd.DataFrame, w4_csv: pd.DataFrame) -> list[str]:
    w13_cols = set(w13_csv.columns) - {"pid"}
    w4_cols = set(w4_csv.columns) - {"pid"}
    return sorted(w13_cols & w4_cols)


# --------------------------------------------------------------------------
# 3. Persona construction (waves 1-3 only, overlap questions dropped)
# --------------------------------------------------------------------------

def iter_questions(obj: Any):
    """Walk the nested persona_json structure, yielding question blocks."""
    if isinstance(obj, dict):
        if obj.get("QuestionID") is not None and "Answers" in obj:
            yield obj
        for v in obj.values():
            yield from iter_questions(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_questions(v)


def strip_answers(question_block: dict) -> dict:
    """Deep-copy a question block and delete Answers/Values/Selected* fields.

    Used both to build the leak-safe persona (drop overlap questions
    entirely -- see build_persona) and to strip the wave-4 *ask* side
    (trap 1: wave4_Q_wave4_A must never carry Answers into the prompt).
    """
    q = copy.deepcopy(question_block)
    for key in list(q.keys()):
        if key == "Answers" or key.startswith("Selected") or key == "Values":
            del q[key]
    return q


def build_persona(wave1_3_persona_json: str, overlap_qids: set[str]) -> str:
    """wave1-3 persona, with any question that is re-asked in wave 4 dropped
    entirely (not just stripped) -- it would otherwise still leak the
    *identity* of what wave 4 asks, and more importantly keeps the
    leak-safe persona strictly disjoint from the diagnostic wave1_3 answer
    used for the copy-last baseline (trap 2 boundary).
    """
    blob = json.loads(wave1_3_persona_json) if isinstance(wave1_3_persona_json, str) else wave1_3_persona_json
    kept = [q for q in iter_questions(blob) if str(q.get("QuestionID")) not in overlap_qids]
    lines = []
    for q in kept:
        text = q.get("QuestionText", "").strip()
        ans = q.get("Answers")
        if text and ans:
            lines.append(f"Q: {text}\nA: {ans}")
    return "\n".join(lines)


def build_prompt(persona_text: str, question_text: str, options_or_range: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Persona:\n{persona_text}\n\n"
        f"Question:\n{question_text}\n"
        f"Options: {options_or_range}\n\n"
        f"Reply with only the answer code or number."
    )


# --------------------------------------------------------------------------
# 4. Person-level split (never row-level -- all of a person's wave-4 items
#    must stay in one split, per docs/02_model_plan.md §2.2)
# --------------------------------------------------------------------------

def person_split(pids: list[int], seed: int = SEED, ratios=SPLIT) -> dict[str, list[int]]:
    rng = random.Random(seed)
    shuffled = sorted(pids)  # sort first so the shuffle is deterministic across pandas versions
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


# --------------------------------------------------------------------------
# 5. Leakage unit test -- MUST pass before any example is written.
#    docs/02_model_plan.md / docs/03_eval_strategy.md pid=1 / QID154 check.
# --------------------------------------------------------------------------

def leakage_unit_test(raw: dict[str, Any], overlap_cols: set[str]) -> None:
    pid, col = 1, "QID154"
    w13_csv, w4_csv = raw["w13_csv"], raw["w4_csv"]

    w13_val = w13_csv.loc[w13_csv.pid == pid, col].values
    w4_val = w4_csv.loc[w4_csv.pid == pid, col].values
    assert len(w13_val) and len(w4_val), f"pid={pid} / {col} missing from CSVs -- fix fixture before trusting the test"
    w13_val, w4_val = w13_val[0], w4_val[0]

    row = raw["wave_split"].loc[raw["wave_split"].pid == pid].iloc[0]
    qid_set = {col.split("_")[0]}  # coarse, matches the single-slider case used here
    persona = build_persona(row["wave1_3_persona_json"], overlap_qids=qid_set)

    assert str(int(w4_val)) not in persona, (
        f"LEAK (trap 1): wave-4 answer {w4_val} found in persona for pid={pid}"
    )
    assert str(int(w13_val)) not in persona, (
        f"LEAK (trap 2): wave1-3 answer {w13_val} for the SAME item found in persona for pid={pid}. "
        f"This is legal only as the copy-last baseline / test-retest diagnostic, never as model input."
    )

    ask_raw = row["wave4_Q_wave4_A"]
    ask_blob = json.loads(ask_raw) if isinstance(ask_raw, str) else ask_raw
    for q in iter_questions(ask_blob):
        if str(q.get("QuestionID")) == "QID154":
            stripped = strip_answers(q)
            assert "Answers" not in stripped and "Values" not in stripped, (
                "LEAK: stripped wave4_Q_wave4_A still carries Answers/Values"
            )

    print(f"[leak test] PASS -- pid={pid} col={col}: neither {int(w13_val)} nor {int(w4_val)} found in persona/prompt.")


# --------------------------------------------------------------------------
# 6. Build examples for one split
# --------------------------------------------------------------------------

def build_examples(
    pids: list[int],
    raw: dict[str, Any],
    col_index: dict[str, dict],
    overlap_cols: list[str],
    max_items_per_pid: int | None = None,
) -> list[dict]:
    wave_split = raw["wave_split"].set_index("pid")
    w4_csv = raw["w4_csv"].set_index("pid")
    overlap_qids = {col_index[c]["QuestionID"] for c in overlap_cols if c in col_index}

    examples = []
    for pid in pids:
        if pid not in wave_split.index or pid not in w4_csv.index:
            continue
        row = wave_split.loc[pid]
        persona = build_persona(row["wave1_3_persona_json"], overlap_qids=overlap_qids)

        ask_raw = row["wave4_Q_wave4_A"]
        ask_blob = json.loads(ask_raw) if isinstance(ask_raw, str) else ask_raw
        questions = list(iter_questions(ask_blob))
        if max_items_per_pid:
            questions = questions[:max_items_per_pid]

        for q in questions:
            qid = str(q.get("QuestionID"))
            meta = col_index.get(q.get("_csv_column", ""), None)
            # Fall back: find the CSV column(s) for this QID among overlap_cols
            candidate_cols = [c for c in overlap_cols if col_index.get(c, {}).get("QuestionID") == qid]
            for col in candidate_cols:
                label_row = w4_csv.loc[pid]
                if col not in label_row or pd.isna(label_row[col]):
                    continue  # between-subject arm this person didn't see
                target = str(label_row[col])
                stripped = strip_answers(q)
                prompt = build_prompt(
                    persona_text=persona,
                    question_text=stripped.get("QuestionText", ""),
                    options_or_range=str(stripped.get("Options", stripped.get("Range", ""))),
                )
                examples.append({"pid": int(pid), "col": col, "prompt": prompt, "target": target})
    return examples


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--max_items_per_pid", type=int, default=None,
                     help="Cap items/person, e.g. 20 for the POC slice (500 pids x <=20 items).")
    ap.add_argument("--poc_pids", type=int, default=None,
                     help="If set, subsample this many train pids for a fast POC run.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    print("Loading wave_split + catalog + CSVs from Hugging Face (never full_persona)...")
    raw = load_raw()
    col_index = build_column_index(raw["catalog"])
    overlap_cols = overlap_columns(raw["w13_csv"], raw["w4_csv"])
    (out_dir / "overlap_columns.txt").write_text("\n".join(overlap_cols))
    print(f"{len(overlap_cols)} overlap columns written to {out_dir / 'overlap_columns.txt'}")

    print("\nRunning leakage unit test (must pass before anything else)...")
    leakage_unit_test(raw, set(overlap_cols))

    pids = sorted(raw["wave_split"]["pid"].unique().tolist())
    splits = person_split(pids)
    if args.poc_pids:
        splits["train"] = splits["train"][: args.poc_pids]
    for name, split_pids in splits.items():
        (out_dir / "splits" / f"pids_{name}.json").write_text(json.dumps(split_pids))
        print(f"{name}: {len(split_pids)} pids")

    for name, split_pids in splits.items():
        examples = build_examples(
            split_pids, raw, col_index, overlap_cols,
            max_items_per_pid=args.max_items_per_pid,
        )
        out_path = out_dir / f"examples_{name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()
