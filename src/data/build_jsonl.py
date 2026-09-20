"""Build leakage-safe JSONL from Hugging Face wave_split + CSVs. Never loads full_persona.

    python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20 --mc_only

Writes:
    data/poc/splits/pids_{train,val,test}.json
    data/poc/overlap_columns.txt
    data/poc/examples_{train,val,test}.jsonl   # {pid, col, qid, qtype, prompt, target}
    data/poc/diag_{train,val,test}.jsonl      # {pid, col, wave1_3_answer} for copy-last
    data/poc/leak_fixture.jsonl               # pid=1 / QID154 prompt, always
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.leak_test import LEAK_COL, LEAK_PID, assert_prompts_leak_free

REPO_ID = "LLM-Digital-Twin/Twin-2K-500"
SEED = 20250319
SPLIT = (0.70, 0.15, 0.15)
SYSTEM = "You are simulating one survey respondent."
MAX_PERSONA_CHARS = 3500


def _hf_home() -> None:
    if "HF_HOME" not in os.environ:
        local = Path(__file__).resolve().parents[2] / "data_raw" / ".cache" / "huggingface"
        if local.exists():
            os.environ["HF_HOME"] = str(local)


def load_raw() -> dict[str, Any]:
    _hf_home()
    wave_split = load_dataset(REPO_ID, "wave_split")["data"].to_pandas()
    wave_split["pid"] = pd.to_numeric(wave_split["pid"], errors="raise").astype("int64")

    catalog_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename="question_catalog_and_human_response_csv/question_catalog.json",
    )
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    w13 = pd.read_csv(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename="question_catalog_and_human_response_csv/wave1_3_response.csv",
        )
    )
    w4 = pd.read_csv(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename="question_catalog_and_human_response_csv/wave4_response.csv",
        )
    )
    w13["pid"] = pd.to_numeric(w13["pid"], errors="raise").astype("int64")
    w4["pid"] = pd.to_numeric(w4["pid"], errors="raise").astype("int64")
    return {"wave_split": wave_split, "catalog": catalog, "w13": w13, "w4": w4}


def column_index(catalog: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for q in catalog:
        for col in q.get("csv_columns") or []:
            out[col] = q
    return out


def overlap_columns(w13: pd.DataFrame, w4: pd.DataFrame) -> list[str]:
    return sorted((set(w13.columns) - {"pid"}) & (set(w4.columns) - {"pid"}))


def _fmt_target(val: Any) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val).strip()


def persona_from_table(row: pd.Series, keep_cols: list[str]) -> str:
    """Short waves 1–3 profile from non-overlap CSV columns only (D2 §2.4)."""
    lines = ["Waves 1-3 profile (no wave-4 items):"]
    for col in keep_cols:
        if col not in row.index:
            continue
        val = _fmt_target(row[col])
        if not val or val.lower() in {"nan", "none"}:
            continue
        lines.append(f"{col}: {val}")
        if sum(len(x) + 1 for x in lines) >= MAX_PERSONA_CHARS:
            break
    text = "\n".join(lines)
    return text[:MAX_PERSONA_CHARS]


def persona_keep_columns(w13: pd.DataFrame, overlap: set[str], catalog: list[dict]) -> list[str]:
    demo_cols: list[str] = []
    other: list[str] = []
    for q in catalog:
        block = str(q.get("BlockName") or "")
        cols = [c for c in (q.get("csv_columns") or []) if c in w13.columns and c not in overlap]
        if not cols:
            continue
        if "demograph" in block.lower() or "background" in block.lower():
            demo_cols.extend(cols)
        else:
            other.extend(cols)
    # Demographics first (D2 §2.4 summary), then a cap of other non-overlap fields.
    seen: set[str] = set()
    ordered: list[str] = []
    for c in demo_cols + other:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered[:40]


def build_prompt(persona: str, question_text: str, options: str) -> str:
    return (
        f"{SYSTEM}\n"
        f"Persona:\n{persona}\n\n"
        f"Question:\n{question_text}\n"
        f"Options: {options}\n\n"
        f"Reply with only the answer code or number."
    )


def person_split(pids: list[int], seed: int = SEED, ratios=SPLIT) -> dict[str, list[int]]:
    rng = random.Random(seed)
    shuffled = sorted(int(p) for p in pids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def example_for_cell(
    pid: int,
    col: str,
    persona: str,
    w4_row: pd.Series,
    w13_row: pd.Series,
    meta: dict,
) -> dict | None:
    if col not in w4_row.index or pd.isna(w4_row[col]):
        return None
    target = _fmt_target(w4_row[col])
    if not target:
        return None
    qtext = str(meta.get("QuestionText") or col)
    options = meta.get("Options")
    if options is None:
        options = meta.get("Range")
    opt_s = json.dumps(options, ensure_ascii=False) if options is not None else ""
    prompt = build_prompt(persona, qtext, opt_s)
    diag = _fmt_target(w13_row[col]) if col in w13_row.index else ""
    return {
        "pid": int(pid),
        "col": col,
        "qid": str(meta.get("QuestionID") or ""),
        "qtype": str(meta.get("QuestionType") or "Unknown"),
        "prompt": prompt,
        "target": target,
        "wave1_3_answer": diag,
    }


def build_for_pids(
    pids: list[int],
    w13: pd.DataFrame,
    w4: pd.DataFrame,
    col_index: dict[str, dict],
    overlap_cols: list[str],
    keep_persona_cols: list[str],
    max_items: int | None,
    mc_only: bool,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    w13_i = w13.set_index("pid")
    w4_i = w4.set_index("pid")
    scored = overlap_cols
    if mc_only:
        scored = [c for c in overlap_cols if col_index.get(c, {}).get("QuestionType") == "MC"]

    examples, diags = [], []
    for pid in pids:
        if pid not in w13_i.index or pid not in w4_i.index:
            continue
        r13, r4 = w13_i.loc[pid], w4_i.loc[pid]
        if isinstance(r13, pd.DataFrame):
            r13 = r13.iloc[0]
        if isinstance(r4, pd.DataFrame):
            r4 = r4.iloc[0]
        persona = persona_from_table(r13, keep_persona_cols)
        cols = list(scored)
        rng.shuffle(cols)
        n_kept = 0
        for col in cols:
            meta = col_index.get(col)
            if not meta:
                continue
            ex = example_for_cell(pid, col, persona, r4, r13, meta)
            if ex is None:
                continue
            examples.append({k: ex[k] for k in ("pid", "col", "qid", "qtype", "prompt", "target")})
            diags.append({"pid": ex["pid"], "col": ex["col"], "wave1_3_answer": ex["wave1_3_answer"]})
            n_kept += 1
            if max_items is not None and n_kept >= max_items:
                break
    return examples, diags


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/poc")
    ap.add_argument("--max_items_per_pid", type=int, default=20)
    ap.add_argument("--poc_train_pids", type=int, default=500)
    ap.add_argument("--poc_eval_pids", type=int, default=100, help="Cap val/test example pids; full split still written.")
    ap.add_argument("--all_types", action="store_true", help="Include Matrix/Slider/TE. Default slice is MC only.")
    args = ap.parse_args()
    mc_only = not args.all_types

    out = Path(args.out_dir)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    print("Loading wave_split + catalog + CSVs (never full_persona)...")
    raw = load_raw()
    col_index = column_index(raw["catalog"])
    overlap = overlap_columns(raw["w13"], raw["w4"])
    (out / "overlap_columns.txt").write_text("\n".join(overlap), encoding="utf-8")
    print(f"{len(overlap)} overlap columns")

    keep_cols = persona_keep_columns(raw["w13"], set(overlap), raw["catalog"])
    (out / "persona_columns.txt").write_text("\n".join(keep_cols), encoding="utf-8")
    splits = person_split(raw["wave_split"]["pid"].unique().tolist())
    for name, pids in splits.items():
        (out / "splits" / f"pids_{name}.json").write_text(json.dumps(pids), encoding="utf-8")
        print(f"{name} split: {len(pids)} pids")

    example_pids = {
        "train": splits["train"][: args.poc_train_pids] if args.poc_train_pids else splits["train"],
        "val": splits["val"][: args.poc_eval_pids] if args.poc_eval_pids else splits["val"],
        "test": splits["test"][: args.poc_eval_pids] if args.poc_eval_pids else splits["test"],
    }

    rng = random.Random(SEED)
    for name, pids in example_pids.items():
        examples, diags = build_for_pids(
            pids,
            raw["w13"],
            raw["w4"],
            col_index,
            overlap,
            keep_cols,
            args.max_items_per_pid,
            mc_only,
            rng,
        )
        write_jsonl(out / f"examples_{name}.jsonl", examples)
        write_jsonl(out / f"diag_{name}.jsonl", diags)
        print(f"{name}: {len(examples)} examples from {len(pids)} pids")

    # Always materialise the D3 unit-test row, whichever split pid=1 landed in.
    w13_i = raw["w13"].set_index("pid")
    w4_i = raw["w4"].set_index("pid")
    if LEAK_PID not in w13_i.index:
        raise SystemExit(f"pid={LEAK_PID} missing from CSV — cannot write leak fixture")
    persona = persona_from_table(w13_i.loc[LEAK_PID], keep_cols)
    meta = col_index[LEAK_COL]
    fixture = example_for_cell(
        LEAK_PID, LEAK_COL, persona, w4_i.loc[LEAK_PID], w13_i.loc[LEAK_PID], meta
    )
    if fixture is None:
        raise SystemExit("Could not build leak fixture for pid=1 / QID154")
    leak_row = {k: fixture[k] for k in ("pid", "col", "qid", "qtype", "prompt", "target")}
    write_jsonl(out / "leak_fixture.jsonl", [leak_row])
    n = assert_prompts_leak_free([leak_row], source=str(out / "leak_fixture.jsonl"))
    print(f"[leak test] PASS on fixture ({n} prompt). JSONL ready under {out}")


if __name__ == "__main__":
    main()
