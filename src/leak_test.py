"""Leak unit test: pid=1 / QID154 persona must contain neither 70 nor 82.

The QID154 *stem* says “70 lawyers”; that is the item text, not the person’s
answer. We therefore search the Persona: … block, not the whole prompt.
D3 hard stop: do not print a score table if this fails.
Fails (does not silently PASS) when the JSONL has no matching example.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LEAK_PID = 1
LEAK_COL = "QID154"
ROUND1 = "70"
WAVE4 = "82"
ANSWER_KEY_RE = re.compile(
    r"""(?ix)(?:^|[\s{"'])"""
    r"""(?:answers?|values|selectedtext|selectedbyposition)"""
    r"""["']?\s*:"""
)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def persona_span(prompt: str) -> str:
    """Text between 'Persona:' and 'Question:'. Falls back to the full prompt."""
    start = prompt.find("Persona:")
    end = prompt.find("\nQuestion:")
    if start == -1:
        return prompt
    start = prompt.find("\n", start)
    start = start + 1 if start != -1 else 0
    if end == -1:
        return prompt[start:]
    return prompt[start:end]


def persona_fields(prompt: str) -> dict[str, str]:
    """Parse the builder's one-field-per-line persona representation."""
    fields = {}
    for line in persona_span(prompt).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def assert_prompts_leak_free(
    examples: list[dict],
    *,
    pid: int = LEAK_PID,
    col: str = LEAK_COL,
    round1: str = ROUND1,
    wave4: str = WAVE4,
    source: str = "jsonl",
    require_hit: bool = True,
) -> int:
    target_cols = {str(ex["col"]) for ex in examples}
    for ex in examples:
        prompt = ex["prompt"]
        fields = persona_fields(prompt)
        leaked_cols = sorted(target_cols & fields.keys())
        if leaked_cols:
            raise AssertionError(
                f"LEAK: repeated target columns are present in the no-copy persona "
                f"({source}, pid={ex['pid']}): {leaked_cols[:5]}"
            )
        if ANSWER_KEY_RE.search(prompt):
            raise AssertionError(
                f"LEAK: prompt contains an answer-bearing key ({source}, "
                f"pid={ex['pid']} {ex['col']})."
            )

    hits = [ex for ex in examples if int(ex["pid"]) == pid and ex["col"] == col]
    if not hits:
        if not require_hit:
            return 0
        raise AssertionError(
            f"Leak test has nothing to check: no pid={pid} / {col} in {source}. "
            "That is a vacuous PASS, not a green test. "
            "Run src.data.build_jsonl (writes leak_fixture.jsonl) and pass that file here."
        )
    for ex in hits:
        prompt = ex["prompt"]
        fields = persona_fields(prompt)
        if fields.get(col) == wave4:
            raise AssertionError(
                f"LEAK trap 1: wave-4 answer {wave4} is in the persona ({source}, pid={pid} {col})."
            )
        if fields.get(col) == round1:
            raise AssertionError(
                f"LEAK trap 2: first-round answer {round1} is in the persona ({source}, pid={pid} {col}). "
                "Legal as copy-last baseline only, never as X."
            )
    return len(hits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+", type=Path, help="JSONL files that contain prompts")
    args = ap.parse_args()
    n = 0
    for path in args.jsonl:
        # Fixture must contain pid=1/QID154. Other split files may not (MC-only slice).
        require_hit = path.name == "leak_fixture.jsonl" or len(args.jsonl) == 1
        n += assert_prompts_leak_free(
            load_jsonl(path), source=str(path), require_hit=require_hit
        )
    if n == 0:
        raise SystemExit(
            "No pid=1 / QID154 prompt was checked. Pass data/poc/leak_fixture.jsonl."
        )
    print(f"[leak test] PASS — {n} pid={LEAK_PID}/{LEAK_COL} persona(s); neither {ROUND1} nor {WAVE4}.")


if __name__ == "__main__":
    main()
