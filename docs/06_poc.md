# Deliverable 6 — POC (bonus)

This is a **runnable slice**, not a reproduction of Toubia et al. Figure 2. Gate (D3): leak test green, parseable codes, slice MAD **> random**. The run may be weak. It must not be leaky.

**Not verified:** official 17-task MAD script / paper **81.72%**. Numbers here (once the scripts are run) are slice-mean MAD with **train-only** empirical ranges. No scores are claimed in this note until `results/poc/metrics.json` exists.

**References:** D2 §2.3 / §2.5 (`docs/02_model_plan.md`); D3 gates (`docs/03_eval_strategy.md`). Base model: `Qwen/Qwen2.5-0.5B-Instruct`.

## Layout

```text
src/leak_test.py           # pid=1 / QID154: persona has neither 70 nor 82
src/data/build_jsonl.py    # wave_split + CSVs; short tabular persona; never full_persona
src/baselines.py           # random, majority, copy-last (never fills missing with gold)
src/train.py               # Qwen2.5-0.5B-Instruct LoRA
src/evaluate.py            # baselines + optional adapter
data/poc/*.jsonl           # generated; not committed
results/poc/metrics.json   # generated after evaluate
```

Persona = non-overlap CSV fields only (demographics first, capped). One JSONL row = one `(pid, column)`. Person split, seed `20250319`, 70 / 15 / 15. Default slice: **MC only**, 500 train pids × ≤20 items, 100 val/test pids. Copy-last diagnostics live in `diag_*.jsonl`, never in the prompt. Leak check is on the **persona** span: the QID154 stem itself says “70 lawyers”.

## Run (from repo root)

```text
python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20
python -m src.leak_test data/poc/leak_fixture.jsonl
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc/adapter --out_dir results/poc
```

GPU: Colab T4 with `--qlora` (fp16; T4 is not bf16). CPU: JSONL + leak + baselines only (omit `--adapter_dir`).

If leak test is red, do not print a score table. If slice MAD > copy-last/ceiling, treat it as a leak alarm, not SOTA.

## Deps (POC)

`transformers` `peft` `accelerate` `bitsandbytes` `datasets` `pandas` `torch` — see `requirements.txt`. If `HF_HOME` is unset, the builder uses `data_raw/.cache/huggingface` when that folder exists.
