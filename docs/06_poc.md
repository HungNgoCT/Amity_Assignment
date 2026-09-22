# Deliverable 6 — POC (bonus)

This is a **runnable slice**, not a reproduction of Toubia et al. Figure 2. The minimal Deliverable 3 gate is: leakage checks pass, parse rate is reported, and slice MAD is compared with random. Beating random is an aspirational performance check, not a requirement for completing the bonus POC. The run may be weak, but it must not be leaky. This minimal take-home evaluation deliberately does not implement Deliverable 3's full research protocol, such as 100 random-baseline seeds, participant-bootstrap confidence intervals, or the stronger follow-up thresholds.

**Not verified:** official 17-task MAD script / paper **81.72%**. The results below are slice-mean MAD with **train-only** empirical ranges and are not directly comparable with the paper's score.

**References:** D2 §2 data pipeline and §5 training details (`docs/02_model_plan.md`); D3 §2.3 split protocol and §2.5 acceptance criteria (`docs/03_eval_strategy.md`). Base model: `Qwen/Qwen2.5-0.5B-Instruct`.

**Model-size disclosure:** the assignment asks for a model with **fewer than 0.5B parameters**. Although Qwen markets this checkpoint as “0.5B,” the current Transformers environment counts `630,167,424` base parameters; this LoRA configuration adds `8,798,208`, for `638,965,632` total parameters (`8,798,208` trainable). Therefore, this implementation does **not** satisfy a strict total-parameter interpretation of the threshold. The bonus deliverable itself is optional, but the `<0.5B` threshold should not be described as optional when claiming strict compliance. I retained Qwen2.5-0.5B-Instruct as a near-boundary, locally runnable instruction-model POC and disclose the deviation rather than hiding it. A strict replacement would be an instruction model such as `HuggingFaceTB/SmolLM2-360M-Instruct`, with LoRA target modules and hyperparameters revalidated; the leakage-safe data and evaluation pipeline can remain the same.

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

This POC implements Deliverable 2's **no-copy** condition. Persona = non-overlap CSV fields only (demographics first, capped). One JSONL row = one `(pid, column)`. Person split, seed `20250319`, 70 / 15 / 15. Default slice: **MC only**, 500 train pids × ≤20 items, 100 val/test pids. The random baseline samples uniformly from each column's train-observed answer codes. Copy-last diagnostics live in `diag_*.jsonl`, never in the prompt. Leak check is on the **persona** span: the QID154 stem itself says “70 lawyers”.

Unlike the research-scale D2 pipeline, this POC constructs a safe question payload directly from the catalog's `QuestionText`, `Options`, and `Range`; it does not load and recursively strip `wave4_Q_wave4_A`. It also uses a plain prompt, 2,048-token sequences, 2 epochs, learning rate `2e-4`, and no retrieval. These are explicit POC shortcuts rather than implementations of D2's 7B summary-plus-retrieval recipe.

## Run (from repo root)

```text
python -m unittest discover -s tests
python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20
python -m src.leak_test data/poc/leak_fixture.jsonl data/poc/examples_train.jsonl data/poc/examples_val.jsonl
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc/adapter --out_dir results/poc
```

## POC run report

**Status:** completed validation rerun. Results were generated from `runs/poc/bundle_manifest.json` and `results/poc/metrics.json`; the paper's published numbers were not copied into this section.

### Run identity

- Evaluation rerun date: `2026-09-23`
- Training bundle date: `2026-09-21`
- Hardware: `NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM`
- Bundle ID: `20260921-3e3eaf7`
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training method: `QLoRA (4-bit NF4) with LoRA rank 16, alpha 32, dropout 0.05`
- Input condition: `no-copy`
- Evaluation split: `validation`

### Data used

- Train participants / examples: `500` / `10,000`
- Validation participants / examples: `100` / `2,000`
- Slice: `MC only`, at most `20` items per participant
- Participant split seed: `20250319`
- Model/training seed: `42`

### Integrity checks

- Unit tests: `7/7 passed`
- Known-fixture leak test: `PASS`
- Validation-prompt structural leak scan: `PASS`
- Train/validation participant overlap: `0`

If any integrity check fails, stop here and do not report the model score as valid.

### Results

- Uniform random: slice MAD `0.5357`; MC exact match `0.4685`; parse rate `100.00%`; people `100`
- Train majority: slice MAD `0.5797`; MC exact match `0.5245`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: slice MAD `0.8461`; MC exact match `0.7920`; coverage/parse rate `99.95%`; people `100`
- LoRA SFT: slice MAD `0.5274`; MC exact match `0.4670`; parse rate `100.00%`; people `100`
- LoRA MAD / copy-last MAD: `0.623`

### Interpretation

> All required integrity checks passed. On the validation slice, LoRA achieved slice MAD `0.5274` with parse rate `100.00%`, compared with random `0.5357`, train majority `0.5797`, and copy-last `0.8461`. The model did **not** beat the aspirational random-baseline performance check and also trailed the train-majority baseline. The fully parseable outputs show that the training and generation loop worked, but this small model and short tabular persona representation did not learn enough person-specific signal to beat simple population priors. It is therefore a completed end-to-end POC, not a candidate for research-scale promotion or deployment.

These are validation-slice results using train-only empirical ranges. They are not the official 17-task MAD score, do not include participant-bootstrap confidence intervals, and must not be compared directly with the paper's published **81.72%** human result. Copy-last is an empirical same-pair test–retest benchmark, not a mathematical upper bound.

> **External-use disclosure:** Simulated survey responses. Not observed behavior. U.S. online panel, not population-weighted. Short-term evidence only: wave 4 launched approximately two weeks after wave 3, with longer intervals for items originating in earlier waves.

Hardware note: This run used an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM. QLoRA training requires CUDA. On a Colab T4, use fp16 because T4 does not support bf16. CPU is practical for JSONL generation, leakage checks, and baseline evaluation; adapter inference on CPU is supported but very slow.

If leak test is red, do not print a score table. If slice MAD > copy-last/ceiling, treat it as a leak alarm, not SOTA.

## Deps (POC)

`transformers` `peft` `accelerate` `bitsandbytes` `datasets` `pandas` `torch` — see `requirements.txt`. If `HF_HOME` is unset, the builder uses `data_raw/.cache/huggingface` when that folder exists.
