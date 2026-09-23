# Deliverable 6 — POC (bonus)

This is a **runnable slice**, not a reproduction of Toubia et al. Figure 2. The minimal Deliverable 3 gate is: leakage checks pass, parse rate is reported, and slice MAD is compared with random. Beating random is an aspirational performance check, not a requirement for completing the bonus POC. The run may be weak, but it must not be leaky. This minimal take-home evaluation deliberately does not implement Deliverable 3's full research protocol, such as 100 random-baseline seeds, participant-bootstrap confidence intervals, or the stronger follow-up thresholds.

**Not verified:** official 17-task MAD script / paper **81.72%**. The results below are slice-mean MAD with **train-only** empirical ranges and are not directly comparable with the paper's score.

**References:** D2 §2 data pipeline and §5 training details (`docs/02_model_plan.md`); D3 §2.3 split protocol and §2.5 acceptance criteria (`docs/03_eval_strategy.md`). Default local checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`. The strict `<0.5B` Colab run uses `HuggingFaceTB/SmolLM2-360M-Instruct`.

**Model-size disclosure:** the assignment asks for a model with **fewer than 0.5B parameters**. Although Qwen markets this checkpoint as “0.5B,” the current Transformers environment counts `630,167,424` base parameters; this LoRA configuration adds `8,798,208`, for `638,965,632` total parameters (`8,798,208` trainable). Therefore, this implementation does **not** satisfy a strict total-parameter interpretation of the threshold. The bonus deliverable itself is optional, but the `<0.5B` threshold should not be described as optional when claiming strict compliance. I retained Qwen2.5-0.5B-Instruct as a near-boundary, locally runnable instruction-model POC and disclose the deviation rather than hiding it. A strict replacement would be an instruction model such as `HuggingFaceTB/SmolLM2-360M-Instruct`, with LoRA target modules and hyperparameters revalidated; the leakage-safe data and evaluation pipeline can remain the same.

## Layout

```text
src/leak_test.py           # pid=1 / QID154: persona has neither 70 nor 82
src/data/build_jsonl.py    # wave_split + CSVs; short tabular persona; never full_persona
src/baselines.py           # random, majority, copy-last (never fills missing with gold)
src/train.py               # QLoRA SFT; default Qwen2.5-0.5B-Instruct
src/evaluate.py            # baselines + optional adapter
data/poc/*.jsonl           # generated; not committed
results/poc*.json          # generated after evaluate; not committed
```

This POC implements Deliverable 2's **no-copy** condition. Persona = non-overlap CSV fields only (demographics first, capped). One JSONL row = one `(pid, column)`. Person split, seed `20250319`, 70 / 15 / 15. Default slice: **MC only**, 500 train pids × ≤20 items, 100 val/test pids. The random baseline samples uniformly from each column's train-observed answer codes. Copy-last diagnostics live in `diag_*.jsonl`, never in the prompt. Leak check is on the **persona** span: the QID154 stem itself says “70 lawyers”.

Unlike the research-scale D2 pipeline, this POC constructs a safe question payload directly from the catalog's `QuestionText`, `Options`, and `Range`; it does not load and recursively strip `wave4_Q_wave4_A`. It also uses a plain prompt, 2,048-token sequences, and no retrieval. Run 1 used the script defaults (2 epochs, learning rate `2e-4`). Runs 2 and 3 used 3 epochs and `1e-4`. These are explicit POC shortcuts rather than implementations of D2's 7B summary-plus-retrieval recipe.

## Run (from repo root)

Default local loop (Run 1). Run 2 and Run 3 commands are under each results block.

```text
python -m unittest discover -s tests
python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20
python -m src.leak_test data/poc/leak_fixture.jsonl data/poc/examples_train.jsonl data/poc/examples_val.jsonl
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc/adapter --out_dir results/poc
```

## POC run report

**Status:** three completed validation runs. Run identity (dates, bundle, base model, hardware) is recorded per run below. Shared fields that do not change across runs are listed once. The paper's published numbers were not copied into this section.

### Shared contract

- Input condition: `no-copy`
- Evaluation split: `validation`
- Training method family: QLoRA 4-bit NF4; LoRA rank 16, alpha 32, dropout 0.05
- Model/training seed: `42`

### Data used

- Train participants / examples: `500` / `10,000`
- Validation participants / examples: `100` / `2,000`
- Slice: `MC only`, at most `20` items per participant
- Participant split seed: `20250319`

### Integrity checks

- Unit tests: `7/7 passed`
- Known-fixture leak test: `PASS`
- Validation-prompt structural leak scan: `PASS`
- Train/validation participant overlap: `0`

If any integrity check fails, stop here and do not report the model score as valid.

### Results

Runs 1 and 2 share the local `data/poc` JSONL, so their baselines match. Run 3 rebuilt JSONL on Colab with the same caps and seeds; its baselines are therefore similar but not identical. All scores use train-only empirical ranges.

#### Run 1 — Qwen2.5-0.5B-Instruct, QLoRA, 2 epochs, lr `2e-4` (local RTX 3060)

- Status: completed
- Evaluation date: `2026-09-23`
- Training bundle date: `2026-09-21`
- Bundle ID: `20260921-3e3eaf7`
- Command: `python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora`
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tune: QLoRA 4-bit NF4; LoRA rank 16, alpha 32, dropout 0.05
- Epochs / learning rate: `2` / `2e-4`
- Training wall time: about `2.5` hours
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM
- Adapter: `runs/poc/adapter`
- Uniform random: slice MAD `0.5357`; MC exact match `0.4685`; parse rate `100.00%`; people `100`
- Train majority: slice MAD `0.5797`; MC exact match `0.5245`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: slice MAD `0.8461`; MC exact match `0.7920`; coverage/parse rate `99.95%`; people `100`
- LoRA SFT: slice MAD `0.5274`; MC exact match `0.4670`; parse rate `100.00%`; people `100`
- LoRA MAD / copy-last MAD: `0.623`

> Integrity checks passed. This 2-epoch Qwen run did **not** beat random (`0.5274` vs `0.5357`) and trailed train-majority (`0.5797`). The loop produced fully parseable codes, but the small model and short tabular persona did not beat population priors.

#### Run 2 — Qwen2.5-0.5B-Instruct, QLoRA, 3 epochs, lr `1e-4` (local RTX 3060)

- Status: completed
- Evaluation date: `2026-09-23`
- Training bundle date: `2026-09-23`
- Bundle ID: `20260923-3e3eaf7`
- Command:

```text
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --out_dir runs/poc_e3_lr1e4 --epochs 3 --lr 1e-4 --qlora
```

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tune: QLoRA 4-bit NF4; LoRA rank 16, alpha 32, dropout 0.05
- Epochs / learning rate: `3` / `1e-4`
- Training wall time: about `5` hours
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM
- Adapter: `runs/poc_e3_lr1e4/adapter`
- Uniform random: slice MAD `0.5357`; MC exact match `0.4685`; parse rate `100.00%`; people `100`
- Train majority: slice MAD `0.5797`; MC exact match `0.5245`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: slice MAD `0.8461`; MC exact match `0.7920`; coverage/parse rate `99.95%`; people `100`
- LoRA SFT: slice MAD `0.5411`; MC exact match `0.4765`; parse rate `100.00%`; people `100`
- LoRA MAD / copy-last MAD: `0.640`

> Same validation JSONL as Run 1. This schedule **did** beat random (`0.5411` vs `0.5357`) by a thin margin and stayed below train-majority (`0.5797`). The gain is too small to treat as a research-scale success; D3's stronger follow-up bar is mean-random + 0.01.

#### Run 3 — SmolLM2-360M-Instruct, QLoRA, 3 epochs, lr `1e-4` (Colab T4)

- Status: completed
- Evaluation date: `2026-09-23`
- Training bundle date: `2026-09-23`
- Bundle ID: not copied from Colab (`/content/runs/smollm360m_e3/bundle_manifest.json`)
- Command:

```text
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --model_name HuggingFaceTB/SmolLM2-360M-Instruct --out_dir /content/runs/smollm360m_e3 --epochs 3 --lr 1e-4 --qlora
```

- Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Fine-tune: QLoRA 4-bit NF4; LoRA rank 16, alpha 32, dropout 0.05
- Epochs / learning rate: `3` / `1e-4`
- Training wall time: about `3` hours
- Hardware: Colab T4
- Adapter: `/content/runs/smollm360m_e3/adapter`
- Source: Colab `metrics.json` (JSONL rebuilt on Colab, so baselines are not identical to Run 1)
- Uniform random: slice MAD `0.5256`; MC exact match `0.4545`; parse rate `100.00%`; people `100`
- Train majority: slice MAD `0.5954`; MC exact match `0.5425`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: slice MAD `0.8565`; MC exact match `0.8045`; coverage/parse rate `99.85%`; people `100`
- LoRA SFT: slice MAD `0.4775`; MC exact match `0.4145`; parse rate `100.00%`; people `100`
- LoRA MAD / copy-last MAD: `0.558`

> This is the strict `<0.5B` run. Integrity held and every output parsed, but SmolLM2-360M did **not** beat random (`0.4775` vs `0.5256`) and trailed train-majority (`0.5954`).

### Interpretation

> All three runs completed with green leakage checks and 100% parse rate. Run 1 (Qwen, 2 epochs / `2e-4`) lost to random. Run 2 (Qwen, 3 epochs / `1e-4`, same local JSONL) is the only schedule that beat random, `0.5411` vs `0.5357`, but it still trailed train-majority and missed D3's stronger +0.01 follow-up margin. Run 3 (SmolLM2-360M on Colab) lost to random. The POC loop works; none of these tiny-model slices is a candidate for research-scale promotion.

These are validation-slice results using train-only empirical ranges. They are not the official 17-task MAD score, do not include participant-bootstrap confidence intervals, and must not be compared directly with the paper's published **81.72%** human result. Copy-last is an empirical same-pair test–retest benchmark, not a mathematical upper bound.

> **External-use disclosure:** Simulated survey responses. Not observed behavior. U.S. online panel, not population-weighted. Short-term evidence only: wave 4 launched approximately two weeks after wave 3, with longer intervals for items originating in earlier waves.

Hardware note: Runs 1–2 used an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM. Run 3 used a Colab T4. QLoRA training requires CUDA. On a T4, use fp16 because T4 does not support bf16. CPU is practical for JSONL generation, leakage checks, and baseline evaluation; adapter inference on CPU is supported but very slow.

If leak test is red, do not print a score table. If slice MAD > copy-last/ceiling, treat it as a leak alarm, not SOTA.

## Deps (POC)

`transformers` `peft` `accelerate` `bitsandbytes` `datasets` `pandas` `torch` — see `requirements.txt`. If `HF_HOME` is unset, the builder uses `data_raw/.cache/huggingface` when that folder exists.

Colab may ship an old `torchao` that PEFT rejects (`Found version 0.10.0, but only versions above 0.16.0 are supported`). This POC does not use `torchao`. Uninstall it, then rerun evaluate:

```text
pip uninstall -y torchao
```
