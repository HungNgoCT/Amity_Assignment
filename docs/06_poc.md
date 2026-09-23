# Deliverable 6 — POC (bonus)

This is a **runnable slice**, not a reproduction of Toubia et al. Figure 2. The minimal Deliverable 3 gate is: leakage checks pass, parse rate is reported, and slice MAD is compared with random. Beating random is an aspirational performance check, not a requirement for completing the bonus POC. The run may be weak, but it must not be leaky. This minimal take-home evaluation deliberately does not implement Deliverable 3's full research protocol, such as 100 random-baseline seeds, participant-bootstrap confidence intervals, or the stronger follow-up thresholds.

**Not verified:** official 17-task MAD script / paper **81.72%**. The results below are slice-mean MAD with **train-only** empirical ranges and are not directly comparable with the paper's score.

**References:** D2 §2 data pipeline and §5 training details (`docs/02_model_plan.md`); D3 §3 train / test protocol and §5 acceptance criteria (`docs/03_eval_strategy.md`). I first chose `HuggingFaceTB/SmolLM2-360M-Instruct` to stay under 0.5B (Experiment 3). That run scored too low, so I then used `Qwen/Qwen2.5-0.5B-Instruct` as the default local checkpoint (Experiments 1–2). The JSONL builder pins Hugging Face revision `f883165a3026fde855dfd448e0cd16443ab257b6`, the same revision as the Deliverable 1 notebook.

**Model-size disclosure:** the assignment asks for a model with **fewer than 0.5B parameters**. I first chose `HuggingFaceTB/SmolLM2-360M-Instruct` (Experiment 3) to meet a strict reading of that threshold. That run stayed leak-safe and fully parseable but scored too low (slice MAD `0.4775` vs random `0.5256`). I therefore also trained `Qwen/Qwen2.5-0.5B-Instruct` and treat it as the default local checkpoint. Although Qwen markets this checkpoint as “0.5B,” Transformers counts `630,167,424` base parameters; this LoRA configuration adds `8,798,208`, for `638,965,632` total parameters (`8,798,208` trainable). Experiments 1–2 therefore do **not** satisfy a strict total-parameter reading. The bonus deliverable itself is optional, but the `<0.5B` threshold should not be described as optional when claiming strict compliance. The leakage-safe data and evaluation pipeline stayed the same across all three experiments.

## Layout

```text
src/leak_test.py           # pid=1 / QID154: persona has neither 70 nor 82
src/data/build_jsonl.py    # wave_split + CSVs; short tabular persona; never full_persona
src/baselines.py           # random, majority, copy-last (never fills missing with gold)
src/train.py               # QLoRA SFT; default Qwen2.5-0.5B-Instruct
src/evaluate.py            # baselines + optional adapter
tests/test_poc.py          # 7 unit tests
data/poc/*.jsonl           # reported-run JSONL; committed
runs/<run>/adapter         # final PEFT files; sibling bundle_manifest.json; committed
results/<run>/metrics.json # after evaluate; committed
```

This POC implements Deliverable 2's **no-copy** condition. Persona = non-overlap CSV fields only (demographics first, capped). One JSONL row = one `(pid, column)`. Person split, seed `20250319`, 70 / 15 / 15. Default slice: **MC only**, 500 train pids × ≤20 items, 100 val and 100 test pids (this report scores validation). The random baseline samples uniformly from each column's train-observed answer codes. Copy-last diagnostics live in `diag_*.jsonl`, never in the prompt. Leak check is on the **persona** span: the QID154 stem itself says “70 lawyers”.

Unlike the research-scale D2 pipeline, this POC constructs a safe question payload directly from the catalog's `QuestionText`, `Options`, and `Range`; it does not load and recursively strip `wave4_Q_wave4_A`. It also uses a plain prompt, 2,048-token sequences, and no retrieval. Experiment 1 used the script defaults (2 epochs, learning rate `2e-4`). Experiment 2 used 3 epochs and `1e-4`. Experiment 3 used 4 epochs and `1e-4` (from the Colab `bundle_manifest.json`). These are explicit POC shortcuts rather than implementations of D2's 7B summary-plus-retrieval recipe.

## Commands (from repo root)

Default local loop (Experiment 1). Experiment 2 and Experiment 3 commands are under each results block.

```text
python -m unittest discover -s tests
python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20
python -m src.leak_test data/poc/leak_fixture.jsonl data/poc/examples_train.jsonl data/poc/examples_val.jsonl
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc/adapter --out_dir results/poc
```

## POC experiment report

**Status:** three completed validation experiments. Experiment identity (dates, bundle, base model, hardware) is recorded per experiment below. Shared fields that do not change across experiments are listed once. The paper's published numbers were not copied into this section.

### Shared contract

- Input condition: `no-copy`
- Evaluation split: `validation`
- Training method family: QLoRA — 4-bit NF4 base model plus LoRA adapters (rank 16, alpha 32, dropout 0.05)
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

Experiments 1 and 2 share the local `data/poc` JSONL, so their baselines match. Experiment 3 rebuilt JSONL on Colab with the same caps and seeds; its baselines are therefore similar but not identical. All scores use train-only empirical ranges.

#### Experiment 1 — Qwen2.5-0.5B-Instruct, QLoRA, 2 epochs, lr `2e-4` (local RTX 3060)

- Status: completed
- Evaluation date: `2026-09-23`
- Training bundle date: `2026-09-21`
- Bundle ID: `20260921-3e3eaf7`
- Command: `python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora`
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tune: QLoRA (4-bit NF4 base + LoRA adapters, rank 16, alpha 32, dropout 0.05)
- Epochs / learning rate: `2` / `2e-4`
- Training wall time: about `2.5` hours
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM
- Adapter: `runs/poc/adapter`
- Source: `results/poc/metrics.json` and `runs/poc/bundle_manifest.json`
- Uniform random: slice MAD `0.5357`; MC exact match `0.4685`; parse rate `100.00%`; people `100`
- Train majority: slice MAD `0.5797`; MC exact match `0.5245`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: slice MAD `0.8461`; MC exact match `0.7920`; coverage/parse rate `99.95%`; people `100`
- LoRA SFT: slice MAD `0.5274`; MC exact match `0.4670`; parse rate `100.00%`; people `100`
- LoRA MAD / copy-last MAD: `0.623`

> Integrity checks passed. This 2-epoch Qwen run did **not** beat random (`0.5274` vs `0.5357`) and trailed train-majority (`0.5797`). The loop produced fully parseable codes, but the small model and short tabular persona did not beat population priors.

#### Experiment 2 — Qwen2.5-0.5B-Instruct, QLoRA, 3 epochs, lr `1e-4` (local RTX 3060)

- Status: completed
- Evaluation date: `2026-09-23`
- Training bundle date: `2026-09-23`
- Bundle ID: `20260923-3e3eaf7`
- Command:

```text
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --out_dir runs/poc_e3_lr1e4 --epochs 3 --lr 1e-4 --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc_e3_lr1e4/adapter --out_dir results/poc_e3_lr1e4
```

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tune: QLoRA (4-bit NF4 base + LoRA adapters, rank 16, alpha 32, dropout 0.05)
- Epochs / learning rate: `3` / `1e-4`
- Training wall time: about `5` hours
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM
- Adapter: `runs/poc_e3_lr1e4/adapter`
- Source: `results/poc_e3_lr1e4/metrics.json` and `runs/poc_e3_lr1e4/bundle_manifest.json`
- Uniform random: slice MAD `0.5357`; MC exact match `0.4685`; parse rate `100.00%`; people `100`
- Train majority: slice MAD `0.5797`; MC exact match `0.5245`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: slice MAD `0.8461`; MC exact match `0.7920`; coverage/parse rate `99.95%`; people `100`
- LoRA SFT: slice MAD `0.5411`; MC exact match `0.4765`; parse rate `100.00%`; people `100`
- LoRA MAD / copy-last MAD: `0.640`

> Same validation JSONL as Experiment 1. This schedule **did** beat random (`0.5411` vs `0.5357`) by a thin margin and stayed below train-majority (`0.5797`). The gain is too small to treat as a research-scale success; D3's stronger follow-up bar is mean-random + 0.01.

#### Experiment 3 — SmolLM2-360M-Instruct, QLoRA, 4 epochs, lr `1e-4` (Colab T4)

- Status: completed
- Evaluation date: `2026-09-23`
- Training bundle date: `2026-09-22`
- Bundle ID: `20260922-3107dfc`
- Commands run in the Colab session from the repository root (T4 uses fp16 automatically). Epochs below follow `bundle_manifest.json` (`--epochs 4`). Colab shipped `torchao` 0.10.0, which PEFT rejects (it wants >0.16.0); this POC does not use `torchao`, so uninstall it before evaluate:

```text
pip install -r requirements.txt
pip uninstall -y torchao
python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20
python -m src.leak_test data/poc/leak_fixture.jsonl data/poc/examples_train.jsonl data/poc/examples_val.jsonl
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --model_name HuggingFaceTB/SmolLM2-360M-Instruct --out_dir /content/runs/smollm360m_e3 --epochs 4 --lr 1e-4 --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir /content/runs/smollm360m_e3/adapter --out_dir /content/results/smollm360m_e3
```

Train, leak checks, and evaluate for this experiment were all executed on Colab. Nothing in this experiment was trained or scored on the local machine. After Colab finished, I copied the outputs to Google Drive and downloaded them. The Drive copy placed PEFT files flat in `runs/smollm360m_e3/`; in this repository they sit under `runs/smollm360m_e3/adapter` with `bundle_manifest.json` beside that folder, matching Experiments 1–2. JSONL was rebuilt on Colab, so baselines are not identical to Experiments 1–2.

- Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Fine-tune: QLoRA (4-bit NF4 base + LoRA adapters, rank 16, alpha 32, dropout 0.05)
- Epochs / learning rate: `4` / `1e-4`
- Training wall time: about `3` hours
- Hardware: Colab T4
- Adapter: `runs/smollm360m_e3/adapter` (trained on Colab at `/content/runs/smollm360m_e3/adapter`)
- Source: `results/smollm360m_e3/metrics.json` and `runs/smollm360m_e3/bundle_manifest.json` (Colab → Drive → repo; JSONL rebuilt on Colab, so baselines are not identical to Experiment 1)
- Uniform random: slice MAD `0.5256`; MC exact match `0.4545`; parse rate `100.00%`; people `100`
- Train majority: slice MAD `0.5954`; MC exact match `0.5425`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: slice MAD `0.8565`; MC exact match `0.8045`; coverage/parse rate `99.85%`; people `100`
- LoRA SFT: slice MAD `0.4775`; MC exact match `0.4145`; parse rate `100.00%`; people `100`
- LoRA MAD / copy-last MAD: `0.558`

> This is the first model I chose for the `<0.5B` request. Integrity held and every output parsed, but SmolLM2-360M did **not** beat random (`0.4775` vs `0.5256`) and trailed train-majority (`0.5954`). That low result is why the default local checkpoint is now Qwen2.5-0.5B-Instruct.

### Interpretation

> All three experiments completed with green leakage checks and 100% parse rate. Ranked by LoRA slice MAD, Experiment 2 is highest (`0.5411`), then Experiment 1 (`0.5274`), then Experiment 3 (`0.4775`). The same order holds against each experiment's own random baseline: Experiment 2 `+0.0054`, Experiment 1 `−0.0083`, Experiment 3 `−0.0481`. Experiment 2 is the only schedule that beat random, but it still trailed train-majority and missed D3's stronger +0.01 follow-up margin. Experiment 3 rebuilt JSONL on Colab, so its absolute MAD is not a paired comparison with Experiments 1–2; even so, it is the weakest of the three against its own random and majority scores. That gap is consistent with the smaller SmolLM2-360M base, not with T4 versus the 3060. The POC loop works; none of these tiny-model slices is a candidate for research-scale promotion.

These are validation-slice results using train-only empirical ranges. They are not the official 17-task MAD score, do not include participant-bootstrap confidence intervals, and must not be compared directly with the paper's published **81.72%** human result. Copy-last is an empirical same-pair test–retest benchmark, not a mathematical upper bound.

> **External-use disclosure:** Simulated survey responses. Not observed behavior. U.S. online panel, not population-weighted. Short-term evidence only: wave 4 launched approximately two weeks after wave 3, with longer intervals for items originating in earlier waves.

Hardware note: Experiments 1–2 used an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM. Experiment 3 used a Colab T4. QLoRA training requires CUDA. On a T4, use fp16 because T4 does not support bf16. CPU is practical for JSONL generation, leakage checks, and baseline evaluation; adapter inference on CPU is supported but very slow.

If leak test is red, do not print a score table. If slice MAD > copy-last/ceiling, treat it as a leak alarm, not SOTA.

### Discussion

**Observation**: larger models appear to predict better on this slice: both Qwen2.5-0.5B-Instruct runs outscored SmolLM2-360M-Instruct, including when each experiment is compared with its own random baseline. That is why Qwen is now the default local checkpoint, with the size deviation disclosed above. More training epochs also appear to help within the Qwen pair: the 3-epoch / `1e-4` schedule (Experiment 2) beat the 2-epoch / `2e-4` schedule (Experiment 1).

**These are initial observations only**. The three experiments are not a controlled ablation — Experiment 2 also changed the learning rate, and Experiment 3 used a different base model, rebuilt JSONL, different hardware, and 4 epochs. Establishing either pattern with higher confidence would require more matched experiments and the stronger Deliverable 3 benchmarks (locked test, more seeds, participant-bootstrap intervals).

## Future work

With more time I would keep this leakage-safe slice fixed and test the Discussion observations under cleaner conditions. Next steps on the prototype:

1. Test larger instruction models such as 1.5B, 7B, or bigger if hardware allows, on the same JSONL, seeds, and QLoRA recipe.
2. Compare frozen-prompt / prompt-engineering baselines with QLoRA fine-tuning, and with the combination, before treating fine-tuning as the default.
3. Score the same slice with the official 17-task MAD script and participant-bootstrap intervals. A win over train-majority would count only if that paired interval stayed above zero.

## Deps (POC)

`transformers` `peft` `accelerate` `bitsandbytes` `datasets` `pandas` `torch` — see `requirements.txt`. If `HF_HOME` is unset, the builder uses `data_raw/.cache/huggingface` when that folder exists (Hugging Face cache only; anonymized wave files are under `raw_data/`). On Colab, uninstall the stock `torchao` as in the Experiment 3 commands.
