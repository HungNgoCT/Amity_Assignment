# Deliverable 6 — POC (bonus)

This is a **runnable slice**, not a reproduction of Toubia et al. Figure 2. The slice is multiple-choice (MC) only, so the headline is **exact match**. Slice mean absolute deviation (MAD) is reported beside it and is not the headline: 19 of the 68 MC columns have a train-observed code range above 1, so MAD gives partial credit for a nearby code. Deliverable 3 treats that as the wrong headline unless the options are ordinal. The minimal gate is: leakage checks pass, parse rate is reported, and exact match is compared with random and train-majority. A score above random is not required to complete the bonus. The run may be weak, but it must not be leaky. This evaluation does not implement Deliverable 3's full protocol, such as 100 random-baseline seeds, participant-bootstrap confidence intervals, or the stronger follow-up thresholds.

**Not verified:** official 17-task script / paper **81.72%** (Deliverable 1). Exact match below is the slice headline. Slice MAD uses **train-only** empirical ranges and is not comparable with the paper's score.

**References:** D2 §2 data pipeline and §5 training details (`docs/02_model_plan.md`); D3 §3 train / test protocol and §5 acceptance criteria (`docs/03_eval_strategy.md`). The JSONL builder pins Hugging Face revision `f883165a3026fde855dfd448e0cd16443ab257b6`, the same revision as the Deliverable 1 notebook. Model choice is in the disclosure below.

**Model-size disclosure:** the assignment asks for a model with **fewer than 0.5B parameters**. I first chose `HuggingFaceTB/SmolLM2-360M-Instruct` (Experiment 3). That run stayed leak-safe and fully parseable but scored too low on the slice headline (exact match `0.4145` vs random `0.4545`). I then trained `Qwen/Qwen2.5-0.5B-Instruct` and treat it as the default local checkpoint. Its [model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) reports **0.49B** parameters (`494,032,768`) with tied word embeddings, so the base model is under 0.5B. `PreTrainedModel.num_parameters()` reports `630,167,424` because it counts that tied embedding matrix a second time (`494,032,768 + 136,134,656`). This LoRA configuration adds `8,798,208` trainable parameters. Unique base weights plus the adapter are `502,830,976`, which exceeds 0.5B only if the adapter is added to the base count. The leakage-safe data and evaluation pipeline stayed the same across all three experiments.

## Layout

```text
src/leak_test.py           # pid=1 / QID154: persona has neither 70 nor 82
src/data/build_jsonl.py    # wave_split + CSVs; unlabeled column codes; never full_persona
src/baselines.py           # random, majority, copy-last (never fills missing with gold)
src/train.py               # QLoRA SFT; default Qwen2.5-0.5B-Instruct
src/evaluate.py            # baselines + optional adapter
tests/test_poc.py          # 7 unit tests
data/poc/*.jsonl           # reported-run JSONL; committed
runs/<run>/adapter         # final PEFT files; sibling bundle_manifest.json; committed
results/<run>/metrics.json # after evaluate; committed
```

This POC implements Deliverable 2's **no-copy** condition. Persona = non-overlap CSV fields only, demographics first, then a hard cap of 40 columns. In the committed `data/poc/persona_columns.txt`, that cap is `QID11`–`QID24` and then `QID25_1`–`QID25_26`, so 26 of the 40 lines are one QuestionID. Those fields are unlabeled Qualtrics codes, for example `QID12: 1`, with no question text and no option labels. A model of this size cannot tell that a code is sex, age, or income.

The question side has a second legend gap. MC options are printed as text, for example `["more", "less"]` or the six “favor program A/B” sentences on `QID157`, while the supervised target is the numeric Qualtrics code (`"2"`, `"1"`). On all 10,000 training rows the target string is absent from that option list, and the prompt never says `1 = more`. The model has to emit an integer the prompt does not define, from a persona it cannot read. That is why these runs sit near random on exact match and trail train-majority. The three reported runs used this prompt. The JSONL was not rebuilt afterward, and the adapters were not retrained.

One JSONL row = one `(pid, column)`. Person split, seed `20250319`, 70 / 15 / 15. Default slice: **MC only**, 500 train pids × ≤20 items, 100 validation and 100 test pids. This report scores validation only. `data/poc/examples_test.jsonl` (100 people, 2,000 rows) was built with the same split and was not used to choose a schedule or to report a score. The random baseline samples uniformly from each column's train-observed answer codes. Copy-last diagnostics live in `diag_*.jsonl`, never in the prompt. Leak check is on the **persona** span: the QID154 stem itself says “70 lawyers”.

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
- Validation participants / examples: `100` / `2,000` (the scored split)
- Test participants / examples built and not scored: `100` / `2,000` (`data/poc/examples_test.jsonl`)
- Slice: `MC only`, at most `20` items per participant
- Participant split seed: `20250319`
- Headline: MC exact match. Slice MAD is secondary (train-only code range; partial credit on 19 columns)

### Integrity checks

- Unit tests: `7/7 passed`
- Known-fixture leak test: `PASS`
- Validation-prompt structural leak scan: `PASS`
- Train/validation participant overlap: `0`

If any integrity check fails, stop here and do not report the model score as valid.

### Results

Experiments 1 and 2 share the local `data/poc` JSONL, so their baselines match. Experiment 3 rebuilt JSONL on Colab with the same caps and seeds; its baselines are therefore similar but not identical. Exact match does not use a range. Slice MAD uses train-only empirical ranges. Copy-last exact match of **79.2%** on Experiments 1–2 is not Deliverable 1's **73.5%**: that figure is all 68 repeated MC columns and all 2,058 people.

#### Experiment 1 — Qwen2.5-0.5B-Instruct, QLoRA, 2 epochs, lr `2e-4` (local RTX 3060)

- Status: completed
- Evaluation date: `2026-09-23`
- Training bundle date: `2026-09-21`
- Bundle ID: `20260921-3e3eaf7`
- Command: `python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora`
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tune: QLoRA (4-bit NF4 base + LoRA adapters, rank 16, alpha 32, dropout 0.05)
- Epochs / learning rate: `2` / `2e-4`. The command above does not pass `--lr`, so `2e-4` is the script default. `runs/poc/bundle_manifest.json` does not record the learning rate.
- Training wall time: about `2.5` hours
- Hardware: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM
- Adapter: `runs/poc/adapter`
- Source: `results/poc/metrics.json` and `runs/poc/bundle_manifest.json`
- LoRA SFT: exact match `0.4670` (headline); slice MAD `0.5274`; parse rate `100.00%`; people `100`
- Uniform random: exact match `0.4685`; slice MAD `0.5357`; parse rate `100.00%`; people `100`
- Train majority: exact match `0.5245`; slice MAD `0.5797`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: exact match `0.7920`; slice MAD `0.8461`; diagnostic coverage `99.95%`; people `100`
- LoRA MAD / copy-last MAD (secondary): `0.623`

> Integrity checks passed. Headline exact match `0.4670` is level with random `0.4685` and below train-majority `0.5245`. Secondary MAD `0.5274` is also below random MAD `0.5357`. Every output parsed. The prompt gaps above — unlabeled persona codes, and numeric targets with text options — leave the model little readable individual information.

#### Experiment 2 — Qwen2.5-0.5B-Instruct, QLoRA, 3 epochs, lr `1e-4` (local RTX 3060)

The directory `runs/poc_e3_lr1e4` means 3 epochs and learning rate `1e-4`. It is Experiment 2, not Experiment 3.

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
- Reproducibility: that manifest records `git_dirty: true` and short hash `3e3eaf7`, which is not the current tree. Rescoring the saved adapter with the current evaluator should match the metrics file. Retraining from today's tree will not reproduce that adapter bit for bit.
- LoRA SFT: exact match `0.4765` (headline); slice MAD `0.5411`; parse rate `100.00%`; people `100`
- Uniform random: exact match `0.4685`; slice MAD `0.5357`; parse rate `100.00%`; people `100`
- Train majority: exact match `0.5245`; slice MAD `0.5797`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: exact match `0.7920`; slice MAD `0.8461`; diagnostic coverage `99.95%`; people `100`
- LoRA MAD / copy-last MAD (secondary): `0.640`

> Same validation JSONL as Experiment 1. Exact match `0.4765` is `0.008` above this split's random exact match and still below train-majority `0.5245`. Secondary MAD is `0.0054` above random MAD. Both gaps are from one seed, on the validation split used to choose this schedule, and the test JSONL was not scored. I do not treat that as beating random. Deliverable 3's stronger follow-up bar (mean random + 0.01, with a paired interval) is not met.

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
- LoRA SFT: exact match `0.4145` (headline); slice MAD `0.4775`; parse rate `100.00%`; people `100`
- Uniform random: exact match `0.4545`; slice MAD `0.5256`; parse rate `100.00%`; people `100`
- Train majority: exact match `0.5425`; slice MAD `0.5954`; parse rate `100.00%`; people `100`
- Copy-last / same-pair human benchmark: exact match `0.8045`; slice MAD `0.8565`; diagnostic coverage `99.85%`; people `100`
- LoRA MAD / copy-last MAD (secondary): `0.558`

> This is the first model I chose for the `<0.5B` request. Integrity held and every output parsed. Exact match `0.4145` is below random `0.4545` and below train-majority `0.5425`. Secondary MAD is also below random. That is why the default local checkpoint is now Qwen2.5-0.5B-Instruct. The Qwen runs use the same prompt gaps and still trail majority on exact match.

### Interpretation

> All three experiments completed with green leakage checks and 100% parse rate. Ranked by exact match, Experiment 2 is `0.4765`, Experiment 1 is `0.4670`, and Experiment 3 is `0.4145`. Against each run's own random exact match that is `+0.008`, `−0.0015`, and `−0.040`. All three remain below their own train-majority exact match (`0.5245`, `0.5245`, and `0.5425`). Experiment 3 rebuilt JSONL on Colab, so its absolute score is not a paired comparison with Experiments 1–2; against its own random and majority scores it is still the weakest. The POC loop works. None of these slices is a candidate for research-scale promotion, and the shared prompt gaps above are a reason the scores sit near the population prior.

These are validation-slice results. The test JSONL was not scored. They are not the official 17-task score, do not include participant-bootstrap confidence intervals, and must not be compared with the paper's **81.72%** (Deliverable 1). Copy-last on this slice is an empirical same-pair retest on the capped MC sample, not that paper figure and not a mathematical upper bound.

> **External-use disclosure:** Simulated survey responses. Not observed behavior. U.S. online panel, not population-weighted. Short-term evidence only: wave 4 launched approximately two weeks after wave 3, with longer intervals for items originating in earlier waves.

Hardware note: Experiments 1–2 used an NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM. Experiment 3 used a Colab T4. QLoRA training requires CUDA. On a T4, use fp16 because T4 does not support bf16. CPU is practical for JSONL generation, leakage checks, and baseline evaluation; adapter inference on CPU is supported but very slow.

If leak test is red, do not print a score table. If slice MAD > copy-last/ceiling, treat it as a leak alarm, not SOTA.

### Discussion

**Observation**: on exact match, both Qwen2.5-0.5B-Instruct runs outscored SmolLM2-360M-Instruct, including against each run's own random baseline. That is why Qwen is the default local checkpoint; the parameter count is disclosed above. Within the Qwen pair, the 3-epoch / `1e-4` schedule (Experiment 2) is higher than the 2-epoch / `2e-4` schedule (Experiment 1). Both still trail train-majority, and both received the same unlabeled persona codes and the same text-option / numeric-target prompt, so the gap is not evidence that either model interpreted demographics.

**These are initial observations only**. The three experiments are not a controlled ablation. Experiment 2 also changed the learning rate, and Experiment 3 used a different base model, rebuilt JSONL, different hardware, and 4 epochs. The validation split was used both to pick the schedule and to report the score. A later claim would need the option legend fixed, then the locked test split, more seeds, and participant-bootstrap intervals from Deliverable 3.

## Future work

With more time I would keep this leakage-safe slice fixed and test the Discussion observations under cleaner conditions. Next steps on the prototype:

1. Fix the prompt before scaling: print `1 = <option text>` for each MC choice, and put question text on the persona fields instead of bare `QID` codes. Rebuild the JSONL, retrain one Qwen run on the same caps and seed, and score exact match on validation. Open `examples_test.jsonl` only after that recipe is locked.
2. Test larger instruction models such as 1.5B, 7B, or bigger if hardware allows, on that corrected JSONL, the same seeds, and the same QLoRA recipe.
3. Compare frozen-prompt / prompt-engineering baselines with QLoRA fine-tuning, and with the combination, before treating fine-tuning as the default.
4. Score the same slice with the official 17-task MAD script and participant-bootstrap intervals. A win over train-majority would count only if that paired interval stayed above zero.

## Deps (POC)

`transformers` `peft` `accelerate` `bitsandbytes` `datasets` `pandas` `torch` — see `requirements.txt`. If `HF_HOME` is unset, the builder uses `data_raw/.cache/huggingface` when that folder exists (Hugging Face cache only; anonymized wave files are under `raw_data/`). On Colab, uninstall the stock `torchao` as in the Experiment 3 commands.
