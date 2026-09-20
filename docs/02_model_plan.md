# Deliverable 2 — Concrete plan to build the behavior model

This is a **build plan**, not a trained production model. We adapt a public instruction-tuned LLM to Twin-2K-500; we do not pretrain a transformer from scratch.

**How to read this note.** Part 1 lists the major steps and the core ideas (including the trade-off in step 9). Part 2 unpacks each step — pipeline, hyperparameters, and risks. EDA: `notebooks/data_exploration.ipynb`. Metrics: Deliverable 3.

**References**

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500.* [arXiv:2505.17479](https://arxiv.org/abs/2505.17479). Figure 2: mean test–retest accuracy **81.72%** over 17 tasks; GPT-4.1-mini twins **71.72%**; paper-stated ratio **87.67%**. Those three figures are as published; 71.72÷81.72 of the *rounded* percentages is not 87.67. We did not recompute them.
- Twin-2K-500 dataset card (`wave_split` vs `full_persona`). [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).

---

## Part 1 — Summary of steps

1. **Frame the task.** Condition on a person’s waves 1–3 persona → predict their **wave 4** answers → score against wave 4 and against the human 2-week test–retest ceiling.

2. **Build a leakage-safe dataset.** Use Hugging Face `wave_split`, never `full_persona`. Strip `Answers` from wave-4 questions before they enter the prompt. Do not put first-round answers to the same items (`wave4_Q_wave1_3_A`, or the 126 overlapping CSV columns) into the model input. Split by **person** (70 / 15 / 15). Sanity check: pid=1, `QID154` — the prompt must contain neither **70** (round 1) nor **82** (wave 4). **Trap 1** = wave-4 label in the persona (`full_persona` / unstripped `Answers` → 82). **Trap 2** = wave 1–3 answer to the *same* item used as a feature (70).

3. **Try methods cheap → expensive.** (1) Frozen prompting + a short persona summary, as a baseline (or the dataset’s precomputed LLM CSVs). (2) Retrieve only the persona *blocks* relevant to the question being asked — full persona text is too long. (3) LoRA fine-tune a **&lt; 0.5B** instruct model (`Qwen2.5-0.5B-Instruct`) — this is the actual “build.” (4) DPO only after SFT already emits parseable codes.

4. **Fit the context window.** Never dump 130k characters of persona. Use a structured summary + top-k retrieved blocks. One training example = **one** wave-4 item, not 88 items concatenated.

5. **Write down a runnable SFT recipe.** LoRA, loss on answer tokens only, ~2 epochs, Colab/RunPod. Deliverable 6 (bonus), if we do it, is this recipe on a data slice — it may be weak; it may not be leaky or unrunnable.

6. **Evaluate on the same yardsticks we will use for the model** (detail in D3). Headline: paper-style MAD accuracy. Diagnostics by question type. Compare to random, majority, **copy last answer**, and the human ceiling. Beating the human ceiling is a leak alarm, not a win.

7. **Name the risks.** Trap 1, trap 2, tiny-model underfit, between-subject missingness, context truncation. Mitigations live in Part 2.

8. **If we had more time.** Same recipe on 7B–8B, then DPO against “copy last answer,” then reproduce Toubia et al. Figure 2 in official MAD units.

9. **Trade-off analysis.** Order is **prompt → retrieve → SFT → DPO** because each step only pays for what the previous one cannot do: prompting is a cheap, leak-safe baseline (Toubia et al. already saw similar scores across prompt tricks, so we do not bet on prompt engineering); retrieval exists only to fit a 130k-character persona into a small context window; LoRA-SFT is the actual “build” (bind persona to a short answer code without pretraining from scratch); DPO comes last, to penalize “copy last round,” and only after SFT already emits parseable codes. **Not RLHF:** reward design across 126 heterogeneous columns is out of take-home scope, and RLHF without a working SFT parse loop is theatre. **Not `full_persona`:** it already substitutes wave-4 labels (pid=1 / `QID154` stores **82**, not 70). Detail: **§2.3–2.5**.

---

## Part 2 — Implementation details

### 2.1 Problem framing

**Unit of prediction.** One `(pid, wave-4 item)` pair. An item is a CSV column (e.g. `QID154`, `QID287_1`), not a Qualtrics QuestionID. Wave 4 has **126** scored CSV columns. Join via catalog `csv_columns` ∩ `wave4_response.csv` (not `startswith(QuestionID)`): those 126 columns map to **84** QuestionIDs (MC 68 · Matrix 7 · TE 6 · Slider 3). Instructional `DB` QIDs have empty `csv_columns`, so they are **not** in this 84. We counted this in EDA; it is not a paper table.

**Allowed input.** `wave_split.wave1_3_persona_json` / `wave1_3_persona_text`, and/or `wave1_3_response.csv` columns **not** in the wave-4 set. `full_persona.persona_summary`: 82 absent on **this one pid/item** (pid=1 / `QID154`); not yet swept across pids — verify other hold-out items before using it.

**Forbidden as model condition.**

| Field | Why |
|---|---|
| `full_persona.persona_text` / `persona_json` | Dataset card: repeated questions carry **wave-4** answers; json follows the same structure as text. Empirically **this row** (pid=1 / `QID154`): json `Values: ['82']`; text has `Answer: 82`. |
| `full_persona.persona_summary` | Checked **one pid / one item** only (pid=1 / `QID154`): **82 is not in the summary**. Not swept across pids. Verify other wave-4 items before using it as a persona. |
| `wave4_Q_wave4_A` **with `Answers` left in** | Same JSON is prompt **and** label. HF usage says to strip `Answers`. Raw use leaks 82. |
| `wave4_Q_wave1_3_A` as a feature for the **same** item | First-round answer (70). Legal as copy-last-answer **baseline** and test–retest; illegal as persona. |
| 126 CSV columns in both `wave1_3_response.csv` and `wave4_response.csv` | Tabular form of the same first-round hold-out. |

**Output.** Canonical codes as in `wave4_response.csv`, not free prose.

**“Good.”** Systematically beating 2-week human test–retest is treated as leak/overfit. Toubia et al. (2025, Figure 2): GPT-4.1-mini twins **71.72%**; test–retest ceiling **81.72%**; paper-stated ratio **87.67%**, not 100%. We did not recompute. Target: beat trivial baselines, approach the ceiling, keep the leakage unit test green.

Toubia et al. (2025): persona = non-hold-out waves 1–3; evaluation vs retest blocks are separate; headline metric is MAD 1 − |a−b| / range (deciles on unbounded anchoring).

---

### 2.2 Data pipeline

```text
HF wave_split + catalog + CSVs
        │
        ├─ persona: wave1_3_persona_json
        │     drop any question whose csv_columns intersect wave-4 columns
        ├─ ask: wave4_Q_wave4_A  → deep-copy, delete Answers / Values / Selected*
        ├─ label: wave4_response.csv[pid, col]
        └─ diagnostics (never in X):
              wave4_Q_wave1_3_A, overlap columns in wave1_3_response.csv
```

Join types via catalog field `csv_columns`, never `col.startswith(QuestionID)` (`QID290_5` would hit `QID2` / `QID29`).

**Person split.** Shuffle `pid`s, seed `20250319`, **70 / 15 / 15** (~1,440 / 309 / 309). All wave-4 items for a person stay in one split.

**Between-subject items.** Train/eval only on `(pid, col)` with a non-null wave-4 label. Do not impute the unseen arm.

**Canonicalization.** JSON `Answers` → CSV codes (1-based option index; sliders 0–100; multi-select `{QID}_{option}`). SFT target = that short code.

**Leakage unit test (before any train/eval claim).**

```text
pid=1, column=QID154
  CSV w1–3 = 70, CSV w4 = 82
  Assert 82 not in persona / prompt
  Assert 70 not in persona / prompt
  Assert stripped test question has no Answers
```

70 present → trap 2. 82 present → trap 1 / unstripped `wave4_Q_wave4_A`.

**Artifacts.** `data/poc/splits/pids_{train,val,test}.json`, `data/poc/overlap_columns.txt`, `src/data/build_jsonl.py` → JSONL `{pid, col, prompt, target}` plus `diag_*.jsonl` (copy-last only) and `leak_fixture.jsonl`.

---

### 2.3 Modeling approaches — order and why

Toubia et al. found a dozen prompt / format / light-FT variants landed in a **similar** band. Do not bet the take-home on a clever prompt. Keep a prompting baseline so a 0.5B SFT model can be compared honestly.

**Frozen prompting + compression (baseline).** Hosted instruct model, or skip API cost and use the dataset’s precomputed LLM simulation CSVs. Prompt = compressed persona + one stripped wave-4 question + “code/number only.” First because it costs no GPU and forces the leak-safe prompt to exist. Not last: the assignment asks how we would *build*, including fine-tuning.

**Retrieval over persona blocks.** Split JSON by `BlockName`. Embed once per pid (`gte-small` or BM25). For a target item, retrieve top-k blocks vs. `(BlockName + QuestionText)`. Full persona is 126k–134k characters; 0.5B cannot hold it. Retrieval is a **pipeline feature**, not a product: it packs prompts for both frozen and SFT models.

**LoRA-SFT (the build, and the bonus POC).** Base: **`Qwen/Qwen2.5-0.5B-Instruct`** (fallback `SmolLM2-360M-Instruct`). Not from-scratch pretraining: the data are small and structured; we need to bind a persona to a short code. Not full FT: it overfits 2k people and wrecks instruction following. Loss = causal LM on **answer tokens only**. Target e.g. `82` or `3`.

A tabular MLP on the 634 non-overlap columns is a cheap neural **baseline**. It cannot take a new natural-language question, so it is not the LBM.

**DPO after SFT works.** Chosen = gold CSV code; rejected = random legal code, or the wave 1–3 answer when it differs from wave 4 (teaches “don’t just copy last time”). Full RLHF/GRPO is out of scope. DPO before a working parse loop is theatre.

**Sequence.** Leak test + JSONL → prompting baseline → majority / copy-last / MLP → LoRA-SFT on a slice (500 pids × ≤20 items) → scale SFT if healthy → DPO only if SFT loses to copy-last on items humans actually changed.

---

### 2.4 Long personae and the context window

| Stage | Budget | Packing |
|---|---|---|
| Frozen API model | Prefer ≤ 8k tokens in | Structured summary + 3–5 retrieved blocks |
| 0.5B SFT (`max_length=2048`) | ~1.5k prompt + 8 answer | Always summary + retrieval; never raw `wave1_3_persona_text` |
| 70B / 128k (if we had it) | Still retrieve | Length is not a license to dump hold-out answers |

**Summary (deterministic).** 400–800 words from non-overlap tabular columns: demographics, political items, Big Five facet means, Need for Cognition, selected cognitive scores. Do not dump 44 raw BFI rows unless retrieval picked that block.

**One item per example.** Concatenating 88 wave-4 questions would let later items condition on earlier gold/predicted wave-4 answers inside the same sequence.

---

### 2.5 Training details (SFT POC)

| Knob | Value | Rationale |
|---|---|---|
| Base | `Qwen/Qwen2.5-0.5B-Instruct` | Instruct, &lt; 0.5B, Apache-2.0 |
| Method | LoRA on `q,k,v,o,gate,up,down_proj` | Standard decoder LoRA |
| Rank / alpha / dropout | 16 / 32 / 0.05 | Rank 8 underfits short codes; 64 wastes T4 VRAM |
| Quantization | NF4 QLoRA if GPU &lt; 16 GB; bf16 LoRA if 24 GB | Free Colab = T4 16 GB |
| Objective | Token CE, **labels = -100 on prompt** | Score answers, not persona parroting |
| Max length | 2048 | Fits T4; forces §2.4 compression |
| Batch / accum | 4 × 8 (effective 32) | Stabilize 0.5B |
| LR / schedule | 2e-4, cosine, 3% warmup | LoRA wants higher LR than full FT |
| Epochs | 2 (early-stop on val MAD) | 5+ memorizes train pids |
| First POC size | 500 pids × ≤ 20 items (~10k rows) | Loop must run, not SOTA |
| Seed | 42 | Repro |
| Decode | greedy, stop at newline | Illegal parse → miss |

```text
You are simulating one survey respondent.
Persona:
{compressed_persona}

Question:
{stripped_question_text}
Options: {options_or_range}

Reply with only the answer code or number.
```

Target: CSV value as string (`82`). Stack: `transformers` + `peft` + `SFTTrainer`, `bitsandbytes` if QLoRA. Script for D6: `src/train.py`. Compute: assignment’s Colab/RunPod credits; &lt; 2 h on a T4 for the POC slice.

---

### 2.6 Evaluation (preview — full spec in D3)

Do **not** headline a single exact-match average across sliders and yes/no items.

1. **Headline (paper-comparable):** MAD 1 − |ŷ−y| / range; anchoring deciles from **train** percentiles only. Ceiling = same MAD, wave 1–3 hold-out vs wave 4 (published task-average **81.72%**).
2. **Diagnostics:** MC accuracy + κ; multi-select Jaccard; slider native MAE **and** MAD; numeric TE MAE.

Baselines: uniform random, train majority, **copy wave 1–3**, prompting, tabular MLP.

On the **full** scored set, ceiling and copy-last are the **same number** (ŷ = the wave 1–3 answer), read two ways: human self-consistency vs a trivial model. Random and majority sit below that number. Do not beat the ceiling. Where the LBM must beat copy-last is the **subset of items that changed** (D3).

---

### 2.7 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Label leak (`full_persona` / unstripped `Answers`) | Never load `full_persona` for prompts; unit test pid=1 `QID154` |
| Trap 2 in X | Drop list of 126 names; copy-last is a **named baseline**, not a feature |
| 0.5B << GPT-4.1-mini on heuristics | POC bar = runnable + beat random; production = same recipe on 7B–8B LoRA |
| Truncation drops demographics | Structured summary **first** in the prompt; truncate retrieved blocks, not the summary |
| Between-subject NaNs treated as dropout | Score only non-null labels; report n per column |
| Concatenated wave-4 items | One item per example |
| Invented MAD ranges | Reuse official `mad_accuracy_evaluation.py` ranges via `wave4_formatted_to_catalog_mapping.json` |
| Val MAD &gt; human ceiling | Fail the leak test first; do not report as SOTA |
| Social-desirability / panel bias | Cannot be trained away; scope in D4/D5 |

---

### 2.8 If we had more time

LoRA-SFT 7B/8B on the full pid split; DPO with rejected = last-round answer when it differs; per-block adapters (pricing vs heuristics); prove `persona_summary` leak-free before using it; reproduce Toubia et al. Figure 2 on **our** split with the official MAD script.

Deliverable 6, if implemented, is only the slice SFT in §2.3 / §2.5.
