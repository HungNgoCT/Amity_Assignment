# Deliverable 3 — Evaluation strategy

## Questions

1. Which metrics for which question types
2. What we compare against (ceiling and trivial baselines)
3. Train / test protocol
4. How we prevent the dataset’s leakage trap
5. Explicit acceptance criteria

## Solution

Modeling recipe: `docs/02_model_plan.md`. Part 1 is the overall summary (same numbering as the questions). Part 2 is the detail.

**References**

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500.* [arXiv:2505.17479](https://arxiv.org/abs/2505.17479). Headline MAD 1 − |a−b| / range. Figure 2: mean test–retest accuracy **81.72%** over 17 tasks; GPT-4.1-mini twins **71.72%**; paper-stated ratio **87.67%**. Those three figures are as published; 71.72÷81.72 of the *rounded* percentages is not 87.67. We did not recompute them.
- Twin-2K-500 dataset card: `full_persona` already uses **wave-4** answers on repeated items; `wave_split` usage says to strip `Answers` from `wave4_Q_wave4_A`. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- Official MAD script / ranges: [Digital-Twin-Simulation](https://github.com/tianyipeng-lab/Digital-Twin-Simulation) `evaluation/mad_accuracy_evaluation.py`. Column types and pid=1 / `QID154` (70 vs 82): `notebooks/data_exploration.ipynb`.

### Part 1 — Overall summary

1. **Metrics by question type.** Do not headline one exact-match average (slider 0–100 ≠ yes/no). System headline = paper **MAD accuracy**, averaged at **task** level (17 tasks). Under that: single-label MC = accuracy (+ κ); multi-select type headline = **Jaccard**, also report per-option F1; matrix = MAD + ordinal MAE; slider = MAD + native MAE (not exact-match); unbounded anchoring = MAD **after train-only deciles**. `DB` excluded. Join types via catalog `csv_columns`.

2. **What we compare against.** **Ceiling:** human 2-week test–retest on the **same MAD** (Toubia et al. Figure 2: **81.72%**). **Trivial baselines:** uniform-in-range random; train majority; **copy last answer**. On the full scored set, copy-last equals the ceiling — two readings (§2.2). **Published twin baseline:** the dataset’s precomputed LLM CSVs. Beating the human ceiling **systematically** is a leak alarm, not a win. Same figure: twins **71.72%**; paper-stated ratio **87.67%**, not 100%. We did not recompute.

3. **Train / test protocol.** Split by **person**, not by row: 70 / 15 / 15 `pid`s, seed `20250319`. All wave-4 items for a person stay in one split. Score only non-null labels (between-subject arms). Fit anchoring deciles and majority baselines on **train pids only**. Tune on val; **touch test once**.

4. **Leakage trap.** Same question text on wave 4 is not a leak (the paper’s retest). **Answers** in the wrong field are. Trap 1: `full_persona` / unstripped `wave4_Q_wave4_A` already contain the wave-4 label (pid=1 / `QID154` stores **82**). Trap 2: `wave4_Q_wave1_3_A` and the 126 overlapping CSV columns are the **first** answer (**70**) — legal as copy-last **baseline** and as the human side of test–retest, illegal as model input. Unit test: eval prompt contains neither 70 nor 82. One item per sequence.

5. **Acceptance criteria (stated before scores).** Leak unit test must pass. POC (&lt;0.5B): MAD **> random** and mostly parseable. “Worth scaling”: MAD > majority **and** not just copy-last on items where humans *changed*. Production-shaped (plan): MAD on the order of **~80% of the human ceiling**, in the band of the paper’s **87.67%** ratio (Figure 2). Fail hard if MAD > ceiling while the leak test is red — do not report it. POC may be weak; skip a type → write “not verified.”

---

### Part 2 — Detail

#### 2.1 Metrics for which question types

Join columns to types via catalog `csv_columns`, never `startswith(QuestionID)`.

| Type (catalog) | Count (wave-4 scored cols, EDA) | Headline | Also report | Do not headline |
|---|---|---|---|---|
| MC, SAVR | **53** QIDs / 53 cols | Accuracy | Cohen’s κ vs chance | MAE on option codes unless the scale is ordinal |
| MC, SAHR | **15** QIDs / 15 cols | Accuracy | Cohen’s κ vs chance | MAE on option codes unless the scale is ordinal |
| MC, MAVR/MAHR | **0** of the 126 wave-4 cols (21 such QIDs exist in the full catalog; BDI / Wason) | **Jaccard** if they appear | Per-option F1 | Single-label accuracy |
| Matrix | 40 overlap cols in EDA | MAD; mean \|code₁ − code₄\| | Exact-match | Exact-match alone |
| Slider (`QID154`, `QID156`, `QID290_*`) | 12 overlap cols | MAD; MAE in 0–100 | Pearson *r*, ICC(A,1) | Exact-match (~single-digit % even when *r* is fine) |
| TE numeric / anchoring | 6 overlap TE cols | MAD after **deciles** | Native MAE | Raw unbounded MAE mixed with 0–1 MC |
| `DB` | 0 of the 126 wave-4 CSV columns (empty `csv_columns`; 14 DBs in the full catalog) | Exclude | — | “Case by case” scoring |

**System headline (comparable to Toubia et al. Figure 2).**

For each scored column, with range R = max − min from the official MAD table (mapped through `wave4_formatted_to_catalog_mapping.json`):

acc(i, c) = 1 − |ŷ(i,c) − y(i,c)| / R_c

only where y(i,c) is non-null. Binary items have R = 1, so this **is** exact-match. Unbounded anchoring (`Q164`/`Q166`/`Q168`/`Q170` in Qualtrics names; map to catalog QIDs): replace values by **decile 1–10** using thresholds fit on **train** wave 1–3 (paper used wave 2; we freeze the rule on train pids so test cannot move the bins).

Then we do **not** re-derive the 17-task reduction. Paper text (Figure 2): within a task, mean the item accuracies **for that respondent**, then average **across respondents** (person-equal). The official `compute_task_mad` table instead concatenates all `(person, item)` pairs in the task (item-weighted if some people have more non-null items). Headline numbers we report = **whatever that script writes**; if we skip the script, write “not verified.” We do not invent a third reduction.

Worked example (pid=1, `QID154`, range 0–100): human 70 vs 82 → MAD acc 1 − 12/100 = 0.88. Exact-match = 0. Those two numbers answer different questions; only MAD shares a scale with MC accuracy — that is why MAD is the headline, not because it looks better.

Reuse `evaluation/mad_accuracy_evaluation.py` rather than inventing R_c.

---

#### 2.2 What we compare against

| Comparator | Definition | Role |
|---|---|---|
| **Ceiling** | MAD(wave 1–3 hold-out answers, wave 4) on the **test pids**, same ranges/deciles as the model | Upper bound we expect to *approach*. Paper: **81.72%** across 17 tasks. On this full set it equals copy-last |
| Uniform random | Draw uniformly in [min_c, max_c] (paper’s random) | Floor. POC must beat this |
| Train majority | Argmax of train labels per column (mode; multi-select: independent bit majority) | “Ignore the persona” |
| **Copy last answer** | Predict wave 4 = wave 1–3 value of that column (`wave4_Q_wave1_3_A` / overlap CSV) | Same MAD as the ceiling on the **full** set; two readings (self-consistency vs trivial model). Trap-2 policy, named. The LBM must add value where humans **changed**. Baseline, **not** a feature — putting 70 in the prompt turns eval into “remember last time” |
| HF precomputed LLM CSVs | Official GPT / Gemini / light-FT dumps | Prompting-class twin without API spend |
| Frozen prompt / SFT / DPO | Our models, leakage-safe input only | The thing under test |

Ratio next to every model number: MAD_model / MAD_ceiling (Toubia et al. Figure 2 states that ratio as **87.67%**; we did not recompute, and we do not treat it as 71.72÷81.72 of the rounded percentages).

If MAD_model > MAD_ceiling on many items, run §2.4 before writing “SOTA.”

**A result row** (each of: random, majority, copy-last, HF LLM dump, our prompt, our SFT): task-mean MAD, 95% CI, n people, mean items/person, MAD / ceiling, parse rate. Optional MC accuracy / slider MAE / Jaccard as **separate columns, never averaged into MAD**. Notebook exact-match histograms stay as diagnostics, not the D3 headline.

---

#### 2.3 Train / test protocol

- Shuffle unique `pid`s, seed **`20250319`**, **70% / 15% / 15%** (~1,440 / 309 / 309).
- No person in two splits. All wave-4 columns for a pid travel together. A random **row** split would put the same person’s other wave-4 items in train.
- Example grain: `(pid, column)` with non-null wave-4 label.
- Decile cutpoints, majority baselines, and MLP (if used) fit on **train** only.
- Val: early-stop, hyper, “do we scale 0.5B?” Test: **one** locked eval after that decision.
- Optional subgroup slices (not gates): age/education cells from D1 ACS gaps — report if n allows, do not chase.

---

#### 2.4 Leakage prevention (the assignment’s trap)

Wave 4 **repeats** waves 1–3 heuristics/pricing. Same question text is not, by itself, a leak. **Answers** in the wrong field are.

**Trap 1 — wave-4 label in the persona or the prompt.**

- `full_persona` (pid=1 / `QID154` only, not swept): `persona_json` has `Values: ['82']`; `persona_text` has `Answer: 82` on this row (wave-4 label). `persona_summary` does **not** contain 82. Other pids/items still to check before using the summary as a persona.
- `wave4_Q_wave4_A` used as `test_questions` **without deleting `Answers`**: the HF usage snippet is the warning. Same field is prompt *and* `ground_truth`.

**Trap 2 — first-round answer to the item being predicted, used as a feature.**

- `wave4_Q_wave1_3_A` and the **126 / 760** overlapping CSV columns hold **70** for that example.
- That is the human test–retest pair and the copy-last baseline. It is **not** allowed in the model input X.

**Eval-time unit test (must be green before any table).**

```text
pid=1, column=QID154
  Assert 82 not in persona, not in prompt
  Assert 70 not in persona, not in prompt
  Assert stripped question JSON has no Answers / Values / Selected*
```

Also: never concatenate 88 wave-4 items in one sequence (later items would see earlier gold/pred wave-4 answers). Eval script loads `wave_split` only, not `full_persona`.

---

#### 2.5 Explicit acceptance criteria

Stated **before** test scores. POC (D6) is allowed to be weak; it is not allowed to skip these gates.

| Gate | Pass | Fail |
|---|---|---|
| **Leak** | §2.4 unit test green on the eval pipeline | Any 70/82 in prompts; `full_persona` used as condition |
| **POC (&lt;0.5B slice)** | Task-mean MAD **> random** on val; parse rate ≥ 80% of non-null items | Unparseable dump; MAD ≤ random |
| **Worth 7B / more data** | Val MAD > train-majority **and** MAD **> copy-last** on the subset of items with wave 1–3 ≠ wave 4 | Only wins by copying 70 |
| **Production-shaped (plan, not this take-home)** | Test MAD / ceiling ≳ **0.80**, in the band of the paper’s **87.67%** ratio (Figure 2); no systematic above-ceiling items | Above-ceiling + red leak test |
| **Hard stop** | No reported number unless leak test is green | Any published score while the leak test is red |

Type diagnostics are **not** gates for the POC (a 0.5B model may collapse sliders). They **are** required in the report so we do not hide MC behind slider exact-match. If we skip the official MAD script or a type, write “not verified.” Per-column *n* always: missingness is experimental assignment, not dropout.

**Not** an OKR: “beat the human ceiling.” Humans are the noise floor over two weeks.
