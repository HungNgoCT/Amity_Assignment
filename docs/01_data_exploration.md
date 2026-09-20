# Deliverable 1 — Data exploration

## Questions

1. Dataset structure and the two Hugging Face subsets
2. Question types and how they would be scored
3. Sample distributions / representativeness
4. Human test–retest reliability
5. Biases and limitations

## Solution

<span style="color:#2563eb;font-size:1.5em;font-weight:600">Part 1</span> is the overall summary (same numbering as the questions). <span style="color:#2563eb;font-size:1.5em;font-weight:600">Part 2</span> is only the notebook pointer (plots and the full ACS table).

**References**

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500.* [arXiv:2505.17479](https://arxiv.org/abs/2505.17479). N=2,058; ~500 questions; wave 4 = ~2-week retest of heuristics/pricing. Figure 2: mean test–retest **accuracy** over 17 tasks = **81.72%** (accuracy = 1 − |a−b| / range; binary = exact-match).
- Twin-2K-500 dataset card (two loadable configs; `full_persona` already uses wave-4 answers on repeats). [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- ACS 2023 1-year marginals: Census S0101, B01001, PEP, S1501, S1901 (bins recoded onto survey options in the notebook).
- https://github.com/tianyipeng-lab/Digital-Twin-Simulation

<h2 style="color:#2563eb;font-size:1.5em;font-weight:600">Part 1 — Overall summary</h2>

| Key finding | In one line |
|---|---|
| Leakage trap | `full_persona` text/json already store the wave-4 answer on repeats (pid=1 / `QID154` = **82**). Modeling input = `wave_split`. |
| Test–retest ceiling | Paper Figure 2: **81.72%** mean accuracy over 17 tasks (1 − \|a−b\| / range). Empirical 2-week self-agreement, not a theoretical max. |
| Representativeness | Described as representative; ACS 2023: gender/region ≈; education, age, income gaps **> 5 pp**. |
| Scoring per type | No pooled exact-match. MC accuracy; multi-select **Jaccard**; slider MAD + MAE. System headline = paper MAD (D3). |
| Biases | Online panel; social desirability; ceiling is **two weeks**; wave-4 NaNs are mostly assignment, not attrition. |

1. **Dataset structure / two HF subsets.**

   Twin-2K-500 = four-wave US survey. On Hugging Face, `load_dataset` has **two subsets**: `full_persona` and `wave_split`. Catalog/CSVs, LLM dumps, and raw Qualtrics are separate files.

   `full_persona` = table (**2058 rows × 4 columns**).
   - 2058 = US adults in the final sample (one row = one person).
   - Row = `pid`. What we predict is that person's item responses, not a population share. Scoring unit = `(pid, item)` (D2).
   - 4 columns = `pid`; `persona_text` (~126k–134k characters); `persona_summary` (~12k–18k); `persona_json`.
   - `persona_text` / `persona_json` = **item-level Q–A** (same survey; json is structured, text is prose). They **merge waves 1–3 and wave 4** into one profile. On questions asked twice, wave 4 **replaces** the earlier answer (dataset card). The card also says json follows the same structure as `persona_text`. On this row, `persona_text` looks like the prose rendering of `persona_json`.
   - `persona_summary` = a narrative, not a shortened Q–A dump (card: “concise summary of key characteristics”). Empirically (pid=1, same template on the first few pids): demographics, computed scale scores + percentiles (Big Five, CRT, games, literacy, …), plus a few qualitative self-descriptions. It is **not** a column on `wave_split`.
   - Check **one pid / one item** (pid=1 / `QID154`; CSV wave 1–3 = **70**, wave 4 = **82**; not swept across pids):
     - `persona_json` has `Values: ['82']` — wave-4 label, leak.
     - `persona_text` has `Answer: 82` on that slider — same leak (our read of this row, matching the json).
     - `persona_summary` does **not** contain 82 (or 70) on this item. Deliverable 2 still checks other wave-4 items / pids before using it as a persona.

   Assignment task = waves 1–3 → held-out wave 4. Do not use `full_persona.persona_text` / `persona_json` as that persona: on repeated items they already store the wave-4 answer. `wave_split` is the modeling interface (already split in time). Do not reconstruct the split from `full_persona`.

   `wave_split` = table (**2058 rows × 5 columns**).
   - Same 2058 people, split in time (`pid`s match; cast `full_persona.pid` from string to int before joining).
   - 5 columns: `pid`; `wave1_3_persona_text`, `wave1_3_persona_json` (persona from waves 1–3); `wave4_Q_wave4_A` (wave-4 questions + answers = labels); `wave4_Q_wave1_3_A` (same questions with the **first** answer = test–retest / copy-last). No `persona_summary`.

   Tabular files (not `load_dataset` configs): `question_catalog.json` (256 QuestionIDs); `wave1_3_response.csv` **(2058, 761)** = pid + 760 items; `wave4_response.csv` **(2058, 127)** = pid + 126 items; plus label CSVs, LLM simulation dumps, raw Qualtrics. Same question text on wave 4 is the paper’s retest, not a leak; putting **70** or **82** in the prompt is the leak (Deliverable 3).

2. **Question types and scoring.**

   One survey, three counts (not three datasets):

   | Count | What it counts |
   |---|---|
   | **256** | Catalog QuestionIDs (Qualtrics blocks) |
   | **~500** | Questions a person actually answers (paper / dataset name) |
   | **760** | CSV data columns after expanding matrix rows and multi-select options |

   Types (by QuestionID): MC **175** · Matrix **36** · TE **28** · DB **14** · Slider **3**.

   How we score:
   - MC single-select (154) → accuracy
   - MC multi-select (21) → **Jaccard** as the type headline; per-option F1 as diagnostic (not single-label accuracy). Gold is a **set** of options, so ordinary accuracy is the wrong unit; both give partial credit (miss 1 of 5 ≠ miss all 5). The **system** headline is still paper MAD (D3), not Jaccard.
   - Matrix → MAD / ordinal MAE
   - Slider → MAD + native MAE (exact-match is the wrong headline). MAD is comparable to the paper’s 17-task ceiling; native MAE (points on 0–100) is the miss size in the slider’s own units. We keep both; they answer different questions.
   - TE numeric / anchoring → native MAE (EDA; same idea as slider). Unbounded anchoring in the **paper** is MAD after **deciles** (they fit bins on wave 2). **Train-only** cuts are a D3 rule so the test set cannot move the bins — not a dataset field.
   - DB (instructional) → exclude

   Join a CSV column to its type via catalog `csv_columns`. Model headline = paper MAD (D3), not one pooled exact-match.

3. **Distributions / representativeness.**

   Assignment / HF card: “representative sample of N = 2,058 US adults.” That is their description. Below is the ACS 2023 check (our EDA), not a verdict that the sample is or is not “representative.”

   Gender and census region ≈ ACS 2023.

   Gaps **> 5 pp** (sample − ACS):
   - college / some postgrad **+13.9**
   - high school only **−12.7**
   - income $100k+ **−11.0**
   - less than high school **−9.4**
   - age 65+ **−9.1**
   - age 50–64 **+8.2**
   - income $30k–$50k **+5.7** (ACS does not split at $30k; this bin is interpolated from the $25–35k ACS row — same as the notebook)

   The measured marginals are uneven on education, age, and income. Sampling frame = online panel, not a census draw.

4. **Human test–retest (the important number).**

   Wave 4 = same heuristics/pricing items, ~**two weeks** later.
   **126 / 760** wave 1–3 CSV columns are asked again — that overlap is the **item set** on which we measure person-vs-self agreement, **and** the leak surface if those first-round values enter the model input. The agreement number on that set is the **human test–retest ceiling**: empirical 2-week consistency, not a theoretical maximum. A model should not be expected to match a later answer more consistently than the person matches themselves (noise / leak aside).

   People vs themselves (our EDA, type-appropriate — not MAD):
   - MC exact-match **~0.74**
   - Matrix **~0.60**
   - Slider **~0.12** (MAE ~15.5 / 100)
   - TE **~0.25**

   Do not average those four into one “accuracy.”
   Paper (Figure 2): mean **test–retest accuracy** over 17 tasks = **81.72%**, where accuracy = 1 − |a−b| / range (binary = exact-match; repo name: MAD accuracy). We did not recompute that number here; the four rates above are our EDA. A twin should *approach* the paper figure, not beat it.

5. **Biases / limitations.**

   - Panel self-selection (habitual survey-takers; almost no < high school).
   - Social-desirability on self-report (risk, honesty, spend).
   - The test–retest ceiling is **two weeks**, not months.
   - Snapshot in time (HF card).
   - Wave-4 missingness is mostly **between-subject assignment**, not dropout (same 2,058 pids in both CSVs). So missingness in wave 4 should not automatically be read as longitudinal attrition.

---

<h2 style="color:#2563eb;font-size:1.5em;font-weight:600">Part 2</h2>

Plots and full ACS table: `notebooks/data_exploration.ipynb`.
