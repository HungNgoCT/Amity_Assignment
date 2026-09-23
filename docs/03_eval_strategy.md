# Deliverable 3 — Evaluation strategy

## Requirements addressed

1. Metrics for each question type
2. Comparison baselines and the human benchmark
3. Train, validation, and test protocol
4. Leakage prevention
5. Explicit acceptance criteria

## Proposed evaluation protocol

Modeling recipe: `docs/02_model_plan.md`. The summary below follows the same numbering as the requirements above, followed by the detailed protocol.

### Evaluation summary

1. **Metrics by question type.** Do not headline one exact-match average (slider 0–100 ≠ yes/no). The system headline is the paper's **mean absolute deviation (MAD) accuracy**, aggregated over 17 tasks. Under that: single-label Multiple Choice (MC) = accuracy plus Cohen's kappa (κ; Cohen, 1960); multi-select headline = Jaccard, with per-option F1 as a diagnostic; matrix = MAD plus ordinal mean absolute error (MAE); slider = MAD plus native-scale MAE and intraclass correlation as a diagnostic (McGraw & Wong, 1996), not exact match; unbounded anchoring = MAD after train-only deciles. Display/Instruction (DB) screens are excluded. Join types through the catalog's `csv_columns`.

2. **Comparators and baselines.** The model is evaluated against a project-specific human test–retest benchmark, trivial baselines, and staged model ablations. The human benchmark is recomputed on the same held-out test participants and items used to evaluate the model. Toubia et al. Figure 2 reports **81.72% across the full study sample**; this is an external published reference, not necessarily the value for this project's test split. Wave 4 was launched approximately two weeks after wave 3, although items originating in waves 1 or 2 have longer intervals, so this is best described as a short-term empirical reliability benchmark rather than a mathematical upper bound. The **trivial baselines** are seeded uniform-in-range random, train majority, and **copy last answer**. The staged ablations are question-only, summary-only, summary plus retrieval under frozen prompting, and a tabular neural baseline. On any fixed scored set, copy-last equals the human test–retest calculation—two readings of the same earlier-versus-later pairs (§2.2). The **published twin reference** is the dataset's precomputed large language model (LLM) CSVs, but only after their persona and prompt provenance records pass the leakage checks. The paper reports twins at **71.72%** and a stated model-to-human ratio of **87.67%** on its evaluation population. These published figures are references; I did not recompute them.

3. **Train, validation, and test protocol.** Shuffle participant identifiers using the fixed random seed `20250319` for reproducibility, then split them by **person**, not by row: 70% train, 15% validation, and 15% test. All wave-4 items for one person remain in the same split. Score only non-null labels from assigned between-subject conditions. Fit anchoring deciles and majority baselines using train participants only, tune on validation, and evaluate the locked test set once. Report the no-copy and full-history input conditions separately.

4. **Leakage trap and historical-answer policy.** Same question text on wave 4 is not a leak (the paper’s retest). A **wave-4 answer** in `full_persona` or an unstripped `wave4_Q_wave4_A` payload is leakage (pid=1 / `QID154` stores **82**). By contrast, the earlier answer **70** in `wave4_Q_wave1_3_A` and the overlapping waves 1–3 CSV column is temporally valid history. It is excluded from the primary no-copy condition to prevent a copy-last shortcut, but explicitly allowed in the separately reported full-history condition. Unit tests enforce the declared condition. One item per sequence.

5. **Acceptance criteria.** Leakage and input-condition checks are hard gates: no result is valid unless they pass. Deliverable 6 is intentionally a simplified proof of concept (POC; fewer than 0.5B parameters): it demonstrates the end-to-end loop and reports slice MAD against random plus parse rate, but it does not implement every resampling and uncertainty procedure in this research-scale protocol. For a stronger follow-up prototype, the proposed engineering thresholds are at least **0.01 MAD accuracy** above mean random and at least **80%** parse rate. Scaling to the planned 7B model is justified if the lower bounds of the paired 95% validation confidence intervals for no-copy minus train-majority and no-copy minus question-only are both above zero. For full-history, the corresponding lower bound against copy-last must also be above zero on items where the participant changed their answer. The research-scale target is test MAD / test-split human benchmark ≥ **0.80**. These thresholds are design choices, not values required by the paper or dataset. Any unexpectedly above-benchmark result triggers a leakage and aggregation audit.

---

### Detailed protocol

#### 2.1 Metrics for which question types

Join columns to types via catalog `csv_columns`, never `startswith(QuestionID)`. In the Qualtrics selector codes, SAVR/SAHR denote single-answer vertical/horizontal layouts, while MAVR/MAHR denote multiple-answer vertical/horizontal layouts. Counts below come from the exploratory data analysis (EDA).

| Type (catalog) | Count (wave-4 scored cols, EDA) | Headline | Also report | Do not headline |
|---|---|---|---|---|
| MC, SAVR | **53** QuestionIDs / 53 cols | Accuracy | Cohen’s κ vs chance | MAE on option codes unless the scale is ordinal |
| MC, SAHR | **15** QuestionIDs / 15 cols | Accuracy | Cohen’s κ vs chance | MAE on option codes unless the scale is ordinal |
| MC, MAVR/MAHR | **0** of the 126 wave-4 cols (21 such QuestionIDs exist in the full catalog; Beck Depression Inventory / Wason) | **Jaccard** if they appear | Per-option F1 | Single-label accuracy |
| Matrix | 40 overlap cols in EDA | MAD; mean \|code₁ − code₄\| | Exact-match | Exact-match alone |
| Slider (`QID154`, `QID156`, `QID290_*`) | 12 overlap cols | MAD; MAE in 0–100 | Pearson *r*; intraclass correlation coefficient (ICC(A,1)) | Exact-match (~single-digit % even when *r* is fine) |
| Text Entry (TE) numeric / anchoring | 6 overlap TE cols | MAD after **deciles** | Native MAE | Raw unbounded MAE mixed with 0–1 MC |
| `DB` | 0 of the 126 wave-4 CSV columns (empty `csv_columns`; 14 DBs in the full catalog) | Exclude | — | “Case by case” scoring |

**System headline (comparable to Toubia et al. Figure 2).**

For each scored column, use R = max − min from `get_default_column_ranges()` in the pinned external official script. Use `get_default_qid_to_task()` for its task assignments and the external `evaluation/column_mapping.csv` when translating official column names:

acc(i, c) = 1 − |ŷ(i,c) − y(i,c)| / R_c

only where y(i,c) is non-null. Binary items have R = 1, so this **is** exact match. Unbounded anchoring (`Q164`/`Q166`/`Q168`/`Q170` in Qualtrics names; map to catalog QuestionIDs) is converted to deciles 1–10.

For model selection and the primary leakage-safe evaluation, fit the decile thresholds on train-pid waves 1–3 values and freeze them before validation or test. This differs from the current official script, which derives thresholds from the loaded waves 1–3 evaluation population. Therefore, I would also run a separately labeled **paper-replication** evaluation with the pinned official script. I would not present the train-only and paper-replication values as if they were identical.

The official script produces two distinct task outputs:

1. `compute_task_mad` pools valid `(person, item)` values within each task; use this as the per-task diagnostic table.
2. Its final task summary first averages item deviations within each task for a respondent, then averages across that respondent's available tasks, and finally averages across respondents. Use this person-equal summary as the **17-task headline**, matching the paper's written aggregation.

Pin the official repository commit, range table, task mapping, and decile implementation. Do not select whichever output is more favorable.

Worked example (pid=1, `QID154`, range 0–100): human 70 vs 82 → MAD acc 1 − 12/100 = 0.88. Exact-match = 0. Those two numbers answer different questions; only MAD shares a scale with MC accuracy — that is why MAD is the headline, not because it looks better.

The official script and mapping CSV are **not stored in this repository**. Fetch them from the pinned Digital-Twin-Simulation revision linked in References, verify their hashes, and either vendor those exact files into the evaluation environment or record their immutable external paths. Reuse their ranges and task mapping rather than inventing R_c.

**Invalid outputs and denominator.** Freeze the scored `(pid, column)` rows before loading predictions. Missing ground-truth rows are excluded, but an unparseable, out-of-range, or illegal model output remains in the denominator and receives headline MAD accuracy 0. For type metrics, an invalid categorical or multi-select output receives accuracy or Jaccard 0; report native-scale numerical MAE only on parseable values, alongside parse rate and the all-row headline. Do not pass invalid predictions to the official script as `NaN`, because its `notna()` mask would silently drop them.

**Uncertainty and model comparisons.** Compute 95% confidence intervals (CIs) with a participant-cluster bootstrap (Efron & Tibshirani, 1993): sample `pid`s with replacement from the split being evaluated, retain all their scored items, and recompute the complete 17-task aggregation for each of 2,000 resamples using a fixed seed. Use validation participants for model-selection CIs and test participants only for the final locked report. For model-versus-baseline differences, use the same bootstrap samples for both systems and report the paired difference CI. The official script's analytic intervals may be retained only as paper-replication diagnostics.

---

#### 2.2 What we compare against

| Comparator | Definition | Role |
|---|---|---|
| **Human test–retest benchmark** | MAD(waves 1–3 earlier answers, wave 4) on the **same test pids and items as the model**, using the same ranges/deciles | Project-specific short-term reliability reference and denominator for model/benchmark ratios. Report its computed test-split value; cite the paper's full-sample **81.72%** separately. On a fixed repeated-item set it is numerically the same calculation as copy-last |
| Uniform random | For the paper-compatible baseline, draw an integer uniformly from each column's legal inclusive range | Floor. Report the mean over 100 fixed random seeds so the acceptance decision does not depend on one lucky draw |
| Train majority | Argmax of train labels per column (mode; multi-select: independent bit majority) | “Ignore the persona” |
| **Copy last answer** | Predict wave 4 = wave 1–3 value of that column (`wave4_Q_wave1_3_A` / overlap CSV) | On the same fixed test participants and repeated items, this is numerically the same MAD calculation as the human benchmark—two readings (self-consistency vs trivial model). It is a diagnostic reference, not a same-input baseline, for no-copy because that model cannot see the earlier value. It is the direct trivial baseline for full-history, which must add value where humans **changed** |
| Hugging Face precomputed LLM CSVs | Official GPT / Gemini / lightly fine-tuned dumps | Use only after verifying and recording which persona fields, prompt, model, and dataset revision generated each dump. If wave-4-derived information was available, label the result non-comparable and exclude it from acceptance decisions |
| Question-only frozen prompt | Stripped question and output schema, with no participant persona | Tests how much performance comes from question wording and population priors rather than individual information |
| Summary-only frozen prompt | Safe deterministic persona summary plus stripped question, without retrieved chunks | Tests the value of compressed persona information |
| Summary + retrieval frozen prompt | Safe summary and retrieved persona chunks, without parameter updates | Tests retrieval value before fine-tuning |
| Tabular multilayer perceptron (MLP) | Safe non-overlap columns, fit on train participants only | Neural baseline for known columns; cannot naturally answer newly worded questions |
| Supervised fine-tuning (SFT) / Direct Preference Optimization (DPO) | Proposed systems evaluated under a declared no-copy or full-history condition | The trainable systems under test |

Ratio next to every project model number: model MAD divided by the human-benchmark MAD recomputed on the same test split. Toubia et al. Figure 2 separately states a full-sample ratio of **87.67%**; I did not recompute it and do not replace it with a division of the rounded 71.72% and 81.72% figures.

If model MAD exceeds the human benchmark on many items, run §2.4 and verify the aggregation before interpreting the result. Exceeding an empirical benchmark is possible, but it requires evidence that the gain is not caused by leakage, denominator changes, or incompatible scoring.

**A result row** for every eligible comparator reports the **17-task person-equal MAD accuracy**, participant-bootstrap 95% CI, number of people, mean items per person, MAD / test-split human benchmark, and parse rate. The per-task pooled MAD table is reported separately as a diagnostic. Optional MC accuracy, slider MAE, and Jaccard remain separate columns and are never averaged into MAD. Notebook exact-match histograms remain diagnostics, not the Deliverable 3 headline.

---

#### 2.3 Train / test protocol

- Shuffle unique `pid`s using the fixed random seed **`20250319`** so the split is reproducible, then assign **70% / 15% / 15%** to train, validation, and test. With `int(n * ratio)` and the remainder going to test, 2,058 participants become **1,440 / 308 / 310**.
- No person in two splits. All wave-4 columns for a pid travel together. A random **row** split would put the same person’s other wave-4 items in train.
- Example grain: `(pid, column)` with non-null wave-4 label.
- Freeze `input_condition=no_copy|full_history` before building prompts and never pool results across the two conditions.
- Decile cutpoints, majority baselines, and any tabular MLP fit on **train** only.
- Validation: early stopping, hyperparameter selection, and the decision to scale from the sub-0.5B POC to the planned 7B model. Test: **one** locked evaluation after those decisions.
- Optional subgroup slices, not acceptance gates: age and education groups motivated by Deliverable 1's American Community Survey (ACS) gaps. Report only when sample sizes are sufficient and always include the group count.

---

#### 2.4 Leakage prevention and input-condition enforcement

Wave 4 **repeats** waves 1–3 heuristics/pricing. Same question text is not, by itself, a leak. **Answers** in the wrong field are.

**The leakage trap — wave-4 label in the persona or the prompt.**

- `full_persona` (pid=1 / `QID154` only, not swept): `persona_json` has `Values: ['82']`; `persona_text` has `Answer: 82` on this row (wave-4 label). `persona_summary` does **not** contain 82. Other pids/items still to check before using the summary as a persona.
- `wave4_Q_wave4_A` used as `test_questions` **without deleting `Answers`**: the Hugging Face usage snippet is the warning. The same field would otherwise provide both the prompt and `ground_truth`.

**Historical-answer policy — two valid but different tasks.**

- `wave4_Q_wave1_3_A` and the **126 / 760** overlapping CSV columns hold **70** for that example.
- This is an earlier waves 1–3 answer, so it is not future-label leakage. In **no-copy**, it is withheld from model input and used only for copy-last and human test–retest comparisons. In **full-history**, it is allowed as a source-tagged historical feature.
- The two conditions answer different questions and must never be pooled: no-copy tests inference from the rest of the persona, while full-history tests whether the model improves on simply repeating the prior answer.

**Dataset-wide structural sweep (must be green before any score).**

- For every generated `(pid, column)` row, recursively assert that the stripped question contains no answer-bearing keys such as `Answers`, `Values`, `SelectedText`, or `SelectedByPosition`.
- Track provenance for every prompt field and assert that no field originates from `wave4_Q_wave4_A` answer values or `wave4_response.csv`.
- Under no-copy, assert that the persona and retrieval index have an empty `csv_columns` intersection with the 126 repeated target columns.
- Under full-history, assert that every repeated-item answer comes only from the source-tagged waves 1–3 table and that the model artifact records `input_condition=full_history`.
- Assert disjoint train, validation, and test pid sets; fit summaries, retrievers, deciles, and baselines without wave-4 test labels.
- Freeze the expected test `(pid, column)` index before prediction so missing or invalid model outputs cannot reduce the denominator.

**Regression test for the known example.**

```text
pid=1, column=QID154
  Both conditions:
    Assert wave-4 answer field 82 is absent from persona and prompt
    Assert stripped question JSON has no Answers / Values / Selected*
  No-copy:
    Assert parsed historical answer fields do not contain QID154 = 70
  Full-history:
    Assert the only QID154 answer exposed is source-tagged waves 1–3 value 70
```

Assertions operate on parsed fields rather than raw substring searches because the QID154 question stem independently contains the number 70. Also, never concatenate all wave-4 items in one sequence (later items would see earlier gold/predicted wave-4 answers). The evaluation script loads `wave_split` only, not `full_persona`.

---

#### 2.5 Explicit acceptance criteria

The POC in Deliverable 6 is allowed to be weak and uses a deliberately simplified evaluation, but it is not allowed to skip the data-integrity gates. The full research protocol below adds repeated random baselines, bootstrap uncertainty, and stronger scaling criteria. Scaling decisions use validation results; the locked test set is evaluated only after those decisions are final.

The **0.01 MAD margin**, **80% parse rate**, and **0.80 benchmark ratio** are design choices for a stronger follow-up evaluation, not published Twin-2K-500 standards or requirements imposed on the minimal take-home POC. The 0.01 margin prevents a negligible numerical win over random from being called success; the 80% parse-rate floor requires the prototype to produce usable outputs most of the time while allowing a small model to fail; and the 0.80 ratio is a preliminary research target relative to the human benchmark computed on the same test split. A different organization could choose stricter or looser thresholds based on risk, compute, and application requirements, but it should freeze them before the final test evaluation.

| Gate | Pass | Fail |
|---|---|---|
| **Leak / condition integrity** | §2.4 tests green; no wave-4 label in either condition; earlier answers match the declared condition | Any wave-4 answer in input, any unstripped question payload, or earlier same-item answer silently entering a no-copy run |
| **Take-home POC (&lt;0.5B slice)** | Leakage gate passes; report slice MAD against random and report parse rate | Leakage failure, no baseline comparison, or parse rate omitted |
| **Stronger follow-up prototype** | Validation 17-task person-equal MAD accuracy ≥ mean random MAD + 0.01 (one percentage point); parse rate ≥ 80%; report the paired CI even if it crosses zero | Unparseable output, smaller margin, or uncertainty omitted |
| **Worth 7B / more data** | Lower bounds of the no-copy minus train-majority and no-copy minus question-only paired 95% validation CIs are above zero; the full-history minus copy-last lower bound is also above zero on the changed-answer subset | No-copy gain over either non-personal baseline is indistinguishable from zero, or full-history only repeats the prior answer |
| **Research-scale target** | Test MAD / test-split human benchmark ≥ **0.80** under the fixed scoring contract; this is an engineering threshold, not a value derived by the paper | Threshold missed, scoring contract changed after test access, or result cannot be reproduced |
| **Hard stop** | No model score is reported as valid unless leakage, condition-integrity, and fixed-denominator tests pass | Any headline score reported while one of those tests is red |

Type diagnostics are **not** gates for the POC (a 0.5B model may collapse sliders). They **are** required in the report so we do not hide MC behind slider exact-match. If we skip the official MAD script or a type, write “not verified.” Per-column *n* always: missingness is experimental assignment, not dropout.

**Not** an objective: “beat the human ceiling.” The published 81.72% value is a full-sample, short-term human reliability reference, not necessarily this project's test-split benchmark, a mathematical maximum, or a production target.

## References

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A Dataset for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions*. [arXiv:2505.17479](https://arxiv.org/abs/2505.17479). Figure 2 reports human test–retest accuracy of **81.72%**, GPT-4.1-mini twins at **71.72%**, and a stated ratio of **87.67%**. These published figures were not recomputed here.
- LLM-Digital-Twin. *Twin-2K-500 dataset card and data files*. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- Toubia et al. *Digital-Twin-Simulation reference implementation*, revision [`f1eed51`](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/tree/f1eed510c9a4fb47aaa9bb46e178aaa2d9224623): [`evaluation/mad_accuracy_evaluation.py`](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/blob/f1eed510c9a4fb47aaa9bb46e178aaa2d9224623/evaluation/mad_accuracy_evaluation.py) and [`evaluation/column_mapping.csv`](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/blob/f1eed510c9a4fb47aaa9bb46e178aaa2d9224623/evaluation/column_mapping.csv).
- Column-type counts and the pid 1 / `QID154` example (70 in waves 1–3; 82 in wave 4): [notebooks/data_exploration.ipynb](../notebooks/data_exploration.ipynb).
- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37–46. [https://doi.org/10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104).
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. [https://doi.org/10.1007/978-1-4899-4541-9](https://doi.org/10.1007/978-1-4899-4541-9).
- McGraw, K. O., & Wong, S. P. (1996). *Forming Inferences About Some Intraclass Correlation Coefficients*. Psychological Methods, 1(1), 30–46. [https://doi.org/10.1037/1082-989X.1.1.30](https://doi.org/10.1037/1082-989X.1.1.30).
