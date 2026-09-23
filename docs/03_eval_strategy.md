# Deliverable 3 — Evaluation strategy

## Requirements addressed

1. Metrics for each question type
2. Comparison baselines and the human benchmark
3. Train, validation, and test protocol
4. Leakage prevention
5. Explicit acceptance criteria

Modeling recipe: `docs/02_model_plan.md`. The summary answers those five points; the numbered sections under Detailed protocol are the scoring contract.

### Evaluation summary

1. **Metrics.** The system headline is the paper's **mean absolute deviation (MAD) accuracy** over 17 tasks, not one exact-match average: a 0–100 slider and a yes/no item are not the same scale. Under that: single-label Multiple Choice (MC) = accuracy plus Cohen's κ (Cohen, 1960); multi-select = Jaccard, with per-option F1 as a diagnostic; matrix = MAD plus ordinal MAE; slider = MAD plus native-scale MAE, not exact match; unbounded anchoring = MAD after train-only deciles. Exclude Display/Instruction (DB) screens. Join types through the catalog's `csv_columns`.

2. **Comparators.** I would score every system on the same test people and items.

   - **Human test–retest** is the project benchmark: MAD between each person's earlier waves 1–3 answer and their wave-4 answer. The paper's **81.72%** is that idea on the full study sample, not on my 15% test split, and it is not a mathematical maximum.
   - **Copy-last** predicts wave 4 = the earlier answer. On a fixed repeated-item set this is the same MAD calculation as human test–retest, read as a trivial model.
   - **Other trivial baselines:** uniform random, and the most common train answer (majority).
   - **Ablations:** question-only, summary-only, summary plus retrieval without fine-tuning, a small tabular neural net, and persona-shuffling. If shuffling people does not drop the score, the model is not using individual information.

   The paper also reports GPT twins at **71.72%** and a twin-to-human ratio of **87.67%**. Those are citations. I did not recompute them on this split, so they are not a row in my result table.

3. **Split.** Person split, seed `20250319`, **70% / 15% / 15%**. Score only assigned non-null cells. Fit deciles and majority on train, tune on validation, lock test once. Report no-copy and full-history as two separate tables; never pool them.

4. **Leakage.** Repeating the same question text on wave 4 is not leakage. Putting the **wave-4 answer** into the model input is leakage. Example for pid=1 / `QID154`: the person first answered **70**, then **82** on wave 4. `full_persona` already stores **82**, so that field cannot be used as the persona. `wave4_Q_wave4_A` is the held-out question plus that same **82**; if those `Answers` are not stripped, the prompt contains the target and that is leakage. The earlier **70** is history, not a future label: withhold it in no-copy, allow it in full-history. Each training or inference sequence contains only one held-out wave-4 item, so a later item cannot see an earlier gold or predicted wave-4 answer.

5. **Acceptance.** Leakage and input-condition checks are hard gates. Deliverable 6 reports slice MAD versus random and parse rate. Stronger follow-up: at least **0.01 MAD** (can change) above mean random, so a tiny win over chance does not count, and **80%** parse rate (can change), so most outputs are usable. Scale to 7B if validation CI lower bounds beat train-majority and question-only (and copy-last on changed-answer items for full-history). Research-scale target: test MAD / test-split human ≥ **0.80** (can change). These are my choices, not paper values; freeze them until the next cycle. An above-benchmark result triggers a leakage audit.

---

### Detailed protocol

#### 1. Metrics for which question types

Join columns to types via `csv_columns`, never `startswith(QuestionID)`. SAVR/SAHR are single-choice layouts; MAVR/MAHR are multi-select. Counts are from the exploratory data analysis (EDA).

| Type (catalog) | Count (wave-4 scored cols, EDA) | Headline | Also report | Wrong headline |
|---|---|---|---|---|
| MC, SAVR/SAHR | **68** QuestionIDs / 68 cols (53 SAVR + 15 SAHR) | Accuracy | Cohen’s κ vs chance | MAE on option codes unless the scale is ordinal |
| MC, MAVR/MAHR | **0** of the 126 wave-4 cols (21 such QuestionIDs exist in the full catalog; Beck Depression Inventory / Wason) | **Jaccard** if they appear | Per-option F1 | Single-label accuracy |
| Matrix | 40 overlap cols in EDA | MAD; ordinal MAE (\|predicted Likert code − wave-4 Likert code\|) | Exact-match | Exact-match alone |
| Slider (`QID154`, `QID156`, `QID290_*`) | 12 overlap cols | MAD; MAE in 0–100 | Pearson *r*; ICC(A,1) (McGraw & Wong, 1996) | Exact-match |
| Text Entry (TE) numeric / anchoring | 6 overlap TE cols | MAD after **deciles** | Native MAE | Raw unbounded MAE mixed with 0–1 MC |
| `DB` | 0 of the 126 wave-4 columns (14 DBs in the catalog) | Exclude | — | “Case by case” scoring |

**System headline (comparable to Toubia et al. Figure 2).**

For each scored column, MAD uses a response range R_c = max − min. Take those ranges, the 17-task grouping, and official column names from the pinned Digital-Twin-Simulation files in References. The headline is that repository's **person-equal 17-task** summary; the per-task pooled table is a diagnostic only.

acc(i, c) = 1 − |ŷ(i,c) − y(i,c)| / R_c

only where y(i,c) is non-null. Binary items have R = 1, so this equals exact match. Unbounded anchoring is converted to deciles 1–10. For model selection, fit those deciles on train-pid waves 1–3 values and freeze them. The official script uses the loaded waves 1–3 evaluation population instead, so I would also run a separately labeled **paper-replication** pass and not treat the two numbers as identical.

Worked example (pid=1, `QID154`, range 0–100): human 70 vs 82 → MAD acc 1 − 12/100 = 0.88. Exact-match = 0. Only MAD shares a scale with MC accuracy.

**Invalid outputs.** Freeze the scored `(pid, column)` rows before loading predictions. Missing ground truth is excluded, but an unparseable or illegal output stays in the denominator and scores headline MAD 0. Passing invalid predictions to the official script as `NaN` would drop them from the denominator.

**Uncertainty.** Participant-cluster bootstrap 95% CIs (Efron & Tibshirani, 1993): 2,000 resamples of `pid`s, fixed seed. Use validation for model-selection CIs and test only for the locked report. Compare systems on paired differences from the same resamples.

---

#### 2. What we compare against

| Comparator | Definition | Role |
|---|---|---|
| **Human test–retest** | MAD(earlier waves 1–3 answers, wave 4) on the same test pids and items, same ranges/deciles | Project benchmark. Same calculation as copy-last. Cite the paper's full-sample **81.72%** separately |
| **Copy-last** | Wave 4 = earlier value of that column | Diagnostic for no-copy. Direct trivial baseline for full-history, which must add value where humans **changed** |
| Uniform random | Integer drawn uniformly from each column's legal range | Floor. Mean over 100 fixed seeds |
| Train majority | Train-set mode per column | “Ignore the persona” |
| Hugging Face precomputed LLM CSVs | Official GPT / Gemini dumps | Use only after leakage provenance checks; not a row unless those checks pass |
| Question-only / summary-only / summary + retrieval | Frozen prompts with increasing persona information | Ablations before fine-tuning |
| Tabular multilayer perceptron (MLP) | Safe non-overlap columns, train pids only | Neural baseline for known columns |
| Persona-shuffling | Pair person A's persona with person B's wave-4 items | If the score does not fall, the model is not using individual information |

Report model MAD / same-split human MAD next to every project number. Type metrics stay in separate columns and are never averaged into MAD. If model MAD exceeds the human benchmark on many items, run §4 before interpreting the result.

---

#### 3. Train / test protocol

- Shuffle unique `pid`s with seed **`20250319`**, then assign **70% / 15% / 15%** to train, validation, and test. Counts use whole people only (`int`): train = `int(2058 × 0.70)` = **1,440**, validation = `int(2058 × 0.15)` = **308**, and the remaining **310** go to test.
- No person in two splits. Grain: `(pid, column)` with a non-null wave-4 label.
- Freeze `input_condition=no_copy|full_history` before building prompts; never pool the two conditions.
- Deciles, majority, and any tabular MLP fit on **train** only. Validation for selection; test **once** after those decisions.
- Optional ACS-motivated age and education slices are diagnostics, not gates.

---

#### 4. Leakage prevention and input-condition enforcement

Repeating the same question text on wave 4 is not leakage. Putting the **wave-4 answer** into the persona or prompt is leakage. Deliverable 1 documents the spot check: pid=1 / `QID154` stores **82** in `full_persona` and **70** as the earlier answer.

- Strip `Answers` / `Values` / `Selected*` from every question payload. Do not use `full_persona`.
- **No-copy:** persona and retrieval have an empty `csv_columns` intersection with the 126 repeated target columns. The earlier answer is used only for copy-last and the human benchmark.
- **Full-history:** the earlier answer may appear only as a source-tagged waves 1–3 feature. The two conditions are never pooled.
- Disjoint train / validation / test pids. Freeze the test `(pid, column)` index so invalid outputs cannot shrink the denominator.

The block below is the **unit-test checklist** for that same Deliverable 1 example, not a score table. Every generated prompt for pid=1 / `QID154` must pass these asserts before any MAD is reported.

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

Assert on parsed fields, not raw substrings: the QID154 question stem itself contains the number 70. Never concatenate all wave-4 items in one sequence.

---

#### 5. Explicit acceptance criteria

The Deliverable 6 POC may be weak, but it may not skip leakage or input-condition gates. The numbers below are my engineering choices (can change). Once chosen, they stay frozen until the next evaluation cycle; they are not Twin-2K-500 standards.

| Gate | Pass | Fail |
|---|---|---|
| **Leakage / condition integrity** | §4 tests green; earlier answers match the declared condition | Wave-4 answer in input, unstripped question, earlier answer silently entering no-copy, or a headline reported while these tests are red |
| **Take-home POC (&lt;0.5B slice)** | Leakage gate passes; report slice MAD against random and parse rate | Leakage failure, no baseline comparison, or parse rate omitted |
| **Stronger follow-up prototype** | Validation 17-task person-equal MAD ≥ mean random + **0.01**; parse rate ≥ **80%**; report the paired CI | Smaller margin, unparseable output, or uncertainty omitted |
| **Worth 7B / more data** | Validation CI lower bounds: no-copy minus train-majority and no-copy minus question-only both > 0; full-history minus copy-last > 0 on changed-answer items | Gain indistinguishable from zero, or full-history only repeats the prior answer |
| **Research-scale target** | Test MAD / test-split human benchmark ≥ **0.80** under a fixed scoring contract | Threshold missed, or scoring changed after test access |

POC type diagnostics are required in the report, not gates. “Beat the human ceiling” is not an objective.

## References

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A Dataset for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions*. [arXiv:2505.17479](https://arxiv.org/abs/2505.17479). Figure 2 reports human test–retest accuracy of **81.72%**, GPT-4.1-mini twins at **71.72%**, and a stated ratio of **87.67%**. These published figures were not recomputed here.
- LLM-Digital-Twin. *Twin-2K-500 dataset card and data files*. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- Toubia et al. *Digital-Twin-Simulation reference implementation*, revision [`f1eed51`](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/tree/f1eed510c9a4fb47aaa9bb46e178aaa2d9224623): [`evaluation/mad_accuracy_evaluation.py`](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/blob/f1eed510c9a4fb47aaa9bb46e178aaa2d9224623/evaluation/mad_accuracy_evaluation.py) and [`evaluation/column_mapping.csv`](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/blob/f1eed510c9a4fb47aaa9bb46e178aaa2d9224623/evaluation/column_mapping.csv).
- Column-type counts and the pid 1 / `QID154` example (70 in waves 1–3; 82 in wave 4): [notebooks/data_exploration.ipynb](../notebooks/data_exploration.ipynb).
- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37–46. [https://doi.org/10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104).
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC. [https://doi.org/10.1007/978-1-4899-4541-9](https://doi.org/10.1007/978-1-4899-4541-9).
- McGraw, K. O., & Wong, S. P. (1996). *Forming Inferences About Some Intraclass Correlation Coefficients*. Psychological Methods, 1(1), 30–46. [https://doi.org/10.1037/1082-989X.1.1.30](https://doi.org/10.1037/1082-989X.1.1.30).
