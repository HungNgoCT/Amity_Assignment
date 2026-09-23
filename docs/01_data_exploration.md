# Deliverable 1 — Data exploration

## Questions addressed

1. Dataset structure and the two Hugging Face (HF) subsets
2. Question types and how they would be scored
3. Sample distributions / representativeness
4. Human test–retest reliability
5. Biases and limitations

## Key findings

| Key finding | Description |
|---|---|
| Leakage trap | `full_persona` contains wave-4 answers for repeated items. Use `wave_split`, remove answers from wave-4 question payloads, and retain them only as prediction targets. |
| Test–retest benchmark | Paper Figure 2 reports **81.72% mean human test–retest accuracy across 17 tasks**. The assignment calls this a ceiling, but it is an empirical short-term benchmark rather than a mathematical maximum. |
| Representativeness | Gender/sex and region are close to 2023 Census benchmarks, but education, age, and income contain gaps above **5 percentage points**. |
| Scoring per type | Use metrics appropriate to each response type. For comparison with the paper, report range-normalized accuracy rather than pooling exact match across all items. |
| Biases / limitations | Online-panel self-selection, completion and self-report bias, limited geographic/time scope, and structured missingness from experimental assignment. |

## Detailed findings

1. **Dataset structure / two HF subsets.**

   Twin-2K-500 = four-wave US survey. On Hugging Face, `load_dataset` has **two subsets**: `full_persona` and `wave_split`. Other dataset artifacts—including the catalog, comma-separated values (CSV) response files, large language model (LLM) outputs, and raw Qualtrics exports—are provided as separate files.

   **`full_persona`** = table (**2,058 rows × 4 columns**), with one row per participant identifier (`pid`) and their survey responses.
   - 4 columns = `pid`; `persona_text`; `persona_summary`; `persona_json`.
   - Across all 2,058 rows, `persona_text` is **125,622–133,627** characters (median 128,665) and `persona_summary` is **11,602–18,482** characters (median 12,993). The paper and dataset card do not report these lengths; the notebook computes them on the pinned dataset revision.
   - The dataset card describes `persona_text` and `persona_json` as item-level survey Q–A in prose and structured JavaScript Object Notation (JSON) formats. Both merge waves 1–3 with wave 4 into one profile; for repeated questions, wave 4 replaces the earlier answer.
   - For the first few participants inspected, `persona_summary` is a prose profile containing demographics, derived scores, and selected self-descriptions—not a question-by-question response list. It is available only in `full_persona` and **not** a column on `wave_split`.
   - Spot check: For **pid=1**, the earlier `QID154` response in waves 1–3 was 70, while the wave-4 response was 82. Both `persona_json` (`Values: ['82']`) and `persona_text` (`Answer: 82`) contain the wave-4 response, so they would leak the prediction target. `persona_summary` does not expose this item in this example, but one case is insufficient to establish that it is leakage-safe.

   The assignment predicts held-out wave-4 responses from waves 1–3. Therefore, the modeling data should be constructed from **wave_split**: use waves 1–3 as the persona, remove answers from the wave-4 questions, and retain those answers only as prediction targets. `full_persona.persona_text` and `full_persona.persona_json` are unsuitable because repeated items already contain their wave-4 answers.

   **`wave_split`** = table (**2,058 rows × 5 columns**), with one row per participant. Its fields separate earlier responses from the later retest:
   - **pid**: participant identifier.
   - **wave1_3_persona_text** and **wave1_3_persona_json**: responses from waves 1–3, represented as text and structured JSON, respectively.
   - **wave4_Q_wave1_3_A**: held-out questions paired with the participant’s **earlier** answers, used to measure test–retest consistency.
   - **wave4_Q_wave4_A**: those questions paired with the participant’s **wave-4** answers—the prediction targets.

   Additional files outside the two Hugging Face configs include:
   - **question_catalog.json**: metadata for 256 QuestionIDs.
   - **wave1_3_response.csv**: 2,058 participants × 760 response columns, plus pid.
   - **wave4_response.csv**: 2,058 participants × 126 response columns, plus pid.

   Label-form CSVs, precomputed LLM outputs, and anonymized raw Qualtrics exports were not used in the analyses below.

2. **Question types and scoring.**

   The dataset has three different counts because they refer to different units:

   | Count | What it counts |
   |---|---|
   | **256** | QuestionIDs in the catalog; one QuestionID may contain several sub-items |
   | **~500** | Questions administered across waves 1–3 |
   | **760** | Response columns after matrix rows and multi-select options are expanded |

   The full catalog contains **175 Multiple Choice (MC)**, **36 Matrix**, **28 Text Entry (TE)**, **14 Display/Instruction (DB)**, and **3 Slider** QuestionIDs. The held-out wave-4 set contains **84 QuestionIDs**: 68 MC, 7 Matrix, 6 TE, and 3 Slider; DB screens are not prediction targets. The paper describes 88 holdout questions across 17 tasks. I have not reconciled that count with these 84 QuestionIDs and 126 response columns; the two units may differ. A response column is mapped to its question type through the catalog's `csv_columns` field.

   Proposed scoring by question type:
   - **154 single-choice MC QuestionIDs in the full catalog** (68 of the 84 wave-4 QuestionIDs): exact-match accuracy—the predicted option either matches or does not (e.g., target = option 3, prediction = option 3 → correct).
   - **Multi-select MC (21 in the full catalog; 0 of the 126 wave-4 scored columns):** Jaccard similarity measures the overlap between the predicted and selected option sets (e.g., target = `{A, C}`, prediction = `{A, B}` → overlap `{A}` divided by combined set `{A, B, C}` = **1/3**). Per-option F1 can be reported as a diagnostic: treat each choice as a yes/no (selected or not) and score that binary decision. This take-home does not compute it, because no multi-select column appears in the wave-4 scored set.
   - **Ordered Matrix responses:** ordinal mean absolute error (MAE)—the average distance between the predicted and target scale values—and a range-normalized accuracy score (e.g., target Likert score = 5, prediction = 3 → error = **2 points**).
   - **Slider:** MAE in the original scale and range-normalized accuracy (e.g., `QID154`: wave-4 target = 82, prediction = 70 → MAE = **12** and normalized accuracy = `1 − 12/100` = **0.88**). Exact match is too strict for a 0–100 slider.
   - **Numeric TE / anchoring:** MAE; following the paper, unbounded numerical answers are converted into ten ordered groups (deciles) from wave-2 answers before scoring. Two answers in the same decile have zero decile distance. Free-form text is not part of the held-out wave-4 item set and would require a separate scoring rubric.
   - **DB:** instruction-only screens with no response columns, such as a page explaining the study procedure, so they are excluded.

   The main metric used to compare the model with the paper's **81.72% human test–retest consistency across 17 tasks** is range-normalized accuracy: `1 − |prediction − target| / response range`. A score of 1 means an exact match; lower scores mean larger errors. The type-specific metrics above are also reported to show where the model performs well or poorly. Full evaluation details are in Deliverable 3.

3. **Distributions / representativeness.**

   The paper and dataset card describe the final sample as representative of U.S. adults. Recruitment used a Prolific online panel with quotas for age, sex, and ethnicity. To examine this claim, the exploratory data analysis (EDA) compares the sample's unweighted demographic shares with 2023 U.S. Census benchmarks, primarily the American Community Survey (ACS). This comparison is a diagnostic, not proof that the sample is or is not representative.

   **Gender/sex** and **census-region** shares are within **2.5** percentage points of the Census benchmarks.

   I flag absolute gaps above **5** percentage points as descriptively important (this is a reporting threshold, not a statistical significance test):

   **More common in the sample than in the Census benchmark (overrepresented):**
   - College graduate / some postgraduate: **+13.9 percentage points**
   - Age 50–64: **+8.2**
   - Family income $30k–$50k: **+5.7**

   **Less common in the sample than in the Census benchmark (underrepresented):**
   - High-school education only: **−12.7 percentage points**
   - Income $100k+: **−11.0**
   - Less than high-school education: **−9.4**
   - Age 65+: **−9.1**

   These comparisons are approximate: ACS education statistics cover adults aged 25+, while this survey includes ages 18–24; ACS reports household income, whereas the survey asks about family income; and the $30k income boundary is interpolated from ACS bins.

   Recruitment also used an ethnicity quota, but this EDA did not add an external race/ethnicity benchmark. Alignment on that dimension therefore remains unverified rather than being inferred from the quota.

   Overall, gender and region align reasonably well, but education, age, and income do not. The sample should therefore be treated as a quota-based U.S. online panel, not as a census draw.

4. **Human test–retest (the important number).**

   Wave 4 was launched approximately **two weeks after wave 3** and repeats heuristics, behavioral-economics, and pricing items from across waves 1–3. Because the repeated items originated in different waves, the elapsed time may be longer for items first administered earlier. In the standardized CSVs, **126 of the 760** waves 1–3 response columns also appear in wave 4. For each participant and repeated item, the earlier answer can therefore be compared with the later answer.

   **This comparison measures how consistently a person answers the same item over time**. It provides the empirical **human test–retest benchmark** for the prediction task. The earlier answer is also the copy-last prediction. It is withheld from the primary no-copy model input, but Deliverables 2 and 3 also define a separately reported full-history condition in which this temporally valid waves 1–3 answer is explicitly available. The two conditions must not be pooled.

   Descriptive results from my EDA (not the paper's overall score):
   - **MC:** exact-match rate ≈ **73.5%**.
   - **Matrix:** exact-match rate ≈ **60.3%**; mean ordinal error ≈ **0.52 scale points**.
   - **Slider:** mean absolute error ≈ **15.5 points** on a 0–100 scale. Exact match is only ≈ **12.0%**, which is too strict to use as the headline.
   - **Numeric TE:** exact-match rate ≈ **25.3%**; this is only a diagnostic because the questions use different numerical scales.

   These values use different units and should not be averaged into one accuracy number.

   Paper Figure 2 reports a mean human test–retest accuracy of **81.72% across 17 tasks**. For binary items, this is exact match. For bounded numerical items, accuracy is `1 − |earlier answer − wave-4 answer| / response range`; unbounded anchoring answers are first converted to deciles. I did not recompute the official 17-task result here. It is an empirical short-term benchmark rather than a mathematical maximum; a model that unexpectedly exceeds it should trigger a leakage audit.

5. **Biases / limitations.**

   - **Selection and completion bias:** The dataset card warns of self-selection bias, and the paper reports that recruitment used Prolific. The final sample contains the 2,058 people who completed all four waves, down from 2,509 wave-1 completions. This may make the final sample differ from people who do not join online panels or who did not complete every wave. My ACS comparison also finds that people with less than high-school education are strongly underrepresented.
   - **Self-report bias:** The dataset card warns of social-desirability bias and states that survey responses may not accurately reflect actual behavior or characteristics. The dataset should therefore be interpreted as reported survey responses, not observed behavior.
   - **Short and limited retest:** Wave 4 was launched approximately two weeks after wave 3, although items first administered in earlier waves have a longer interval. The retest covers only repeated heuristics, behavioral-economics, and pricing tasks. It does not establish consistency over months or years, or for the personality and demographic questions that wave 4 did not repeat.
   - **Time and geographic scope:** The dataset card describes a specific time and geographic context; the paper reports a U.S. panel surveyed from January to February 2025. Results may not generalize to other countries, periods, or populations.
   - **Structured wave-4 missingness:** My EDA finds the same 2,058 final-sample `pid`s in both response CSVs. Many blank wave-4 cells arise because between-subject experiments assign each participant to only one condition, not because that participant dropped out. Evaluation should score only non-null assigned items. This item-level missingness is separate from the reduction from 2,509 wave-1 completions to the 2,058-person final sample.

## Reproducibility

Reproducible code, plots, full tables, and ACS benchmark recoding: [notebooks/data_exploration.ipynb](../notebooks/data_exploration.ipynb).

## References

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A Dataset for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions*. [arXiv:2505.17479](https://arxiv.org/abs/2505.17479).
- LLM-Digital-Twin. *Twin-2K-500 dataset card and data files*. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- U.S. Census Bureau. *2023 American Community Survey 1-Year Estimates*: [S0101](https://data.census.gov/table/ACSST1Y2023.S0101) (age), [B01001](https://data.census.gov/table/ACSDT1Y2023.B01001) (sex), [S1501](https://data.census.gov/table/ACSST1Y2023.S1501) (education), and [S1901](https://data.census.gov/table/ACSST1Y2023.S1901) (income); [2023 Population Estimates Program](https://www.census.gov/programs-surveys/popest.html) (region).
