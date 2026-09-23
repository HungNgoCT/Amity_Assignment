# Deliverable 5 — Long-run maintenance

## Requirements addressed

1. Monitoring
2. Drift as people and contexts change
3. Retraining triggers
4. Reproducible versioning and rollback
5. Governance and ethics

## Proposed maintenance plan

Business-use boundaries remain in [Deliverable 4](04_business_apps.md), while metrics and evaluation gates remain in [Deliverable 3](03_eval_strategy.md). This document defines the operating loop after an initial model has been selected.

### Maintenance summary

1. **Monitoring.** Maintain two separate loops. **Model and stack regression monitoring** uses a frozen labeled slice to check structural leakage, input-condition integrity, parse rate, absolute mean absolute deviation (MAD) accuracy, question-type slices, latency, and cost whenever code or model components change. **Human behavior monitoring** requires a newly collected retest wave; only new human labels can reveal person, item, world, or population drift. Always report absolute MAD beside the ratio to the current test-split human benchmark.

2. **Drift.** Track four sources: **person drift** (preferences change), **item/world drift** (prices, events, medical or political context change), **sample drift** (the served population moves away from the validated panel), and **stack drift** (base large language model (LLM), tokenizer, prompt, or retrieval changes). The paper's **81.72%** is a published full-sample, short-term reference—not a permanent benchmark. Wave 4 launched approximately two weeks after wave 3, while items originating in waves 1–2 have longer intervals. Re-estimate reliability on each new retest cohort rather than assuming how a six-month benchmark will move.

3. **Retraining triggers.** Block release or suspend serving when a leakage or input-condition test fails. Retrain or collect new human data when a candidate bundle loses more than 2 percentage points of absolute 17-task person-equal MAD accuracy on the frozen slice, model MAD / current human benchmark falls below 0.80 on a new retest, the paired 95% confidence-interval lower bounds for no-copy versus train-majority and question-only are not both above zero, the full-history lower bound against copy-last is not above zero on changed-answer items, or the catalog or population changes. Revalidate the complete bundle whenever a stack component changes. These are engineering decision thresholds, not paper values, and must be frozen before each evaluation cycle.

4. **Versioning and rollback.** Release an immutable bundle containing model and adapter identifiers, code and environment versions, prompt and tokenizer versions, retrieval-index provenance, data hashes and temporal cutoff, participant split, input condition, scoring assets, and a signed evaluation report. A newly observed answer may become history only in a later bundle with a new temporal cutoff and a subsequent holdout wave. Retain the last approved bundle for rollback.

5. **Governance and ethics.** Deliverable 4's prohibited uses remain in force after every retrain. Require named owners for data, model release, privacy, and rollback; least-privilege access to participant-level artifacts; retention and deletion rules; an incident log; and human approval before external use (National Institute of Standards and Technology, 2020, 2023). Any new data collection requires appropriate consent and privacy/legal review. Do not scrape external personal data into a persona without an approved purpose and consent basis.

---

### Detailed operating protocol

#### 2.1 Monitoring

Use the same scoring contract as Deliverable 3. A frozen labeled slice detects regressions caused by code or model changes; it cannot detect changes in people without new human labels.

| Signal | Source and cadence | Alert / action |
|---|---|---|
| Structural leakage and input-condition tests | Every data build and model evaluation | Any wave-4 answer in an input, any unstripped question, or an earlier same-item answer in no-copy → block release. Full-history may contain only source-tagged earlier answers |
| Parse rate | Every candidate bundle on the frozen labeled slice | &lt; 80% of non-null items → block research-scale release; this is an engineering threshold |
| Absolute 17-task person-equal MAD accuracy and per-task diagnostic table | Every candidate bundle on the same frozen slice | Drop &gt; 2 percentage points from the last approved bundle → investigate and block promotion |
| Question-type slices, including Multiple Choice (MC), slider, and anchoring | Every candidate bundle | Any type materially degrades while aggregate MAD appears stable → investigate before promotion |
| Latency, graphics processing unit (GPU) memory, and cost per 1,000 predictions | Every candidate bundle on fixed hardware or a normalized benchmark | Exceeds the approved service budget → optimize or reject the bundle |
| Current human test–retest benchmark and model/benchmark ratio | Only when a new human retest wave is available | Ratio &lt; 0.80 after checking absolute MAD → investigate data and model drift |
| No-copy minus train-majority and no-copy minus question-only | New human retest wave | Either paired 95% confidence-interval lower bound ≤ 0 → no demonstrated gain over both non-personal baselines |
| Full-history minus copy-last on changed-answer items | New human retest wave | Lower bound of the paired confidence interval ≤ 0 → the model has not shown incremental value over copying |
| Sample composition and subgroup diagnostics | At recruitment and each new human wave | Material movement from the validated population or insufficient subgroup counts → narrow claims or recruit new data |

Subgroup diagnostics may include education and age 65+ categories motivated by Deliverable 1's American Community Survey (ACS) comparison, but only when the group count is large enough to report responsibly.

---

#### 2.2 Drift

The Hugging Face dataset card states that the data represent a specific **point in time** and geographic context. Wave 4 provides short-term retest evidence; it does not establish 12-month stability. The monitoring plan treats drift as a change in the relationship between available inputs and the behavior being predicted, following the general concept-drift framing of Gama et al. (2014).

| Drift | What moves | How we see it | What we do |
|---|---|---|---|
| **Person** | The same participant answers the same item differently over time | Human self-agreement on the new retest differs from the previous cohort-specific benchmark | Recompute the current benchmark and report absolute model MAD beside the ratio; do not call the model better merely because the denominator fell |
| **Item / world** | Prices, news, and medical or political context change | Performance loss is concentrated in particular items or catalog blocks | Investigate, relabel, or retire affected items until a new human wave validates them |
| **Sample** | The recruited or served population moves away from the validated panel | Repeat the representativeness analysis from Deliverable 1 and compare cohort composition | Narrow population claims or recruit and validate a new sample; model retraining alone cannot repair coverage |
| **Stack** | Base LLM, tokenizer, chat template, prompt, embedding model, or retrieval policy changes | The same frozen labeled slice changes under a new bundle | Treat the change as a new model and rerun the full Deliverable 3 validation protocol |

**How the system keeps up with people changing:** collect a small human retest using the same item wording and experimental condition where possible. Keep a held-out set of participant identifiers (`pid`s), recompute the cohort-specific human benchmark, and then decide whether to retrain the Large Behavior Model (LBM) on older human waves. The newest wave remains evaluation-only for that bundle. Model-generated responses are never treated as human labels.

---

#### 2.3 Retraining triggers

Freeze trigger definitions before opening the next evaluation wave. The 2-percentage-point drop, 80% parse rate, and 0.80 benchmark ratio are engineering thresholds for this plan, not values published by Twin-2K-500.

| Trigger | Action |
|---|---|
| Leakage or input-condition test fails | **Stop release or serving.** Inspect field provenance and prompt construction; do not publish the affected score |
| Frozen-slice 17-task person-equal MAD accuracy drops &gt; 2 percentage points | Diagnose code, model, prompt, and retrieval changes; rollback or retrain even if the benchmark ratio still looks acceptable |
| Model/current-human-benchmark ratio &lt; 0.80 on a new retest wave | Check absolute MAD and the newly measured human benchmark, then improve the model or collect more human persona data. Do not continue dividing by the published 81.72% |
| No-copy does not beat both train-majority and question-only with positive paired 95% confidence-interval lower bounds, or full-history does not beat copy-last on changed-answer items | Audit persona use, retrieval, and temporal features, then try supervised fine-tuning (SFT) or collect more longitudinal human labels. Consider Direct Preference Optimization (DPO) only if valid train-split preference pairs exist and error analysis justifies it |
| Scored catalog or item wording changes | Build new training examples, collect or verify matching human labels, refit train-only transforms, and issue a new bundle; do not compare scores as if the task were unchanged |
| Base model, tokenizer, prompt, or retrieval changes | Run the full validation protocol; evaluate the locked test set once only after acceptance decisions are final |
| Population no longer matches Deliverable 4's intended users | Narrow the claim or recruit a new sample; do not try to retrain away a coverage problem |
| Unauthorized access, provenance failure, or consent-scope concern | Stop affected processing, preserve audit evidence, notify privacy/legal owners, and follow the incident-response process |

Never train on model-generated response codes as human ground truth. Moving a previously evaluated human wave into persona history requires a new temporal cutoff and a newer holdout wave.

---

#### 2.4 Versioning

Every release is an immutable, content-addressed bundle. Record the Hugging Face (HF) revision and comma-separated values (CSV) file hashes rather than relying on mutable filenames. Capturing code, data, configuration, and downstream dependencies together reduces the hidden technical debt created by undeclared machine-learning system dependencies (Sculley et al., 2015).

```text
bundle_id:  YYYYMMDD-<short-hash>
  model:       base model + adapter ids/hashes
  code_env:    git commit + dependency lock + hardware/runtime
  tokenizer:   tokenizer revision + chat-template hash
  prompt:      prompt-template hash
  retrieval:   method + embedding revision + index hash + k
  data:        HF revision + CSV hashes + temporal cutoff + provenance/consent-purpose record
  split:       seed 20250319 + train/validation/test pid-list hashes
  condition:   no_copy | full_history
  scoring:     evaluation-code commit + range/task-mapping hashes
  deciles:     train-pid cutpoint-file hash
  evaluation:  signed report hash + approver + timestamp
  rollback:    previous approved bundle_id
```

Every report quotes `bundle_id`. Compare bundles only on a documented common evaluation set and never average scores from incompatible bundles. Changing the split seed to obtain a more favorable test result is forbidden; a legitimate split change requires written rationale, a new bundle, and re-evaluation of every comparator.

When a new human wave arrives, it is either an evaluation label or persona history for a given `(pid, item)` in one bundle, never both. After promoting that wave into history, create a new temporal cutoff and reserve a later human wave as the new holdout.

---

#### 2.5 Governance

| Rule | Owner | Escalation |
|---|---|---|
| Deliverable 4 prohibitions, including high-stakes decisions, impersonation, and participant-level productization | Research and product owners | Remain in force after every retrain; any scope change requires ethics, privacy, and legal review |
| Participant-level data access, encryption, retention, and deletion | Data and privacy owners | Access or retention violation → stop affected processing and open an incident |
| New survey wave or external persona source | Data owner | Require documented purpose, appropriate consent, provenance, and privacy/legal review before collection |
| Structural leakage tests in the automated build pipeline | Engineering owner | Any failure → block bundle promotion and follow §2.3 |
| Model card (Mitchell et al., 2019), change log, evaluation report, and Deliverable 4 disclosure | Release owner and independent reviewer | Missing artifact or disclosure → reject or withdraw the release |
| Red monitoring or retraining trigger without action | Research owner | Escalate to a backup reviewer; keep the affected bundle stopped until disposition is documented |
| Above-benchmark result | Evaluation owner | Run the Deliverable 3 leakage, denominator, and aggregation audit before making any claim |
| Rollback and incident response | Engineering, privacy, and product owners | Restore the last approved bundle, preserve logs, notify affected stakeholders, and document corrective action |

Ethics is part of the operating process, not a one-time checklist. Publish the current cohort-specific human benchmark beside model MAD, disclose the evidenced time and population scope, and never improve a report by silently moving labels into persona history or replacing human labels with model outputs.

## References

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A Dataset for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions*. [arXiv:2505.17479](https://arxiv.org/abs/2505.17479).
- LLM-Digital-Twin. *Twin-2K-500 dataset card and data files*. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- National Institute of Standards and Technology. (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. [https://doi.org/10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1).
- National Institute of Standards and Technology. (2020). *NIST Privacy Framework: A Tool for Improving Privacy through Enterprise Risk Management, Version 1.0*. [https://doi.org/10.6028/NIST.CSWP.01162020](https://doi.org/10.6028/NIST.CSWP.01162020).
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). *Model Cards for Model Reporting*. [https://doi.org/10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596).
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys, 46(4), Article 44. [https://doi.org/10.1145/2523813](https://doi.org/10.1145/2523813).
- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). *Hidden Technical Debt in Machine Learning Systems*. Advances in Neural Information Processing Systems 28. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html).
- Evaluation protocol: [docs/03_eval_strategy.md](03_eval_strategy.md). Business-use boundaries: [docs/04_business_apps.md](04_business_apps.md). Representativeness analysis: [notebooks/data_exploration.ipynb](../notebooks/data_exploration.ipynb).
