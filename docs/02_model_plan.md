# Deliverable 2 — Concrete plan to build the behavior model

This document describes the model I would build, not a claim that the complete system has already been trained. The goal is to adapt a public instruction-tuned language model to predict one participant's held-out wave-4 response from a leakage-safe representation of that participant's waves 1–3 data.

This is a full-scale plan for a longer-term research project with sufficient time and compute for systematic data validation, model training, ablation studies, and error analysis. It is intentionally broader than the optional Deliverable 6 proof of concept (POC). Because the take-home assignment is time-limited, that POC uses a small model and a smaller data slice to demonstrate that the leakage-safe training and evaluation pipeline works end to end. The assignment asks for a model with fewer than 0.5B parameters, so I first chose `HuggingFaceTB/SmolLM2-360M-Instruct`. That run scored too low on the slice, so I then tried `Qwen/Qwen2.5-0.5B-Instruct` and treat it as the default local checkpoint. Transformers counts about 630M base parameters for that Qwen model; Deliverable 6 discloses this and reports both runs. The POC does not attempt to implement or validate every component of the 7B-scale research plan below.

The central design is:

1. Use `wave_split`, not `full_persona`.
2. Represent each example as one `(participant, held-out response column)` pair.
3. Report two explicit input conditions: a primary **no-copy** condition that excludes earlier answers to repeated wave-4 items, and a **full-history** condition that treats those waves 1–3 answers as historical inputs.
4. Compress the remaining persona with a deterministic summary and retrieval over safe persona blocks.
5. Establish a frozen prompting baseline, then train a shared 7B-scale model with supervised fine-tuning (SFT) using Quantized Low-Rank Adaptation (QLoRA).
6. Evaluate against wave-4 ground truth, trivial baselines, and human test–retest using the protocol in Deliverable 3.

The model is shared across participants; it is not one separately trained model or adapter per person.

## 1. Problem framing

**Unit of prediction.** One example is one `(pid, response column)` pair, where `pid` is the participant identifier: a leakage-safe persona, one held-out wave-4 question, and one target answer. A response column (for example, `QID154` or `QID287_1`) is the scoring unit; one Qualtrics QuestionID may expand into several response columns.

The held-out set contains **126 response columns** mapped through the catalog's `csv_columns` field to **84 QuestionIDs**: 68 Multiple Choice (MC), 7 Matrix, 6 Text Entry (TE), and 3 Slider. Display/Instruction (DB) screens have no response columns and are excluded. These are exploratory data analysis (EDA) counts, not figures reported by the paper.

**Two interpretations of the historical answers.** The assignment can reasonably be read in two ways. An earlier answer to the same question is temporally valid waves 1–3 history, not a future-label leak. However, allowing it creates a strong copy-last shortcut and changes the question from whether the model can infer an unseen response from the rest of the persona to whether it can update a known prior response. I would therefore report both conditions and never mix them within one result:

- **No-copy (primary):** use the provided `wave1_3_persona_json` or `wave1_3_persona_text`, which separates the repeated target-item answers, plus non-overlapping waves 1–3 response columns. This tests prediction from the rest of the persona.
- **Full-history (secondary):** additionally expose the source-tagged earlier answers from `wave4_Q_wave1_3_A` or the 126 overlapping waves 1–3 columns. This follows the literal “all waves 1–3 history” interpretation. It must be compared with copy-last, especially on items where the participant changed their answer.

For a controlled comparison, I would train and evaluate separate checkpoints for the two conditions while holding participant splits, target rows, and other hyperparameters fixed. Copy-last uses information unavailable to the no-copy model, so it is a diagnostic reference there; it becomes a direct trivial baseline for full-history.

Both conditions may use:

- The wave-4 question text, options, and response range after all answer fields have been removed.

I would not use any field from `full_persona`, including `persona_summary`, until a systematic sweep established that it contains no wave-4-derived information.

**Always forbidden as model input.**

| Field | Why |
|---|---|
| `full_persona.persona_text` / `persona_json` | Repeated questions already contain wave-4 answers. For pid 1, `QID154` stores the wave-4 value 82 rather than the earlier value 70. |
| `full_persona.persona_summary` | Only one participant/item has been inspected. Absence of 82 in that example does not establish that the field is safe. |
| `wave4_Q_wave4_A` with `Answers` left in | The target would appear directly in the question payload. |

The same question text appearing in wave 4 is expected and is not leakage. A wave-4 **answer** entering either condition is leakage. An earlier same-item answer is permitted only in the explicitly labeled full-history condition; accidentally including it in a no-copy run is a condition-contamination error, not future-label leakage.

**Output.** The model returns a canonical answer code or number matching the schema used by `wave4_response.csv`, not an explanation in free prose.

**Success.** The model should beat question-only and population baselines and approach the empirical human test–retest benchmark without suspicious above-benchmark behavior. Metrics, baselines, confidence intervals, and acceptance criteria are specified in Deliverable 3.

---

## 2. Data pipeline

```text
Pinned wave_split revision + question catalog + response CSVs
        │
        ├─ identify the 126 held-out response columns through csv_columns
        ├─ split unique pids into train / validation / test
        ├─ extract and source-tag the earlier repeated-item responses
        │     copy-last / test–retest: always retain as comparison data
        ├─ build the waves 1–3 input under one declared policy
        │     no-copy: exclude the 126 earlier repeated-item responses
        │     full-history: include those source-tagged earlier responses
        ├─ build one question payload
        │     deep-copy wave4_Q_wave4_A and recursively remove answer fields
        └─ isolate label = wave4_response.csv[pid, column]
```

Wave-4 labels remain isolated in every condition. The earlier responses remain available for copy-last and test–retest comparisons even when they are also exposed as source-tagged history in the full-history condition.

**Schema mapping.** I would build an explicit `response_column → QuestionID → type/options/range/block` index from `question_catalog.json`. Prefix matching is unsafe because, for example, `QID290_5` could be incorrectly matched to `QID2` or `QID29`.

**Participant split.** Shuffle unique `pid`s with seed `20250319` and split them **70% / 15% / 15%** into train, validation, and test sets. All items for one participant stay in one split. This tests generalization to unseen people and prevents the model from seeing other wave-4 labels from a test participant during training.

**Row construction.** Create a row only when the wave-4 target is non-null. Null cells produced by between-subject assignment are not negative labels and are never imputed. The training sampler should balance across tasks or response columns so large matrix blocks do not dominate the objective.

**Canonical targets.**

- Single-choice MC: one legal option code.
- Matrix: one code for each expanded matrix response column.
- Slider: a number within the catalog range.
- Numeric TE: the raw numeric response; decile conversion is evaluation-only.
- Multi-select, if later included: either one binary target per expanded option column or a canonical sorted list, chosen once and used consistently.

**Leakage gates.** These checks run before training and before every evaluation:

- Train, validation, and test pid sets are disjoint.
- In a no-copy run, no question object whose `csv_columns` intersect the held-out set is used to build a persona summary or retrieval index.
- In a full-history run, every repeated-item answer is explicitly tagged as coming from waves 1–3, and no value from `wave4_Q_wave4_A` or `wave4_response.csv` can enter the summary or retrieval index.
- Before a wave-4 question is added to the model prompt, create a **stripped question**: a copy with every field containing the participant's answer removed, including `Answers`, `Values`, `SelectedText`, and `SelectedByPosition`. The stripped question keeps only the question text, response options or range, and other non-answer metadata.
- For pid 1 / `QID154`, the no-copy input contains neither the earlier answer 70 nor the wave-4 target 82. The full-history input may contain the source-tagged earlier answer 70 but must never contain 82. Because the question stem may independently contain the number 70 as scenario text, these assertions operate on parsed answer fields rather than global substrings.
- Wave-4 labels and copy-last predictions are stored separately from model inputs; any earlier values used by full-history come from a versioned, source-tagged historical feature table.

I would version the dataset revision, catalog hash, split pid lists, input condition, preprocessing configuration, and prompt template so every model checkpoint can be traced to the exact data contract that produced it.

---

## 3. Modeling approaches — order and why

I would advance from low-cost baselines to trainable models. Each stage answers a specific question before more compute is committed.

**1. Frozen prompting.** Start with a capable instruction-tuned model using a safe compressed persona, one stripped question, and an explicit output schema. The dataset's precomputed large language model (LLM) simulations provide a reference without paying for external model calls. This establishes whether the data contract and prompt are useful before training.

**2. Summary plus retrieval.** Add the deterministic summary and retrieve relevant safe persona blocks. Begin with BM25, a keyword-based text-ranking algorithm (Robertson & Zaragoza, 2009), because it is transparent and cheap; compare it with a small dense embedding model only if lexical retrieval misses semantically related blocks. The retrieval policy is shared by frozen prompting and fine-tuning.

**3. QLoRA supervised fine-tuning—the main build.** Fine-tune **`Qwen/Qwen2.5-7B-Instruct`** (Qwen Team, 2024) as the primary open model using QLoRA (Dettmers et al., 2023). A 7B model is large enough to reason over heterogeneous survey questions while still being practical to adapt in 4-bit precision. The objective is causal language modeling with loss only on the answer tokens. I would train one shared model conditioned on the persona rather than one adapter per participant.

**Why 7B rather than the largest available model?** A 70B–90B model may improve general language reasoning, but it does not automatically improve person-specific prediction: performance may instead be limited by persona quality, retrieval errors, the 2,058-participant sample, and genuine test–retest variation. A 7B model can be adapted on a single 24–48 GB graphics processing unit (GPU), making it feasible to run the retrieval, hyperparameter, and persona-shuffling ablations needed to verify that the model actually uses individual information. I would scale to 14B, 32B, or 70B only if the validation scaling curve indicates that model capacity—not the data pipeline—is the bottleneck. A vision-oriented 90B model would also add capacity that this text-only survey task does not use.

I would compare this with one smaller model, such as `Qwen2.5-1.5B-Instruct`, to measure the quality/compute trade-off. A model below 0.5B is appropriate for the optional bonus POC, but it would not be my primary architecture for the full behavior model.

**4. Preference tuning only if error analysis justifies it.** Direct Preference Optimization (DPO) is optional, not part of the default path. I would consider it only after SFT produces reliably parseable answers and evidence shows a specific preference failure, such as repeatedly choosing an attractive but invalid response format. Preference pairs would be constructed from train-split examples only: the canonical ground-truth answer is the chosen response and an observed incorrect or invalid SFT sample is the rejected response. For short categorical targets, constrained decoding and supervised training may solve the problem more directly.

A tabular multilayer perceptron (MLP) over safe non-overlap columns is a useful neural baseline, but it is not the proposed behavior model because it cannot naturally answer a newly worded question.

**Experiment order.** Question-only prompt → summary prompt → summary + BM25 retrieval → summary + dense retrieval → QLoRA SFT → optional preference tuning. An ablation at each step shows whether extra complexity produces measurable value.

**Key trade-offs.**

- I would not pretrain from scratch: 2,058 participants are far too few to learn general language and reasoning capabilities.
- I would prefer QLoRA to full fine-tuning: it reduces memory and checkpoint cost and lowers the risk of damaging the base model's instruction-following behavior.
- I would not train one model per participant: each person provides too little data, and the resulting system could not generalize to a new participant.
- I would not rely on raw long-context prompting: it is expensive, difficult to audit, and likely to dilute the relevant evidence.
- I would not start with reinforcement learning from human feedback (RLHF) or Group Relative Policy Optimization (GRPO): the dataset supplies ground-truth answers, not human preference labels, so supervised learning is the direct objective.

---

## 4. Long personae and the context window

The raw persona text is approximately 126k–134k characters, so placing it directly in every prompt would be expensive, difficult to audit, and likely to bury relevant evidence. A long context window does not remove those problems.

**Deterministic summary.** Build a compact summary under the declared input condition. Both conditions may summarize safe non-overlap fields such as stable demographics, aggregate psychological-scale scores computed from permitted items, broad preferences, and selected cognitive measures. In full-history, repeated-item answers may be added in a separately labeled historical section; they remain absent in no-copy. Store field provenance with every summary value so it can be audited. Budget: approximately **400–600 tokens**.

**Retrieval index.**

1. Apply the declared input condition before indexing: remove the 126 earlier repeated-item responses for no-copy, or retain and source-tag them for full-history.
2. Split `wave1_3_persona_json` by `BlockName`; split oversized blocks into smaller question–answer chunks.
3. Index chunks separately for each participant.
4. Query with the held-out question text, options, and catalog block name.
5. Retrieve **3–5 chunks**, deduplicate them, and pack them after the deterministic summary.

BM25 is the first retriever because survey terminology often repeats exactly and its matches are easy to inspect. A dense retriever is adopted only if validation ablations show a gain.

**Planned training sequence budget: 4,096 tokens.** This is an initial engineering choice, not a fixed requirement of the dataset or model. It can be reduced when GPU memory or training throughput is constrained, or increased when hardware permits and validation experiments show that additional retrieved context improves prediction. Any change must remain within the base model's context limit, and the component budgets should be retuned without truncating the question or output schema.

| Component | Approximate budget |
|---|---:|
| System instruction and output schema | 150 tokens |
| Deterministic persona summary | 600 tokens |
| Retrieved persona chunks | 2,700 tokens |
| Question, options, and range | 500 tokens |
| Target answer | 16 tokens |

The target allocation is the supervised label during training and reserved generation space during inference. It is not part of the inference prompt.

**Hardware fit.** With a 7B model in 4-bit QLoRA, gradient checkpointing, and a micro-batch of 1, the 4,096-token plan is intended to fit on a single 24 GB GPU such as an RTX 3090/4090-class card. A 48 GB GPU provides more headroom for a micro-batch of 2, faster kernels, and fewer out-of-memory adjustments. On a GPU below 24 GB, I would initially reduce the sequence budget to 2,048 tokens or retrieve fewer persona chunks; CPU offloading is possible but substantially slower. These are planning estimates because actual memory use also depends on the training library, attention implementation, precision, and optimizer, so I would confirm the final batch and sequence settings with a short memory-profiling run.

If the complete training sequence exceeds the budget, truncate the lowest-ranked retrieved chunk first; never truncate the question or output schema, and never silently reintroduce fields excluded by the declared input condition.

**One held-out response per example.** Do not concatenate all available wave-4 items for a participant. During training, later items could otherwise see earlier gold answers in the same sequence; during inference, errors could cascade between items.

---

## 5. Training details

Here and below, mean absolute deviation (MAD) accuracy refers to the paper's range-normalized accuracy metric used for model selection and evaluation.

| Knob | Value | Rationale |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | Strong open instruction model with an Apache-2.0 license and manageable QLoRA cost |
| Adaptation | 4-bit NormalFloat (NF4) QLoRA | Fits a 7B model on a single 24–48 GB GPU without full-model updates |
| Low-Rank Adaptation (LoRA) targets | Attention and MLP projection layers | Gives the adapter capacity to use persona and question information |
| Rank / alpha / dropout | 16 / 32 / 0.05 | Conservative starting point; tune rank 8 vs 16 on validation |
| Objective | Causal token cross-entropy with prompt labels set to `-100` | Trains the answer rather than reconstructing the persona |
| Sequence length | 4,096 tokens | Fits the packing budget in §4 |
| Micro-batch / accumulation | `1 × 32` or `2 × 16`; effective batch = 32 | Choose the pair that fits GPU memory while preserving the same effective batch size |
| Learning rate | `1e-4`, cosine schedule, 3% warmup | Reasonable QLoRA starting point; compare with `2e-4` on validation |
| Epochs | Initial trial: 3; initial search range: 1–5 | Evaluate after every epoch and retain the checkpoint with the best validation 17-task person-equal MAD accuracy. Stop after two evaluations without an improvement above a predeclared minimum tied to validation uncertainty; extend beyond 5 only while validation continues to improve without a widening train–validation gap and the additional compute cost remains justified |
| Precision | bfloat16 (bf16) compute on supported GPUs; 16-bit floating point (fp16) otherwise | Stable mixed-precision training |
| Seed | 42 for training; fixed data-split seed from §2 | Separates model randomness from split generation |
| Repeated runs | One seed for screening; three seeds for finalists | Avoids selecting an architecture from one unusually favorable training run |

Training uses the model's official chat template. Rows are sampled by task or response column rather than uniformly from the expanded table, preventing large matrix blocks from dominating. I would log both training loss and validation metrics, but select checkpoints using Deliverable 3's validation 17-task person-equal MAD accuracy rather than token loss alone.

```text
System:
Predict how this survey participant would answer the question.
Return exactly one answer in the required schema.

User:
Participant summary:
{safe_summary}

Relevant prior responses:
{retrieved_safe_chunks}

Question:
{stripped_question}

Allowed response:
{legal_codes_or_numeric_range}

Assistant:
{target}
```

At inference time, decoding is greedy and schema-constrained where possible: MC outputs are restricted to legal codes, sliders are parsed and range-validated, and invalid outputs are counted as failures rather than silently dropped. The generated explanation is suppressed because the evaluation target is the answer, not a rationale.

**NOTE**: The optional bonus POC applies the no-copy version of the same objective to a small model and a smaller data slice. I first used SmolLM2-360M-Instruct to stay under 0.5B; after that run scored too low, the default local checkpoint became `Qwen2.5-0.5B-Instruct`, with the parameter-count deviation disclosed in Deliverable 6. That prototype is a demonstration of the loop, not the primary architecture proposed here.

---

## 6. Evaluation preview

Model selection and final reporting follow Deliverable 3. The important design constraints for training are:

- Split by participant and use validation only for model/retrieval/hyperparameter choices.
- Report no-copy and full-history results separately; never combine their rows or compare them as if they used the same information.
- Use the official paper-compatible MAD ranges and task mapping for the headline result.
- Report type-specific diagnostics instead of pooling exact match across incompatible response scales.
- Compare against question-only, train-majority, random, copy-last, frozen prompting, and the human test–retest benchmark.
- Compare normal predictions with a persona-shuffling ablation; if performance does not fall, the model is not meaningfully using individual-level information.
- Report parse rate and count invalid generations as failures.
- In no-copy results, use the earlier same-item answer only for copy-last and human test–retest comparisons. In full-history results, it is an explicitly labeled historical feature, and performance must also be reported on the subset where the earlier and wave-4 answers differ.

The paper reports **81.72% mean human test–retest accuracy across 17 tasks**. This is an empirical short-term benchmark, not a mathematical upper bound. A model that unexpectedly exceeds it should trigger a leakage and aggregation audit before the result is interpreted.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Wave-4 label enters the persona or question payload | Use only `wave_split`; recursively strip answer fields; enforce structural leakage tests before train/eval |
| Earlier-answer policy is mixed across runs | Store `input_condition=no_copy|full_history` in every data and model artifact; apply condition-specific structural tests |
| Retrieval selects irrelevant blocks | Compare question-only, summary-only, BM25, and dense retrieval on validation; inspect retrieved block IDs |
| Important stable attributes are truncated | Place the deterministic summary first and truncate the lowest-ranked retrieved chunks |
| Large matrix tasks dominate training | Sample by task/column and report the training distribution |
| Between-subject nulls are treated as labels | Train and score only assigned, non-null cells; never impute an unseen condition |
| The model emits illegal codes or prose | Use the official chat template, explicit legal responses, constrained decoding, and strict parsing |
| Overfitting to 2,058 participants | Split by participant, use QLoRA rather than full fine-tuning, select on validation, and test once |
| A high score is caused by leakage or aggregation error | Run leakage gates, inspect per-column results, and reproduce the result with the official evaluation mapping |
| Survey and panel biases are learned by the model | Document scope, report subgroup diagnostics, and apply the guardrails in Deliverables 4 and 5 |

---

## 8. If I had more time

1. Compare 1.5B, 7B, and 14B models under the same persona and retrieval pipeline.
2. Train a supervised retriever using validation performance rather than semantic similarity alone.
3. Test separate adapters or routing for pricing, heuristics, and survey-scale tasks.
4. Calibrate predictive uncertainty and allow abstention when the persona contains little relevant evidence.
5. Evaluate temporal drift using a later data wave rather than only a short retest interval.
6. Consider DPO only for a clearly observed failure that supervised training and constrained decoding do not solve.

## References

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A Dataset for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions*. [arXiv:2505.17479](https://arxiv.org/abs/2505.17479).
- LLM-Digital-Twin. *Twin-2K-500 dataset card*. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- Toubia et al. *Digital-Twin-Simulation reference implementation*. [GitHub](https://github.com/tianyipeng-lab/Digital-Twin-Simulation).
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333–389. [https://doi.org/10.1561/1500000019](https://doi.org/10.1561/1500000019).
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
- Qwen Team. (2024). *Qwen2.5 Technical Report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).
