# Large Behavior Model on Twin-2K-500

This repository is my solution to the **Research Engineer Take-Home Assignment**. It analyzes the Twin-2K-500 longitudinal survey and proposes a Large Behavior Model (LBM) that predicts a participant's held-out wave-4 responses from information collected in waves 1–3.

The main design constraint is temporal data integrity: a wave-4 answer must never enter the model input used to predict that answer. The research plan reports two input conditions separately:

- **No-copy (primary):** excludes earlier answers to questions repeated in wave 4.
- **Full-history (secondary):** allows those earlier answers as temporally valid history and compares the model against the copy-last baseline.

## Deliverables

| Deliverable | File | Scope |
|---|---|---|
| 1 — Data exploration | [`docs/01_data_exploration.md`](docs/01_data_exploration.md) | Dataset structure, leakage risks, human test–retest benchmark, representativeness, scoring, and limitations |
| 1 — Analysis notebook | [`notebooks/data_exploration.ipynb`](notebooks/data_exploration.ipynb) | Reproducible exploratory analysis supporting Deliverable 1 |
| 2 — Model plan | [`docs/02_model_plan.md`](docs/02_model_plan.md) | Data contract, model progression, retrieval, QLoRA fine-tuning, context budget, and risks |
| 3 — Evaluation strategy | [`docs/03_eval_strategy.md`](docs/03_eval_strategy.md) | Metrics, baselines, participant split, uncertainty, leakage tests, and acceptance criteria |
| 4 — Business applications | [`docs/04_business_apps.md`](docs/04_business_apps.md) | Product concepts, suitable organizations, decision boundaries, and guardrails |
| 5 — Maintenance | [`docs/05_maintenance.md`](docs/05_maintenance.md) | Monitoring, drift, retraining triggers, versioning, governance, and incident response |
| 6 — Bonus POC | [`docs/06_poc.md`](docs/06_poc.md) | Commands and limitations for the runnable leakage-safe prototype |

Deliverables 2–5 describe a research-scale system. Deliverable 6 is intentionally a smaller proof of concept and does not implement the complete evaluation protocol in Deliverable 3.

## Repository structure

```text
.
├── README.md
├── Research_Engineer_Assignment.pdf   # Original assignment
├── requirements.txt                   # Notebook and POC dependencies
│
├── docs/
│   ├── 01_data_exploration.md         # Deliverable 1
│   ├── 02_model_plan.md               # Deliverable 2
│   ├── 03_eval_strategy.md            # Deliverable 3
│   ├── 04_business_apps.md            # Deliverable 4
│   ├── 05_maintenance.md              # Deliverable 5
│   └── 06_poc.md                      # Deliverable 6
│
├── notebooks/
│   └── data_exploration.ipynb         # EDA supporting Deliverable 1
│
├── src/
│   ├── data/
│   │   └── build_jsonl.py             # Build person-split, leakage-safe POC JSONL
│   ├── baselines.py                   # Random, train-majority, and copy-last baselines
│   ├── leak_test.py                   # Prompt-level leakage checks
│   ├── train.py                       # Qwen2.5-0.5B-Instruct LoRA/QLoRA training
│   └── evaluate.py                    # Slice evaluation and optional adapter inference
│
├── data/
│   ├── wave4_response.csv             # Wave-4 numeric responses
│   ├── wave4_response_label.csv       # Wave-4 label-form responses
│   └── poc/                           # Generated POC JSONL; ignored by Git
│
├── data_raw/
│   └── raw_data/                      # Anonymized wave exports and questionnaires
│
├── runs/                              # Generated model adapters; ignored by Git
└── results/                           # Generated notebook/POC outputs; ignored by Git
```

Additional top-level scripts and documents are source-inspection or earlier reference artifacts; the submission path is the `docs/`, `notebooks/`, and `src/` structure above.

## Setup

Use Python 3.10 or a compatible environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

CUDA-enabled PyTorch is required for GPU training. Data preparation, leakage tests, and non-model baselines can run without a GPU.

## Reproduce Deliverable 1

Open and run:

```text
notebooks/data_exploration.ipynb
```

The notebook loads the Twin-2K-500 Hugging Face dataset and writes generated tables or exports under `results/`.

## Run the bonus POC

Run these commands from the repository root:

```bash
python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20
python -m src.leak_test data/poc/leak_fixture.jsonl
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc/adapter --out_dir results/poc
```

The POC implements the **no-copy** condition. It uses non-overlapping waves 1–3 fields for the persona, keeps copy-last data in diagnostic files only, and never loads `full_persona` for model prompts.

The POC reports slice-level mean absolute deviation (MAD) accuracy using train-only empirical ranges. It is not a reproduction of the official 17-task paper evaluation. See [`docs/06_poc.md`](docs/06_poc.md) for hardware notes, command variants, and interpretation rules.