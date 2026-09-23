# Take-Home: Building a Large Behavior Model on Twin-2K-500

This repository is my submission for the **Research Engineer Take-Home Assignment**. It is a design-and-reasoning package plus a small bonus prototype, not a production behavior model. It analyzes the Twin-2K-500 longitudinal survey and proposes a Large Behavior Model (LBM) for the dataset's natural task: predict a participant's held-out wave-4 responses from information collected in waves 1–3.

The main design constraint is temporal data integrity: a wave-4 answer must never enter the model input used to predict that answer. The research plan reports two input conditions separately:

- **No-copy (primary):** excludes earlier answers to questions repeated in wave 4.
- **Full-history (secondary):** allows those **earlier** answers as temporally valid history and compares the model against the copy-last baseline.

## Deliverables

| Deliverable | File | Scope |
|---|---|---|
| 1 — Data exploration | [`docs/01_data_exploration.md`](docs/01_data_exploration.md) | Dataset structure, leakage risks, human test–retest benchmark, representativeness, scoring, and limitations |
| 1 — Analysis notebook | [`notebooks/data_exploration.ipynb`](notebooks/data_exploration.ipynb) | Reproducible exploratory analysis supporting Deliverable 1 |
| 2 — Model plan | [`docs/02_model_plan.md`](docs/02_model_plan.md) | Data contract, model progression, retrieval, QLoRA fine-tuning, context budget, and risks |
| 3 — Evaluation strategy | [`docs/03_eval_strategy.md`](docs/03_eval_strategy.md) | Metrics, baselines, participant split, uncertainty, leakage tests, and acceptance criteria |
| 4 — Business applications | [`docs/04_business_apps.md`](docs/04_business_apps.md) | Product concepts, suitable organizations, decision boundaries, and guardrails |
| 5 — Maintenance | [`docs/05_maintenance.md`](docs/05_maintenance.md) | Monitoring, drift, retraining triggers, versioning, governance, and incident response |
| 6 — Bonus POC | [`docs/06_poc.md`](docs/06_poc.md) | Commands, three completed runs, and limitations for the runnable leakage-safe prototype |

Deliverables 2–5 describe a research-scale system. Deliverable 6 is intentionally a smaller proof of concept and does not implement the complete evaluation protocol in Deliverable 3.

## Repository structure

```text
.
├── README.md
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
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── build_jsonl.py             # Build person-split, leakage-safe POC JSONL
│   ├── baselines.py                   # Random, train-majority, and copy-last baselines
│   ├── leak_test.py                   # Prompt-level leakage checks
│   ├── train.py                       # QLoRA SFT; default Qwen2.5-0.5B-Instruct
│   └── evaluate.py                    # Slice evaluation and optional adapter inference
│
├── tests/
│   └── test_poc.py                    # 7 unit tests in 4 groups: scoring, baseline, leakage, split
│
├── data/
│   ├── wave4_response.csv             # Wave-4 numeric responses
│   ├── wave4_response_label.csv       # Wave-4 label-form responses
│   └── poc/                           # POC JSONL used for the reported runs
│
├── raw_data/                          # Reference copies only; not used for reported numbers
│
├── runs/                              # Final LoRA adapters and bundle manifests
└── results/                           # Notebook export and POC metrics.json
```

The submission path is `docs/`, `notebooks/`, `src/`, `tests/`, plus the reported `data/poc/`, `runs/`, and `results/` artifacts. `raw_data/` and `data/wave4_response*.csv` are reference copies of the public dataset. The notebook and the POC load the pinned Hugging Face revision; the reported statistics do not come from those local copies. Intermediate Trainer checkpoints (`runs/**/checkpoint-*`) are not uploaded because of GitHub size limits; I will share a download link for those checkpoints if requested.

## Setup

Use Python 3.10 in a conda environment:

```bash
conda create -n Amity_Assignment python=3.10 -y
conda activate Amity_Assignment
python -m pip install -r requirements.txt
```

`requirements.txt` lists `torch` but does not pin a CUDA wheel. After the commands above, install a CUDA build from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) that matches this machine. The local POC used PyTorch `2.14.0+cu126`; pick the equivalent wheel for Linux, Windows, or Colab. A CUDA 12.6 example is:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

A pip wheel is enough; a full CUDA Toolkit is not required. `--qlora` needs a CUDA GPU and `bitsandbytes`. Data preparation, leakage tests, unit tests, and non-model baselines can run on CPU.

## Reproduce Deliverable 1

Open [`notebooks/data_exploration.ipynb`](notebooks/data_exploration.ipynb) in Jupyter, VS Code, or Cursor and run all cells. Select the `Amity_Assignment` conda kernel so the notebook uses the environment from Setup.

The notebook loads the Twin-2K-500 Hugging Face dataset on first run (internet required) and writes generated tables or exports under `results/`.

## Run the bonus POC

Default local loop (Experiment 1 in [`docs/06_poc.md`](docs/06_poc.md)). From the repository root:

```bash
python -m unittest discover -s tests
python -m src.data.build_jsonl --out_dir data/poc --poc_train_pids 500 --max_items_per_pid 20
python -m src.leak_test data/poc/leak_fixture.jsonl data/poc/examples_train.jsonl data/poc/examples_val.jsonl
python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc/adapter --out_dir results/poc
```

To check results quickly with the already-trained models, open `results/*/metrics.json` or run tests and evaluate:

```bash
python -m unittest discover -s tests
python -m src.leak_test data/poc/leak_fixture.jsonl data/poc/examples_train.jsonl data/poc/examples_val.jsonl
python -m src.evaluate --train_jsonl data/poc/examples_train.jsonl --test_jsonl data/poc/examples_val.jsonl --diag_jsonl data/poc/diag_val.jsonl --leak_jsonl data/poc/leak_fixture.jsonl --adapter_dir runs/poc/adapter --out_dir results/poc
```

Use `--adapter_dir runs/poc_e3_lr1e4/adapter --out_dir results/poc_e3_lr1e4` for Experiment 2. That directory name means 3 epochs, not Experiment 3. Use `--adapter_dir runs/smollm360m_e3/adapter --out_dir results/smollm360m_e3` for Experiment 3. Intermediate Trainer checkpoints are not uploaded here because of GitHub size limits; I will share a download link if requested. If the base model is not already cached, the code downloads it automatically on the first `evaluate` run: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) for Experiments 1–2, or [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) for Experiment 3. It does not reload LoRA from a training run. GPU is faster; CPU works but is slow. Experiment 3's published scores used JSONL rebuilt on Colab, so a local re-score of that adapter on `data/poc` will be close but not identical. The POC implements the **no-copy** condition. It uses non-overlapping waves 1–3 fields for the persona, written as unlabeled column codes such as `QID12: 1`, and it prints MC options as text while the supervised target is a numeric code. Copy-last data stays in diagnostic files only. The builder never loads `full_persona` for model prompts.

The reported experiments are a local Qwen2.5-0.5B-Instruct schedule, a longer lower-LR Qwen schedule, and a Colab `SmolLM2-360M-Instruct` run. Both published base-model counts are under 0.5B. The slice is multiple-choice only, so the headline is exact match. Slice MAD, one minus the absolute code error divided by the train-only empirical range, is a secondary number: 19 of the 68 columns have a train-observed code range above 1, so MAD gives partial credit for a nearby code. Scores are on the validation slice. `data/poc/examples_test.jsonl` was built and not scored. This is not a reproduction of the official 17-task paper evaluation. See [`docs/06_poc.md`](docs/06_poc.md) for the three-experiment table, the prompt limitation, and interpretation rules.

## Further work

The research-scale priorities are in [`docs/02_model_plan.md`](docs/02_model_plan.md) §8: compare 1.5B, 7B, and 14B on the same persona and retrieval pipeline, train a supervised retriever, test task-specific adapters, calibrate abstention, and measure drift on a later wave than this short retest.

For the POC, the next run would fix the option legend and persona text before any larger model, then score exact match on the locked test split with participant-bootstrap intervals. A model strictly smaller than `0.5B` parameters would count only if its paired interval over train-majority stayed above zero.