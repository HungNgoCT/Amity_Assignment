# Large Behavior Model on Twin-2K-500

This repository contains the solution and design framework for the **Research Engineer Take-Home Assignment**. The goal is to design, analyze, and prototype a Large Behavior Model (LBM) capable of predicting individual human responses across time using the **Twin-2K-500** dataset.

---

## 📌 Project Overview

Digital twins attempt to emulate human behavior based on historical responses. Using the `Twin-2K-500` dataset (N=2,058 US adults across 4 waves), this project explores:
1. **Persona Conditioning**: Predicting Wave 4 responses given Wave 1–3 data.
2. **Human Test-Retest Ceiling**: Benchmarking model performance against true human consistency over time.
3. **Data Leakage Mitigation**: Identifying and handling re-administered questions and overlapping constructs.

---

## 📂 Repository Structure

```text
.
├── README.md                  # Main orientation and setup guide
├── LICENSE
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files/folders to ignore in Git
│
├── docs/                      # Core Design & Analysis Deliverables
│   ├── 01_data_exploration.md # [Deliverable 1] Short answers (code: notebooks/data_exploration.ipynb)
│   ├── 02_model_plan.md       # [Deliverable 2] Architecture, Fine-tuning, & Context Window Strategy
│   ├── 03_eval_strategy.md    # [Deliverable 3] Metrics, Baselines, & Data Leakage Prevention
│   ├── 04_business_apps.md    # [Deliverable 4] Use Cases, Guardrails & Ethical Boundaries
│   ├── 05_maintenance.md      # [Deliverable 5] Long-term Maintenance, Drift, & Governance
│   └── 06_poc.md              # [Deliverable 6] Bonus POC slice, commands, honesty notes
│
├── notebooks/                 # Exploratory Data Analysis Code
│   └── data_exploration.ipynb # Notebook generating figures/metrics for Deliverable 1
│
└── src/                       # [Deliverable 6 - Bonus] Leak-safe POC (5 modules)
    ├── leak_test.py           # pid=1 / QID154: persona has neither 70 nor 82
    ├── data/
    │   └── build_jsonl.py     # wave_split + CSVs → JSONL; never loads full_persona
    ├── baselines.py           # random / majority / copy-last (copy-last never uses gold)
    ├── train.py               # Qwen2.5-0.5B-Instruct LoRA-SFT
    └── evaluate.py            # Baselines + optional adapter; slice MAD, not 17-task
```

POC commands: `docs/06_poc.md`. Do not load `full_persona` for prompts.