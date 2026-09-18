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
│   ├── 01_data_exploration.md # [Deliverable 1] EDA, Human Test-Retest Ceiling & Biases
│   ├── 02_model_plan.md       # [Deliverable 2] Architecture, Fine-tuning, & Context Window Strategy
│   ├── 03_eval_strategy.md    # [Deliverable 3] Metrics, Baselines, & Data Leakage Prevention
│   ├── 04_business_apps.md    # [Deliverable 4] Use Cases, Guardrails & Ethical Boundaries
│   ├── 05_maintenance.md      # [Deliverable 5] Long-term Maintenance, Drift, & Governance 
│   └── 06_future_work         # [Deliverable 6] Optional note
│
├── notebooks/                 # Exploratory Data Analysis Code
│   └── data_exploration.ipynb # Notebook generating figures/metrics for Deliverable 1
│
└── src/                       # [Deliverable 6 - Bonus] Minimal Proof-of-Concept (POC)
    ├── data/
    │   └── loader.py          # Hugging Face dataset loader & preprocessor
    ├── train.py               # Fine-tuning script (<0.5B model)
    └── evaluate.py            # Evaluation pipeline vs. trivial baselines