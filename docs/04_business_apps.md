# Deliverable 4 — Business applications

## Questions

1. What the model’s job is
2. Which products / organizations can use this
3. What it must **not** be used for

## Solution

Part 1 is the overall summary (same numbering as the questions). Part 2 is the detail.

**References**

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions.* [arXiv:2505.17479](https://arxiv.org/abs/2505.17479). Intended uses and limits: Introduction (pilot experiments; customer insight / product development); Conclusion (US / social-science scope; risks of dehumanizing research and over-reliance on AI).
- Twin-2K-500 dataset card, **Considerations for Using the Data** — Social Impact, Discussion of Biases, Other Known Limitations. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- Sample vs ACS 2023 gaps: `notebooks/data_exploration.ipynb` §4, §7.

### Part 1 — Overall summary

1. **Job.** Condition on a consenting adult’s prior survey answers (waves 1–3, leakage-safe) → predict a **survey code** for one Twin-2K-500-style item. Score against a later **survey**, not against observed behavior. Survey twin, not person twin. The HF card states responses are **self-reported** and “may not always accurately reflect actual behaviors”; the paper scores twins against **survey** retest, not clickstream or purchase.

2. **Who uses it.** The card’s Social Impact and the paper’s Introduction name two buyers: **researchers** (theory development, experimental design, silicon-sample pilots) and **practitioners** (customer insights, product development). Operationally, survey / UX / product-research teams **pretest wording** on a twin panel, then still run a human pilot. Insights teams run a **first-pass** only on constructs close to this battery (pricing, heuristics, values) — directional, not a board number. Not banks, HR, clinics, or targeting desks. Described as representative (assignment / card); ACS 2023 still shows gaps (under 65+, under low education, over college). That description is not a license to sell “the US public.”

3. **Guardrails (do not).** No credit, insurance, hiring, medicine, legal, or individual targeting — the paper warns against **excessive reliance on AI in decision-making**. No impersonating the person (dehumanizing research). No selling per-person twin files. No deploying off this sample without new data (card: **specific geographic context**; paper: US-focused). No treating codes as what people *do* (card: self-selection + social-desirability bias; self-report ≠ behavior). No pitching scores above the human 2-week ceiling (D3: leak alarm). Charts that leave the lab: **simulated survey response, not observed behavior.**

---

### Part 2 — Detail

#### 2.1 Job of the model

| | In | Out |
|---|---|---|
| **Input** | Leakage-safe persona (`wave_split` waves 1–3); one stripped catalog item | `full_persona`; first-round answer to the *same* item as a feature |
| **Output** | Canonical survey code (option index, 0–100 slider, …) | “Will buy / will vote / is risky”; a prose biography |
| **Ground truth** | A later **survey** answer (paper: wave-4 retest) | Clickstream, purchase, clinical outcome |
| **Horizon we measured** | ~2 weeks (paper wave 4; D1/D3) | Months/years (card: “specific point in time”) |

Toubia et al. (2025, Figure 2): twins **71.72%**; test–retest ceiling **81.72%**; paper-stated ratio **87.67%** (not 71.72÷81.72 of those rounded percentages; we did not recompute). That is not evidence the twin is the person. Copy-last already wins on stable items; the model is only useful where people **change** — wave 4 mostly *repeats* heuristics/pricing, so “any new question” is not what we showed.

---

#### 2.2 Products / organizations

Mapped to the card’s Social Impact and the paper’s Introduction — not invented verticals.

| Use | Organization | What they do | Output they act on | Source |
|---|---|---|---|---|
| **Questionnaire / experiment pretest** | Academic labs; survey, UX, product research | Compare two wordings or stimuli on the same twin panel | Which wording is noisier / more polarized. Then a **human** pretest before fielding | Paper: silicon samples for **pilot experiments** and experimental design. Card: researchers, theory development |
| **First-pass customer insight** | Insights / product teams already on pricing, heuristics, values | Ask items **close to** this battery to design the real instrument | Directional ranking, not market share; not an external report | Paper + card: **customer insights** and **product development** |
| **Methods / academic** | A lab (this take-home) | Twin vs human vs copy-last vs ceiling, by question type | When an LBM adds value | Paper: public **testbed** for LLM persona simulations |

**Who this is *not* for, even as a customer.** A buyer whose population is not this panel (non-US, 65+, low education, people who do not take web surveys) should collect a new persona survey and re-evaluate — not “fine-tune and ship.” Card: geographic context. Paper: US / social-science scope is a limitation. Versus ACS 2023 (our EDA, not the card): 65+ 13.5% vs 22.6%; less than high school 0.8% vs 10.2%; high school only 13.2% vs 25.9%; college / some postgrad 35.7% vs 21.8%; income $30k–$50k 20.0% vs 14.3% (ACS bin interpolated; notebook).

---

#### 2.3 Guardrails — do not use the model for

| Do not | Why, on this dataset / these sources |
|---|---|
| Credit, insurance, hiring, medical, legal, welfare | Individual high-stakes action. Paper conclusion: **excessive reliance on AI in decision-making**. Paper also: twins diverged from humans on **medical** (outcome / omission bias) and **political** items |
| Individual / political targeting | Panel ≠ voter file; survey ≠ behavior |
| Impersonation (twin *is* the user) | Consent was to a survey, not a stand-in. Paper: risk of **dehumanization of research** |
| Selling per-`pid` twin files | Re-identification on a 2k psychometric panel |
| Pitch “US representative” or off-sample deploy | Assignment / card use that description; ACS 2023 marginals differ on education, age, income. Card: specific **geographic context**. Paper: US-focused |
| Pitch “we beat the human ceiling” | D3 treats that as leak, not SOTA |
| Treat answers as revealed behavior | Card **Discussion of Biases:** self-selection, social desirability. Card **Other Known Limitations:** self-report may not reflect actual behavior. D1 §7 |
| Train other models on twin outputs as labels | Launders survey bias |

Required line on any chart that leaves the research team:

> Simulated survey responses. Not observed behavior. US online panel, not population-weighted. Horizon evidenced: ~2 weeks.
