# Deliverable 4 — Business applications

## Requirements addressed

1. The model's concrete job
2. Products and organizations that could use it
3. Uses that must remain out of scope

## Proposed business applications

The summary follows the same numbering as the requirements above, followed by detailed use cases and guardrails.

### Application summary

1. **Job.** Condition on one participant's leakage-safe waves 1–3 survey answers and predict a **survey response code** for one Twin-2K-500-style item. Score the prediction against a later **survey response**, not observed behavior. This is a survey-response twin, not a complete person twin. The dataset card states that responses are **self-reported** and “may not always accurately reflect actual behaviors”; the paper evaluates twins against a survey retest, not clickstream or purchase data.

2. **Products and organizations.** Three concrete product concepts fit the demonstrated capability. A **survey and experiment pretesting tool** would help academic labs, survey teams, and user-experience (UX) researchers generate hypotheses about candidate questions or stimuli before a human pilot. A **customer-research sandbox** would help product and insights teams explore pricing, heuristics, or values close to the observed survey battery when designing a real study. A **behavior-model evaluation platform** would help artificial intelligence (AI) labs and survey-methodology researchers compare prompting, retrieval, and fine-tuning methods against human test–retest and trivial baselines. These products produce research hypotheses and directional evidence, not decision-grade population estimates. The assignment and dataset card describe the sample as representative, but comparison with the 2023 American Community Survey (ACS) still shows age, education, and income gaps. This does not justify presenting the model as a simulation of the entire U.S. population.

3. **Guardrails (do not).** Do not use the model for credit, insurance, hiring, medicine, legal decisions, public-benefit eligibility, or individual targeting. The paper warns against excessive reliance on AI in decision-making, and a risk-based deployment policy should keep these high-impact uses out of scope (National Institute of Standards and Technology, 2023). Do not impersonate participants, productize participant-level twin files, deploy to a different population without new human validation, treat predicted survey codes as observed behavior, or claim causal effects from twin-only experiments. An unexpectedly above-benchmark result is an audit trigger, not a marketing claim. Any chart shared outside the research team must say: **simulated survey responses, not observed behavior**.

---

### Detailed applications and guardrails

#### 2.1 Job of the model

| Aspect | Within scope | Outside scope |
|---|---|---|
| **Input** | Leakage-safe `wave_split` persona under a declared no-copy or full-history policy; one stripped catalog item | `full_persona`; an unstripped wave-4 payload; silently mixing the two input policies |
| **Output** | Canonical survey code (option index, 0–100 slider, …) | “Will buy / will vote / is risky”; a prose biography |
| **Ground truth** | A later **survey** answer (paper: wave-4 retest) | Clickstream, purchase, clinical outcome |
| **Horizon evidenced** | Short-term retest: wave 4 launched approximately two weeks after wave 3; items originating in waves 1–2 have longer intervals | Reliable prediction over months or years |

Toubia et al. (2025, Figure 2) report full-sample twins at **71.72%**, human test–retest at **81.72%**, and a stated ratio of **87.67%**; these published figures were not recomputed here. They do not establish that a twin is the person. In the full-history condition, incremental value over copy-last is clearest on items where the participant changed their answer. The no-copy condition asks a different question: whether the rest of the persona predicts the held-out response without the same-item earlier answer. Wave 4 primarily repeats heuristics and pricing tasks, so this evidence does not support claims about arbitrary new questions.

---

#### 2.2 Product concepts and organizations

The following products directly answer what could use the model's capabilities and how. The first two are grounded in the customer-insight, product-development, pilot-experiment, and experimental-design uses named by the paper and dataset card; the third operationalizes the dataset's role as a testbed for persona-model research.

| Product concept | Primary organizations | Workflow | Output and decision boundary |
|---|---|---|---|
| **Survey and experiment pretesting tool** | Academic labs; survey, UX, and product-research teams | Submit candidate questions or stimuli, run them on the same simulated panel, and identify differences worth investigating | Hypotheses about wording or stimulus effects to verify in a **human** pretest; not a substitute for randomized human evidence |
| **Customer-research sandbox** | Product and insights teams working on pricing, heuristics, or values | Explore items close to the observed survey battery before designing the real questionnaire or study | Directional patterns and candidate questions; not market share, demand forecasts, or a decision-grade external estimate |
| **Behavior-model evaluation platform** | AI labs and survey-methodology researchers | Compare prompting, retrieval, and fine-tuning against human test–retest, train-majority, random, and copy-last baselines by question type | Reproducible evidence about when a Large Behavior Model (LBM) adds predictive value; not a claim that a twin is the person |

**Who this is *not* for, even as a customer.** A buyer whose population is not this panel (non-U.S., 65+, low education, or people who do not take web surveys) should collect a new persona survey and re-evaluate rather than “fine-tune and ship.” The dataset card identifies geographic context as a limitation, and the paper describes a U.S.-focused social-science scope. Deliverable 1's ACS comparison is the evidence: age, education, and income gaps are large enough that this panel is not a stand-in for those groups. The comparability limits of that table stay in Deliverable 1.

---

#### 2.3 Guardrails — do not use the model for

| Do not | Why, on this dataset / these sources |
|---|---|
| Credit, insurance, hiring, medical, legal, welfare | Individual high-stakes action. Paper conclusion: **excessive reliance on AI in decision-making**. Paper also: twins diverged from humans on **medical** (outcome / omission bias) and **political** items |
| Individual / political targeting | Panel ≠ voter file; survey ≠ behavior |
| Impersonation (twin *is* the participant) | The dataset documentation does not establish consent for impersonation or automated decision-making. Paper: risk of **dehumanization of research** |
| Productizing or distributing per-participant twin files identified by `pid` | Rich demographic and psychometric profiles create linkage and re-identification risk even when direct identifiers are removed (National Institute of Standards and Technology, 2020) |
| Pitch “U.S. representative” or deploy off-sample | Assignment and card use that description; ACS 2023 marginals differ on education, age, and income. Card: specific **geographic context**. Paper: U.S.-focused |
| Pitch “we beat the human ceiling” | Deliverable 3 treats an unexpectedly above-benchmark result as a leakage and aggregation audit trigger, not evidence of product superiority |
| Treat answers as revealed behavior | Card **Discussion of Biases:** self-selection and social desirability. Card **Other Known Limitations:** self-report may not reflect actual behavior. See Deliverable 1, **Biases / limitations** |
| Claim causal effects from twin-only experiments | Simulated potential outcomes are not randomized human outcomes; use them only to generate hypotheses for a human experiment |
| Train other models on twin outputs as labels | Propagates and obscures survey and model bias; human responses should remain the ground-truth labels |

Note: Required line on any chart that leaves the research team:

> Simulated survey responses. Not observed behavior. U.S. online panel, not population-weighted. Short-term evidence only: wave 4 launched approximately two weeks after wave 3, with longer intervals for items originating in earlier waves.

## References

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500: A Dataset for Building Digital Twins of over 2,000 People Based on Their Answers to over 500 Questions*. [arXiv:2505.17479](https://arxiv.org/abs/2505.17479).
- LLM-Digital-Twin. *Twin-2K-500 dataset card and data files*. [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- National Institute of Standards and Technology. (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. [https://doi.org/10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1).
- National Institute of Standards and Technology. (2020). *NIST Privacy Framework: A Tool for Improving Privacy through Enterprise Risk Management, Version 1.0*. [https://doi.org/10.6028/NIST.CSWP.01162020](https://doi.org/10.6028/NIST.CSWP.01162020).
- U.S. Census Bureau. *2023 American Community Survey 1-Year Estimates*: [S0101](https://data.census.gov/table/ACSST1Y2023.S0101) (age), [S1501](https://data.census.gov/table/ACSST1Y2023.S1501) (education), and [S1901](https://data.census.gov/table/ACSST1Y2023.S1901) (income).
- Sample-to-ACS benchmark construction and limitations: [notebooks/data_exploration.ipynb](../notebooks/data_exploration.ipynb).
