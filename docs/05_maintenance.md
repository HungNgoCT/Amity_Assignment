# Deliverable 5 — Long-run maintenance

## Questions

1. Monitoring
2. Drift (people’s answers change over time)
3. Retraining triggers
4. Versioning
5. Governance

## Solution

Uses/guardrails stay in D4; metrics/gates stay in D3. This note is only the operating loop. Part 1 is the overall summary (same numbering as the questions). Part 2 is the detail.

**References**

- Toubia, O., Gui, G. Z., Peng, T., Merlau, D. J., Li, A., & Chen, H. (2025). *Twin-2K-500.* [arXiv:2505.17479](https://arxiv.org/abs/2505.17479). Wave 4 is a **~2-week** retest of heuristics/pricing, not a 6-month panel. Conclusion: US / social-science snapshot; risk of over-reliance on AI.
- Twin-2K-500 dataset card, **Other Known Limitations**: self-report; “specific **point in time** and geographic context.” [Hugging Face](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500).
- D3 gates and leak unit test: `docs/03_eval_strategy.md`. Sample vs ACS: `notebooks/data_exploration.ipynb` §4, §7.

### Part 1 — Overall summary

1. **Monitoring.** Reuse the D3 dashboard in production, not a new metric. Watch: leak unit test (must stay green), parse rate, **absolute** task-mean MAD, **MAD / current human ceiling**, copy-last gap on items where people *changed*, plus type slices (MC / slider / anchoring). Do not watch a single exact-match average. Cadence: leak test on every build; MAD vs a frozen labeled slice weekly; full 17-task table when a new human wave lands. The ratio is not a stand-alone health score: if the human ceiling falls (longer gap than two weeks) and the model is unchanged, MAD / ceiling **rises**, so the ~0.80 band looks easier to hit and a ratio **> 1.0** can fire without a new leak. Always publish MAD next to the **current** ceiling; alert on an **absolute** MAD drop vs the last signed bundle as well.

2. **Drift (how people change).** The paper’s ceiling (**81.72%** MAD) is two weeks. A 6-month product should assume a **lower** human ceiling — that is not model failure, it is the data’s time window. Four drifts: **person** (preferences move); **item / world** (prices, politics, medical news); **sample** (panel vs ACS, who still answers); **stack** (base LLM or prompt change). Detect with a small **retest wave** on a held-out pid set, same items, same MAD — not with twin-vs-twin.

3. **Retraining triggers (written before we look).** Stop serving if the leak test is red. Retrain (or collect data) if: MAD / ceiling on a new retest wave falls below the D3 band (~**0.80**) **and** absolute MAD is not merely “looking better” because the ceiling dropped; copy-last beats us again on *changed* items; catalog / wording of scored items changes; base model or prompt template changes; ACS-style sample gaps move enough that D4’s “who this is for” is wrong. Do **not** retrain on twin outputs as labels (D4).

4. **Versioning.** Ship an immutable bundle: weights + prompt/retrieval + `pid` split (seed `20250319`) + data snapshot hash + MAD script / ranges + train-only decile cuts. Every number in a report cites that bundle. New human answers never quietly enter the persona for the same item (trap 1, forever).

5. **Governance.** D4 hard no’s still apply after a retrain. A named human signs off before a new bundle is used outside the research team. Changing the split seed, the MAD script, or “what counts as wave-4” is a reviewed change, not a silent notebook edit. Consent is for **new survey waves**, not for scraping the person later.

---

### Part 2 — Detail

#### 2.1 Monitoring

Same units as D3 so a weekly plot is comparable to the paper and to the take-home.

| Signal | Why | Alert |
|---|---|---|
| Leak unit test (pid=1 / `QID154`: no 70, no 82) | Trap 1–2 do not expire when we retrain | Any fail → stop serving (§2.3) |
| Parse rate | Unparseable codes inflate “error” that is not behavior | &lt; 80% of non-null items |
| Task-mean MAD + 17-task table | Headline the paper uses; absolute floor so a falling ceiling cannot hide a worse model | Drop **> 2 pp** vs last signed bundle (plan threshold, not a paper number) |
| MAD / **current** human ceiling | Ceiling must be re-estimated when the retest gap is no longer two weeks | Ratio &lt; ~0.80 **after** checking absolute MAD. Ratio **> 1.0** is a leak alarm only if the ceiling is still the 2-week retest; a longer-gap ceiling can put an unchanged model above 1.0 without trap 1–2 |
| Copy-last on items with wave *t* ≠ wave *t+1* | Product value is change, not copying | We lose this slice |
| Type slices | Slider exact-match will always look terrible | One type collapses while MAD looks fine |

Optional, not gates: education / 65+ cells from D1 — only if *n* allows.

---

#### 2.2 Drift

The HF card already says the data are a **point in time**. Wave 4 does not prove 12-month stability.

| Drift | What moves | How we see it | What we do |
|---|---|---|---|
| **Person** | Same pid, same item, later answer ≠ 2-week answer | New retest MAD **human vs self** falls below 81.72% | Lower the ceiling we compare twins to; report **absolute** MAD too. Do not call the model “better” just because MAD / ceiling rose when the denominator fell |
| **Item / world** | Prices, news, medical/political items (paper: twins already diverge there) | Item-level MAD drop concentrated on a block | Freeze or retire that block until a new human wave |
| **Sample** | Who is still on the panel vs ACS | Repeat D1 §4 tables | Stop selling “US panel” if gaps blow out; collect a new sample |
| **Stack** | New base LLM, tokenizer, prompt, retrieval *k* | Same labeled slice, new bundle, MAD moves | Treat as a new model; run full D3 once |

**How we keep up with people changing:** we do not “fine-tune on last week’s twins.” We field a **small human retest** (same protocol as wave 4: same condition, labeled CSV) on held-out pids, recompute the human ceiling, then decide whether to retrain the LBM on *older* waves only — newest wave stays hold-out, same trap logic as D2/D3.

---

#### 2.3 Retraining triggers

Written down before looking at next quarter’s scores.

| Trigger | Action |
|---|---|
| Leak test red | **Stop serving.** Find 70/82 in prompts. Do not publish a MAD |
| Task-mean MAD drop **> 2 pp** vs last signed bundle | Retrain or collect data even if MAD / ceiling still looks fine (ceiling may have fallen) |
| MAD / ceiling &lt; ~0.80 on a new retest wave | Retrain SFT (D2 recipe) or collect more persona (non-hold-out) items — **after** updating the ceiling to the current retest gap. Do not keep dividing by 81.72 |
| Copy-last wins on *changed* items | DPO against last-round answers, or admit the LBM is not adding value |
| Scored catalog / wording change | Rebuild JSONL; refit deciles on train pids; new bundle |
| Base model or prompt change | Full D3 on val; one test pass |
| Sample no longer matches D4’s buyer | Do not retrain our way out of it — new recruitment |

Never: mix the evaluation wave into `persona_*` for those items; train on model-generated codes as gold.

---

#### 2.4 Versioning

```text
bundle_id:  YYYYMMDD-<short-hash>
  model:    adapter / base ids + commit
  prompt:   template id + retrieval k
  data:     HF revision + CSV hashes
  split:    seed 20250319, pid lists
  scoring:  mad_accuracy_evaluation.py commit + R_c table
  deciles:  train-pid cutpoints file
```

Reports quote `bundle_id`. Two MAD numbers from different bundles are not averaged. Rolling the split seed to “get a better test” is forbidden without a written change.

When a new human wave arrives: it is **labels + new ceiling**, or it is **persona**, never both for the same `(pid, item)` in the same bundle.

---

#### 2.5 Governance

| Rule | Owner | Escalation |
|---|---|---|
| D4 hard no’s (credit, hiring, medical, impersonation, selling `pid` files) | Named research owner | Still in force after every retrain; no silent waiver |
| Chart that leaves the research team carries the D4 disclosure line | Named reviewer | Missing line → pull the chart |
| New survey wave = new consent; no social-media scrape into persona | Data owner | Legal / privacy if scrape is requested |
| Leak test in CI of `src/evaluate.py` (if D6 exists) | Engineering | Red CI → stop serving (§2.3) |
| Red retraining trigger (§2.3) with no action | Named research owner | Still red at the next weekly MAD check → backup reviewer; bundle stays **stopped** (no silent keep-serving) |
| “Beat the human ceiling” is not a KPI | Same as D3 | Ratio > 1.0 on a 2-week ceiling → leak review, not a press line |

Ethics here is operational: the 2-week ceiling will **look** worse at six months. The honest move is to publish the new human ceiling next to the twin, not to hide the drop or to refill labels from the model.
