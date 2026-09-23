---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---

# Experiment 2 — Qwen2.5-0.5B-Instruct, 3 epochs

QLoRA adapter for the Twin-2K-500 no-copy proof of concept. The directory name `poc_e3_lr1e4` means 3 epochs and learning rate `1e-4`. This is Experiment 2, not the SmolLM Experiment 3.

- Bundle: `20260923-3e3eaf7`
- The manifest records `git_dirty: true`. Rescoring this saved adapter should match the published metrics. Retraining from a later tree will not reproduce the adapter bit for bit.
- Scores: [`results/poc_e3_lr1e4/metrics.json`](../../../results/poc_e3_lr1e4/metrics.json)
- Write-up: [`docs/06_poc.md`](../../../docs/06_poc.md)

This is a take-home slice, not a general-purpose chat model. Simulated survey responses are not observed behavior.
