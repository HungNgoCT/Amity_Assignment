---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---

# Experiment 1 — Qwen2.5-0.5B-Instruct, 2 epochs

QLoRA adapter for the Twin-2K-500 no-copy proof of concept. It predicts one held-out wave-4 multiple-choice code from unlabeled waves 1–3 column codes.

- Bundle: `20260921-3e3eaf7`
- Training used the script defaults: 2 epochs and learning rate `2e-4`. This older manifest does not store the learning rate.
- Scores: [`results/poc/metrics.json`](../../../results/poc/metrics.json)
- Write-up: [`docs/06_poc.md`](../../../docs/06_poc.md)

This is a take-home slice, not a general-purpose chat model. Simulated survey responses are not observed behavior.
