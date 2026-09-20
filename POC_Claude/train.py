"""
src/train.py

LoRA-SFT fine-tune of a <0.5B instruct model on the Twin-2K-500 behavior
task. Hyperparameters follow docs/02_model_plan.md §2.5 exactly.

    python src/train.py \
        --train_jsonl data/examples_train.jsonl \
        --val_jsonl   data/examples_val.jsonl \
        --out_dir     runs/poc_0.5b

Expected runtime: < 2h on a single T4 (free Colab) for the POC slice
(500 pids x <=20 items, ~10k rows -- see docs/02_model_plan.md §2.5).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05

MAX_LENGTH = 2048
SEED = 42


def git_short_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def make_bundle_id() -> str:
    """Immutable bundle id, per docs/05_maintenance.md §2.4. A POC run gets
    one too, so results are traceable the same way production ones would be.
    """
    return f"{date.today():%Y%m%d}-{git_short_hash()}"


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model_and_tokenizer(model_name: str, use_qlora: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_kwargs = {}
    if use_qlora:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **quant_kwargs)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def build_supervised_dataset(examples: list[dict], tokenizer) -> Dataset:
    """One example = one (pid, col) wave-4 item -- never concatenate
    multiple wave-4 items in one sequence (docs/02_model_plan.md §2.4):
    that would let later items condition on earlier gold/predicted
    wave-4 answers inside the same context.

    Loss is computed on the answer tokens only: prompt tokens are masked
    with label = -100 so the model is scored on producing the code, not
    on reconstructing the persona.
    """

    def _tokenize(ex):
        prompt = ex["prompt"] + "\n"
        target = ex["target"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, truncation=True, max_length=64, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    ds = Dataset.from_list(examples)
    return ds.map(_tokenize, remove_columns=ds.column_names)


class PadCollator:
    """Pads input_ids/labels/attention_mask to the batch max length.
    Kept separate from DataCollatorForLanguageModeling since our labels
    are already masked (-100 on the prompt) and must not be re-derived
    from input_ids (mlm=False would otherwise just copy input_ids).
    """

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attn.append(f["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/poc")
    ap.add_argument("--model_name", type=str, default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)  # effective batch 32
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--qlora", action="store_true", help="Use NF4 QLoRA (for GPUs < 16GB).")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_id = make_bundle_id()
    print(f"bundle_id: {bundle_id}")

    print(f"Loading base model: {args.model_name}")
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_name, use_qlora=args.qlora)
    except Exception as e:
        print(f"Failed to load {args.model_name} ({e}); falling back to {FALLBACK_MODEL}")
        model, tokenizer = load_model_and_tokenizer(FALLBACK_MODEL, use_qlora=args.qlora)
        args.model_name = FALLBACK_MODEL

    train_examples = load_jsonl(args.train_jsonl)
    val_examples = load_jsonl(args.val_jsonl)
    print(f"train examples: {len(train_examples)}  val examples: {len(val_examples)}")

    train_ds = build_supervised_dataset(train_examples, tokenizer)
    val_ds = build_supervised_dataset(val_examples, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to=[],
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer),
    )

    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    manifest = {
        "bundle_id": bundle_id,
        "base_model": args.model_name,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "targets": LORA_TARGET_MODULES},
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "epochs": args.epochs,
        "seed": SEED,
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved adapter + bundle manifest to {out_dir}")


if __name__ == "__main__":
    main()
