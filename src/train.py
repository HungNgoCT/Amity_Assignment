"""LoRA-SFT on the POC JSONL. Hyperparameters: docs/02_model_plan.md §2.5.

    python -m src.train --train_jsonl data/poc/examples_train.jsonl --val_jsonl data/poc/examples_val.jsonl --out_dir runs/poc --qlora
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import subprocess
from datetime import date
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.leak_test import assert_prompts_leak_free, load_jsonl

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
MAX_LENGTH = 2048
SEED = 42


def bundle_id() -> str:
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        h = "nogit"
    return f"{date.today():%Y%m%d}-{h}"


def git_is_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def cuda_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    # Ampere+ : bf16. T4 is 7.5 → fp16.
    use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    return use_bf16, not use_bf16


def load_model(model_name: str, qlora: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if qlora:
        if not torch.cuda.is_available():
            raise SystemExit("QLoRA needs CUDA. Drop --qlora, or run leak test + baselines only.")
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        dtype = torch.bfloat16 if cuda_flags()[0] else (torch.float16 if torch.cuda.is_available() else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


def tokenize_supervised(examples: list[dict], tokenizer) -> Dataset:
    def _tok(ex):
        target_ids = tokenizer(
            str(ex["target"]) + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        budget = max(32, MAX_LENGTH - len(target_ids))
        prompt_ids = tokenizer(
            ex["prompt"] + "\n",
            truncation=True,
            max_length=budget,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    ds = Dataset.from_list(examples)
    return ds.map(_tok, remove_columns=ds.column_names)


def make_training_args(n_train: int, **kwargs) -> TrainingArguments:
    """Transformers 4 vs 5: drop unknown kwargs; 5.x has warmup_steps, not warmup_ratio."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    if "warmup_ratio" in kwargs and "warmup_ratio" not in params:
        ratio = float(kwargs.pop("warmup_ratio"))
        if "warmup_steps" in params:
            eff = max(1, int(kwargs.get("per_device_train_batch_size", 1)) * int(kwargs.get("gradient_accumulation_steps", 1)))
            steps = max(1, (n_train + eff - 1) // eff) * int(kwargs.get("num_train_epochs", 1))
            kwargs["warmup_steps"] = max(1, int(ratio * steps))
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return TrainingArguments(**kwargs)


class PadCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        ids, labs, attn = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * n)
            labs.append(f["labels"] + [-100] * n)
            attn.append(f["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(attn),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--leak_jsonl", default="data/poc/leak_fixture.jsonl")
    ap.add_argument("--out_dir", default="runs/poc")
    ap.add_argument("--model_name", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--per_device_batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--qlora", action="store_true")
    args = ap.parse_args()

    train_ex = load_jsonl(args.train_jsonl)
    val_ex = load_jsonl(args.val_jsonl)
    leak_path = Path(args.leak_jsonl)
    if not leak_path.exists():
        raise SystemExit(f"Missing {leak_path}. Run: python -m src.data.build_jsonl")
    assert_prompts_leak_free(load_jsonl(leak_path), source=str(leak_path))
    assert_prompts_leak_free(train_ex, source=args.train_jsonl, require_hit=False)
    assert_prompts_leak_free(val_ex, source=args.val_jsonl, require_hit=False)

    torch.manual_seed(SEED)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bid = bundle_id()
    print("bundle_id", bid)

    model, tokenizer = load_model(args.model_name, args.qlora)
    use_bf16, use_fp16 = cuda_flags()
    targs_kw = dict(
        output_dir=str(out),
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
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to=[],
        seed=SEED,
    )
    targs = make_training_args(len(train_ex), **targs_kw)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenize_supervised(train_ex, tokenizer),
        eval_dataset=tokenize_supervised(val_ex, tokenizer),
        data_collator=PadCollator(tokenizer),
    )
    trainer.train()

    adapter = out / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    (out / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_id": bid,
                "base_model": args.model_name,
                "qlora": args.qlora,
                "train_examples": len(train_ex),
                "val_examples": len(val_ex),
                "epochs": args.epochs,
                "per_device_batch_size": args.per_device_batch_size,
                "gradient_accumulation_steps": args.grad_accum,
                "learning_rate": args.lr,
                "max_length": MAX_LENGTH,
                "lora": {
                    "r": LORA_R,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "target_modules": LORA_TARGETS,
                },
                "seed": SEED,
                "git_dirty": git_is_dirty(),
                "packages": package_versions(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", adapter)


if __name__ == "__main__":
    main()
