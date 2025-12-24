from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load a JSONL file (one JSON object per line)."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _tokenize_example(
    tokenizer: AutoTokenizer,
    context: str,
    target: str,
    max_length: int,
) -> Dict[str, List[int]]:
    """Tokenize context/target and mask labels so only target contributes to loss."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False) if context else []
    tgt_ids = tokenizer.encode(target, add_special_tokens=False) if target else []

    if max_length <= 0:
        raise ValueError("max_length must be positive")

    if len(tgt_ids) >= max_length:
        input_ids = tgt_ids[-max_length:]
        labels = input_ids[:]
    else:
        available_ctx = max_length - len(tgt_ids)
        ctx_ids = ctx_ids[-available_ctx:]
        input_ids = ctx_ids + tgt_ids
        labels = [-100] * len(ctx_ids) + tgt_ids

    if tokenizer.bos_token_id is not None and len(input_ids) < max_length:
        input_ids = [tokenizer.bos_token_id] + input_ids
        labels = [-100] + labels
    elif tokenizer.bos_token_id is not None and len(input_ids) == max_length:
        # Avoid exceeding max_length when a BOS token is present.
        input_ids = [tokenizer.bos_token_id] + input_ids[:-1]
        labels = [-100] + labels[:-1]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class ChatDataset(Dataset):
    """Dataset that yields tokenized chat examples with masked context labels."""

    def __init__(self, rows: List[Dict[str, str]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.rows[idx]
        context = item.get("context", "").strip()
        target = item.get("target", "").strip()
        return _tokenize_example(self.tokenizer, context, target, self.max_length)


def _pad_sequences(
    sequences: List[List[int]],
    pad_value: int,
) -> List[List[int]]:
    """Pad sequences to the same length with the provided pad value."""
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]


class DataCollator:
    """Pad and batch tokenized examples for the Trainer."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = _pad_sequences(
            [f["input_ids"] for f in features], self.pad_token_id
        )
        attention_mask = _pad_sequences([f["attention_mask"] for f in features], 0)
        labels = _pad_sequences([f["labels"] for f in features], -100)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def maybe_wrap_lora(model, cfg: Dict[str, Any]):
    """Optionally wrap a model with LoRA adapters based on config."""
    lora_cfg = cfg.get("lora", {})
    if not lora_cfg.get("enabled", False):
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA enabled but peft is not installed") from exc

    config = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", None),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve a path relative to a base directory when value is not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def build_model(model_name: str, cfg: Dict[str, Any]):
    """Load a base model, optionally using 4-bit/8-bit quantization."""
    quant_cfg = cfg.get("quantization", {})
    quant_mode = str(quant_cfg.get("mode", "none")).lower()

    use_cuda = torch.cuda.is_available()
    if quant_mode in {"4bit", "8bit"} and not use_cuda:
        raise RuntimeError(
            "Quantization requires CUDA; set quantization.mode=none for CPU"
        )

    if quant_mode == "none":
        return AutoModelForCausalLM.from_pretrained(model_name)

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Quantization requires transformers with bitsandbytes support"
        ) from exc

    if quant_mode == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant_mode == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unknown quantization.mode: {quant_mode}")

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
    )


def apply_smoke_test_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Override config with tiny settings for quick local verification."""
    cfg = dict(cfg)
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["training"] = dict(cfg.get("training", {}))
    cfg["quantization"] = dict(cfg.get("quantization", {}))

    cfg["model"]["name_or_path"] = "sshleifer/tiny-gpt2"
    cfg["quantization"]["mode"] = "none"

    cfg["data"]["max_train_examples"] = 200
    cfg["data"]["max_length"] = 128

    cfg["training"]["max_steps"] = 20
    cfg["training"]["per_device_train_batch_size"] = 1
    cfg["training"]["logging_steps"] = 1
    cfg["training"]["save_steps"] = 20
    cfg["training"]["fp16"] = False
    cfg["training"]["bf16"] = False

    return cfg


def main() -> None:
    """Entrypoint for training with YAML-configured settings."""
    parser = argparse.ArgumentParser(description="Train fizzbot chat model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "train_config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny local training smoke test.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = args.config.resolve().parent
    if args.smoke_test:
        cfg = apply_smoke_test_overrides(cfg)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    model_name = model_cfg.get("name_or_path")
    if not model_name:
        raise ValueError("model.name_or_path is required in config")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=model_cfg.get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(model_name, cfg)
    model = maybe_wrap_lora(model, cfg)

    train_path = resolve_path(
        config_dir,
        data_cfg.get("train_jsonl", "train_data/training_examples.jsonl"),
    )
    rows = load_jsonl(train_path)
    max_train_examples = int(data_cfg.get("max_train_examples", 0))
    if max_train_examples > 0:
        rows = rows[:max_train_examples]

    max_length = int(data_cfg.get("max_length", 512))
    val_path_value = data_cfg.get("val_jsonl")
    val_split = float(data_cfg.get("val_split", 0.0))
    seed = int(train_cfg.get("seed", 42))

    if val_path_value:
        val_path = resolve_path(config_dir, val_path_value)
        val_rows = load_jsonl(val_path)
    elif val_split > 0:
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        cut = max(1, int(len(indices) * val_split))
        val_rows = [rows[i] for i in indices[:cut]]
        rows = [rows[i] for i in indices[cut:]]
    else:
        val_rows = []

    dataset = ChatDataset(rows, tokenizer, max_length=max_length)
    eval_dataset = (
        ChatDataset(val_rows, tokenizer, max_length=max_length) if val_rows else None
    )

    use_cuda = torch.cuda.is_available()
    use_fp16 = bool(train_cfg.get("fp16", False)) and use_cuda
    use_bf16 = bool(train_cfg.get("bf16", False)) and use_cuda

    base_output_dir = resolve_path(
        config_dir, train_cfg.get("output_dir", "llm/runs/fizzbot")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(base_output_dir / timestamp)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(
            train_cfg.get("per_device_train_batch_size", 4)
        ),
        per_device_eval_batch_size=int(
            train_cfg.get("per_device_eval_batch_size", 4)
        ),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 1)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        seed=seed,
        max_steps=int(train_cfg.get("max_steps", -1)),
        fp16=use_fp16,
        bf16=use_bf16,
        evaluation_strategy="steps" if eval_dataset else "no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
