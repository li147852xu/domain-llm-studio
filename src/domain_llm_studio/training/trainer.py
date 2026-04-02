"""Unified training entry point using HF SFTTrainer with LoRA/QLoRA."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

from domain_llm_studio.config import AdapterType, TrainConfig
from domain_llm_studio.data.stats import load_jsonl
from domain_llm_studio.training.callbacks import LossLoggerCallback, TrainingSummaryCallback
from domain_llm_studio.training.model_loader import (
    detect_device,
    load_base_model,
    load_tokenizer,
    prepare_model_for_training,
)

logger = logging.getLogger(__name__)


def _build_chat_dataset(data_dir: Path, tokenizer, max_seq_length: int) -> tuple[Dataset, Dataset]:
    """Load and format training data as chat-template conversations."""
    train_samples = load_jsonl(data_dir / "train.jsonl")
    dev_samples = load_jsonl(data_dir / "dev.jsonl")

    def to_conversations(samples: list[dict]) -> list[dict]:
        result = []
        for s in samples:
            messages = [
                {"role": "system", "content": s.get("instruction", "")},
                {"role": "user", "content": s.get("input", "")},
                {"role": "assistant", "content": s.get("output", "")},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            result.append({"text": text})
        return result

    train_data = Dataset.from_list(to_conversations(train_samples))
    dev_data = Dataset.from_list(to_conversations(dev_samples))
    return train_data, dev_data


def _make_training_args(cfg: TrainConfig, output_dir: Path, use_bf16: bool, use_fp16: bool):
    """Build training arguments, auto-detecting TRL version for compatibility."""
    common = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    try:
        from trl import SFTConfig
        logger.info("Using TRL SFTConfig (TRL >= 1.0)")
        return SFTConfig(
            **common,
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
        ), False
    except ImportError:
        from transformers import TrainingArguments
        logger.info("Using TrainingArguments + SFTTrainer max_seq_length (TRL < 1.0)")
        return TrainingArguments(**common), True


def run_training(cfg: TrainConfig) -> Path:
    """Execute the full training pipeline. Returns path to saved adapter."""
    from trl import SFTTrainer

    logging.basicConfig(level=logging.INFO)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    logger.info("Training on device: %s", device)

    config_path = output_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.model_dump(), f, indent=2, default=str)

    quantize = cfg.adapter_type == AdapterType.QLORA and device == "cuda"
    if cfg.adapter_type == AdapterType.QLORA and device != "cuda":
        logger.warning("QLoRA requires CUDA. Falling back to standard LoRA on %s.", device)

    tokenizer = load_tokenizer(cfg.base_model)
    model = load_base_model(cfg.base_model, device_map=cfg.device_map, quantize_4bit=quantize)
    model = prepare_model_for_training(
        model,
        lora_r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    train_dataset, dev_dataset = _build_chat_dataset(
        Path(cfg.data_dir), tokenizer, cfg.max_seq_length
    )
    logger.info("Train samples: %d, Dev samples: %d", len(train_dataset), len(dev_dataset))

    use_bf16 = cfg.bf16 and device == "cuda"
    use_fp16 = not use_bf16 and device != "cpu"

    training_args, pass_max_seq_to_trainer = _make_training_args(cfg, output_dir, use_bf16, use_fp16)

    callbacks = [
        LossLoggerCallback(output_dir),
        TrainingSummaryCallback(output_dir, cfg.model_dump()),
    ]

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if pass_max_seq_to_trainer:
        trainer_kwargs["max_seq_length"] = cfg.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training...")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Adapter saved to %s", adapter_dir)

    return adapter_dir
