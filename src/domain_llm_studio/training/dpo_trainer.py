"""Direct Preference Optimization (DPO) trainer on top of an SFT LoRA.

Design choices
--------------

We default to the **merge-then-LoRA** pattern: load the base model, merge
the existing SFT LoRA adapter into the base weights via
``PeftModel.merge_and_unload()``, then attach a *new*, freshly-initialized
LoRA on top of the merged model and run DPO. This:

1. Lets DPO see a strong starting policy (the SFT model) without needing
   to keep two LoRA adapters loaded simultaneously.
2. Avoids fragile "stack two LoRAs" code paths in PEFT.
3. Keeps the trainable parameter count small (only the new LoRA), so the
   7B model fits comfortably on a single 32 GB GPU.

The DPO reference model is intentionally left as ``None``; trl will
internally clone the merged model with frozen weights as the reference,
so the implicit reward = log p_policy(y) - log p_ref(y) baseline is
exactly the SFT model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

from domain_llm_studio.config import DpoConfig
from domain_llm_studio.data.stats import load_jsonl
from domain_llm_studio.training.callbacks import (
    LossLoggerCallback,
    TrainingSummaryCallback,
)
from domain_llm_studio.training.model_loader import (
    detect_device,
    load_base_model,
    load_tokenizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _load_preference_dataset(data_dir: Path) -> tuple[Dataset, Dataset]:
    train_records = load_jsonl(data_dir / "train.jsonl")
    dev_path = data_dir / "dev.jsonl"
    dev_records = load_jsonl(dev_path) if dev_path.exists() else []

    if not train_records:
        raise ValueError(
            f"No preference training data at {data_dir}/train.jsonl. "
            "Run `python -m domain_llm_studio.data.preference_pairs` first."
        )

    def _project(rec: dict) -> dict:
        return {
            "prompt": rec["prompt"],
            "chosen": rec["chosen"],
            "rejected": rec["rejected"],
        }

    train_dataset = Dataset.from_list([_project(r) for r in train_records])
    dev_dataset = (
        Dataset.from_list([_project(r) for r in dev_records])
        if dev_records
        else None
    )
    return train_dataset, dev_dataset


# ---------------------------------------------------------------------------
# Model construction (merge SFT adapter, attach fresh LoRA)
# ---------------------------------------------------------------------------

def _build_dpo_policy_model(cfg: DpoConfig):
    """Load base, merge SFT adapter, attach new LoRA. Returns the model."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import PeftModel, get_peft_model

    base_model = load_base_model(cfg.base_model, device_map=cfg.device_map)

    if cfg.sft_adapter_path:
        logger.info("Loading SFT adapter from %s", cfg.sft_adapter_path)
        peft_model = PeftModel.from_pretrained(base_model, cfg.sft_adapter_path)
        logger.info("Merging SFT adapter into base weights (merge_and_unload)")
        merged = peft_model.merge_and_unload()
    else:
        logger.warning(
            "No sft_adapter_path provided — DPO will start from the raw base "
            "model. This is unusual for post-SFT alignment."
        )
        merged = base_model

    new_lora = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    if cfg.lora.target_modules is None:
        from domain_llm_studio.training.model_loader import _auto_detect_target_modules
        new_lora.target_modules = _auto_detect_target_modules(merged)

    policy = get_peft_model(merged, new_lora)
    trainable, total = policy.get_nb_trainable_parameters()
    logger.info(
        "DPO policy LoRA attached: %d trainable / %d total (%.3f%%)",
        trainable, total, 100.0 * trainable / total,
    )
    return policy


# ---------------------------------------------------------------------------
# DPOConfig args (cross-version compatibility for trl 0.x → 1.x)
# ---------------------------------------------------------------------------

def _build_dpo_args(cfg: DpoConfig, output_dir: Path, use_bf16: bool):
    """Construct trl DPOConfig with version-tolerant kwargs."""
    from trl import DPOConfig

    base_kwargs = dict(
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
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        beta=cfg.beta,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_length,
    )

    try:
        return DPOConfig(**base_kwargs)
    except TypeError as e:
        # trl 0.x used max_target_length instead of max_length; trl 1.x dropped
        # max_prompt_length on some sub-classes. Strip and retry.
        logger.warning("DPOConfig kwargs incompatible (%s), retrying minimal", e)
        for opt in ("max_prompt_length", "max_length"):
            base_kwargs.pop(opt, None)
        return DPOConfig(**base_kwargs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dpo(cfg: DpoConfig) -> Path:
    from trl import DPOTrainer

    logging.basicConfig(level=logging.INFO)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_config.json", "w") as f:
        json.dump(cfg.model_dump(), f, indent=2, default=str)

    device = detect_device()
    logger.info("DPO training on device: %s", device)

    tokenizer = load_tokenizer(cfg.base_model)

    policy = _build_dpo_policy_model(cfg)

    train_dataset, dev_dataset = _load_preference_dataset(Path(cfg.data_dir))
    logger.info(
        "Preference samples: train=%d, dev=%d",
        len(train_dataset),
        len(dev_dataset) if dev_dataset else 0,
    )

    use_bf16 = cfg.bf16 and device == "cuda"
    args = _build_dpo_args(cfg, output_dir, use_bf16)

    callbacks = [
        LossLoggerCallback(output_dir),
        TrainingSummaryCallback(output_dir, cfg.model_dump()),
    ]

    trainer_kwargs = dict(
        model=policy,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    try:
        trainer = DPOTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = DPOTrainer(**trainer_kwargs)

    logger.info("Starting DPO training...")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("DPO adapter saved to %s", adapter_dir)
    return adapter_dir
