"""Model loading utilities with automatic device detection (CUDA > MPS > CPU).

Supports loading base models, applying LoRA adapters, and optional QLoRA
quantization (CUDA-only).
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name_or_path: str,
    device_map: str = "auto",
    quantize_4bit: bool = False,
) -> AutoModelForCausalLM:
    """Load a base model with optional 4-bit quantization."""
    device = detect_device()
    dtype = detect_dtype(device)

    kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if quantize_4bit:
        if device != "cuda":
            logger.warning(
                "4-bit quantization requires CUDA. Falling back to full precision on %s.",
                device,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config

    if device == "mps":
        kwargs["device_map"] = {"": "mps"}
    elif device == "cuda":
        kwargs["device_map"] = device_map
    else:
        kwargs["device_map"] = {"": "cpu"}

    logger.info("Loading model %s on %s (dtype=%s)", model_name_or_path, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model


def load_model_with_adapter(
    base_model_path: str,
    adapter_path: str,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """Load a base model and merge a LoRA adapter."""
    from peft import PeftModel

    model = load_base_model(base_model_path, device_map=device_map)
    logger.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model


def prepare_model_for_training(
    model: AutoModelForCausalLM,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> AutoModelForCausalLM:
    """Apply LoRA configuration to a model for training."""
    from peft import LoraConfig, TaskType, get_peft_model

    task_type_enum = TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.SEQ_2_SEQ_LM

    if target_modules is None:
        target_modules = _auto_detect_target_modules(model)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type_enum,
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied: %d trainable params / %d total (%.2f%%)",
        trainable, total, 100.0 * trainable / total,
    )
    return model


def _auto_detect_target_modules(model: AutoModelForCausalLM) -> list[str]:
    """Detect common linear layer names for LoRA targeting."""
    target_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parts = name.split(".")
            layer_name = parts[-1]
            if layer_name in ("q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",
                              "query_key_value", "dense", "dense_h_to_4h",
                              "dense_4h_to_h"):
                target_names.add(layer_name)

    if not target_names:
        target_names = {"q_proj", "v_proj"}
        logger.warning("Could not auto-detect target modules, defaulting to %s", target_names)

    return sorted(target_names)
