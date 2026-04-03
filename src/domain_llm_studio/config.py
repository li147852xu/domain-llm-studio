"""Pydantic v2 configuration models for all project components."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SUMMARIZATION = "fin_summary"
    EXTRACTION = "event_extraction"
    QA = "doc_qa"
    GENERATION = "analysis_gen"


class AdapterType(str, Enum):
    LORA = "lora"
    QLORA = "qlora"


class ModelVariant(str, Enum):
    BASE = "base"
    PROMPT_ONLY = "prompt_only"
    TUNED = "tuned"


# ---------------------------------------------------------------------------
# Task Configuration
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    task_name: str
    task_type: TaskType
    instruction_template: str = ""
    few_shot_template: str = ""
    input_fields: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    eval_metrics: list[str] = Field(default_factory=list)
    max_input_len: int = 1024
    max_output_len: int = 512


# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    seed_dir: Path = Path("data/seed")
    output_dir: Path = Path("data/processed")
    seed: int = 42
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    max_input_len: int = 1024
    max_output_len: int = 512
    languages: list[str] = Field(default_factory=lambda: ["zh", "en"])
    num_seed_samples_per_task: int = 50


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

class LoraConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_type: AdapterType = AdapterType.LORA
    lora: LoraConfig = Field(default_factory=LoraConfig)
    data_dir: Path = Path("data/processed")
    output_dir: Path = Path("experiments/train")
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    max_seq_length: int = 1536
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    bf16: bool = True
    seed: int = 42
    device_map: str = "auto"


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

class EvalConfig(BaseModel):
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str | None = None
    model_variant: str = "base"
    data_dir: Path = Path("data/processed")
    output_dir: Path = Path("experiments/eval")
    tasks: list[TaskType] = Field(
        default_factory=lambda: list(TaskType)
    )
    num_samples: int | None = None
    batch_size: int = 1
    compare_models: list[ModelVariant] = Field(
        default_factory=lambda: list(ModelVariant)
    )
    seed: int = 42


# ---------------------------------------------------------------------------
# Serving Configuration
# ---------------------------------------------------------------------------

class ServeConfig(BaseModel):
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    device_map: str = "auto"


# ---------------------------------------------------------------------------
# YAML Loader Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_cls: type[BaseModel], path: Path) -> BaseModel:
    data = load_yaml(path)
    return config_cls(**data)


def load_task_configs(config_dir: Path) -> dict[str, TaskConfig]:
    configs = {}
    for f in sorted(config_dir.glob("*.yaml")):
        data = load_yaml(f)
        cfg = TaskConfig(**data)
        configs[cfg.task_type.value] = cfg
    return configs
