"""Custom training callbacks for logging and evaluation tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class LossLoggerCallback(TrainerCallback):
    """Logs training loss to a JSON file for later visualization."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_entries: list[dict] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if logs is None:
            return
        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        }
        self.log_entries.append(entry)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        log_path = self.output_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.log_entries, f, indent=2)
        logger.info("Training log saved to %s", log_path)


class TrainingSummaryCallback(TrainerCallback):
    """Saves a training summary at the end of training."""

    def __init__(self, output_dir: Path, config_dict: dict):
        self.output_dir = Path(output_dir)
        self.config_dict = config_dict

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        summary = {
            "total_steps": state.global_step,
            "num_epochs": state.epoch,
            "best_metric": state.best_metric,
            "training_config": self.config_dict,
        }
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Training summary saved to %s", summary_path)
