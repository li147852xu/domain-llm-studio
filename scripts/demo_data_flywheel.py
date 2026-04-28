"""End-to-end data-flywheel demo.

Runs:
    1. Import a fixture ResearchOps run into 4-task SFT samples
    2. Merge those samples into the existing data/processed/* splits
    3. Mini LoRA training on Qwen2.5-1.5B (1 epoch, ~5-10 steps total)
    4. Evaluate base vs flywheel_tuned, write a summary

This is intentionally a *toy-scale* demo to prove the pipeline works
end-to-end on a single GPU in <10 minutes. Real data flywheel runs would
import 100s of ResearchOps runs and train for full epochs.

Output: experiments/flywheel_demo/summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_step(name: str, cmd: list[str]) -> dict:
    logger.info("=== %s ===", name)
    logger.info("$ %s", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        logger.error("STDOUT:\n%s", proc.stdout[-2000:])
        logger.error("STDERR:\n%s", proc.stderr[-2000:])
        raise RuntimeError(f"step '{name}' failed (exit={proc.returncode})")
    return {"name": name, "elapsed_s": elapsed, "tail": proc.stdout[-300:]}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _safe_load_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("tests/fixtures/researchops_run"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/flywheel_demo"),
    )
    parser.add_argument(
        "--base-model",
        default="models/Qwen2.5-1.5B-Instruct",
        help="Local path to a base model (cloned via hf download).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Cap mini-training to this many optimizer steps.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace = args.output_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    timeline: list[dict] = []

    # Step 1: ResearchOps importer ------------------------------------------
    imported_dir = workspace / "from_researchops"
    if imported_dir.exists():
        shutil.rmtree(imported_dir)
    timeline.append(
        _run_step(
            "1. researchops_importer",
            [
                sys.executable,
                "-m",
                "domain_llm_studio.data.researchops_importer",
                "--runs-dir",
                str(args.runs_dir),
                "--output",
                str(imported_dir),
            ],
        )
    )
    imported_counts = {
        s: _count_jsonl(imported_dir / f"{s}.jsonl") for s in ("train", "dev", "test")
    }

    # Step 2: build a tiny merged dataset for the mini-training -------------
    merged_dir = workspace / "merged_data"
    merged_dir.mkdir(parents=True, exist_ok=True)
    base_processed = Path("data/processed")
    for split in ("train", "dev", "test"):
        src = base_processed / f"{split}.jsonl"
        if not src.exists():
            logger.warning("base processed split missing: %s", src)
            continue
        # tiny snapshot: keep only first 16 base samples (keeps mini training fast)
        with open(src, encoding="utf-8") as f:
            base_lines = [next(f) for _ in range(16) if f.readable()]
        # ResearchOps additions
        ro_src = imported_dir / f"{split}.jsonl"
        ro_lines = ro_src.read_text(encoding="utf-8").splitlines() if ro_src.exists() else []
        out = merged_dir / f"{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for line in base_lines:
                if not line.endswith("\n"):
                    line = line + "\n"
                f.write(line)
            for line in ro_lines:
                if line.strip():
                    f.write(line + "\n")
    merged_counts = {
        s: _count_jsonl(merged_dir / f"{s}.jsonl") for s in ("train", "dev", "test")
    }

    # Step 3: write a tiny training config ----------------------------------
    train_cfg = workspace / "flywheel_train.yaml"
    train_dir = workspace / "tuned_adapter"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    train_cfg.write_text(
        f"""base_model: "{args.base_model}"
adapter_type: "lora"
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: null
  bias: "none"
  task_type: "CAUSAL_LM"
data_dir: "{merged_dir.as_posix()}"
output_dir: "{train_dir.as_posix()}"
learning_rate: 2.0e-4
num_epochs: 1
per_device_batch_size: 1
gradient_accumulation_steps: 1
warmup_ratio: 0.0
max_seq_length: 1024
eval_steps: 50
save_steps: 100
logging_steps: 1
bf16: true
seed: 42
device_map: "auto"
""",
        encoding="utf-8",
    )

    timeline.append(
        _run_step(
            "3. mini LoRA training (1.5B)",
            [
                sys.executable,
                "-m",
                "domain_llm_studio.cli",
                "train",
                "--config",
                str(train_cfg),
            ],
        )
    )

    # Step 4: evaluate base + flywheel_tuned --------------------------------
    eval_dir = workspace / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_eval_cfg = workspace / "eval_base.yaml"
    base_eval_cfg.write_text(
        f"""model_path: "{args.base_model}"
adapter_path: null
model_variant: "base"
data_dir: "{merged_dir.as_posix()}"
output_dir: "{eval_dir.as_posix()}"
tasks:
  - "fin_summary"
  - "event_extraction"
  - "doc_qa"
  - "analysis_gen"
num_samples: 4
batch_size: 1
seed: 42
""",
        encoding="utf-8",
    )
    fly_eval_cfg = workspace / "eval_flywheel.yaml"
    fly_eval_cfg.write_text(
        f"""model_path: "{args.base_model}"
adapter_path: "{(train_dir / 'adapter').as_posix()}"
model_variant: "flywheel_tuned"
data_dir: "{merged_dir.as_posix()}"
output_dir: "{eval_dir.as_posix()}"
tasks:
  - "fin_summary"
  - "event_extraction"
  - "doc_qa"
  - "analysis_gen"
num_samples: 4
batch_size: 1
seed: 42
""",
        encoding="utf-8",
    )

    timeline.append(
        _run_step(
            "4a. eval base",
            [sys.executable, "-m", "domain_llm_studio.cli", "eval", "--config", str(base_eval_cfg)],
        )
    )
    timeline.append(
        _run_step(
            "4b. eval flywheel_tuned",
            [sys.executable, "-m", "domain_llm_studio.cli", "eval", "--config", str(fly_eval_cfg)],
        )
    )

    base_eval = _safe_load_eval(eval_dir / "eval_base.json")
    fly_eval = _safe_load_eval(eval_dir / "eval_flywheel_tuned.json")

    def _avg_metric(eval_data: dict, metric: str) -> float | None:
        if not eval_data:
            return None
        vals = []
        for task_metrics in eval_data.get("per_task", {}).values():
            v = task_metrics.get(metric)
            if isinstance(v, (int, float)):
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    base_rouge_l = _avg_metric(base_eval, "rougeL")
    fly_rouge_l = _avg_metric(fly_eval, "rougeL")
    base_grounding = _avg_metric(base_eval, "grounding_rate")
    fly_grounding = _avg_metric(fly_eval, "grounding_rate")

    def _delta(b, f):
        if b is None or f is None:
            return "n/a"
        return f"{(f - b):+.4f}"

    summary_lines = [
        "# Data Flywheel Demo (toy scale)",
        "",
        f"- ResearchOps runs imported from: `{args.runs_dir}`",
        f"- Imported samples: {imported_counts}",
        f"- Merged splits (toy snapshot 16 base + ResearchOps): {merged_counts}",
        f"- Mini LoRA: r=8, 1 epoch on {merged_counts['train']} samples",
        "",
        "## Eval (n=4 samples per task, deterministic)",
        "",
        "| Metric | base | flywheel_tuned | Δ |",
        "|---|---|---|---|",
        f"| ROUGE-L (avg over fin_summary etc.) | {base_rouge_l!s} | {fly_rouge_l!s} | {_delta(base_rouge_l, fly_rouge_l)} |",
        f"| grounding_rate (doc_qa) | {base_grounding!s} | {fly_grounding!s} | {_delta(base_grounding, fly_grounding)} |",
        "",
        "## Timeline (wall-clock)",
        "",
        "| Step | Seconds |",
        "|---|---|",
    ]
    for step in timeline:
        summary_lines.append(f"| {step['name']} | {step['elapsed_s']:.1f} |")

    flywheel_summary = (
        f"Imported {sum(imported_counts.values())} samples from "
        f"{len(list(args.runs_dir.iterdir()))} ResearchOps runs and trained a "
        f"toy-scale LoRA on top of {sum(merged_counts.values())} merged samples; "
        f"this demonstrates the data-flywheel loop end-to-end."
    )
    summary_lines += ["", "## TL;DR", "", flywheel_summary, ""]

    summary_path = args.output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "imported_counts": imported_counts,
                "merged_counts": merged_counts,
                "base_eval": base_eval,
                "flywheel_eval": fly_eval,
                "timeline": timeline,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("Flywheel demo complete. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
