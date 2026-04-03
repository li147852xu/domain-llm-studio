"""Evaluation runner: orchestrates model inference and metric computation."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from rich.console import Console

from domain_llm_studio.config import EvalConfig
from domain_llm_studio.data.stats import load_jsonl
from domain_llm_studio.evaluation.error_analysis import analyze_errors
from domain_llm_studio.evaluation.metrics.extraction_metrics import compute_extraction_metrics
from domain_llm_studio.evaluation.metrics.generation_metrics import compute_generation_metrics
from domain_llm_studio.evaluation.metrics.qa_metrics import compute_qa_metrics
from domain_llm_studio.evaluation.metrics.rouge_metrics import (
    compute_keypoint_coverage,
    compute_rouge,
)

logger = logging.getLogger(__name__)
console = Console()


def _get_few_shot_examples(task: str, lang: str) -> list[dict]:
    """Retrieve few-shot examples for prompt_only variant."""
    from domain_llm_studio.inference.predictor import FEW_SHOT_EXAMPLES

    examples = FEW_SHOT_EXAMPLES.get(task, {}).get(lang, [])
    if not examples:
        examples = FEW_SHOT_EXAMPLES.get(task, {}).get("en", [])
    return examples


def _detect_lang(text: str) -> str:
    return "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"


def _run_inference_batch(
    model,
    tokenizer,
    samples: list[dict],
    max_new_tokens: int = 512,
    device: str = "mps",
    model_variant: str = "base",
) -> list[str]:
    """Run inference on a batch of samples."""
    import torch

    predictions = []
    for sample in samples:
        messages = [
            {"role": "system", "content": sample.get("instruction", "")},
        ]

        if model_variant == "prompt_only":
            task = sample.get("task", "")
            lang = _detect_lang(sample.get("input", ""))
            for ex in _get_few_shot_examples(task, lang):
                messages.append({"role": "user", "content": ex["input"]})
                messages.append({"role": "assistant", "content": ex["output"]})

        messages.append({"role": "user", "content": sample.get("input", "")})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(prediction.strip())

    return predictions


def _compute_task_metrics(
    task: str,
    predictions: list[str],
    references: list[str],
    inputs: list[str],
) -> dict[str, float]:
    """Compute task-specific metrics."""
    if task == "fin_summary":
        metrics = compute_rouge(predictions, references)
        metrics.update(compute_keypoint_coverage(predictions, references))
        return metrics
    elif task == "event_extraction":
        return compute_extraction_metrics(predictions, references)
    elif task == "doc_qa":
        return compute_qa_metrics(predictions, references, contexts=inputs)
    elif task == "analysis_gen":
        return compute_generation_metrics(predictions, references, inputs=inputs)
    return {}


def run_evaluation(cfg: EvalConfig) -> dict:
    """Run full evaluation pipeline."""
    from domain_llm_studio.training.model_loader import (
        load_base_model,
        load_model_with_adapter,
        load_tokenizer,
    )

    logging.basicConfig(level=logging.INFO)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = Path(cfg.data_dir) / "test.jsonl"
    if not test_path.exists():
        console.print(f"[red]Test data not found at {test_path}[/red]")
        return {}

    test_samples = load_jsonl(test_path)
    if cfg.num_samples:
        test_samples = test_samples[:cfg.num_samples]

    console.print(f"Loaded {len(test_samples)} test samples")

    tokenizer = load_tokenizer(cfg.model_path)

    model_variant = getattr(cfg, "model_variant", "base")

    if cfg.adapter_path:
        model = load_model_with_adapter(cfg.model_path, cfg.adapter_path)
        model_label = "tuned"
    else:
        model = load_base_model(cfg.model_path)
        model_label = model_variant if model_variant != "tuned" else "base"

    model.eval()
    console.print(f"Model loaded: {model_label} (variant={model_variant})")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for s in test_samples:
        task = s.get("task", "unknown")
        by_task[task].append(s)

    all_results = {}
    all_predictions = []
    all_references = []
    all_inputs = []
    all_tasks = []

    for task_name, task_samples in sorted(by_task.items()):
        if cfg.tasks and task_name not in [t.value for t in cfg.tasks]:
            continue

        console.print(f"\n[bold cyan]Evaluating {task_name}[/bold cyan] ({len(task_samples)} samples)...")
        predictions = _run_inference_batch(
            model, tokenizer, task_samples, model_variant=model_variant
        )
        references = [s["output"] for s in task_samples]
        inputs = [s["input"] for s in task_samples]

        metrics = _compute_task_metrics(task_name, predictions, references, inputs)
        all_results[task_name] = metrics

        all_predictions.extend(predictions)
        all_references.extend(references)
        all_inputs.extend(inputs)
        all_tasks.extend([task_name] * len(task_samples))

        console.print(f"  Metrics: {json.dumps(metrics, indent=2)}")

    console.print("\n[bold cyan]Running error analysis...[/bold cyan]")
    error_report = analyze_errors(all_predictions, all_references, all_inputs, all_tasks)
    console.print(f"  Error rate: {error_report['error_rate']:.1%}")
    console.print(f"  Distribution: {error_report['error_distribution']}")

    results = {
        "model": model_label,
        "model_path": cfg.model_path,
        "adapter_path": cfg.adapter_path,
        "model_variant": model_variant,
        "per_task": all_results,
        "error_analysis": error_report,
    }

    results_path = output_dir / f"eval_{model_label}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]Results saved to {results_path}[/green]")

    return results
