"""FinanceBench external benchmark evaluation.

FinanceBench (PatronusAI/financebench) is a 150-question QA benchmark
over real SEC filings. We use it purely for generalization evaluation —
no training on this data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from rich.console import Console

from domain_llm_studio.evaluation.metrics.qa_metrics import compute_qa_metrics
from domain_llm_studio.inference.predictor import FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)
console = Console()


def load_financebench(num_samples: int | None = None) -> list[dict]:
    """Load FinanceBench dataset and convert to internal format.

    Returns list of dicts with keys: instruction, input, output, task.
    Falls back to a small built-in sample if the datasets library or
    the HuggingFace dataset is unavailable.
    """
    samples = []

    try:
        from datasets import load_dataset

        ds = load_dataset("PatronusAI/financebench", split="train")
        for row in ds:
            question = row.get("question", "")
            answer = row.get("answer", "")
            evidence = row.get("evidence", "") or row.get("context", "") or ""

            if isinstance(evidence, list):
                evidence = " ".join(str(e) for e in evidence)
            evidence = str(evidence)

            if not question or not answer:
                continue

            context = evidence if evidence else f"[Financial document context for: {question}]"

            samples.append({
                "task": "doc_qa",
                "instruction": "Answer the financial question based on the provided context. Return JSON with 'answer' and 'evidence_span' fields.",
                "input": json.dumps({"context": context, "question": question}, ensure_ascii=False),
                "output": json.dumps({"answer": answer, "evidence_span": answer[:200]}, ensure_ascii=False),
            })

        logger.info(f"Loaded {len(samples)} samples from FinanceBench HuggingFace dataset")

    except Exception as e:
        logger.warning(f"Could not load FinanceBench from HuggingFace: {e}")
        logger.info("Using built-in FinanceBench sample for demonstration")
        samples = _builtin_financebench_samples()

    if num_samples and len(samples) > num_samples:
        samples = samples[:num_samples]

    return samples


def _builtin_financebench_samples() -> list[dict]:
    """Small built-in sample for offline/demo use."""
    raw = [
        {
            "question": "What was Apple's total revenue for fiscal year 2023?",
            "answer": "$383.3 billion",
            "context": "Apple Inc. reported total net revenue of $383.3 billion for fiscal year 2023, compared to $394.3 billion in fiscal year 2022, a decrease of 3%.",
        },
        {
            "question": "What was Microsoft's cloud revenue growth in FY2023?",
            "answer": "22%",
            "context": "Microsoft's Intelligent Cloud segment revenue grew 22% year-over-year to $87.9 billion in fiscal year 2023, driven by Azure and other cloud services growth of 29%.",
        },
        {
            "question": "What was Amazon's operating income in 2023?",
            "answer": "$36.9 billion",
            "context": "Amazon.com reported operating income of $36.9 billion for the full year 2023, compared to $12.2 billion in 2022. AWS operating income was $24.6 billion.",
        },
        {
            "question": "What was JPMorgan's net interest income in 2023?",
            "answer": "$89.3 billion",
            "context": "JPMorgan Chase reported record net interest income of $89.3 billion in 2023, up 34% year-over-year, benefiting from higher interest rates and loan growth.",
        },
        {
            "question": "What was Tesla's total vehicle deliveries in 2023?",
            "answer": "1.81 million vehicles",
            "context": "Tesla delivered approximately 1.81 million vehicles in 2023, representing a 38% increase from 2022. Model Y was the best-selling vehicle globally.",
        },
        {
            "question": "What was Google's advertising revenue in Q4 2023?",
            "answer": "$65.5 billion",
            "context": "Alphabet reported Google advertising revenue of $65.5 billion in Q4 2023, up 11% year-over-year. YouTube ads contributed $9.2 billion to this total.",
        },
        {
            "question": "What was Meta's Reality Labs operating loss in 2023?",
            "answer": "$16.1 billion",
            "context": "Meta Platforms' Reality Labs segment reported an operating loss of $16.1 billion in 2023, widening from $13.7 billion in 2022. Total Reality Labs revenue was $1.9 billion.",
        },
        {
            "question": "What was NVIDIA's data center revenue in Q4 FY2024?",
            "answer": "$18.4 billion",
            "context": "NVIDIA reported record data center revenue of $18.4 billion in Q4 fiscal year 2024, up 409% year-over-year, driven by unprecedented demand for AI training and inference chips.",
        },
    ]

    samples = []
    for r in raw:
        samples.append({
            "task": "doc_qa",
            "instruction": "Answer the financial question based on the provided context. Return JSON with 'answer' and 'evidence_span' fields.",
            "input": json.dumps({"context": r["context"], "question": r["question"]}, ensure_ascii=False),
            "output": json.dumps({"answer": r["answer"], "evidence_span": r["answer"]}, ensure_ascii=False),
        })
    return samples


def _get_few_shot_messages(task: str = "doc_qa") -> list[dict]:
    """Build few-shot messages for prompt_only variant."""
    examples = FEW_SHOT_EXAMPLES.get(task, {}).get("en", [])
    messages = []
    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    return messages


def run_financebench_eval(
    model_path: str,
    adapter_path: str | None,
    model_variant: str,
    samples: list[dict],
) -> dict:
    """Run FinanceBench evaluation and return results dict."""
    from domain_llm_studio.training.model_loader import (
        load_base_model,
        load_model_with_adapter,
        load_tokenizer,
    )

    tokenizer = load_tokenizer(model_path)

    if adapter_path and model_variant == "tuned":
        model = load_model_with_adapter(model_path, adapter_path)
    else:
        model = load_base_model(model_path)

    model.eval()
    console.print(f"Model loaded for benchmark: {model_variant}")

    predictions = []
    for sample in samples:
        messages = [
            {"role": "system", "content": sample.get("instruction", "")},
        ]

        if model_variant == "prompt_only":
            messages.extend(_get_few_shot_messages())

        messages.append({"role": "user", "content": sample.get("input", "")})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        predictions.append(prediction)

    references = [s["output"] for s in samples]
    contexts = [s["input"] for s in samples]

    metrics = compute_qa_metrics(predictions, references, contexts=contexts)

    return {
        "benchmark": "financebench",
        "model_variant": model_variant,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "num_samples": len(samples),
        "metrics": metrics,
        "predictions_sample": [
            {"input": s["input"][:200], "prediction": p[:200], "reference": s["output"][:200]}
            for s, p in list(zip(samples, predictions))[:5]
        ],
    }


def save_benchmark_results(results: dict, output_dir: Path) -> None:
    """Save benchmark results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = results.get("model_variant", "base")
    path = output_dir / f"eval_{variant}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Benchmark results saved to {path}[/green]")
