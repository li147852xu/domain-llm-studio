"""Error analysis module: classifies prediction errors into actionable categories."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class ErrorCase:
    sample_id: int
    task: str
    error_type: str
    input_text: str
    prediction: str
    reference: str
    detail: str = ""


ERROR_TYPES = [
    "hallucination",
    "missing_extraction",
    "format_violation",
    "truncation",
    "grounding_failure",
    "wrong_answer",
    "partial_match",
]


def detect_hallucination(prediction: str, input_text: str) -> bool:
    """Check if prediction contains entities not present in input."""
    try:
        pred_data = json.loads(prediction)
        if isinstance(pred_data, list):
            for item in pred_data:
                company = str(item.get("company", ""))
                if company and company.lower() not in input_text.lower():
                    return True
        elif isinstance(pred_data, dict):
            for val in pred_data.values():
                if isinstance(val, str) and len(val) > 20:
                    words = val.split()
                    novel_words = [w for w in words if len(w) > 5 and w.lower() not in input_text.lower()]
                    if len(novel_words) > len(words) * 0.5:
                        return True
    except (json.JSONDecodeError, TypeError):
        pass
    return False


def detect_format_violation(prediction: str, task: str) -> bool:
    """Check if prediction violates the expected format."""
    if task in ("fin_summary", "event_extraction", "doc_qa"):
        try:
            json.loads(prediction)
            return False
        except (json.JSONDecodeError, TypeError):
            return True
    return False


def detect_truncation(prediction: str, reference: str) -> bool:
    """Check if prediction is suspiciously shorter than reference."""
    if not reference:
        return False
    return len(prediction) < len(reference) * 0.3


def detect_grounding_failure(prediction: str, input_text: str, task: str) -> bool:
    """For QA tasks, check if answer is grounded in context."""
    if task != "doc_qa":
        return False
    try:
        pred_data = json.loads(prediction)
        answer = str(pred_data.get("answer", ""))
        if answer.lower() == "unanswerable":
            return False
        return answer.lower() not in input_text.lower()
    except (json.JSONDecodeError, TypeError):
        return False


def detect_missing_extraction(prediction: str, reference: str, task: str) -> bool:
    """For extraction tasks, check if key entities from reference are missing."""
    if task != "event_extraction":
        return False
    try:
        pred_events = json.loads(prediction) if isinstance(json.loads(prediction), list) else []
        ref_events = json.loads(reference) if isinstance(json.loads(reference), list) else []
        pred_companies = {str(e.get("company", "")).lower() for e in pred_events}
        ref_companies = {str(e.get("company", "")).lower() for e in ref_events}
        missing = ref_companies - pred_companies
        return len(missing) > 0
    except (json.JSONDecodeError, TypeError):
        return False


def analyze_errors(
    predictions: list[str],
    references: list[str],
    inputs: list[str],
    tasks: list[str],
    max_examples_per_type: int = 5,
) -> dict:
    """Run error analysis across all predictions.

    Returns:
        {
            "error_distribution": {error_type: count},
            "error_rate": float,
            "examples": {error_type: [ErrorCase, ...]},
            "per_task_errors": {task: {error_type: count}},
        }
    """
    error_counts = Counter()
    error_examples: dict[str, list[dict]] = defaultdict(list)
    per_task_errors: dict[str, Counter] = defaultdict(Counter)
    total_errors = 0

    for i, (pred, ref, inp, task) in enumerate(zip(predictions, references, inputs, tasks)):
        errors_found = []

        if detect_format_violation(pred, task):
            errors_found.append("format_violation")

        if detect_hallucination(pred, inp):
            errors_found.append("hallucination")

        if detect_truncation(pred, ref):
            errors_found.append("truncation")

        if detect_grounding_failure(pred, inp, task):
            errors_found.append("grounding_failure")

        if detect_missing_extraction(pred, ref, task):
            errors_found.append("missing_extraction")

        if not errors_found and pred.strip() != ref.strip():
            errors_found.append("partial_match")

        for err_type in errors_found:
            error_counts[err_type] += 1
            per_task_errors[task][err_type] += 1
            total_errors += 1

            if len(error_examples[err_type]) < max_examples_per_type:
                error_examples[err_type].append({
                    "sample_id": i,
                    "task": task,
                    "input": inp[:200] + "..." if len(inp) > 200 else inp,
                    "prediction": pred[:300] + "..." if len(pred) > 300 else pred,
                    "reference": ref[:300] + "..." if len(ref) > 300 else ref,
                })

    return {
        "error_distribution": dict(error_counts),
        "total_errors": total_errors,
        "total_samples": len(predictions),
        "error_rate": total_errors / len(predictions) if predictions else 0.0,
        "examples": dict(error_examples),
        "per_task_errors": {k: dict(v) for k, v in per_task_errors.items()},
    }
