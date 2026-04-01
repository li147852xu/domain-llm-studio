"""Generation quality metrics: format compliance, field completeness, schema match."""

from __future__ import annotations

import json
import re


def _has_required_sections(memo: str) -> dict[str, bool]:
    """Check if a memo has the expected structural elements."""
    memo_lower = memo.lower()  # noqa: F841
    return {
        "has_company_mention": bool(re.search(r"[A-Z][a-z]+|[\u4e00-\u9fff]{2,}", memo)),
        "has_period_mention": bool(re.search(r"(Q[1-4]|FY|季度|年度|\d{4})", memo)),
        "has_metrics": bool(re.search(r"(\d+\.?\d*\s*(billion|million|%|亿|万))", memo, re.IGNORECASE)),
        "has_analysis": len(memo.split()) > 20 or len(memo) > 50,
    }


def format_compliance_score(prediction: str) -> float:
    """Score how well the output follows expected memo format (0-1)."""
    sections = _has_required_sections(prediction)
    return sum(sections.values()) / len(sections) if sections else 0.0


def field_completeness_score(prediction: str, input_data: str) -> float:
    """Check if key fields from the input appear in the output."""
    try:
        data = json.loads(input_data)
    except (json.JSONDecodeError, TypeError):
        return 0.5

    if not isinstance(data, dict):
        return 0.5

    check_fields = ["company", "period", "revenue"]
    found = 0
    total = 0

    for field in check_fields:
        if field in data and data[field]:
            total += 1
            value = str(data[field]).lower()
            if value in prediction.lower():
                found += 1

    return found / total if total > 0 else 0.5


def schema_match_score(prediction: str, input_data: str) -> float:
    """Combined score of format compliance and field completeness."""
    fc = format_compliance_score(prediction)
    fcs = field_completeness_score(prediction, input_data)
    return (fc + fcs) / 2


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
    inputs: list[str] | None = None,
) -> dict[str, float]:
    """Compute generation quality metrics across all samples."""
    fc_scores = []
    fcs_scores = []
    sm_scores = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        fc_scores.append(format_compliance_score(pred))

        input_data = inputs[i] if inputs and i < len(inputs) else ""
        fcs_scores.append(field_completeness_score(pred, input_data))
        sm_scores.append(schema_match_score(pred, input_data))

    return {
        "format_compliance": sum(fc_scores) / len(fc_scores) if fc_scores else 0.0,
        "field_completeness": sum(fcs_scores) / len(fcs_scores) if fcs_scores else 0.0,
        "schema_match": sum(sm_scores) / len(sm_scores) if sm_scores else 0.0,
    }
