"""QA metrics: exact match, token F1, and evidence grounding rate."""

from __future__ import annotations

import json
import re
import string
from collections import Counter


def _normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the|的|了|是)\b", " ", text)
    text = " ".join(text.split())
    return text


def _extract_answer_field(text: str) -> str:
    """Try to parse JSON and extract the 'answer' field, else return raw text."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"])
    except (json.JSONDecodeError, TypeError):
        pass
    return text


def _extract_evidence_span(text: str) -> str | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data.get("evidence_span")
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def exact_match_score(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


def token_f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def grounding_score(pred_text: str, context: str) -> float:
    """Check if the predicted answer (or evidence span) can be found in the context."""
    answer = _extract_answer_field(pred_text)
    evidence = _extract_evidence_span(pred_text)

    if answer.lower() == "unanswerable":
        return 1.0

    check_text = evidence if evidence else answer
    if not check_text:
        return 0.0

    return 1.0 if check_text.lower() in context.lower() else 0.0


def compute_qa_metrics(
    predictions: list[str],
    references: list[str],
    contexts: list[str] | None = None,
) -> dict[str, float]:
    """Compute QA metrics across all samples."""
    em_scores = []
    f1_scores = []
    grounding_scores = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_answer = _extract_answer_field(pred)
        ref_answer = _extract_answer_field(ref)

        em_scores.append(exact_match_score(pred_answer, ref_answer))
        f1_scores.append(token_f1_score(pred_answer, ref_answer))

        if contexts and i < len(contexts):
            grounding_scores.append(grounding_score(pred, contexts[i]))

    result = {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }
    if grounding_scores:
        result["grounding_rate"] = sum(grounding_scores) / len(grounding_scores)

    return result
