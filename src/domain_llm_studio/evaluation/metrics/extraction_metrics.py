"""Extraction metrics: entity-level and event-level P/R/F1."""

from __future__ import annotations

import json


def _safe_parse_json(text: str) -> list[dict] | None:
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _normalize_entity(entity: dict) -> tuple:
    """Create a hashable normalized key from an entity dict."""
    return (
        str(entity.get("company", "")).lower().strip(),
        str(entity.get("event_type", "")).lower().strip(),
    )


def _normalize_event(event: dict) -> tuple:
    """Create a hashable normalized key including all main fields."""
    return (
        str(event.get("company", "")).lower().strip(),
        str(event.get("event_type", "")).lower().strip(),
        str(event.get("date", "")).strip(),
        str(event.get("sentiment", "")).lower().strip(),
    )


def _precision_recall_f1(pred_set: set, gold_set: set) -> dict[str, float]:
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_extraction_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Compute entity-level and event-level extraction metrics."""
    entity_scores = {"precision": [], "recall": [], "f1": []}
    event_scores = {"precision": [], "recall": [], "f1": []}
    parse_failures = 0

    for pred_text, ref_text in zip(predictions, references):
        pred_events = _safe_parse_json(pred_text)
        ref_events = _safe_parse_json(ref_text)

        if pred_events is None:
            parse_failures += 1
            for k in entity_scores:
                entity_scores[k].append(0.0)
                event_scores[k].append(0.0)
            continue

        if ref_events is None:
            continue

        pred_entities = {_normalize_entity(e) for e in pred_events}
        gold_entities = {_normalize_entity(e) for e in ref_events}
        e_scores = _precision_recall_f1(pred_entities, gold_entities)
        for k, v in e_scores.items():
            entity_scores[k].append(v)

        pred_events_full = {_normalize_event(e) for e in pred_events}
        gold_events_full = {_normalize_event(e) for e in ref_events}
        ev_scores = _precision_recall_f1(pred_events_full, gold_events_full)
        for k, v in ev_scores.items():
            event_scores[k].append(v)

    result = {}
    for prefix, scores in [("entity", entity_scores), ("event", event_scores)]:
        for metric, values in scores.items():
            avg = sum(values) / len(values) if values else 0.0
            result[f"{prefix}_{metric}"] = avg

    result["parse_failure_rate"] = parse_failures / len(predictions) if predictions else 0.0
    return result
