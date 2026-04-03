"""Extraction metrics: entity-level and event-level P/R/F1 + diagnostic metrics."""

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


_EXPECTED_KEYS = {"company", "event_type", "date", "metric", "change_direction", "sentiment"}


def _compute_key_presence_rate(predictions: list[list[dict] | None]) -> float:
    """Fraction of samples where prediction contains all expected keys."""
    if not predictions:
        return 0.0
    count = 0
    for pred_events in predictions:
        if pred_events is None or not pred_events:
            continue
        event = pred_events[0]
        if _EXPECTED_KEYS.issubset(event.keys()):
            count += 1
    return count / len(predictions)


def _compute_entity_match_rate(
    predictions: list[list[dict] | None],
    references: list[list[dict] | None],
) -> float:
    """Fraction of samples where company name matches (case-insensitive substring)."""
    if not predictions:
        return 0.0
    count = 0
    total = 0
    for pred_events, ref_events in zip(predictions, references):
        if ref_events is None:
            continue
        total += 1
        if pred_events is None or not pred_events:
            continue
        pred_company = str(pred_events[0].get("company", "")).lower().strip()
        ref_company = str(ref_events[0].get("company", "")).lower().strip()
        if pred_company and ref_company and (pred_company in ref_company or ref_company in pred_company):
            count += 1
    return count / total if total else 0.0


def _compute_partial_field_match(
    predictions: list[list[dict] | None],
    references: list[list[dict] | None],
) -> float:
    """Average fraction of fields that match per sample."""
    if not predictions:
        return 0.0
    field_ratios = []
    for pred_events, ref_events in zip(predictions, references):
        if ref_events is None:
            continue
        if pred_events is None or not pred_events:
            field_ratios.append(0.0)
            continue
        pred = pred_events[0]
        ref = ref_events[0]
        fields = list(_EXPECTED_KEYS)
        matched = 0
        for f in fields:
            pv = str(pred.get(f, "")).lower().strip()
            rv = str(ref.get(f, "")).lower().strip()
            if pv and rv and (pv == rv or pv in rv or rv in pv):
                matched += 1
        field_ratios.append(matched / len(fields) if fields else 0.0)
    return sum(field_ratios) / len(field_ratios) if field_ratios else 0.0


def compute_extraction_metrics(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Compute entity-level and event-level extraction metrics + diagnostics."""
    entity_scores = {"precision": [], "recall": [], "f1": []}
    event_scores = {"precision": [], "recall": [], "f1": []}
    parse_failures = 0

    parsed_preds: list[list[dict] | None] = []
    parsed_refs: list[list[dict] | None] = []

    for pred_text, ref_text in zip(predictions, references):
        pred_events = _safe_parse_json(pred_text)
        ref_events = _safe_parse_json(ref_text)
        parsed_preds.append(pred_events)
        parsed_refs.append(ref_events)

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

    result["key_presence_rate"] = _compute_key_presence_rate(parsed_preds)
    result["entity_match_rate"] = _compute_entity_match_rate(parsed_preds, parsed_refs)
    result["partial_field_match"] = _compute_partial_field_match(parsed_preds, parsed_refs)

    return result
