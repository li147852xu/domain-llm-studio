"""Tests for evaluation metrics."""

from __future__ import annotations

import json

from domain_llm_studio.evaluation.metrics.extraction_metrics import compute_extraction_metrics
from domain_llm_studio.evaluation.metrics.generation_metrics import (
    compute_generation_metrics,
    field_completeness_score,
    format_compliance_score,
)
from domain_llm_studio.evaluation.metrics.qa_metrics import (
    compute_qa_metrics,
    exact_match_score,
    token_f1_score,
)
from domain_llm_studio.evaluation.metrics.rouge_metrics import compute_rouge


class TestRouge:
    def test_identical_texts(self):
        r = compute_rouge(["hello world"], ["hello world"])
        assert r["rouge1"] == 1.0

    def test_no_overlap(self):
        r = compute_rouge(["cat"], ["dog"])
        assert r["rouge1"] == 0.0

    def test_partial_overlap(self):
        r = compute_rouge(["hello world test"], ["hello world demo"])
        assert 0.0 < r["rouge1"] < 1.0


class TestQAMetrics:
    def test_exact_match(self):
        assert exact_match_score("42 billion", "42 billion") == 1.0
        assert exact_match_score("42 billion", "43 billion") == 0.0

    def test_token_f1(self):
        assert token_f1_score("the revenue was 42 billion", "the revenue was 42 billion") == 1.0
        f1 = token_f1_score("revenue 42 billion", "the revenue was 42 billion dollars")
        assert 0.0 < f1 < 1.0

    def test_compute_qa_metrics(self):
        preds = ['{"answer": "100 billion"}', '{"answer": "unknown"}']
        refs = ['{"answer": "100 billion"}', '{"answer": "200 billion"}']
        m = compute_qa_metrics(preds, refs)
        assert m["exact_match"] == 0.5
        assert m["token_f1"] > 0.0


class TestExtractionMetrics:
    def test_perfect_match(self):
        events = [{"company": "Apple", "event_type": "earnings", "date": "2024-01-01", "sentiment": "positive"}]
        pred = json.dumps(events)
        ref = json.dumps(events)
        m = compute_extraction_metrics([pred], [ref])
        assert m["entity_f1"] == 1.0
        assert m["event_f1"] == 1.0

    def test_parse_failure(self):
        m = compute_extraction_metrics(["not json"], [json.dumps([{"company": "X"}])])
        assert m["parse_failure_rate"] == 1.0
        assert m["entity_f1"] == 0.0

    def test_soft_metrics_perfect(self):
        events = [{"company": "Apple", "event_type": "earnings", "date": "2024-01-01",
                    "metric": "revenue", "change_direction": "increase", "sentiment": "positive"}]
        pred = json.dumps(events)
        ref = json.dumps(events)
        m = compute_extraction_metrics([pred], [ref])
        assert m["key_presence_rate"] == 1.0
        assert m["entity_match_rate"] == 1.0
        assert m["partial_field_match"] == 1.0

    def test_soft_metrics_partial(self):
        ref = [{"company": "Apple", "event_type": "earnings", "date": "2024-01-01",
                "metric": "revenue", "change_direction": "increase", "sentiment": "positive"}]
        pred = [{"company": "Apple", "event_type": "acquisition", "date": "2024-01-01",
                 "metric": "wrong", "change_direction": "decrease", "sentiment": "positive"}]
        m = compute_extraction_metrics([json.dumps(pred)], [json.dumps(ref)])
        assert m["entity_match_rate"] == 1.0
        assert 0.0 < m["partial_field_match"] < 1.0


class TestGenerationMetrics:
    def test_format_compliance(self):
        good = "Apple reported Q3 2024 revenue of $85 billion, up 5% year-over-year."
        score = format_compliance_score(good)
        assert score > 0.5

    def test_field_completeness(self):
        pred = "Apple Q3 2024 revenue was strong"
        inp = json.dumps({"company": "Apple", "period": "Q3 2024", "revenue": "$85B"})
        score = field_completeness_score(pred, inp)
        assert score > 0.0

    def test_compute_generation_metrics(self):
        preds = ["Meta Q3 2024 revenue $40 billion, growth 23%."]
        refs = ["Reference memo text"]
        inputs = [json.dumps({"company": "Meta", "period": "Q3 2024", "revenue": "$40B"})]
        m = compute_generation_metrics(preds, refs, inputs)
        assert "format_compliance" in m
        assert "field_completeness" in m
