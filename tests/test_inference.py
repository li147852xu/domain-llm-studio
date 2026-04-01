"""Tests for inference schemas and server configuration."""

from __future__ import annotations

from domain_llm_studio.inference.schemas import (
    CompareRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from domain_llm_studio.inference.server import TASK_REGISTRY


class TestSchemas:
    def test_predict_request(self):
        req = PredictRequest(
            task="fin_summary",
            model_type="base",
            input_text="Test document",
        )
        assert req.task == "fin_summary"
        assert req.question is None

    def test_predict_response(self):
        resp = PredictResponse(task="fin_summary", model_type="base", output="result")
        assert resp.output == "result"

    def test_compare_request(self):
        req = CompareRequest(task="doc_qa", input_text="context", question="what?")
        assert req.question == "what?"

    def test_health_response(self):
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.model_loaded is False


class TestTaskRegistry:
    def test_all_tasks_registered(self):
        expected = {"fin_summary", "event_extraction", "doc_qa", "analysis_gen"}
        assert set(TASK_REGISTRY.keys()) == expected

    def test_task_info_fields(self):
        for task_type, info in TASK_REGISTRY.items():
            assert info.name
            assert info.description
            assert len(info.input_fields) > 0
