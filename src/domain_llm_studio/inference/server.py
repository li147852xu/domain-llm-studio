"""FastAPI inference server with predict, compare, and task listing endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from domain_llm_studio.inference.predictor import DomainPredictor
from domain_llm_studio.inference.schemas import (
    CompareRequest,
    CompareResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TaskInfo,
)

logger = logging.getLogger(__name__)

TASK_REGISTRY = {
    "fin_summary": TaskInfo(
        task_type="fin_summary",
        name="Financial Document Summarization",
        description="Generate structured summaries from financial documents with key points, risks, and opportunities.",
        input_fields=["document"],
        output_format="JSON: {summary, key_points, risks, opportunities}",
    ),
    "event_extraction": TaskInfo(
        task_type="event_extraction",
        name="Event & Entity Extraction",
        description="Extract structured events and entities from financial news and announcements.",
        input_fields=["text"],
        output_format="JSON array: [{company, event_type, date, metric, change_direction, sentiment}]",
    ),
    "doc_qa": TaskInfo(
        task_type="doc_qa",
        name="Document Question Answering",
        description="Answer questions based strictly on the provided document context.",
        input_fields=["context", "question"],
        output_format="JSON: {answer, evidence_span}",
    ),
    "analysis_gen": TaskInfo(
        task_type="analysis_gen",
        name="Structured Analysis Generation",
        description="Generate professional analysis memos from structured financial data points.",
        input_fields=["structured_data"],
        output_format="Text: professional analysis paragraph",
    ),
}

_predictor: DomainPredictor | None = None


def create_app(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _predictor
        logger.info("Loading model: %s", model_path)
        _predictor = DomainPredictor(model_path, adapter_path)
        yield
        _predictor = None

    app = FastAPI(
        title="Domain LLM Studio API",
        description="Inference API for domain-adapted LLM serving",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            model_loaded=_predictor is not None,
            model_path=model_path,
            adapter_loaded=adapter_path is not None,
        )

    @app.get("/tasks", response_model=list[TaskInfo])
    async def list_tasks():
        return list(TASK_REGISTRY.values())

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if request.task not in TASK_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task: {request.task}. Available: {list(TASK_REGISTRY.keys())}",
            )

        try:
            output = _predictor.predict(
                task=request.task,
                input_text=request.input_text,
                model_type=request.model_type,
                question=request.question,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return PredictResponse(
            task=request.task,
            model_type=request.model_type,
            output=output,
        )

    @app.post("/compare", response_model=CompareResponse)
    async def compare(request: CompareRequest):
        if _predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if request.task not in TASK_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task: {request.task}. Available: {list(TASK_REGISTRY.keys())}",
            )

        try:
            results = _predictor.compare(
                task=request.task,
                input_text=request.input_text,
                question=request.question,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return CompareResponse(task=request.task, results=results)

    return app
