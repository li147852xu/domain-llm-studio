"""Pydantic v2 request/response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    task: str = Field(description="Task type: fin_summary, event_extraction, doc_qa, analysis_gen")
    model_type: str = Field(
        default="base",
        description="Model variant: base, prompt_only, or tuned",
    )
    input_text: str = Field(description="Input document or text")
    question: str | None = Field(default=None, description="Question for doc_qa task")


class PredictResponse(BaseModel):
    task: str
    model_type: str
    output: str
    metadata: dict = Field(default_factory=dict)


class CompareRequest(BaseModel):
    task: str
    input_text: str
    question: str | None = None


class CompareResponse(BaseModel):
    task: str
    results: dict[str, str] = Field(
        description="Mapping of model_type -> output for each variant"
    )
    metadata: dict = Field(default_factory=dict)


class TaskInfo(BaseModel):
    task_type: str
    name: str
    description: str
    input_fields: list[str]
    output_format: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    model_path: str = ""
    adapter_loaded: bool = False
