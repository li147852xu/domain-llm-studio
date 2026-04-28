"""Tests for the vLLM inference backend.

Most tests are CUDA-only because vLLM requires a GPU. We verify the
non-GPU surface area (prompt construction, factory selection, ImportError
behavior) without instantiating the engine.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch


def test_vllm_backend_module_imports_lazily():
    """The module itself must import on a CPU-only / no-vllm machine."""
    from domain_llm_studio.inference import vllm_backend

    assert hasattr(vllm_backend, "VllmPredictor")


def test_factory_returns_transformers_predictor_by_default():
    from domain_llm_studio.inference.predictor import (
        DomainPredictor,
        create_predictor,
    )

    predictor = create_predictor(model_path="/no/such/path", backend="transformers")
    assert isinstance(predictor, DomainPredictor)


def test_factory_returns_vllm_predictor_for_vllm_backend():
    from domain_llm_studio.inference.predictor import create_predictor
    from domain_llm_studio.inference.vllm_backend import VllmPredictor

    predictor = create_predictor(model_path="/no/such/path", backend="vllm")
    assert isinstance(predictor, VllmPredictor)
    assert predictor.model_path == "/no/such/path"
    assert predictor.adapter_path is None
    assert predictor._llm is None


def test_vllm_predictor_raises_clear_error_when_vllm_missing(monkeypatch):
    """If vllm is not installed, accessing .vllm should raise our message."""
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm is installed; cannot test missing-vllm path")

    from domain_llm_studio.inference.vllm_backend import VllmPredictor

    predictor = VllmPredictor(model_path="/no/such/path")
    with pytest.raises(ImportError, match="vLLM backend requested"):
        _ = predictor.vllm


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="vLLM end-to-end test requires CUDA",
)
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vllm package not installed",
)
def test_vllm_predict_smoke():
    """Real generation smoke test (CUDA + vllm only).

    Uses an extremely small model alias if available; otherwise marked
    expensive and skipped to avoid 7B downloads in CI.
    """
    from pathlib import Path

    from domain_llm_studio.inference.vllm_backend import VllmPredictor

    model_path = "models/Qwen2.5-1.5B-Instruct"
    if not Path(model_path).exists():
        pytest.skip(f"local model {model_path} missing")

    predictor = VllmPredictor(
        model_path=model_path,
        max_model_len=1024,
        gpu_memory_utilization=0.5,
    )
    output = predictor.predict(
        task="fin_summary",
        input_text="Acme Corp Q3 revenue grew 12% YoY.",
        model_type="base",
    )
    assert isinstance(output, str)
    assert len(output) > 0
