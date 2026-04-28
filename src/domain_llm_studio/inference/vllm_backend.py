"""vLLM-based inference backend for high-throughput serving.

Mirrors the public interface of :class:`DomainPredictor` so callers can swap
the transformers backend for vLLM with minimal changes (e.g.
``backend="vllm"`` in benchmark / serving config).

vLLM offers paged-attention KV cache reuse, continuous batching, and CUDA
graph capture, which typically yields 3-10x throughput vs naive
``transformers.generate`` on the same GPU. We keep the chat-template
formatting consistent with the transformers backend so that generated
outputs are directly comparable in benchmarks.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from domain_llm_studio.data.formatters import TASK_INSTRUCTIONS
from domain_llm_studio.inference.predictor import FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)


_VLLM_IMPORT_ERROR = (
    "vLLM backend requested but vllm is not installed.\n"
    "Install via: pip install -e \".[vllm]\"  (requires CUDA + matching torch)."
)


def _import_vllm():
    """Import vllm lazily so the rest of the package doesn't hard-depend on it."""
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        raise ImportError(_VLLM_IMPORT_ERROR) from e
    return vllm


class VllmPredictor:
    """vLLM-backed predictor mirroring :class:`DomainPredictor`.

    Notes:
        - LoRA adapters are loaded lazily through ``LoRARequest`` per call so
          a single ``vllm.LLM`` instance can serve multiple adapters.
        - Chat template is applied via the HF tokenizer (vLLM accepts raw
          prompts) to keep parity with the transformers backend.
        - Greedy decoding is used by default to make benchmark numbers
          reproducible and to mirror the transformers predictor's
          ``do_sample=False``.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 512,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype

        self._vllm = None
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None
        self._lora_request = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------
    @property
    def vllm(self):
        if self._vllm is None:
            self._vllm = _import_vllm()
        return self._vllm

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from domain_llm_studio.training.model_loader import load_tokenizer
            self._tokenizer = load_tokenizer(self.model_path)
        return self._tokenizer

    @property
    def llm(self):
        if self._llm is None:
            vllm = self.vllm
            kwargs: dict[str, Any] = {
                "model": self.model_path,
                "dtype": self.dtype,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": True,
                "disable_log_stats": True,
            }
            if self.adapter_path:
                kwargs["enable_lora"] = True
                kwargs["max_lora_rank"] = 64
            logger.info("Initializing vLLM backend: %s", kwargs)
            self._llm = vllm.LLM(**kwargs)
        return self._llm

    @property
    def sampling_params(self):
        if self._sampling_params is None:
            self._sampling_params = self.vllm.SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.max_new_tokens,
                stop_token_ids=self._collect_stop_token_ids(),
            )
        return self._sampling_params

    @property
    def lora_request(self):
        if self.adapter_path is None:
            return None
        if self._lora_request is None:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest(
                lora_name="domain_adapter",
                lora_int_id=1,
                lora_path=self.adapter_path,
            )
        return self._lora_request

    def _collect_stop_token_ids(self) -> list[int]:
        ids: list[int] = []
        for token in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(token)
                if isinstance(tid, int) and tid >= 0:
                    ids.append(tid)
            except Exception:
                continue
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return list(dict.fromkeys(ids))

    # ------------------------------------------------------------------
    # Prompt construction (mirrors DomainPredictor._build_prompt)
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        task: str,
        input_text: str,
        model_type: str,
        question: str | None = None,
    ) -> str:
        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in input_text) else "en"
        instruction = TASK_INSTRUCTIONS.get(task, {}).get(lang, "")

        if task == "doc_qa" and question:
            try:
                data = json.loads(input_text)
                user_content = (
                    f"Context: {data.get('context', input_text)}\n\n"
                    f"Question: {data.get('question', question)}"
                )
            except (json.JSONDecodeError, TypeError):
                user_content = f"Context: {input_text}\n\nQuestion: {question}"
        else:
            user_content = input_text

        messages = [{"role": "system", "content": instruction}]

        if model_type == "prompt_only":
            examples = FEW_SHOT_EXAMPLES.get(task, {}).get(lang, [])
            if not examples:
                examples = FEW_SHOT_EXAMPLES.get(task, {}).get("en", [])
            for ex in examples:
                messages.append({"role": "user", "content": ex["input"]})
                messages.append({"role": "assistant", "content": ex["output"]})

        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        task: str,
        input_text: str,
        model_type: str = "base",
        question: str | None = None,
    ) -> str:
        """Run a single-sample inference. Returns generated text only."""
        prompt = self._build_prompt(task, input_text, model_type, question)
        outputs = self.llm.generate(
            [prompt],
            self.sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()

    def predict_batch(
        self,
        prompts: list[str],
    ) -> list[str]:
        """Run a batch of pre-formatted prompts. Returns generated texts.

        This bypasses :meth:`_build_prompt` so callers (e.g. benchmark) can
        precompute prompts once and time only the generation phase.
        """
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        return [o.outputs[0].text.strip() for o in outputs]

    def build_prompt(
        self,
        task: str,
        input_text: str,
        model_type: str = "base",
        question: str | None = None,
    ) -> str:
        """Public alias for prompt construction (used by benchmark)."""
        return self._build_prompt(task, input_text, model_type, question)
