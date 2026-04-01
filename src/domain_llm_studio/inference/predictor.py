"""Model inference wrapper supporting base, prompt-only, and tuned variants."""

from __future__ import annotations

import json
import logging

import torch

from domain_llm_studio.data.formatters import TASK_INSTRUCTIONS

logger = logging.getLogger(__name__)

FEW_SHOT_EXAMPLES = {
    "fin_summary": {
        "en": [
            {
                "input": "Tesla reported Q2 2024 revenue of $25.5 billion, up 2% year-over-year. Net income was $1.8 billion. The company faced pricing pressure but maintained strong delivery volumes.",
                "output": '{"summary": "Tesla Q2 2024 revenue reached $25.5B (+2% YoY) with net income of $1.8B despite pricing headwinds.", "key_points": ["Revenue $25.5B, +2% YoY", "Net income $1.8B"], "risks": ["pricing pressure"], "opportunities": ["strong delivery volumes"]}',
            }
        ],
        "zh": [
            {
                "input": "比亚迪2024年上半年营收3011亿元，同比增长15.8%。净利润136.3亿元，新能源汽车销量持续领先。",
                "output": '{"summary": "比亚迪2024上半年营收3011亿元，同比增长15.8%，净利润136.3亿元。", "key_points": ["营收3011亿元", "同比增长15.8%", "净利润136.3亿元"], "risks": ["市场竞争加剧"], "opportunities": ["新能源汽车销量领先"]}',
            }
        ],
    },
    "event_extraction": {
        "en": [
            {
                "input": "NVIDIA announced record Q4 2024 revenue of $22.1 billion on Feb 21, driven by surging AI chip demand.",
                "output": '[{"company": "NVIDIA", "event_type": "earnings", "date": "2024-02-21", "metric": "revenue", "change_direction": "increase", "sentiment": "positive"}]',
            }
        ],
    },
    "doc_qa": {
        "en": [
            {
                "input": '{"context": "Amazon Web Services generated $24.2 billion in Q3 revenue, growing 12% year-over-year.", "question": "What was AWS Q3 revenue?"}',
                "output": '{"answer": "$24.2 billion", "evidence_span": "Amazon Web Services generated $24.2 billion in Q3 revenue"}',
            }
        ],
    },
    "analysis_gen": {
        "en": [
            {
                "input": '{"company": "Meta Platforms", "period": "Q3 2024", "revenue": "$40.6 billion", "yoy_growth": "+23%", "net_income": "$15.7 billion", "segment_highlight": "AI investments driving ad efficiency"}',
                "output": "Meta Platforms reported Q3 2024 revenue of $40.6 billion, a robust 23% year-over-year increase. Net income reached $15.7 billion, reflecting strong operational leverage. AI investments continue to drive advertising efficiency improvements. Looking ahead, sustained AI-driven ad targeting enhancements should support continued revenue momentum.",
            }
        ],
    },
}


class DomainPredictor:
    """Unified predictor supporting multiple model variants."""

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._base_model = None
        self._tuned_model = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from domain_llm_studio.training.model_loader import load_tokenizer
            self._tokenizer = load_tokenizer(self.model_path)
        return self._tokenizer

    @property
    def base_model(self):
        if self._base_model is None:
            from domain_llm_studio.training.model_loader import load_base_model
            self._base_model = load_base_model(self.model_path)
            self._base_model.eval()
        return self._base_model

    @property
    def tuned_model(self):
        if self._tuned_model is None:
            if self.adapter_path:
                from domain_llm_studio.training.model_loader import load_model_with_adapter
                self._tuned_model = load_model_with_adapter(self.model_path, self.adapter_path)
            else:
                self._tuned_model = self.base_model
            self._tuned_model.eval()
        return self._tuned_model

    def _build_prompt(
        self, task: str, input_text: str, model_type: str, question: str | None = None
    ) -> str:
        lang = "zh" if any("\u4e00" <= c <= "\u9fff" for c in input_text) else "en"
        instruction = TASK_INSTRUCTIONS.get(task, {}).get(lang, "")

        if task == "doc_qa" and question:
            try:
                data = json.loads(input_text)
                user_content = f"Context: {data.get('context', input_text)}\n\nQuestion: {data.get('question', question)}"
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

    def predict(
        self,
        task: str,
        input_text: str,
        model_type: str = "base",
        question: str | None = None,
    ) -> str:
        """Run inference with the specified model variant."""
        model = self.tuned_model if model_type == "tuned" else self.base_model
        prompt = self._build_prompt(task, input_text, model_type, question)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def compare(
        self, task: str, input_text: str, question: str | None = None
    ) -> dict[str, str]:
        """Run inference with all model variants and return results."""
        results = {}
        for variant in ["base", "prompt_only", "tuned"]:
            try:
                results[variant] = self.predict(task, input_text, variant, question)
            except Exception as e:
                results[variant] = f"Error: {e}"
        return results
