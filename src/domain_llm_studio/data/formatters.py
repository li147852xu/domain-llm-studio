"""Instruction-tuning data formatters for each task type.

Converts raw samples into {instruction, input, output} format suitable
for supervised fine-tuning with chat-template models.
"""

from __future__ import annotations

import json
from pathlib import Path

from domain_llm_studio.config import TaskConfig, load_task_configs

# ---------------------------------------------------------------------------
# Per-task formatting
# ---------------------------------------------------------------------------

TASK_INSTRUCTIONS = {
    "fin_summary": {
        "en": (
            "You are a financial analyst assistant. Given the following financial document excerpt, "
            "produce a structured summary in JSON format with fields: summary, key_points, risks, opportunities. "
            "Respond ONLY with valid JSON."
        ),
        "zh": (
            "你是一名金融分析助手。请根据以下金融文档片段，生成结构化摘要，包含以下JSON字段："
            "summary（摘要）、key_points（要点列表）、risks（风险列表）、opportunities（机会列表）。"
            "请仅输出有效JSON。"
        ),
    },
    "event_extraction": {
        "en": (
            "You are an information extraction assistant specializing in financial events. "
            "Given the following text, extract all events into a JSON array. Each event should have: "
            "company, event_type, date, metric, change_direction, sentiment. "
            "Respond ONLY with a valid JSON array."
        ),
        "zh": (
            "你是一名专注于金融事件的信息抽取助手。请从以下文本中提取所有事件，"
            "以JSON数组格式输出。每个事件包含：company（公司）、event_type（事件类型）、"
            "date（日期）、metric（指标）、change_direction（变化方向）、sentiment（情感倾向）。"
            "请仅输出有效JSON数组。"
        ),
    },
    "doc_qa": {
        "en": (
            "You are a document QA assistant. Answer the question based STRICTLY on the provided context. "
            "Respond in JSON format with fields: answer, evidence_span. "
            "If unanswerable, set answer to 'unanswerable' and evidence_span to null."
        ),
        "zh": (
            "你是一名文档问答助手。请严格根据给定的上下文回答问题。"
            "以JSON格式输出，包含：answer（答案）、evidence_span（证据片段）。"
            "如果无法从上下文中找到答案，请将answer设为'unanswerable'，evidence_span设为null。"
        ),
    },
    "analysis_gen": {
        "en": (
            "You are a financial analysis assistant. Given structured data points about a company's "
            "performance, generate a professional analysis memo paragraph covering: "
            "opening, key metrics, analysis, and outlook."
        ),
        "zh": (
            "你是一名金融分析助手。请根据以下公司业绩结构化数据，"
            "生成一段专业的分析备忘录，涵盖：开头概述、关键指标、分析解读、未来展望。"
        ),
    },
}


def format_sample(sample: dict, task_configs: dict[str, TaskConfig] | None = None) -> dict:
    """Convert a raw sample into instruction-tuning format.

    Returns:
        {
            "instruction": str,
            "input": str,
            "output": str,
            "task": str,
            "lang": str,
        }
    """
    task = sample["task"]
    lang = sample.get("lang", "en")
    raw_input = sample["input"]
    raw_output = sample["output"]

    instruction = TASK_INSTRUCTIONS.get(task, {}).get(lang, "")

    if task_configs and task in task_configs:
        cfg = task_configs[task]
        if cfg.instruction_template:
            instruction = cfg.instruction_template

    if task == "doc_qa":
        try:
            qa_data = json.loads(raw_input)
            formatted_input = f"Context: {qa_data['context']}\n\nQuestion: {qa_data['question']}"
        except (json.JSONDecodeError, KeyError):
            formatted_input = raw_input
    else:
        formatted_input = raw_input

    return {
        "instruction": instruction,
        "input": formatted_input,
        "output": raw_output,
        "task": task,
        "lang": lang,
    }


def format_as_chat_messages(sample: dict) -> list[dict]:
    """Convert an instruction-tuning sample into chat message format
    suitable for Qwen2.5 and similar chat models."""
    return [
        {"role": "system", "content": sample["instruction"]},
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": sample["output"]},
    ]


def format_dataset(
    samples: list[dict],
    config_dir: Path | None = None,
) -> list[dict]:
    """Format all samples in a dataset."""
    task_configs = None
    if config_dir and config_dir.exists():
        task_configs = load_task_configs(config_dir)

    return [format_sample(s, task_configs) for s in samples]
