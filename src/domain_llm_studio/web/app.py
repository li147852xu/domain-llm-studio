"""Gradio web demo with task selection, model comparison, and preset examples."""

from __future__ import annotations

import json
import logging

import gradio as gr

logger = logging.getLogger(__name__)

PRESET_EXAMPLES = [
    {
        "task": "fin_summary",
        "lang": "en",
        "label": "[EN] Apple Q3 Earnings Summary",
        "input_text": (
            "Apple Inc. reported Q3 2024 financial results. Total revenue reached $85.8 billion, "
            "representing a 4.9% increase year-over-year. Net income came in at $21.4 billion, "
            "with gross margin at 46.3%. The company highlighted strong performance in its Services "
            "segment, which grew 14% to $24.2 billion. iPhone revenue was $42.3 billion, slightly "
            "above analyst expectations. Management raised full-year guidance citing growing demand "
            "for AI-enabled devices. However, the company noted headwinds from foreign exchange "
            "fluctuations and increased competition in the China market."
        ),
        "question": None,
    },
    {
        "task": "fin_summary",
        "lang": "zh",
        "label": "[ZH] 贵州茅台半年报摘要",
        "input_text": (
            "贵州茅台发布2024年半年度业绩报告。报告期内，公司实现营业收入819.31亿元，"
            "同比增长17.76%。归属于上市公司股东的净利润为416.96亿元，同比增长15.88%。"
            "毛利率保持在92%以上的高水平。公司直销渠道收入占比持续提升，线上渠道增长显著。"
            "管理层表示将持续优化产品结构，推动品牌国际化。期间费用率有所优化，"
            "经营现金流保持健康水平。不过，高端白酒市场竞争加剧，库存管理面临挑战。"
        ),
        "question": None,
    },
    {
        "task": "event_extraction",
        "lang": "en",
        "label": "[EN] NVIDIA Earnings Event",
        "input_text": (
            "NVIDIA reported record fiscal Q4 2025 revenue of $22.1 billion on February 21, "
            "a 265% surge from the year-ago period, driven by explosive demand for its AI training "
            "chips. The company announced a new strategic partnership with Microsoft to accelerate "
            "enterprise AI deployment. CEO Jensen Huang said demand for Blackwell GPUs far exceeds supply."
        ),
        "question": None,
    },
    {
        "task": "event_extraction",
        "lang": "zh",
        "label": "[ZH] 宁德时代投资事件",
        "input_text": (
            "宁德时代于2024年8月15日宣布，将投资100亿元在匈牙利建设第二座欧洲电池工厂，"
            "预计2027年投产。同日，公司发布了最新一代神行超充电池，充电速度提升50%。"
            "此外，公司与特斯拉续签了为期五年的电池供应合同。"
        ),
        "question": None,
    },
    {
        "task": "doc_qa",
        "lang": "en",
        "label": "[EN] Tesla Delivery QA",
        "input_text": (
            "Tesla delivered 435,059 vehicles in Q3 2024, a 6.4% increase from Q2 2024. "
            "Model Y remained the best-selling vehicle globally. The company produced 469,796 "
            "vehicles during the quarter. Energy storage deployments reached a record 6.9 GWh. "
            "The Cybertruck began volume production at Gigafactory Texas. Tesla's automotive "
            "gross margin excluding regulatory credits was 17.1%."
        ),
        "question": "How many vehicles did Tesla deliver in Q3 2024?",
    },
    {
        "task": "doc_qa",
        "lang": "zh",
        "label": "[ZH] 招商银行财报问答",
        "input_text": (
            "招商银行2024年上半年实现营业收入1729.47亿元，同比下降3.09%。"
            "净利润为747.43亿元，同比增长1.15%。不良贷款率为0.94%，较年初持平。"
            "零售贷款余额为3.12万亿元，其中住房按揭贷款占比35%。"
            "理财业务管理资产规模突破4万亿元。资本充足率为17.88%。"
        ),
        "question": "招商银行上半年净利润是多少？",
    },
    {
        "task": "analysis_gen",
        "lang": "en",
        "label": "[EN] Meta Performance Analysis",
        "input_text": json.dumps({
            "company": "Meta Platforms",
            "period": "Q3 2024",
            "revenue": "$40.6 billion",
            "yoy_growth": "+23%",
            "net_income": "$15.7 billion",
            "margin": "43.1%",
            "segment_highlight": "AI-driven ad targeting improved click-through rates by 18%",
        }),
        "question": None,
    },
    {
        "task": "analysis_gen",
        "lang": "zh",
        "label": "[ZH] 比亚迪业绩分析",
        "input_text": json.dumps({
            "company": "比亚迪",
            "period": "2024年上半年",
            "revenue": "3011.3亿元",
            "yoy_growth": "+15.8%",
            "net_income": "136.3亿元",
            "margin": "20.0%",
            "segment_highlight": "新能源汽车销量161万辆，同比增长28.5%",
        }, ensure_ascii=False),
        "question": None,
    },
]

TASK_DESCRIPTIONS = {
    "fin_summary": "Financial Document Summarization — Generate structured summaries with key points, risks, and opportunities",
    "event_extraction": "Event & Entity Extraction — Extract structured events from financial text",
    "doc_qa": "Document QA — Answer questions based strictly on the given context",
    "analysis_gen": "Structured Analysis Generation — Generate professional analysis memos from data",
}


def _build_demo(predictor=None):
    """Build the Gradio interface. If predictor is None, runs in display-only mode."""

    def predict_fn(task, model_type, input_text, question):
        if predictor is None:
            return "(Model not loaded. Start with: domain-llm-studio web --model-path ...)"
        if not input_text.strip():
            return "Please enter input text."
        return predictor.predict(
            task=task, input_text=input_text, model_type=model_type, question=question or None
        )

    def compare_fn(task, input_text, question):
        if predictor is None:
            return "(Model not loaded)", "(Model not loaded)", "(Model not loaded)"
        if not input_text.strip():
            return "Please enter input text.", "", ""
        results = predictor.compare(task=task, input_text=input_text, question=question or None)
        return results.get("base", ""), results.get("prompt_only", ""), results.get("tuned", "")

    def load_example(example_label):
        for ex in PRESET_EXAMPLES:
            if ex["label"] == example_label:
                return ex["task"], ex["input_text"], ex.get("question", "") or ""
        return "fin_summary", "", ""

    example_labels = [ex["label"] for ex in PRESET_EXAMPLES]

    with gr.Blocks(title="Domain LLM Studio") as demo:
        gr.HTML('<h1 style="text-align:center">Domain LLM Adaptation & Evaluation Studio</h1>')
        gr.HTML(
            '<p style="text-align:center;color:#666;font-size:0.95em">'
            "A reproducible system for domain-specific LLM adaptation, "
            "evaluation, and serving</p>"
        )

        with gr.Tab("Single Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    task_dropdown = gr.Dropdown(
                        choices=list(TASK_DESCRIPTIONS.keys()),
                        value="fin_summary",
                        label="Task Type",
                        info="Select the task to perform",
                    )
                    task_desc = gr.Textbox(
                        value=TASK_DESCRIPTIONS["fin_summary"],
                        label="Task Description",
                        interactive=False,
                        lines=2,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=["base", "prompt_only", "tuned"],
                        value="base",
                        label="Model Variant",
                        info="base: zero-shot | prompt_only: few-shot | tuned: LoRA fine-tuned",
                    )
                    preset_dropdown = gr.Dropdown(
                        choices=example_labels,
                        label="Preset Examples",
                        info="Load a preset example",
                    )

                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Input Text",
                        lines=8,
                        placeholder="Paste financial document, news, or data here...",
                    )
                    question_text = gr.Textbox(
                        label="Question (for Document QA only)",
                        lines=2,
                        placeholder="Enter your question here...",
                    )

            predict_btn = gr.Button("Generate", variant="primary", size="lg")
            output_text = gr.Textbox(label="Model Output", lines=10)

            task_dropdown.change(
                lambda t: TASK_DESCRIPTIONS.get(t, ""),
                inputs=[task_dropdown],
                outputs=[task_desc],
            )
            preset_dropdown.change(
                load_example,
                inputs=[preset_dropdown],
                outputs=[task_dropdown, input_text, question_text],
            )
            predict_btn.click(
                predict_fn,
                inputs=[task_dropdown, model_dropdown, input_text, question_text],
                outputs=[output_text],
            )

        with gr.Tab("Model Comparison"):
            gr.Markdown(
                "### Side-by-Side Comparison\n"
                "Compare outputs from **Base** (zero-shot), **Prompt-Only** (few-shot), "
                "and **Tuned** (LoRA fine-tuned) model variants on the same input."
            )
            with gr.Row():
                cmp_task = gr.Dropdown(
                    choices=list(TASK_DESCRIPTIONS.keys()),
                    value="fin_summary",
                    label="Task Type",
                )
                cmp_preset = gr.Dropdown(
                    choices=example_labels,
                    label="Preset Examples",
                )

            cmp_input = gr.Textbox(label="Input Text", lines=6)
            cmp_question = gr.Textbox(label="Question (QA only)", lines=2)

            cmp_btn = gr.Button("Compare All Models", variant="primary", size="lg")

            with gr.Row():
                cmp_base = gr.Textbox(label="Base (Zero-Shot)", lines=10)
                cmp_prompt = gr.Textbox(label="Prompt-Only (Few-Shot)", lines=10)
                cmp_tuned = gr.Textbox(label="Tuned (LoRA)", lines=10)

            cmp_preset.change(
                load_example,
                inputs=[cmp_preset],
                outputs=[cmp_task, cmp_input, cmp_question],
            )
            cmp_btn.click(
                compare_fn,
                inputs=[cmp_task, cmp_input, cmp_question],
                outputs=[cmp_base, cmp_prompt, cmp_tuned],
            )

        with gr.Tab("About"):
            gr.Markdown("""
## Domain LLM Adaptation & Evaluation Studio

This system demonstrates a complete pipeline for adapting open-source LLMs
to domain-specific tasks in **financial and enterprise document intelligence**.

### Supported Tasks

| Task | Input | Output |
|------|-------|--------|
| **Financial Summarization** | Document excerpt | Structured summary (JSON) |
| **Event Extraction** | News/announcement | Structured events (JSON array) |
| **Document QA** | Context + Question | Answer + Evidence span |
| **Analysis Generation** | Structured data | Professional memo paragraph |

### Model Variants

- **Base**: Zero-shot inference with the base Qwen2.5 model
- **Prompt-Only**: Few-shot in-context learning with curated examples
- **Tuned**: LoRA fine-tuned adapter on domain instruction data

### Technical Stack

PyTorch, Transformers, PEFT, TRL, FastAPI, Gradio
""")

    return demo


def launch_demo(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: str | None = None,
    port: int = 7860,
    share: bool = False,
):
    """Launch the Gradio demo with a loaded model."""
    from domain_llm_studio.inference.predictor import DomainPredictor

    logger.info("Loading model for demo: %s", model_path)
    predictor = DomainPredictor(model_path, adapter_path)
    demo = _build_demo(predictor)
    demo.launch(server_port=port, share=share)
