# Domain LLM Adaptation & Evaluation Studio

**A reproducible system for adapting open-source LLMs to domain-specific tasks such as financial and enterprise document understanding, information extraction, and structured summarization.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)]()

---

## Why This Project?

Large language models excel at general tasks, but domain-specific applications — financial analysis, enterprise document processing, compliance review — demand models that understand specialized terminology, follow strict output formats, and produce verifiable results.

This project demonstrates the **complete lifecycle** of adapting an open-source LLM to domain tasks:

```
Dataset Construction → Parameter-Efficient Fine-Tuning → Systematic Evaluation → Error Analysis → Inference Serving → Web Demo
```

It is designed to complement agent/workflow platform projects by showing depth in the **model adaptation layer** — not just calling APIs, but understanding how to make models work better for specific domains.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Domain LLM Studio                          │
├──────────────┬───────────────┬──────────────┬────────────────┤
│ Data Pipeline│ Training      │ Evaluation   │ Serving        │
├──────────────┼───────────────┼──────────────┼────────────────┤
│ Seed Gen     │ Qwen2.5 Base  │ ROUGE/F1/EM  │ FastAPI Server │
│ Cleaning     │ LoRA / QLoRA  │ BERTScore    │ Gradio Demo    │
│ Formatting   │ SFTTrainer    │ Error Types  │ Model Compare  │
│ Splitting    │ Config-driven │ Comparison   │ Preset Examples│
└──────────────┴───────────────┴──────────────┴────────────────┘
```

## Task Definitions

The system targets four concrete, evaluatable tasks in **financial/enterprise document intelligence** (bilingual: Chinese + English):

### Task 1: Financial Document Summarization (`fin_summary`)

| | |
|---|---|
| **Input** | Financial document excerpt (earnings report, announcement, research brief) |
| **Output** | JSON: `{summary, key_points[], risks[], opportunities[]}` |
| **Metrics** | ROUGE-1/2/L, BERTScore, key-point coverage |

### Task 2: Event & Entity Extraction (`event_extraction`)

| | |
|---|---|
| **Input** | News paragraph or announcement excerpt |
| **Output** | JSON array: `[{company, event_type, date, metric, change_direction, sentiment}]` |
| **Metrics** | Entity P/R/F1, event-level exact match F1 |

### Task 3: Document Question Answering (`doc_qa`)

| | |
|---|---|
| **Input** | Context document + question |
| **Output** | JSON: `{answer, evidence_span}` |
| **Metrics** | Exact match, token F1, evidence grounding rate |

### Task 4: Structured Analysis Generation (`analysis_gen`)

| | |
|---|---|
| **Input** | Structured data points (company, metrics, period, changes) |
| **Output** | Professional analysis memo paragraph |
| **Metrics** | Format compliance, field completeness, schema match |

## Quick Start

### Prerequisites

- macOS / Linux
- 16GB+ RAM (for local inference with 1.5B model)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd domain-llm-studio
uv sync

# Download model (optional — auto-downloads on first use)
uv run python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct
```

### End-to-End Pipeline

```bash
# 1. Build instruction-tuning dataset (bilingual, 400 samples)
uv run domain-llm-studio build-data --num-samples 50

# 2. Inspect dataset statistics
uv run domain-llm-studio inspect-data

# 3. Train LoRA adapter (local: 1.5B model on MPS/CPU)
uv run domain-llm-studio train --config configs/training/lora_1.5b_local.yaml

# 4. Run evaluation
uv run domain-llm-studio eval --config configs/eval/default.yaml

# 5. Generate comparison report
uv run domain-llm-studio compare

# 6. Launch web demo
uv run domain-llm-studio web

# 7. Launch API server
uv run domain-llm-studio serve
```

## Data Pipeline

The data pipeline generates, cleans, formats, and splits bilingual instruction-tuning data:

```
Seed Generation → Cleaning & Dedup → Instruction Formatting → Stratified Split → Statistics
    (Jinja2)        (normalize,         (task templates,        (80/10/10,         (per-task
     templates)      dedup, filter)       chat format)           seed=42)           reports)
```

**Key design decisions:**
- Template-based synthesis with financial domain vocabulary banks (20 companies, 12 metrics, 8 event types per language)
- Deterministic generation with fixed random seed for full reproducibility
- Content-hash deduplication and JSON validation for quality control
- Stratified splitting ensures balanced task/language distribution

```bash
# Build with custom sample count
uv run domain-llm-studio build-data --num-samples 100

# View statistics
uv run domain-llm-studio inspect-data --data-dir data/processed
```

## Training

### Model Strategy

| Variant | Model | Use Case |
|---------|-------|----------|
| Local dev | `Qwen/Qwen2.5-1.5B-Instruct` | M4 Mac, MPS, rapid iteration |
| Cloud | `Qwen/Qwen2.5-7B-Instruct` | A100/4090, production quality |

### Three Experiment Modes

1. **Base model** — zero-shot inference with task instruction only
2. **Prompt-only** — few-shot in-context learning with curated examples
3. **LoRA fine-tuned** — trained adapter on domain instruction data

### LoRA Configuration

- `r=16, alpha=32, dropout=0.05`
- Auto-detected target modules (q/k/v/o/gate/up/down projections)
- SFTTrainer from TRL with chat-template formatting
- Device auto-detection: CUDA > MPS > CPU

```bash
# Local training (Apple Silicon)
uv run domain-llm-studio train --config configs/training/lora_1.5b_local.yaml

# Cloud training (NVIDIA GPU)
uv run domain-llm-studio train --config configs/training/lora_7b_cloud.yaml

# QLoRA (4-bit, CUDA only)
uv run domain-llm-studio train --config configs/training/qlora_7b_cloud.yaml
```

## Evaluation System

The evaluation system is the **project centerpiece** — demonstrating not just training ability but systematic assessment.

### Metrics by Task

| Task | Metrics |
|------|---------|
| `fin_summary` | ROUGE-1/2/L, BERTScore-F1, key-point coverage |
| `event_extraction` | Entity P/R/F1, event exact-match F1, parse failure rate |
| `doc_qa` | Exact match, token F1, evidence grounding rate |
| `analysis_gen` | Format compliance %, field completeness %, schema match |

### Error Analysis Categories

The system classifies prediction failures into actionable categories:

| Error Type | Description |
|------------|-------------|
| **Hallucination** | Output contains entities/claims not present in input |
| **Missing extraction** | Gold entities not found in prediction |
| **Format violation** | Output fails to parse as expected JSON schema |
| **Truncation** | Output significantly shorter than expected |
| **Grounding failure** | QA answer not traceable to context |

### Comparison Reports

```bash
# Run evaluation with base model
uv run domain-llm-studio eval

# Generate side-by-side comparison report
uv run domain-llm-studio compare

# View latest results
uv run domain-llm-studio inspect-results
```

Reports include per-task metric tables, delta columns, error distribution charts, and failure examples.

## Inference & Demo

### FastAPI Server

```bash
uv run domain-llm-studio serve --model-path Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

**Endpoints:**
- `POST /predict` — single model inference
- `POST /compare` — run all model variants, return side-by-side results
- `GET /tasks` — list available tasks with descriptions
- `GET /health` — status check

### Gradio Web Demo

```bash
uv run domain-llm-studio web --port 7860
```

**Features:**
- Task type selector (4 tasks)
- Model variant selector (base / prompt-only / tuned)
- 8 bilingual preset examples (2 per task, CN + EN)
- Side-by-side comparison tab
- Professional styling for interview demos

## Project Structure

```
domain-llm-studio/
├── pyproject.toml                      # Dependencies, CLI entry points
├── configs/
│   ├── tasks/                          # Per-task YAML configs
│   ├── training/                       # Training profiles (1.5B local, 7B cloud, QLoRA)
│   └── eval/                           # Evaluation configs
├── data/
│   ├── seed/                           # Generated seed data (committed)
│   └── processed/                      # Cleaned instruction data (train/dev/test JSONL)
├── src/domain_llm_studio/
│   ├── cli.py                          # Typer CLI (9 commands)
│   ├── config.py                       # Pydantic v2 configuration models
│   ├── data/
│   │   ├── seed_generator.py           # Template-based bilingual data synthesis
│   │   ├── cleaners.py                 # Dedup, normalize, validate
│   │   ├── formatters.py               # Instruction-tuning format conversion
│   │   ├── splitter.py                 # Stratified train/dev/test split
│   │   ├── stats.py                    # Dataset statistics & reporting
│   │   └── builder.py                  # Pipeline orchestrator
│   ├── training/
│   │   ├── model_loader.py             # Model loading with device auto-detection
│   │   ├── trainer.py                  # SFTTrainer with LoRA/QLoRA
│   │   └── callbacks.py               # Loss logging, training summary
│   ├── evaluation/
│   │   ├── metrics/                    # ROUGE, extraction F1, QA metrics, generation quality
│   │   ├── runner.py                   # Evaluation orchestrator
│   │   ├── comparator.py              # Multi-model comparison
│   │   ├── error_analysis.py          # Error classification
│   │   └── report.py                  # Markdown/HTML report generation
│   ├── inference/
│   │   ├── server.py                   # FastAPI application
│   │   ├── predictor.py               # Model inference wrapper
│   │   └── schemas.py                 # Pydantic request/response models
│   └── web/
│       └── app.py                      # Gradio demo with comparison view
├── tests/                              # 31 tests covering data, metrics, inference
├── scripts/
│   └── download_model.py              # Model download utility
└── experiments/                        # Training outputs, eval results (gitignored)
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Model** | Qwen2.5-1.5B/7B-Instruct |
| **Training** | PyTorch, Transformers, PEFT (LoRA/QLoRA), TRL (SFTTrainer) |
| **Evaluation** | rouge-score, bert-score, custom task metrics |
| **Serving** | FastAPI, Uvicorn, Gradio |
| **Infrastructure** | Pydantic v2, Typer, Rich, Pandas, Matplotlib |
| **Dev** | uv, pytest, ruff |

## CLI Reference

```
domain-llm-studio build-data     Build and clean instruction-tuning dataset
domain-llm-studio inspect-data   Show dataset statistics and summary
domain-llm-studio train          Run LoRA / QLoRA fine-tuning
domain-llm-studio eval           Run evaluation on the test set
domain-llm-studio compare        Generate base vs tuned comparison report
domain-llm-studio serve          Launch FastAPI inference server
domain-llm-studio web            Launch Gradio web demo
domain-llm-studio demo           Run preset examples through all models
domain-llm-studio inspect-results Show latest evaluation summary
```

## Resume Talking Points

This project demonstrates:

- **Domain-specific LLM adaptation** — not just calling APIs, but adapting models to specialized tasks
- **Instruction data construction** — bilingual template-based synthesis with quality controls
- **Parameter-efficient fine-tuning** — LoRA/QLoRA with config-driven experiment management
- **Task-specific evaluation design** — custom metrics per task, not just perplexity
- **Error analysis** — systematic classification of model failures (hallucination, format errors, grounding)
- **Base vs. tuned comparison** — quantified improvement with side-by-side demonstration
- **Model serving & deployment** — FastAPI API + Gradio demo ready for production patterns
- **Financial/enterprise document intelligence** — relevant to fintech, banking, enterprise AI roles

## Non-Goals

This project is explicitly **not**:

- A chatbot or conversational AI system
- A multi-agent or RAG platform
- A stock prediction or trading system
- A pre-training framework (we adapt, not train from scratch)
- A pure research paper reproduction
- A prompt engineering demo without model adaptation

## Applicable Roles

- LLM Application Engineer
- Model Fine-tuning / Adaptation Engineer
- NLP / AI Engineer (Financial Domain)
- AI Evaluation Engineer
- Machine Learning Engineer (LLM)
- Financial AI / Document Intelligence

## License

MIT
