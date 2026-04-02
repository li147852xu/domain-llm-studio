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

### End-to-End Pipeline (1.5B Local)

```bash
# 1. Build instruction-tuning dataset (bilingual, 400 samples)
uv run domain-llm-studio build-data --num-samples 50

# 2. Inspect dataset statistics
uv run domain-llm-studio inspect-data

# 3. Train LoRA adapter (1.5B model on MPS/CPU)
uv run domain-llm-studio train --config configs/training/lora_1.5b_local.yaml

# 4. Evaluate base model + tuned model
uv run domain-llm-studio eval --config configs/eval/base_1.5b.yaml
uv run domain-llm-studio eval --config configs/eval/tuned_1.5b.yaml

# 5. Generate comparison report
uv run domain-llm-studio compare --config configs/eval/base_1.5b.yaml --output experiments/comparison_1.5b

# 6. Launch web demo
uv run domain-llm-studio web --adapter-path experiments/train/lora_1.5b/adapter

# 7. Launch API server
uv run domain-llm-studio serve --adapter-path experiments/train/lora_1.5b/adapter
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
| Cloud | `Qwen/Qwen2.5-7B-Instruct` | RTX 5090, production quality |

### Training Results

Both models were trained for 3 epochs with LoRA (r=16, α=32) on the same 320-sample instruction dataset:

| | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|
| **Device** | Apple M4 MPS | RTX 5090 CUDA |
| **Training time** | 25 min | **2 min** |
| **Trainable params** | 18.5M (1.18%) | 40.4M (0.53%) |
| **Final train loss** | 0.207 | 0.273 |
| **Final eval loss** | 0.208 | 0.267 |
| **Token accuracy** | 93.2% | 92.0% |
| **Learning rate** | 2e-4 | 1e-4 |

> Loss curve (7B): 2.114 → 1.144 → 0.641 → 0.382 → 0.306 → 0.273. No overfitting observed.

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
# === 1.5B Local (Apple Silicon / CPU) ===
uv run domain-llm-studio train --config configs/training/lora_1.5b_local.yaml

# === 7B Cloud (NVIDIA GPU) ===
uv run domain-llm-studio train --config configs/training/lora_7b_cloud.yaml

# === 7B QLoRA (4-bit, CUDA only) ===
uv run domain-llm-studio train --config configs/training/qlora_7b_cloud.yaml
```

### Cloud Workflow (7B on NVIDIA GPU)

```bash
git clone https://github.com/li147852xu/domain-llm-studio.git && cd domain-llm-studio
uv sync

# Build data + Train + Evaluate (zero config edits needed)
uv run domain-llm-studio build-data --num-samples 50
uv run domain-llm-studio train --config configs/training/lora_7b_cloud.yaml
uv run domain-llm-studio eval --config configs/eval/base_7b.yaml
uv run domain-llm-studio eval --config configs/eval/tuned_7b.yaml
uv run domain-llm-studio compare --config configs/eval/base_7b.yaml --output experiments/comparison_7b
```

After training completes, download these files to your local machine:

```
experiments/train/lora_7b/adapter/           # LoRA weights (~200-400MB)
experiments/train/lora_7b/training_log.json  # Training curve
experiments/train/lora_7b/training_summary.json
experiments/eval_7b/                         # Eval result JSONs
experiments/comparison_7b/                   # Comparison report
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

### Results: Base vs LoRA-Tuned (Qwen2.5-1.5B, 3 epochs, M4 Mac MPS)

**Financial Summarization** — ROUGE-2 nearly doubled:

| Metric | Base | LoRA-Tuned | Delta |
|--------|------|------------|-------|
| ROUGE-1 | 0.658 | **0.918** | +0.260 |
| ROUGE-2 | 0.475 | **0.902** | **+0.427** |
| ROUGE-L | 0.598 | **0.914** | +0.316 |
| Keypoint Coverage | 0.545 | **0.805** | +0.260 |

**Event Extraction** — from zero to functional:

| Metric | Base | LoRA-Tuned | Delta |
|--------|------|------------|-------|
| Entity F1 | 0.000 | **0.200** | +0.200 |
| Event F1 | 0.000 | **0.200** | +0.200 |

**Document QA** — reached perfect score:

| Metric | Base | LoRA-Tuned | Delta |
|--------|------|------------|-------|
| Exact Match | 0.900 | **1.000** | +0.100 |
| Token F1 | 0.955 | **1.000** | +0.045 |
| Grounding Rate | 1.000 | 1.000 | — |

**Structured Analysis** — field completeness fully resolved:

| Metric | Base | LoRA-Tuned | Delta |
|--------|------|------------|-------|
| Format Compliance | 1.000 | 1.000 | — |
| Field Completeness | 0.400 | **1.000** | **+0.600** |
| Schema Match | 0.700 | **1.000** | +0.300 |

**Error Analysis** — error rate dropped from 100% to 57.5%:

| | Base | LoRA-Tuned |
|---|------|------------|
| Error Rate | 100% | **57.5%** |
| Format Violation | 10 | **0** |
| Grounding Failure | 1 | **0** |
| Partial Match | 29 | 18 |
| Hallucination | 0 | 5 |

> Training: 3 epochs, 60 steps, 25 min on Apple M4 MPS. LoRA r=16/α=32, 18.5M trainable params (1.18% of 1.56B). Train loss converged from 1.878 → 0.207 with no overfitting (eval loss 0.208).

### Comparison Reports

```bash
# --- 1.5B model ---
uv run domain-llm-studio eval --config configs/eval/base_1.5b.yaml
uv run domain-llm-studio eval --config configs/eval/tuned_1.5b.yaml
uv run domain-llm-studio compare --config configs/eval/base_1.5b.yaml --output experiments/comparison_1.5b

# --- 7B model ---
uv run domain-llm-studio eval --config configs/eval/base_7b.yaml
uv run domain-llm-studio eval --config configs/eval/tuned_7b.yaml
uv run domain-llm-studio compare --config configs/eval/base_7b.yaml --output experiments/comparison_7b
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
│   └── eval/                           # Eval configs per model size (base_1.5b, tuned_7b, etc.)
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
└── experiments/                        # Per-model results (weights gitignored, JSONs committed)
    ├── train/lora_1.5b/               # 1.5B training artifacts
    ├── train/lora_7b/                 # 7B training artifacts
    ├── eval_1.5b/                     # 1.5B eval results (eval_base.json, eval_tuned.json)
    ├── eval_7b/                       # 7B eval results
    ├── comparison_1.5b/               # 1.5B comparison report
    └── comparison_7b/                 # 7B comparison report
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
