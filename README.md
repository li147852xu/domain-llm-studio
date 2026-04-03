# Domain LLM Adaptation & Evaluation Studio

**A reproducible system for adapting open-source LLMs to domain-specific tasks such as financial and enterprise document understanding, information extraction, and structured summarization.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![CI](https://github.com/li147852xu/domain-llm-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/li147852xu/domain-llm-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen)]()

---

## Why This Project?

Large language models excel at general tasks, but domain-specific applications — financial analysis, enterprise document processing, compliance review — demand models that understand specialized terminology, follow strict output formats, and produce verifiable results.

This project demonstrates the **complete lifecycle** of adapting an open-source LLM to domain tasks:

```
Data Construction → PEFT Fine-Tuning → Prompt-only vs Tuned Eval → External Benchmark → Error Analysis → Serving
```

It is designed to complement agent/workflow platform projects by showing depth in the **model adaptation layer** — not just calling APIs, but understanding how to make models work better for specific domains.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Domain LLM Studio                               │
├──────────────┬───────────────┬──────────────────┬─────────────────────┤
│ Data Pipeline│ Training      │ Evaluation       │ Serving             │
├──────────────┼───────────────┼──────────────────┼─────────────────────┤
│ Seed Gen     │ Qwen2.5 Base  │ Internal 4-task  │ FastAPI Server      │
│ Cleaning     │ LoRA / QLoRA  │ External Bench   │ Gradio Demo         │
│ Formatting   │ SFTTrainer    │ 3 variants eval  │ Model Compare       │
│ Splitting    │ Config-driven │ Charts & Reports │ Preset Examples     │
└──────────────┴───────────────┴──────────────────┴─────────────────────┘
```

## Task Definitions

The system targets four concrete, evaluatable tasks in **financial/enterprise document intelligence** (bilingual: Chinese + English):

### Task 1: Financial Document Summarization (`fin_summary`)

| | |
|---|---|
| **Input** | Financial document excerpt (earnings report, announcement, research brief) |
| **Output** | JSON: `{summary, key_points[], risks[], opportunities[]}` |
| **Metrics** | ROUGE-1/2/L, keypoint coverage |

### Task 2: Event & Entity Extraction (`event_extraction`)

| | |
|---|---|
| **Input** | News paragraph or announcement excerpt |
| **Output** | JSON array: `[{company, event_type, date, metric, change_direction, sentiment}]` |
| **Metrics** | Entity P/R/F1, event F1, key presence rate, entity match rate, partial field match |

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
curl -LsSf https://astral.sh/uv/install.sh | sh
cd domain-llm-studio
uv sync
```

### End-to-End Pipeline with Makefile

```bash
make install          # Install dependencies
make data             # Build bilingual instruction dataset (400 samples)
make train-1.5b       # Train LoRA adapter on 1.5B (MPS/CPU, ~25 min)
make eval-1.5b        # Evaluate base + prompt_only + tuned
make compare-1.5b     # Generate comparison report with charts
make benchmark-1.5b   # Run FinanceBench external benchmark
make charts           # Generate comprehensive report
```

Or run individual commands:

```bash
uv run domain-llm-studio build-data --num-samples 50
uv run domain-llm-studio train --config configs/training/lora_1.5b_local.yaml
uv run domain-llm-studio eval --config configs/eval/base_1.5b.yaml
uv run domain-llm-studio eval --config configs/eval/prompt_only_1.5b.yaml
uv run domain-llm-studio eval --config configs/eval/tuned_1.5b.yaml
uv run domain-llm-studio compare --results-dir experiments/eval_1.5b
```

## Three-Way Evaluation: Base vs Prompt-Only vs LoRA-Tuned

A key design of this project is the **three-way comparison** that isolates the contribution of each technique:

| Variant | Description | Weights Modified? |
|---------|-------------|-------------------|
| **Base** | Zero-shot with task instruction only | No |
| **Prompt-only** | Few-shot in-context learning with curated examples | No |
| **LoRA-tuned** | Trained adapter on domain instruction data | Yes (1.18% params) |

## Results: Qwen2.5-1.5B (3 epochs, M4 Mac MPS, 25 min)

### Financial Summarization

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| ROUGE-1 | 0.658 | 0.667 | **0.928** | +0.270 |
| ROUGE-2 | 0.475 | 0.432 | **0.914** | **+0.439** |
| ROUGE-L | 0.598 | 0.628 | **0.925** | +0.327 |
| Keypoint Coverage | 0.545 | 0.512 | **0.814** | +0.269 |

### Event Extraction (after data-bug fix)

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| Entity F1 | 0.800 | 0.867 | **1.000** | +0.200 |
| Event F1 | 0.167 | 0.167 | **0.600** | **+0.433** |
| Entity Match Rate | 0.900 | 1.000 | **1.000** | +0.100 |
| Partial Field Match | 0.717 | 0.633 | **0.933** | +0.217 |
| Parse Failure Rate | 0.000 | 0.000 | 0.000 | — |

### Document QA

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| Exact Match | 0.800 | 0.800 | **1.000** | +0.200 |
| Token F1 | 0.917 | 0.800 | **1.000** | +0.083 |
| Grounding Rate | 0.800 | 0.800 | **1.000** | +0.200 |

### Structured Analysis

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| Format Compliance | 1.000 | 1.000 | 1.000 | — |
| Field Completeness | 0.500 | 0.700 | **1.000** | +0.500 |
| Schema Match | 0.750 | 0.850 | **1.000** | +0.250 |

### Error Analysis (1.5B)

| | Base | Prompt-Only | Tuned |
|---|------|-------------|-------|
| Error Rate | 100% | 100% | **45%** |
| Partial Match | 28 | 33 | 13 |
| Format Violation | 10 | 0 | 0 |
| Hallucination | 0 | 5 | 5 |
| Grounding Failure | 2 | 2 | 0 |

> **Training:** 3 epochs, 60 steps, 25 min on Apple M4 MPS. LoRA r=16/α=32, 18.5M trainable params (1.18% of 1.56B). Train loss 1.878 → 0.207, eval loss 0.200.

## External Benchmark: FinanceBench

[FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) (PatronusAI) is a 150-question QA benchmark over real SEC filings. We use it purely for **external generalization evaluation** — no training on this data.

### 1.5B Results on FinanceBench (8 samples, built-in)

| Metric | Base | Prompt-Only | Tuned |
|--------|------|-------------|-------|
| Exact Match | 0.000 | 0.000 | 0.000 |
| Token F1 | 0.062 | 0.086 | **0.158** |
| Grounding Rate | 0.125 | **0.375** | 0.250 |

> FinanceBench is intentionally hard — questions require reading real 10-K filings with complex tables. The low scores are expected for a 1.5B model and demonstrate honest external evaluation. The tuned model shows a 2.5x improvement in Token F1 over base, confirming that domain adaptation transfers to unseen benchmarks.

## Event Extraction: Deep Analysis

Event extraction is the most challenging task in the system. Here's what we learned:

**Data bug found and fixed:** The original `seed_generator.py` randomly assigned `event_type` independently of the narrative template. This meant a "layoff" template could get `event_type: "acquisition"` as the gold label, making the task impossible. After aligning templates with event types, Entity F1 jumped from 0.0 to 0.8 (base) and 1.0 (tuned).

**Why Event F1 remains at 0.6 (not 1.0):** Full event matching requires *all fields* — company, event_type, date, and sentiment — to match exactly. The model often gets the sentiment wrong (e.g., predicting "negative" for a regulatory approval), showing that multi-field structured extraction with nuanced judgment remains inherently difficult.

**Diagnostic metrics tell the full story:**
- `key_presence_rate = 1.0` — the model always produces the right JSON schema
- `entity_match_rate = 1.0` — company names are always correct after tuning
- `partial_field_match = 0.93` — on average, 5.6 out of 6 fields are correct
- `event_f1 = 0.6` — full exact match is the strict bar

## Cloud Workflow (7B on NVIDIA GPU)

```bash
git clone https://github.com/li147852xu/domain-llm-studio.git && cd domain-llm-studio
uv sync

# Download model (if needed)
export HF_ENDPOINT=https://hf-mirror.com
uv run hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# Full pipeline
make data
make train-7b
make eval-7b
make compare-7b
make benchmark-7b
```

After training, download lightweight result files to local:

```
experiments/train/lora_7b/training_log.json
experiments/train/lora_7b/training_summary.json
experiments/eval_7b/eval_base.json
experiments/eval_7b/eval_prompt_only.json
experiments/eval_7b/eval_tuned.json
experiments/comparison_7b/
experiments/benchmark/financebench_7b/
```

## Data Pipeline

```
Seed Generation → Cleaning & Dedup → Instruction Formatting → Stratified Split → Statistics
    (Templates)      (normalize,         (task templates,        (80/10/10,         (per-task
                      dedup, filter)       chat format)           seed=42)           reports)
```

**Key design decisions:**
- Template-based synthesis with financial domain vocabulary banks (20 companies, 12 metrics, 6 event types per language)
- **Template-event alignment**: each event template maps to its correct event_type, preventing label noise
- Deterministic generation with fixed random seed for full reproducibility
- Content-hash deduplication and JSON validation for quality control
- Stratified splitting ensures balanced task/language distribution

## Training

### Model Strategy

| Variant | Model | Use Case |
|---------|-------|----------|
| Local dev | `Qwen/Qwen2.5-1.5B-Instruct` | M4 Mac, MPS, rapid iteration |
| Cloud | `Qwen/Qwen2.5-7B-Instruct` | RTX 5090/A100, production quality |

### LoRA Configuration

- `r=16, alpha=32, dropout=0.05`
- Auto-detected target modules (q/k/v/o/gate/up/down projections)
- SFTTrainer from TRL with chat-template formatting
- Device auto-detection: CUDA > MPS > CPU
- Version-agnostic TRL compatibility (SFTConfig for >= 1.0, fallback for older)

## Evaluation System

### Metrics by Task

| Task | Primary Metrics | Diagnostic Metrics |
|------|----------------|-------------------|
| `fin_summary` | ROUGE-1/2/L, keypoint coverage | — |
| `event_extraction` | Entity P/R/F1, Event F1 | key_presence_rate, entity_match_rate, partial_field_match |
| `doc_qa` | Exact match, Token F1, grounding rate | — |
| `analysis_gen` | Format compliance, field completeness, schema match | — |

### Error Analysis Categories

| Error Type | Description |
|------------|-------------|
| **Hallucination** | Output contains entities/claims not present in input |
| **Missing extraction** | Gold entities not found in prediction |
| **Format violation** | Output fails to parse as expected JSON schema |
| **Truncation** | Output significantly shorter than expected |
| **Grounding failure** | QA answer not traceable to context |
| **Partial match** | Semantically similar but not exact match |

## Inference & Demo

### FastAPI Server

```bash
uv run domain-llm-studio serve --adapter-path experiments/train/lora_1.5b/adapter
```

**Endpoints:** `POST /predict`, `POST /compare`, `GET /tasks`, `GET /health`

### Gradio Web Demo

```bash
uv run domain-llm-studio web --adapter-path experiments/train/lora_1.5b/adapter
```

Features: task selector, model variant selector (base/prompt-only/tuned), 8 bilingual preset examples, side-by-side comparison tab.

### CLI Demo

```bash
uv run domain-llm-studio demo  # Runs all preset examples through base + prompt_only + tuned
```

## Project Structure

```
domain-llm-studio/
├── pyproject.toml                      # Dependencies, CLI entry points
├── Makefile                            # Common targets (install, data, train, eval, charts)
├── .github/workflows/ci.yml           # GitHub Actions (lint + test)
├── configs/
│   ├── tasks/                          # Per-task YAML configs
│   ├── training/                       # Training profiles (1.5B local, 7B cloud, QLoRA)
│   └── eval/                           # Eval configs: base/prompt_only/tuned × 1.5b/7b
├── data/
│   ├── seed/                           # Generated seed data (committed)
│   └── processed/                      # Cleaned instruction data (train/dev/test JSONL)
├── src/domain_llm_studio/
│   ├── cli.py                          # Typer CLI (12 commands)
│   ├── config.py                       # Pydantic v2 configuration models
│   ├── data/                           # Seed gen, cleaning, formatting, splitting
│   ├── training/                       # Model loading, SFTTrainer, callbacks
│   ├── evaluation/
│   │   ├── metrics/                    # ROUGE, extraction F1, QA, generation quality
│   │   ├── runner.py                   # Eval orchestrator (supports prompt_only)
│   │   ├── comparator.py              # Multi-model comparison with chart generation
│   │   ├── error_analysis.py          # Error classification
│   │   └── report.py                  # Charts, markdown, cross-model reports
│   ├── benchmark/
│   │   └── financebench.py            # External FinanceBench evaluation
│   ├── inference/                      # FastAPI server, predictor, schemas
│   └── web/                            # Gradio demo
├── tests/                              # 34 tests covering data, metrics, inference
├── docs/results/                       # Generated charts and reports
└── experiments/                        # Results (weights gitignored, JSONs committed)
    ├── train/lora_{1.5b,7b}/           # Training artifacts
    ├── eval_{1.5b,7b}/                 # eval_base.json, eval_prompt_only.json, eval_tuned.json
    ├── comparison_{1.5b,7b}/           # Comparison reports + charts
    └── benchmark/financebench_{1.5b,7b}/  # External benchmark results
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Model** | Qwen2.5-1.5B/7B-Instruct |
| **Training** | PyTorch, Transformers, PEFT (LoRA/QLoRA), TRL (SFTTrainer) |
| **Evaluation** | rouge-score, custom task metrics, FinanceBench |
| **Serving** | FastAPI, Uvicorn, Gradio |
| **Infrastructure** | Pydantic v2, Typer, Rich, Matplotlib |
| **Dev** | uv, pytest, ruff, GitHub Actions CI |

## CLI Reference

```
domain-llm-studio build-data       Build and clean instruction-tuning dataset
domain-llm-studio inspect-data     Show dataset statistics and summary
domain-llm-studio train            Run LoRA / QLoRA fine-tuning
domain-llm-studio eval             Run evaluation on the test set
domain-llm-studio compare          Generate base vs prompt-only vs tuned comparison
domain-llm-studio benchmark-eval   Run external benchmark evaluation
domain-llm-studio generate-report  Generate comprehensive report with charts
domain-llm-studio serve            Launch FastAPI inference server
domain-llm-studio web              Launch Gradio web demo
domain-llm-studio demo             Run preset examples through all model variants
domain-llm-studio inspect-results  Show latest evaluation summary
```

## Resume Talking Points

This project demonstrates:

- **Domain-specific LLM adaptation** — not just calling APIs, but adapting models to specialized tasks with measurable improvement
- **Three-way evaluation** (base vs prompt-only vs tuned) — isolates the contribution of prompting vs fine-tuning
- **Internal + external evaluation** — custom task metrics plus FinanceBench public benchmark
- **Data quality matters** — discovered and fixed a data generation bug that was causing 0.0 extraction F1, demonstrating systematic debugging
- **Task-specific metrics** — custom evaluation beyond perplexity (entity F1, grounding rate, field completeness, key presence rate)
- **Error analysis** — systematic classification of model failures with actionable categories
- **Parameter-efficient fine-tuning** — LoRA/QLoRA with config-driven experiment management
- **Reproducible pipeline** — Makefile, CI/CD, seed-deterministic data, config-driven experiments
- **Financial/enterprise intelligence** — relevant to fintech, banking, enterprise AI roles

## License

MIT
