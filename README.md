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

## Results: Qwen2.5-7B (3 epochs, RTX 5090 CUDA, ~2 min)

### Financial Summarization

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| ROUGE-1 | 0.706 | 0.725 | **0.917** | +0.211 |
| ROUGE-2 | 0.507 | 0.549 | **0.900** | **+0.393** |
| ROUGE-L | 0.633 | 0.682 | **0.917** | +0.284 |
| Keypoint Coverage | 0.650 | 0.585 | **0.793** | +0.143 |

### Event Extraction

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| Entity F1 | 0.900 | 0.933 | **1.000** | +0.100 |
| Event F1 | 0.067 | 0.367 | **0.400** | +0.333 |
| Entity Match Rate | 1.000 | 1.000 | 1.000 | — |
| Partial Field Match | 0.850 | 0.883 | **0.900** | +0.050 |
| Parse Failure Rate | 0.000 | 0.000 | 0.000 | — |

### Document QA

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| Exact Match | 0.800 | 0.800 | **1.000** | +0.200 |
| Token F1 | 0.956 | 0.956 | **1.000** | +0.044 |
| Grounding Rate | 0.900 | **1.000** | **1.000** | +0.100 |

### Structured Analysis

| Metric | Base | Prompt-Only | Tuned | Δ (Tuned−Base) |
|--------|------|-------------|-------|-----------------|
| Format Compliance | 1.000 | 1.000 | 1.000 | — |
| Field Completeness | 0.867 | 0.800 | **1.000** | +0.133 |
| Schema Match | 0.933 | 0.900 | **1.000** | +0.067 |

### Error Analysis (7B)

| | Base | Prompt-Only | Tuned |
|---|------|-------------|-------|
| Error Rate | 100% | 92.5% | **52.5%** |
| Partial Match | 35 | 32 | 16 |
| Hallucination | 5 | 5 | 5 |

> **Training:** 3 epochs, 60 steps, **~2 min** on RTX 5090 CUDA (bf16). LoRA r=16/α=32, 40.4M trainable params (0.53% of 7.66B). Train loss 2.092 → 0.266, eval loss 0.256.

## Cross-Model Comparison (6 Variants)

| Task | Key Metric | 1.5B Base | 1.5B Prompt | 1.5B Tuned | 7B Base | 7B Prompt | 7B Tuned |
|------|-----------|-----------|-------------|------------|---------|-----------|----------|
| fin_summary | ROUGE-2 | 0.475 | 0.432 | **0.914** | 0.507 | 0.549 | 0.900 |
| event_extraction | Entity F1 | 0.800 | 0.867 | **1.000** | 0.900 | 0.933 | **1.000** |
| event_extraction | Event F1 | 0.167 | 0.167 | **0.600** | 0.067 | 0.367 | 0.400 |
| doc_qa | Exact Match | 0.800 | 0.800 | **1.000** | 0.800 | 0.800 | **1.000** |
| analysis_gen | Field Comp. | 0.500 | 0.700 | **1.000** | 0.867 | 0.800 | **1.000** |
| — | Error Rate | 100% | 100% | **45%** | 100% | 92.5% | 52.5% |

**Key observations:**
- **7B base is stronger out-of-the-box**: higher entity F1 (0.90 vs 0.80), higher field completeness (0.87 vs 0.50), higher ROUGE
- **Both scales benefit dramatically from LoRA tuning**: doc_qa and analysis_gen reach perfect scores at both scales
- **1.5B tuned slightly outperforms 7B tuned on some metrics**: event_f1 (0.60 vs 0.40) and error rate (45% vs 52.5%) — smaller models can be more tightly adapted to the training distribution
- **Prompt-only provides moderate gains for 7B**: event_f1 jumps from 0.067 → 0.367 with few-shot examples on 7B, showing larger models leverage in-context learning better
- **Training speed**: 7B on RTX 5090 (~2 min) vs 1.5B on M4 MPS (25 min) — GPU advantage is ~12x

## External Benchmark: FinanceBench

[FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) (PatronusAI) is a 150-question QA benchmark over real SEC filings. We use it purely for **external generalization evaluation** — no training on this data.

### 1.5B Results on FinanceBench (8 samples, built-in)

| Metric | Base | Prompt-Only | Tuned |
|--------|------|-------------|-------|
| Exact Match | 0.000 | 0.000 | 0.000 |
| Token F1 | 0.062 | 0.086 | **0.158** |
| Grounding Rate | 0.125 | **0.375** | 0.250 |

### 7B Results on FinanceBench (150 samples, full dataset)

| Metric | Base | Prompt-Only | Tuned |
|--------|------|-------------|-------|
| Exact Match | **0.020** | 0.013 | **0.020** |
| Token F1 | **0.167** | 0.133 | 0.119 |
| Grounding Rate | 0.180 | 0.200 | **0.320** |

> FinanceBench is intentionally hard — questions require reading real 10-K filings with complex tables. The low absolute scores are expected and demonstrate honest external evaluation. Key takeaway: the **tuned 7B model achieves 0.32 grounding rate** (78% improvement over base), meaning domain adaptation helps the model stay anchored to source evidence even on completely unseen financial documents.

## Event Extraction: Deep Analysis

Event extraction is the most challenging task in the system. Here's what we learned:

**Data bug found and fixed:** The original `seed_generator.py` randomly assigned `event_type` independently of the narrative template. This meant a "layoff" template could get `event_type: "acquisition"` as the gold label, making the task impossible. After aligning templates with event types, Entity F1 jumped from 0.0 to 0.8+ (base) and 1.0 (tuned).

**Why Event F1 remains moderate (0.4–0.6):** Full event matching requires *all fields* — company, event_type, date, metric, change_direction, and sentiment — to match exactly. The model often gets nuanced fields like sentiment wrong (e.g., predicting "negative" for a regulatory approval), showing that multi-field structured extraction with subjective judgment remains inherently difficult.

**Scale vs adaptation trade-off:** Interestingly, 1.5B tuned (event_f1=0.60) outperforms 7B tuned (0.40) on strict event matching. The smaller model fits the training distribution more tightly, while the 7B model's broader knowledge occasionally introduces plausible but non-exact field values.

**Diagnostic metrics tell the full story (7B tuned):**
- `key_presence_rate = 1.0` — the model always produces the correct JSON schema
- `entity_match_rate = 1.0` — company names are always correct after tuning
- `partial_field_match = 0.90` — on average, 5.4 out of 6 fields are correct
- `event_f1 = 0.40` — full exact match is the strict bar

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

## Inference Backend Comparison

The system ships with **two interchangeable inference backends** behind a single
`create_predictor(backend=...)` factory:

- `transformers` — eager `model.generate`, single request per call (default,
  best for development and small-scale evaluation)
- `vllm` — paged-attention KV cache + continuous batching (production serving)

Benchmark on RTX 5090 (32 GB), bf16, greedy decoding, 40 samples (10 per
task) across 4 financial tasks
(`fin_summary` / `event_extraction` / `doc_qa` / `analysis_gen`),
single-request-per-call (no client-side batching):

<!-- BENCHMARK_TABLE_START -->

| Model | Backend | Total(s) | P50(ms) | P95(ms) | Tokens/s | PeakMem(GB) |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | transformers | 90.8 | 1762 | 5984 | 69.5 | 3.5 |
| Qwen2.5-1.5B | **vLLM** | **19.6** | **372** | **1442** | **317.4** | 30.9 |
| Qwen2.5-7B  | transformers | 85.2 | 1870 | 6631 | 69.0 | 14.9 |
| Qwen2.5-7B  | **vLLM** | **63.1** | **1281** | **4852** | **103.9** | 30.4 |

| Model | Throughput speedup | P95 latency reduction |
|---|---|---|
| 1.5B | **4.57x** | **-75.9%** |
| 7B | **1.51x** | **-26.8%** |

Notes:
- vLLM peak memory is high because vLLM **pre-allocates ~85% of device
  memory as a paged-attention KV-cache pool** at engine init (not because
  inference itself uses more). The pool is what enables 100x+ concurrent
  request capacity in a real serving deployment.
- The 1.5B speedup is dramatic because vLLM's per-token overhead (CUDA
  graphs, paged attention) is amortized over more cheap forward passes.
  For 7B, the model forward dominates and the gap shrinks. Under a
  **batched** workload (the actual production use case), vLLM's
  continuous batching widens the gap further on both sizes.
- Reproduce with `python scripts/benchmark_inference.py --models 1.5b,7b
  --backends transformers,vllm --num-samples 50`. Raw numbers and charts
  live under `experiments/inference_benchmark/`.

<!-- BENCHMARK_TABLE_END -->

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
| **Serving** | FastAPI, Uvicorn, Gradio, vLLM (paged-attention KV cache) |
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

- **Multi-scale domain LLM adaptation** — trained and evaluated both 1.5B (local, Apple M4) and 7B (cloud, RTX 5090) models with LoRA
- **Six-variant evaluation matrix** (2 scales × 3 variants) — isolates the contribution of model scale, prompting, and fine-tuning
- **Internal + external evaluation** — custom 4-task metric suite plus FinanceBench (150 real SEC filing questions) public benchmark
- **Data quality engineering** — discovered and fixed a data generation bug causing 0.0 extraction F1, demonstrating systematic debugging
- **Task-specific metrics** — custom evaluation beyond perplexity: entity F1, grounding rate, field completeness, key presence rate, partial field match
- **Error analysis** — systematic classification of model failures (hallucination, grounding failure, format violation, partial match)
- **Scale vs adaptation insights** — found that 1.5B tuned can outperform 7B tuned on certain metrics, demonstrating cost-efficient adaptation
- **Parameter-efficient fine-tuning** — LoRA with config-driven experiment management, 1.18% params for 1.5B, 0.53% for 7B
- **Reproducible pipeline** — Makefile, GitHub Actions CI, seed-deterministic data, YAML-configured experiments
- **Full-stack ML** — data synthesis → training → evaluation → error analysis → API serving → web demo

## License

MIT
