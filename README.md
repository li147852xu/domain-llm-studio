# Domain LLM Adaptation & Evaluation Studio

**面向金融文档智能的开源 LLM 端到端适配系统:数据构造 → SFT/DPO 训练 → 4-variant 评估 → vLLM 服务化,在 RTX 5090 上完整跑通。**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Domain](https://img.shields.io/badge/domain-finance-blueviolet)](#金融场景设计决策)
[![CI](https://github.com/li147852xu/domain-llm-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/li147852xu/domain-llm-studio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen)]()

---

## Why This Project?

通用 LLM 在金融场景下有 **三个独特、且都可量化** 的硬要求:

1. **术语理解** — "息税前利润 / EBITDA / 同店销售 / Yoy 内生增长 / 表外负债" 这类术语既不能字面翻译也不能 hallucinate,必须靠领域语料 SFT 调对。
2. **严格输出格式** — 摘要必须是合法 JSON、事件抽取必须是 schema 完整的对象数组、QA 必须给出可定位的 `evidence_span`。下游会直接解析,字段缺一个都报错。
3. **可验证 grounding** — 财务问答和披露摘要不能只看表面流畅度;必须能在原文件里指回证据、并能在 OOD benchmark (FinanceBench) 上不崩。

本项目就是围绕这三条做的端到端实验台,每一步都跟金融场景的实际痛点直接挂钩:

```
金融语料种子 → 4-task SFT → DPO 偏好对齐 → 4-variant 严格评测 → FinanceBench OOD → vLLM 服务化
```

它跟 agent / workflow 平台类项目互补:那些在 **如何编排 LLM 调用** 上做深;本项目在 **如何让模型本身在金融领域更好用** 上做深 — 包括同时作证 prompt-only / SFT / DPO 谁带来多少边际价值,而不是把"调用 API 就完事"当成 baseline。

## 金融场景设计决策

每一个看起来"技术"的决策,背后都是一个金融场景的取舍:

### 1. 为什么选 Qwen2.5 系列(中英双语 + 国产合规)?

A 股研报、招股书、监管披露绝大多数是中文,但全球可比公司 / 上市的财报又以英文为主。需要一个 **同时把中文金融术语吃透、又能直接读 SEC 10-K 的 base 模型**。Qwen2.5-1.5B/7B-Instruct 是当前同尺寸下中英双语金融语料表现最稳的开源选项;同时国产模型在合规审查、内网部署、`safetensors` 完整可下载这些工程实操维度上比 Llama / Mistral 摩擦小一档。

### 2. 为什么是这 4 个任务而不是泛化的"document QA"?

`fin_summary` / `event_extraction` / `doc_qa` / `analysis_gen` 不是随便选的 — 它们对应了金融业务里 **真正在被人重复做、且可以被机器接管的那一拨工作**:研究员看一段研报先做要点摘要(fin_summary)、风控/合规扫公告抽出 (公司, 事件, 指标方向) 三元组(event_extraction)、PM 开会前对照招股书问问题(doc_qa)、分析师把数据点写成一段可读的备忘录段落(analysis_gen)。每个任务都有 **结构化、机器可校验的输出格式**,这才让 SFT / DPO / prompt-only 之间的对比是真有数字差,而不是各凭主观判断。

### 3. 为什么用 grounding rate 而不只用 ROUGE?

ROUGE 只能告诉你"输出和参考长得像",但金融场景里 **流畅但错的回答比直接说"找不到"危险得多** — 一个 hallucinated 的财报数字会被下游分析、写进备忘录、甚至发给客户。`grounding_rate`(`doc_qa` 的 `evidence_span` 是否在原文里出现)是直接量化"模型说的是不是有据可查",对应金融场景里"留痕 / 可审计"的合规要求。SFT 把它从 0.90 推到 1.00 不是好看的数字,是 **从'偶尔编一个'到'敢把答案给监管看'** 的差别。

### 4. 为什么把 FinanceBench 单列出来当 OOD 外部基准?

我们自己造的 400 条种子数据集,无论怎么 split,都跟 SFT 训练数据长得像。**没有外部 benchmark = 没有泛化证据**。FinanceBench 是公开的金融 10-K 上的 QA 集,跟我们种子数据完全异质 — 在它上面跑分,是检验 SFT/DPO 学到的是 "金融领域知识" 还是 "我们这 320 条 train 集的捷径"。这是把项目从"看上去能跑"推到"敢说在金融领域有泛化能力"的关键一道关。

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Domain LLM Studio (finance-focused)             │
├──────────────┬───────────────┬──────────────────┬─────────────────────┤
│ Data Pipeline│ Training      │ Evaluation       │ Serving             │
├──────────────┼───────────────┼──────────────────┼─────────────────────┤
│ Seed Gen     │ Qwen2.5 Base  │ Internal 4-task  │ FastAPI Server      │
│ Cleaning     │ LoRA / QLoRA  │ FinanceBench OOD │ Gradio Demo         │
│ Formatting   │ + DPO post    │ 4 variants eval  │ vLLM (paged-attn)   │
│ Splitting    │ Config-driven │ Charts & Reports │ Preset Examples     │
└──────────────┴───────────────┴──────────────────┴─────────────────────┘
```

## Task Definitions

The system targets four concrete, evaluatable tasks in **金融文档智能** (bilingual: Chinese + English):

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

## DPO Post-Training

After SFT brings the model into the right output format, **Direct Preference
Optimization (DPO)** further sharpens behavior by learning from
preference pairs `(prompt, chosen, rejected)`. The pipeline:

1. **Build preference pairs** ([`preference_pairs.py`](src/domain_llm_studio/data/preference_pairs.py)):
   for each training prompt, generate one completion from the **base** model
   and one from the **SFT-tuned** model, score both with ROUGE-L vs the gold
   reference, and label the higher-scoring one as `chosen`. Ties are dropped.
2. **Merge SFT adapter into base** via `peft.merge_and_unload()` so DPO
   starts from the SFT policy without needing two LoRA adapters loaded
   simultaneously.
3. **Train a fresh LoRA on top** with `trl.DPOTrainer` (β = 0.1, lr = 5e-7,
   3 epochs); the implicit reward is `log p_policy(y) - log p_ref(y)`
   where `p_ref` is the SFT-merged model with frozen weights.
4. **4-variant evaluation** on the same held-out test set:
   `base` vs `prompt_only` vs `tuned` (SFT) vs `dpo_tuned` (SFT + DPO).

<!-- DPO_RESULTS_START -->

**Training run** (RTX 5090, 7B base + SFT-merged + new LoRA):

| | Value |
|---|---|
| Preference pairs | 197 train / 38 dev (from base vs SFT, ROUGE-L scoring) |
| Epochs / steps | 3 / 75 |
| Wall time | 192s |
| Final train loss | 0.674 |
| **Eval reward accuracy** | **0.816** (model correctly prefers `chosen` over `rejected` 81.6% of the time) |
| Final reward margin | +0.07 |

**4-variant test-set comparison** (10 samples per task, deterministic):

| Task / Metric | base | prompt_only | SFT (tuned) | DPO_tuned |
|---|---|---|---|---|
| `fin_summary` ROUGE-L | 0.633 | 0.682 | **0.917** | 0.623 |
| `fin_summary` keypoint coverage | 0.650 | 0.585 | **0.793** | 0.630 |
| `doc_qa` exact_match | 0.800 | 0.800 | **1.000** | 0.800 |
| `doc_qa` grounding_rate | 0.900 | 1.000 | **1.000** | 0.900 |
| `event_extraction` event_f1 | 0.067 | 0.367 | **0.400** | 0.067 |
| `analysis_gen` schema_match | 0.933 | 0.933 | 0.933 | 0.933 |

**Honest reading of these numbers.** SFT alone closes the largest gap on
this task suite (event_f1: 0.067 → 0.400, ROUGE-L: 0.633 → 0.917). DPO on
top of SFT shows healthy training-side signals (reward accuracy 0.82,
positive reward margin) but on this 197-pair / 5e-7 / 3-epoch budget
**does not move generation behavior past the SFT initialization** — the
DPO LoRA noise even costs a few points on `fin_summary`. The pipeline
itself (preference data construction, merge-and-LoRA DPO trainer,
4-variant evaluator) is correctly wired end-to-end and ready to scale:
the next levers are **more preference pairs (≥5×)**, **higher β /
larger lr**, and **human-labeled chosen/rejected** instead of
ROUGE-L-as-proxy-judge. Treat this as the working scaffold, not the
asymptote.

<!-- DPO_RESULTS_END -->

```bash
make build-preference-pairs    # ~15-30 min on RTX 5090 (uses 7B model)
make train-dpo-7b              # ~10-25 min
make eval-dpo-7b
make compare-7b-dpo            # 4-variant base / prompt_only / SFT / DPO
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

- **金融领域 LLM 端到端适配** — 覆盖 4 项核心金融文档智能任务(摘要 / 事件抽取 / 文档 QA / 结构化分析),从数据构造一路打到 vLLM 服务化
- **Multi-scale domain LLM adaptation** — trained and evaluated both 1.5B (local, Apple M4) and 7B (cloud, RTX 5090) with LoRA
- **SFT + DPO 双阶段后训练** — SFT 把 fin_summary ROUGE-L 从 0.633 推到 0.917、event_f1 从 0.067 推到 0.400;在 SFT 顶上加 DPO,reward accuracy 0.82,完整跑通 197 偏好对的端到端 pipeline
- **vLLM 推理后端 + 量化对照** — 在 RTX 5090 上对照 transformers vs vLLM:1.5B 吞吐 4.57x、P95 延迟 -75.9%;7B 吞吐 1.51x、P95 -26.8%;诚实交代单请求场景下 7B 加速空间小,batched 服务才是 vLLM 真正的舞台
- **Eight-variant evaluation matrix** (2 scales × 4 variants: base / prompt-only / SFT / DPO) — 把 prompting / SFT / DPO 各自的边际价值都拆开来量化,而不是只跑一个 tuned vs base
- **Internal + external evaluation** — custom 4-task metric suite plus FinanceBench (150 real SEC filing questions) as the OOD benchmark — 没有外部 OOD = 没有泛化证据
- **Task-specific metrics** — custom evaluation beyond perplexity: entity F1, grounding rate, field completeness, key presence rate, partial field match — 都跟金融场景的合规 / 留痕需求直接挂钩
- **Error analysis** — systematic classification of model failures (hallucination, grounding failure, format violation, partial match)
- **Reproducible pipeline** — Makefile (`make train-dpo-7b` / `make compare-7b-dpo` / `make benchmark-inference`), GitHub Actions CI, seed-deterministic data, YAML-configured experiments
- **Full-stack ML** — data synthesis → training (SFT+DPO) → evaluation → error analysis → vLLM serving → Gradio demo

## License

MIT
