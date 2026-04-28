.PHONY: install test lint data train-1.5b train-7b train-dpo-7b eval-1.5b eval-7b eval-dpo-7b compare-1.5b compare-7b compare-7b-dpo charts demo benchmark-1.5b benchmark-7b benchmark-inference build-preference-pairs

install:
	uv sync --extra dev

test:
	uv run pytest tests/ -x -q

lint:
	uv run ruff check src/ tests/

data:
	uv run domain-llm-studio build-data --num-samples 50

# Training
train-1.5b:
	uv run domain-llm-studio train --config configs/training/lora_1.5b_local.yaml

train-7b:
	uv run domain-llm-studio train --config configs/training/lora_7b_cloud.yaml

# DPO post-training (requires SFT adapter)
build-preference-pairs:
	python -m domain_llm_studio.data.preference_pairs \
		--input data/processed --output data/processed/preference \
		--base-model models/Qwen2.5-7B-Instruct \
		--sft-adapter experiments/train/lora_7b/adapter

train-dpo-7b:
	uv run domain-llm-studio dpo --config configs/training/dpo_7b_cloud.yaml

# Evaluation (base + prompt_only + tuned)
eval-1.5b:
	uv run domain-llm-studio eval --config configs/eval/base_1.5b.yaml
	uv run domain-llm-studio eval --config configs/eval/prompt_only_1.5b.yaml
	uv run domain-llm-studio eval --config configs/eval/tuned_1.5b.yaml

eval-7b:
	uv run domain-llm-studio eval --config configs/eval/base_7b.yaml
	uv run domain-llm-studio eval --config configs/eval/prompt_only_7b.yaml
	uv run domain-llm-studio eval --config configs/eval/tuned_7b.yaml

eval-dpo-7b:
	uv run domain-llm-studio eval --config configs/eval/dpo_tuned_7b.yaml

# Comparison reports
compare-1.5b:
	uv run domain-llm-studio compare --results-dir experiments/eval_1.5b --output experiments/comparison_1.5b

compare-7b:
	uv run domain-llm-studio compare --results-dir experiments/eval_7b --output experiments/comparison_7b

# 4-variant comparison: base / prompt_only / tuned (SFT) / dpo_tuned
compare-7b-dpo: compare-7b
	uv run domain-llm-studio compare --results-dir experiments/eval_7b --output experiments/comparison_7b_dpo

# Benchmark
benchmark-1.5b:
	uv run domain-llm-studio benchmark-eval --model-path Qwen/Qwen2.5-1.5B-Instruct --model-variant base --output-dir experiments/benchmark/financebench_1.5b
	uv run domain-llm-studio benchmark-eval --model-path Qwen/Qwen2.5-1.5B-Instruct --model-variant prompt_only --output-dir experiments/benchmark/financebench_1.5b
	uv run domain-llm-studio benchmark-eval --model-path Qwen/Qwen2.5-1.5B-Instruct --adapter-path experiments/train/lora_1.5b/adapter --model-variant tuned --output-dir experiments/benchmark/financebench_1.5b

benchmark-7b:
	uv run domain-llm-studio benchmark-eval --model-path models/Qwen2.5-7B-Instruct --model-variant base --output-dir experiments/benchmark/financebench_7b
	uv run domain-llm-studio benchmark-eval --model-path models/Qwen2.5-7B-Instruct --model-variant prompt_only --output-dir experiments/benchmark/financebench_7b
	uv run domain-llm-studio benchmark-eval --model-path models/Qwen2.5-7B-Instruct --adapter-path experiments/train/lora_7b/adapter --model-variant tuned --output-dir experiments/benchmark/financebench_7b

# Charts and report
charts:
	uv run domain-llm-studio generate-report --results-dir experiments --output docs/results

# Inference backend benchmark (transformers vs vLLM)
benchmark-inference:
	python scripts/benchmark_inference.py \
		--models 1.5b,7b --backends transformers,vllm --num-samples 50

# Demo
demo:
	uv run domain-llm-studio demo
