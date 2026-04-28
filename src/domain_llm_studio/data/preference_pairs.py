"""Build (prompt, chosen, rejected) preference pairs from base vs SFT outputs.

For each input sample:
    1. Format the prompt the same way the SFT model was trained on.
    2. Generate one completion from the *base* model and one from the
       *SFT-tuned* model (greedy, deterministic).
    3. Score both with ROUGE-L vs the gold reference; the higher-scoring
       completion is ``chosen``, the lower-scoring is ``rejected``.
    4. Drop pairs where the two completions are equally scored (no signal).

The output schema follows the trl DPO convention:
``{"prompt": str, "chosen": str, "rejected": str}``.

Usage::

    python -m domain_llm_studio.data.preference_pairs \\
        --input data/processed \\
        --output data/processed/preference \\
        --base-model models/Qwen2.5-7B-Instruct \\
        --sft-adapter experiments/train/lora_7b/adapter \\
        --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from domain_llm_studio.data.formatters import TASK_INSTRUCTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Prompt construction (mirrors training/inference)
# ---------------------------------------------------------------------------

def _detect_lang(text: str) -> str:
    return "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"


def _build_messages(sample: dict) -> tuple[list[dict], str]:
    """Return (messages, gold_output)."""
    task = sample["task"]
    raw_input = sample.get("input", "")
    lang = sample.get("lang") or _detect_lang(raw_input)
    instruction = TASK_INSTRUCTIONS.get(task, {}).get(lang, "")

    if task == "doc_qa":
        try:
            payload = json.loads(raw_input)
            user_content = (
                f"Context: {payload.get('context', raw_input)}\n\n"
                f"Question: {payload.get('question', '')}"
            )
        except (json.JSONDecodeError, TypeError):
            user_content = raw_input
    else:
        user_content = raw_input

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_content},
    ]
    gold = sample.get("output", "")
    return messages, gold


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rouge_l(pred: str, ref: str) -> float:
    """Char-level F-measure ROUGE-L; works equally well for zh and en."""
    if not pred or not ref:
        return 0.0
    pred_tokens = list(pred)
    ref_tokens = list(ref)
    m, n = len(pred_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_preference_split(
    samples: list[dict],
    base_model,
    sft_model,
    tokenizer,
    max_new_tokens: int,
    log_every: int = 25,
) -> list[dict]:
    pairs: list[dict] = []
    skipped_tie = 0
    for i, sample in enumerate(samples):
        messages, gold = _build_messages(sample)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            base_out = _generate(base_model, tokenizer, prompt, max_new_tokens)
        except Exception as e:
            logger.warning("base generate failed at idx=%d: %s", i, e)
            continue
        try:
            sft_out = _generate(sft_model, tokenizer, prompt, max_new_tokens)
        except Exception as e:
            logger.warning("sft generate failed at idx=%d: %s", i, e)
            continue

        base_score = _rouge_l(base_out, gold)
        sft_score = _rouge_l(sft_out, gold)

        if abs(base_score - sft_score) < 1e-6:
            skipped_tie += 1
            continue

        chosen, rejected = (
            (sft_out, base_out) if sft_score > base_score else (base_out, sft_out)
        )

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "task": sample.get("task"),
            "scores": {"base": base_score, "sft": sft_score},
        })

        if (i + 1) % log_every == 0:
            logger.info(
                "Built %d/%d preference pairs (ties skipped: %d)",
                len(pairs), i + 1, skipped_tie,
            )

    logger.info(
        "Final: %d pairs from %d samples (%d ties skipped)",
        len(pairs), len(samples), skipped_tie,
    )
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/preference")
    )
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sft-adapter", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--splits", default="train,dev")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from domain_llm_studio.training.model_loader import (
        load_base_model,
        load_model_with_adapter,
        load_tokenizer,
    )

    tokenizer = load_tokenizer(args.base_model)
    logger.info("Loading base model: %s", args.base_model)
    base_model = load_base_model(args.base_model)
    base_model.eval()
    logger.info("Loading SFT model with adapter: %s", args.sft_adapter)
    sft_model = load_model_with_adapter(args.base_model, args.sft_adapter)
    sft_model.eval()

    args.output.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        in_path = args.input / f"{split}.jsonl"
        if not in_path.exists():
            logger.warning("Skipping missing %s", in_path)
            continue
        samples = _load_jsonl(in_path)[: args.max_samples]
        logger.info("Building preference pairs for %s (%d samples)", split, len(samples))
        pairs = build_preference_split(
            samples, base_model, sft_model, tokenizer, args.max_new_tokens
        )
        out_path = args.output / f"{split}.jsonl"
        _save_jsonl(pairs, out_path)
        logger.info("Wrote %d pairs → %s", len(pairs), out_path)


if __name__ == "__main__":
    main()
