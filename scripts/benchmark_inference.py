"""Benchmark transformers vs vLLM inference backends.

Runs greedy generation over N test samples per task for one or more
(model_size, backend) configurations and records:
  - total wall time
  - per-sample latency P50 / P95 / P99
  - tokens/s throughput (output tokens / generation seconds)
  - peak GPU memory (torch.cuda.max_memory_allocated)

Outputs:
  - experiments/inference_benchmark/results.json   (raw numbers)
  - experiments/inference_benchmark/summary.md     (markdown table)
  - experiments/inference_benchmark/charts/{throughput,latency_p95,memory}.png
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from statistics import median

import torch

from domain_llm_studio.data.formatters import TASK_INSTRUCTIONS  # noqa: F401  (used indirectly)

logger = logging.getLogger(__name__)

MODEL_PATHS = {
    "1.5b": "models/Qwen2.5-1.5B-Instruct",
    "7b": "models/Qwen2.5-7B-Instruct",
}

TASKS = ["fin_summary", "event_extraction", "doc_qa", "analysis_gen"]


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def load_samples(test_path: Path, num_per_task: int) -> dict[str, list[dict]]:
    """Read test.jsonl and group up to ``num_per_task`` samples per task.

    Each returned sample has ``input``, ``task``, optional ``question``.
    """
    by_task: dict[str, list[dict]] = {t: [] for t in TASKS}
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = row.get("task")
            if task in by_task and len(by_task[task]) < num_per_task:
                by_task[task].append(row)

    return by_task


# ---------------------------------------------------------------------------
# Percentile helper (numpy-free to avoid extra deps)
# ---------------------------------------------------------------------------

def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------

def _question_for(sample: dict) -> str | None:
    if sample.get("task") == "doc_qa":
        try:
            payload = json.loads(sample.get("input", "{}"))
            return payload.get("question")
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def run_transformers(
    model_path: str,
    samples_by_task: dict[str, list[dict]],
) -> dict:
    """Run sequential transformers.generate over all samples; collect timings."""
    from domain_llm_studio.inference.predictor import DomainPredictor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    predictor = DomainPredictor(model_path=model_path, adapter_path=None)
    _ = predictor.tokenizer
    _ = predictor.base_model

    all_latencies_ms: list[float] = []
    total_output_tokens = 0
    t_start = time.perf_counter()

    for task, samples in samples_by_task.items():
        for sample in samples:
            t0 = time.perf_counter()
            output = predictor.predict(
                task=task,
                input_text=sample["input"],
                model_type="base",
                question=_question_for(sample),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            all_latencies_ms.append(latency_ms)
            total_output_tokens += len(predictor.tokenizer.encode(output))

    total_seconds = time.perf_counter() - t_start

    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "total_seconds": total_seconds,
        "latencies_ms": all_latencies_ms,
        "total_output_tokens": total_output_tokens,
        "peak_mem_gb": peak_mem_gb,
    }


def run_vllm(
    model_path: str,
    samples_by_task: dict[str, list[dict]],
) -> dict:
    """Run vLLM engine.generate over all samples; collect timings."""
    from domain_llm_studio.inference.vllm_backend import VllmPredictor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    predictor = VllmPredictor(
        model_path=model_path,
        adapter_path=None,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    _ = predictor.tokenizer
    _ = predictor.llm

    all_latencies_ms: list[float] = []
    total_output_tokens = 0
    t_start = time.perf_counter()

    for task, samples in samples_by_task.items():
        for sample in samples:
            prompt = predictor.build_prompt(
                task=task,
                input_text=sample["input"],
                model_type="base",
                question=_question_for(sample),
            )
            t0 = time.perf_counter()
            outputs = predictor.predict_batch([prompt])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            all_latencies_ms.append(latency_ms)
            total_output_tokens += len(predictor.tokenizer.encode(outputs[0]))

    total_seconds = time.perf_counter() - t_start

    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )

    if hasattr(predictor, "_llm"):
        try:
            del predictor._llm
        except AttributeError:
            pass
    del predictor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "total_seconds": total_seconds,
        "latencies_ms": all_latencies_ms,
        "total_output_tokens": total_output_tokens,
        "peak_mem_gb": peak_mem_gb,
    }


# ---------------------------------------------------------------------------
# Aggregation + outputs
# ---------------------------------------------------------------------------

def summarize(raw: dict) -> dict:
    lats = raw["latencies_ms"]
    n = len(lats)
    avg_ms = sum(lats) / n if n else 0.0
    return {
        "num_samples": n,
        "total_seconds": raw["total_seconds"],
        "p50_ms": median(lats) if lats else 0.0,
        "p95_ms": _pct(lats, 95),
        "p99_ms": _pct(lats, 99),
        "avg_ms": avg_ms,
        "throughput_tok_per_s": (
            raw["total_output_tokens"] / raw["total_seconds"]
            if raw["total_seconds"] > 0
            else 0.0
        ),
        "total_output_tokens": raw["total_output_tokens"],
        "peak_mem_gb": raw["peak_mem_gb"],
    }


def write_summary_md(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# Inference Backend Benchmark",
        "",
        "Greedy generation, single-request per call, RTX 5090 (32 GB), bf16.",
        "",
        "| Model | Backend | Samples | Total(s) | P50(ms) | P95(ms) | P99(ms) | Tokens/s | PeakMem(GB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {model} | {backend} | {n} | {total:.1f} | {p50:.0f} | {p95:.0f} | "
            "{p99:.0f} | {tps:.1f} | {mem:.2f} |".format(
                model=r["model"],
                backend=r["backend"],
                n=r["num_samples"],
                total=r["total_seconds"],
                p50=r["p50_ms"],
                p95=r["p95_ms"],
                p99=r["p99_ms"],
                tps=r["throughput_tok_per_s"],
                mem=r["peak_mem_gb"],
            )
        )

    lines += ["", "## Speedup vs transformers", ""]

    by_model: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["backend"]] = r
    lines.append("| Model | Throughput speedup | P95 latency reduction |")
    lines.append("|---|---|---|")
    for model, backends in by_model.items():
        if "transformers" in backends and "vllm" in backends:
            t = backends["transformers"]
            v = backends["vllm"]
            tps_x = (
                v["throughput_tok_per_s"] / t["throughput_tok_per_s"]
                if t["throughput_tok_per_s"] > 0
                else 0.0
            )
            p95_drop = (
                (t["p95_ms"] - v["p95_ms"]) / t["p95_ms"] * 100
                if t["p95_ms"] > 0
                else 0.0
            )
            lines.append(
                f"| {model} | {tps_x:.2f}x | {p95_drop:+.1f}% |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_charts(rows: list[dict], chart_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping charts")
        return

    chart_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['model']}-{r['backend']}" for r in rows]
    colors = ["#1f77b4" if r["backend"] == "transformers" else "#ff7f0e" for r in rows]

    for metric, ylabel, fname in [
        ("throughput_tok_per_s", "Tokens / s", "throughput.png"),
        ("p95_ms", "P95 latency (ms)", "latency_p95.png"),
        ("peak_mem_gb", "Peak GPU memory (GB)", "memory.png"),
    ]:
        values = [r[metric] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Inference benchmark — {ylabel}")
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(chart_dir / fname, dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="1.5b,7b", help="comma-separated: 1.5b,7b")
    parser.add_argument(
        "--backends", default="transformers,vllm", help="comma-separated: transformers,vllm"
    )
    parser.add_argument("--num-samples", type=int, default=50, help="samples per task")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/processed/test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/inference_benchmark"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    samples_by_task = load_samples(args.test_path, args.num_samples)
    n_total = sum(len(v) for v in samples_by_task.values())
    logger.info("Loaded %d samples across %d tasks", n_total, len(samples_by_task))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_size in models:
        if model_size not in MODEL_PATHS:
            logger.warning("Unknown model size %s, skipping", model_size)
            continue
        model_path = MODEL_PATHS[model_size]
        if not Path(model_path).exists():
            logger.error("Model path missing: %s — run hf download first", model_path)
            continue

        for backend in backends:
            logger.info("=== Running %s × %s ===", model_size, backend)
            try:
                if backend == "transformers":
                    raw = run_transformers(model_path, samples_by_task)
                elif backend == "vllm":
                    raw = run_vllm(model_path, samples_by_task)
                else:
                    logger.warning("Unknown backend %s", backend)
                    continue
            except Exception as e:
                logger.exception("Backend %s on %s failed: %s", backend, model_size, e)
                continue

            summary = summarize(raw)
            raw_rows.append({"model": model_size, "backend": backend, **raw})
            summary_rows.append({"model": model_size, "backend": backend, **summary})
            logger.info(
                "%s × %s done: %.1fs total, P95=%.0fms, %.1f tok/s, %.2f GB peak",
                model_size, backend,
                summary["total_seconds"], summary["p95_ms"],
                summary["throughput_tok_per_s"], summary["peak_mem_gb"],
            )

    results_path = args.output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "models": models,
                    "backends": backends,
                    "num_samples_per_task": args.num_samples,
                    "tasks": TASKS,
                },
                "summary": summary_rows,
                "raw": [
                    {k: v for k, v in r.items() if k != "latencies_ms"}
                    for r in raw_rows
                ],
                "latencies_ms": {
                    f"{r['model']}_{r['backend']}": r["latencies_ms"] for r in raw_rows
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_summary_md(summary_rows, args.output_dir / "summary.md")
    write_charts(summary_rows, args.output_dir / "charts")
    logger.info("Benchmark complete. Wrote %s", results_path)


if __name__ == "__main__":
    main()
