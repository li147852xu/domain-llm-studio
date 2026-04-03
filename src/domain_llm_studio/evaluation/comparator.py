"""Model comparison: base vs prompt-only vs tuned side-by-side."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from domain_llm_studio.config import EvalConfig

logger = logging.getLogger(__name__)
console = Console()


def _load_eval_results(eval_dir: Path) -> dict[str, dict]:
    """Load all eval result files from a directory."""
    results = {}
    for f in sorted(eval_dir.glob("eval_*.json")):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        label = data.get("model", f.stem.replace("eval_", ""))
        results[label] = data
    return results


def _build_comparison(results: dict[str, dict]) -> dict:
    """Build comparison data structure from loaded results."""
    models = sorted(results.keys())
    all_tasks = set()
    for r in results.values():
        all_tasks.update(r.get("per_task", {}).keys())
    all_tasks = sorted(all_tasks)

    all_metrics = set()
    for r in results.values():
        for task_metrics in r.get("per_task", {}).values():
            all_metrics.update(task_metrics.keys())
    all_metrics = sorted(all_metrics)

    comparison = {}
    for task in all_tasks:
        task_comparison = {}
        for metric in all_metrics:
            row = {}
            for model in models:
                val = results[model].get("per_task", {}).get(task, {}).get(metric)
                if val is not None:
                    row[model] = val
            if row:
                task_comparison[metric] = row
        if task_comparison:
            comparison[task] = task_comparison

    return comparison


def _print_comparison(comparison: dict, models: list[str], results: dict) -> None:
    """Print comparison tables to console."""
    for task, metrics in comparison.items():
        table = Table(title=f"Comparison — {task}")
        table.add_column("Metric", style="cyan")
        for model in models:
            table.add_column(model.upper(), justify="right")
        if len(models) >= 2:
            table.add_column("Delta (last-first)", justify="right", style="green")

        for metric, values in sorted(metrics.items()):
            row = [metric]
            vals = []
            for model in models:
                v = values.get(model)
                if v is not None:
                    row.append(f"{v:.4f}")
                    vals.append(v)
                else:
                    row.append("-")
                    vals.append(None)

            if len(vals) >= 2 and all(v is not None for v in vals):
                delta = vals[-1] - vals[0]
                sign = "+" if delta >= 0 else ""
                row.append(f"{sign}{delta:.4f}")
            elif len(models) >= 2:
                row.append("-")

            table.add_row(*row)

        console.print(table)
        console.print()

    error_table = Table(title="Error Analysis Comparison")
    error_table.add_column("Model", style="cyan")
    error_table.add_column("Error Rate", justify="right")
    error_table.add_column("Total Errors", justify="right")
    error_table.add_column("Top Error Types", style="yellow")

    for model in models:
        ea = results[model].get("error_analysis", {})
        err_dist = ea.get("error_distribution", {})
        top_errors = sorted(err_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}({v})" for k, v in top_errors)
        error_table.add_row(
            model.upper(),
            f"{ea.get('error_rate', 0):.1%}",
            str(ea.get("total_errors", 0)),
            top_str or "none",
        )

    console.print(error_table)


def run_comparison_from_dir(eval_dir: Path, output_dir: Path) -> dict:
    """Generate comparison report from a directory of eval_*.json files."""
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_eval_results(eval_dir)
    if not results:
        console.print("[yellow]No evaluation results found. Run 'eval' first.[/yellow]")
        return {}

    models = sorted(results.keys())
    comparison = _build_comparison(results)

    _print_comparison(comparison, models, results)

    from domain_llm_studio.evaluation.report import generate_charts, generate_markdown_report

    report = {
        "models": models,
        "comparison": comparison,
        "error_comparison": {
            m: results[m].get("error_analysis", {}) for m in models
        },
    }
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    chart_dir = output_dir / "charts"
    chart_paths = generate_charts(report, chart_dir)
    if chart_paths:
        console.print(f"[green]Charts saved to {chart_dir}/ ({len(chart_paths)} charts)[/green]")

    md_path = output_dir / "report.md"
    generate_markdown_report(report, md_path)
    console.print(f"[green]Markdown report saved to {md_path}[/green]")

    console.print(f"\n[green]Comparison report saved to {report_path}[/green]")
    return report


def run_comparison(cfg: EvalConfig, output_dir: Path) -> dict:
    """Generate comparison report (legacy interface using EvalConfig)."""
    return run_comparison_from_dir(Path(cfg.output_dir), output_dir)
