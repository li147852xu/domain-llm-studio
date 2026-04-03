"""Evaluation report generation: Markdown reports and matplotlib charts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def generate_markdown_report(comparison: dict, output_path: Path) -> None:
    """Generate a Markdown evaluation report with comparison tables."""
    models = comparison.get("models", [])
    comp_data = comparison.get("comparison", {})

    lines = [
        "# Evaluation Report",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "## Model Variants",
        "",
    ]
    for m in models:
        lines.append(f"- **{m.upper()}**")
    lines.append("")

    lines.append("## Per-Task Results\n")
    for task, metrics in sorted(comp_data.items()):
        lines.append(f"### {task}\n")
        header = "| Metric |"
        sep = "|--------|"
        for m in models:
            header += f" {m.upper()} |"
            sep += "--------|"
        if len(models) >= 2:
            header += " Delta |"
            sep += "--------|"
        lines.extend([header, sep])

        for metric, values in sorted(metrics.items()):
            row = f"| {metric} |"
            vals = []
            for m in models:
                v = values.get(m)
                if v is not None:
                    row += f" {v:.4f} |"
                    vals.append(v)
                else:
                    row += " - |"
                    vals.append(None)

            if len(vals) >= 2 and all(v is not None for v in vals):
                delta = vals[-1] - vals[0]
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:.4f} |"
            elif len(models) >= 2:
                row += " - |"
            lines.append(row)
        lines.append("")

    error_data = comparison.get("error_comparison", {})
    if error_data:
        lines.append("## Error Analysis\n")
        for model, ea in sorted(error_data.items()):
            lines.append(f"### {model.upper()}\n")
            lines.append(f"- Error rate: {ea.get('error_rate', 0):.1%}")
            lines.append(f"- Total errors: {ea.get('total_errors', 0)}")
            dist = ea.get("error_distribution", {})
            if dist:
                lines.append("\n**Error Distribution:**\n")
                for err_type, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {err_type}: {count}")

            examples = ea.get("examples", {})
            if examples:
                lines.append("\n**Example Failures:**\n")
                for err_type, cases in sorted(examples.items()):
                    if cases:
                        case = cases[0]
                        lines.append(f"*{err_type}* (task: {case.get('task', 'unknown')}):")
                        lines.append(f"  - Input: `{case.get('input', '')[:100]}...`")
                        lines.append(f"  - Prediction: `{case.get('prediction', '')[:100]}...`")
                        lines.append(f"  - Reference: `{case.get('reference', '')[:100]}...`")
                        lines.append("")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_charts(comparison: dict, output_dir: Path) -> list[Path]:
    """Generate matplotlib comparison charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        console.print("[yellow]matplotlib not available, skipping charts[/yellow]")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    models = comparison.get("models", [])
    comp_data = comparison.get("comparison", {})

    if not models or not comp_data:
        return []

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    for task, metrics in comp_data.items():
        key_metrics = {k: v for k, v in metrics.items()
                       if not k.startswith("parse_") and len(v) == len(models)}
        if not key_metrics:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(key_metrics) * 1.5), 5))
        x_labels = list(key_metrics.keys())
        x = np.arange(len(x_labels))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [key_metrics[m].get(model, 0) for m in x_labels]
            color = colors[i % len(colors)]
            ax.bar(x + i * width, values, width, label=model.upper(), color=color)

        ax.set_ylabel("Score")
        ax.set_title(f"Metrics Comparison — {task}")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        chart_path = output_dir / f"chart_{task}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        chart_paths.append(chart_path)

    # Error distribution stacked bar chart
    error_data = comparison.get("error_comparison", {})
    if error_data:
        all_error_types = set()
        for ea in error_data.values():
            all_error_types.update(ea.get("error_distribution", {}).keys())
        all_error_types = sorted(all_error_types)

        if all_error_types:
            fig, ax = plt.subplots(figsize=(10, 5))
            model_names = sorted(error_data.keys())
            x = np.arange(len(model_names))
            bottom = np.zeros(len(model_names))

            for i, err_type in enumerate(all_error_types):
                values = [error_data[m].get("error_distribution", {}).get(err_type, 0) for m in model_names]
                color = colors[i % len(colors)]
                ax.bar(x, values, 0.5, bottom=bottom, label=err_type, color=color)
                bottom += np.array(values)

            ax.set_ylabel("Error Count")
            ax.set_title("Error Distribution by Model Variant")
            ax.set_xticks(x)
            ax.set_xticklabels([m.upper() for m in model_names])
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            chart_path = output_dir / "error_distribution.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            chart_paths.append(chart_path)

    return chart_paths


def generate_cross_model_chart(
    results_1_5b: dict, results_7b: dict, output_dir: Path
) -> list[Path]:
    """Generate cross-model (1.5B vs 7B) comparison charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    key_metric_map = {
        "fin_summary": "rouge_l",
        "event_extraction": "entity_f1",
        "doc_qa": "token_f1",
        "analysis_gen": "bertscore_f1",
    }

    tasks = sorted(key_metric_map.keys())
    variants = sorted(set(list(results_1_5b.keys()) + list(results_7b.keys())))

    for variant in variants:
        data_1_5b = results_1_5b.get(variant, {}).get("per_task", {})
        data_7b = results_7b.get(variant, {}).get("per_task", {})

        vals_1_5b = []
        vals_7b = []
        for t in tasks:
            m = key_metric_map[t]
            vals_1_5b.append(data_1_5b.get(t, {}).get(m, 0))
            vals_7b.append(data_7b.get(t, {}).get(m, 0))

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(tasks))
        width = 0.35

        ax.bar(x - width / 2, vals_1_5b, width, label="1.5B", color="#4C72B0")
        ax.bar(x + width / 2, vals_7b, width, label="7B", color="#DD8452")

        ax.set_ylabel("Score")
        ax.set_title(f"1.5B vs 7B — {variant.upper()} Key Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n({key_metric_map[t]})" for t in tasks], fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        chart_path = output_dir / f"cross_model_{variant}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        chart_paths.append(chart_path)

    return chart_paths


def generate_full_report(results_dir: Path, output_dir: Path) -> None:
    """Generate comprehensive report from all experiment results."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = output_dir / "charts"

    def load_eval_results(eval_dir: Path) -> dict[str, dict]:
        results = {}
        for f in sorted(eval_dir.glob("eval_*.json")):
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            label = data.get("model", f.stem.replace("eval_", ""))
            results[label] = data
        return results

    all_charts = []
    summary_lines = [
        "# Domain LLM Studio — Comprehensive Evaluation Report",
        f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
    ]

    for model_size in ["1.5b", "7b"]:
        eval_dir = results_dir / f"eval_{model_size}"

        if not eval_dir.exists():
            continue

        results = load_eval_results(eval_dir)
        if not results:
            continue

        summary_lines.append(f"\n## Qwen2.5-{model_size.upper()} Results\n")

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

        report = {
            "models": models,
            "comparison": comparison,
            "error_comparison": {m: results[m].get("error_analysis", {}) for m in models},
        }

        charts = generate_charts(report, chart_dir)
        all_charts.extend(charts)

        for task, metrics in sorted(comparison.items()):
            summary_lines.append(f"\n### {task}\n")
            header = "| Metric |"
            sep = "|--------|"
            for m in models:
                header += f" {m.upper()} |"
                sep += "--------|"
            summary_lines.extend([header, sep])
            for metric, values in sorted(metrics.items()):
                row = f"| {metric} |"
                for m in models:
                    v = values.get(m)
                    row += f" {v:.4f} |" if v is not None else " - |"
                summary_lines.append(row)
            summary_lines.append("")

    # Benchmark results
    bench_dir = results_dir / "benchmark"
    if bench_dir.exists():
        summary_lines.append("\n## External Benchmark: FinanceBench\n")
        for bench_file in sorted(bench_dir.rglob("eval_*.json")):
            with open(bench_file, encoding="utf-8") as fp:
                data = json.load(fp)
            variant = data.get("model_variant", bench_file.stem.replace("eval_", ""))
            metrics = data.get("metrics", {})
            summary_lines.append(f"\n### {variant.upper()}\n")
            summary_lines.append("| Metric | Value |")
            summary_lines.append("|--------|-------|")
            for k, v in sorted(metrics.items()):
                summary_lines.append(f"| {k} | {v:.4f} |")
            summary_lines.append("")

    # Cross-model charts
    eval_1_5b = results_dir / "eval_1.5b"
    eval_7b = results_dir / "eval_7b"
    if eval_1_5b.exists() and eval_7b.exists():
        results_1_5b = load_eval_results(eval_1_5b)
        results_7b = load_eval_results(eval_7b)
        cross_charts = generate_cross_model_chart(results_1_5b, results_7b, chart_dir)
        all_charts.extend(cross_charts)

    # Embed chart references
    if all_charts:
        summary_lines.append("\n## Charts\n")
        for cp in all_charts:
            rel = cp.relative_to(output_dir) if cp.is_relative_to(output_dir) else cp
            summary_lines.append(f"![{cp.stem}]({rel})\n")

    report_path = output_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    console.print(f"[green]Full report saved to {report_path}[/green]")
    if all_charts:
        console.print(f"[green]{len(all_charts)} charts saved to {chart_dir}/[/green]")


def print_latest_summary(results_dir: Path) -> None:
    """Print a summary of the latest evaluation results."""
    results_dir = Path(results_dir)

    eval_files = list(results_dir.rglob("eval_*.json"))
    comparison_files = list(results_dir.rglob("comparison_report.json"))

    if not eval_files and not comparison_files:
        console.print("[yellow]No evaluation results found.[/yellow]")
        return

    for f in sorted(eval_files):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)

        model = data.get("model", "unknown")
        console.rule(f"[bold]{model.upper()} Model Results[/bold]")

        per_task = data.get("per_task", {})
        for task, metrics in sorted(per_task.items()):
            table = Table(title=task)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            for k, v in sorted(metrics.items()):
                table.add_row(k, f"{v:.4f}")
            console.print(table)

        ea = data.get("error_analysis", {})
        if ea:
            console.print(f"\nError rate: [bold]{ea.get('error_rate', 0):.1%}[/bold]")
        console.print()

    for f in sorted(comparison_files):
        console.rule("[bold]Comparison Summary[/bold]")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        models = data.get("models", [])
        console.print(f"Models compared: {', '.join(m.upper() for m in models)}")
