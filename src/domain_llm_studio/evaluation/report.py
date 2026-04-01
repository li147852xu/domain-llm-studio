"""Evaluation report generation: Markdown and HTML with charts."""

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

    # Per-task comparison tables
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

    # Error analysis
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
    except ImportError:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    models = comparison.get("models", [])
    comp_data = comparison.get("comparison", {})

    if not models or not comp_data:
        return []

    # Bar chart: key metrics per task
    for task, metrics in comp_data.items():
        key_metrics = {k: v for k, v in metrics.items()
                       if not k.startswith("parse_") and len(v) == len(models)}
        if not key_metrics:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(key_metrics) * 1.5), 5))
        x_labels = list(key_metrics.keys())
        x = range(len(x_labels))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [key_metrics[m].get(model, 0) for m in x_labels]
            ax.bar([xi + i * width for xi in x], values, width, label=model.upper())

        ax.set_ylabel("Score")
        ax.set_title(f"Metrics Comparison — {task}")
        ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        chart_path = output_dir / f"chart_{task}.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        chart_paths.append(chart_path)

    # Error distribution pie chart
    error_data = comparison.get("error_comparison", {})
    for model, ea in error_data.items():
        dist = ea.get("error_distribution", {})
        if dist:
            fig, ax = plt.subplots(figsize=(7, 5))
            labels = list(dist.keys())
            sizes = list(dist.values())
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
            ax.set_title(f"Error Distribution — {model.upper()}")
            plt.tight_layout()
            chart_path = output_dir / f"errors_{model}.png"
            plt.savefig(chart_path, dpi=150)
            plt.close()
            chart_paths.append(chart_path)

    return chart_paths


def print_latest_summary(results_dir: Path) -> None:
    """Print a summary of the latest evaluation results."""
    results_dir = Path(results_dir)

    eval_files = list(results_dir.rglob("eval_*.json"))
    comparison_files = list(results_dir.rglob("comparison_report.json"))

    if not eval_files and not comparison_files:
        console.print("[yellow]No evaluation results found.[/yellow]")
        return

    # Show individual eval results
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

    # Show comparison if available
    for f in sorted(comparison_files):
        console.rule("[bold]Comparison Summary[/bold]")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        models = data.get("models", [])
        console.print(f"Models compared: {', '.join(m.upper() for m in models)}")
