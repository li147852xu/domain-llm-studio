"""Dataset statistics calculation and reporting."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table


def compute_stats(samples: list[dict]) -> dict:
    """Compute comprehensive statistics for a dataset."""
    if not samples:
        return {"total": 0}

    task_counts = Counter(s.get("task", "unknown") for s in samples)
    lang_counts = Counter(s.get("lang", "unknown") for s in samples)

    input_lengths = [len(s.get("input", "")) for s in samples]
    output_lengths = [len(s.get("output", "")) for s in samples]

    task_input_lens: dict[str, list[int]] = defaultdict(list)
    task_output_lens: dict[str, list[int]] = defaultdict(list)
    for s in samples:
        task = s.get("task", "unknown")
        task_input_lens[task].append(len(s.get("input", "")))
        task_output_lens[task].append(len(s.get("output", "")))

    task_stats = {}
    for task in sorted(task_counts.keys()):
        i_lens = task_input_lens[task]
        o_lens = task_output_lens[task]
        task_stats[task] = {
            "count": task_counts[task],
            "avg_input_len": sum(i_lens) / len(i_lens),
            "avg_output_len": sum(o_lens) / len(o_lens),
            "max_input_len": max(i_lens),
            "max_output_len": max(o_lens),
        }

    return {
        "total": len(samples),
        "by_task": dict(task_counts),
        "by_lang": dict(lang_counts),
        "avg_input_len": sum(input_lengths) / len(input_lengths),
        "avg_output_len": sum(output_lengths) / len(output_lengths),
        "max_input_len": max(input_lengths),
        "max_output_len": max(output_lengths),
        "task_details": task_stats,
    }


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def print_stats(data_dir: Path) -> None:
    """Print dataset statistics to console using rich tables."""
    console = Console()
    data_dir = Path(data_dir)

    for split_name in ["train", "dev", "test"]:
        path = data_dir / f"{split_name}.jsonl"
        if not path.exists():
            console.print(f"[yellow]⚠ {split_name}.jsonl not found in {data_dir}[/yellow]")
            continue

        samples = load_jsonl(path)
        stats = compute_stats(samples)

        console.rule(f"[bold]{split_name.upper()} Split[/bold]")
        console.print(f"  Total samples: [bold]{stats['total']}[/bold]")
        console.print(f"  Avg input length: {stats['avg_input_len']:.0f} chars")
        console.print(f"  Avg output length: {stats['avg_output_len']:.0f} chars")
        console.print()

        table = Table(title=f"{split_name.upper()} — By Task")
        table.add_column("Task", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Avg Input Len", justify="right")
        table.add_column("Avg Output Len", justify="right")
        table.add_column("Max Input Len", justify="right")
        table.add_column("Max Output Len", justify="right")

        for task, detail in sorted(stats.get("task_details", {}).items()):
            table.add_row(
                task,
                str(detail["count"]),
                f"{detail['avg_input_len']:.0f}",
                f"{detail['avg_output_len']:.0f}",
                str(detail["max_input_len"]),
                str(detail["max_output_len"]),
            )
        console.print(table)

        lang_table = Table(title=f"{split_name.upper()} — By Language")
        lang_table.add_column("Language", style="green")
        lang_table.add_column("Count", justify="right")
        for lang, count in sorted(stats.get("by_lang", {}).items()):
            lang_table.add_row(lang, str(count))
        console.print(lang_table)
        console.print()
