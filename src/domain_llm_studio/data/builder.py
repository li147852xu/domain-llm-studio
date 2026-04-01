"""Data pipeline orchestrator — ties seed generation, cleaning, formatting, and splitting."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from domain_llm_studio.data.cleaners import clean_dataset
from domain_llm_studio.data.formatters import format_dataset
from domain_llm_studio.data.seed_generator import generate_seed_data
from domain_llm_studio.data.splitter import save_splits, stratified_split
from domain_llm_studio.data.stats import compute_stats, load_jsonl

console = Console()


def build_all(
    config_dir: Path = Path("configs/tasks"),
    output_dir: Path = Path("data/processed"),
    seed_only: bool = False,
    num_samples: int = 50,
    seed: int = 42,
) -> None:
    """Run the full data construction pipeline."""
    seed_dir = Path("data/seed")

    # Step 1: Generate seed data
    console.print("\n[bold cyan]Step 1:[/bold cyan] Generating seed data...")
    counts = generate_seed_data(
        output_dir=seed_dir,
        num_samples_per_task=num_samples,
        seed=seed,
    )
    for task, n in counts.items():
        console.print(f"  {task}: {n} samples")

    if seed_only:
        console.print("[green]Seed-only mode: done.[/green]")
        return

    # Step 2: Load all seed data
    console.print("\n[bold cyan]Step 2:[/bold cyan] Loading seed data...")
    all_samples = []
    for jsonl_file in sorted(seed_dir.glob("*.jsonl")):
        all_samples.extend(load_jsonl(jsonl_file))
    console.print(f"  Loaded {len(all_samples)} total samples")

    # Step 3: Clean & deduplicate
    console.print("\n[bold cyan]Step 3:[/bold cyan] Cleaning and deduplicating...")
    cleaned, clean_stats = clean_dataset(all_samples)
    console.print(f"  Original: {clean_stats['original']}")
    console.print(f"  After cleaning: {clean_stats['after_cleaning']}")
    console.print(f"  After dedup: {clean_stats['after_dedup']}")
    console.print(f"  Retention rate: {clean_stats['retention_rate']:.1%}")

    # Step 4: Format as instruction-tuning data
    console.print("\n[bold cyan]Step 4:[/bold cyan] Formatting instruction data...")
    formatted = format_dataset(cleaned, config_dir=config_dir)
    console.print(f"  Formatted {len(formatted)} samples")

    # Step 5: Split
    console.print("\n[bold cyan]Step 5:[/bold cyan] Splitting into train/dev/test...")
    train, dev, test = stratified_split(formatted, seed=seed)
    split_counts = save_splits(train, dev, test, output_dir)
    for name, cnt in split_counts.items():
        console.print(f"  {name}: {cnt} samples")

    # Step 6: Stats
    console.print("\n[bold cyan]Step 6:[/bold cyan] Computing statistics...")
    all_formatted = train + dev + test
    stats = compute_stats(all_formatted)

    stats_path = output_dir / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    console.print(f"  Stats saved to {stats_path}")

    # Also save cleaning stats
    clean_stats_path = output_dir / "cleaning_stats.json"
    with open(clean_stats_path, "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=2)
    console.print(f"  Cleaning stats saved to {clean_stats_path}")

    console.print("\n[bold green]Data pipeline complete![/bold green]")
