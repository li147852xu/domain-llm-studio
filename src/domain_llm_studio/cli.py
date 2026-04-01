"""Unified Typer CLI for Domain LLM Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="domain-llm-studio",
    help="Domain LLM Adaptation & Evaluation Studio — "
    "data construction, fine-tuning, evaluation, and serving.",
    add_completion=False,
)
console = Console()


@app.command()
def build_data(
    config_dir: Path = typer.Option(
        "configs/tasks", help="Directory containing task YAML configs"
    ),
    output_dir: Path = typer.Option(
        "data/processed", help="Output directory for processed data"
    ),
    seed_only: bool = typer.Option(
        False, help="Only generate seed data, skip cleaning"
    ),
    num_samples: int = typer.Option(
        50, help="Number of seed samples per task"
    ),
):
    """Build and clean instruction-tuning dataset."""
    from domain_llm_studio.data.builder import build_all

    console.print("[bold green]Building dataset...[/bold green]")
    build_all(
        config_dir=config_dir,
        output_dir=output_dir,
        seed_only=seed_only,
        num_samples=num_samples,
    )
    console.print("[bold green]Dataset built successfully.[/bold green]")


@app.command()
def inspect_data(
    data_dir: Path = typer.Option(
        "data/processed", help="Directory containing processed data"
    ),
):
    """Show dataset statistics and summary."""
    from domain_llm_studio.data.stats import print_stats

    print_stats(data_dir)


@app.command()
def train(
    config: Path = typer.Option(
        "configs/training/lora_1.5b_local.yaml",
        help="Training config YAML",
    ),
):
    """Run LoRA / QLoRA fine-tuning."""
    from domain_llm_studio.config import TrainConfig, load_config
    from domain_llm_studio.training.trainer import run_training

    cfg = load_config(TrainConfig, config)
    console.print(f"[bold green]Starting training with {cfg.base_model}...[/bold green]")
    run_training(cfg)


@app.command(name="eval")
def evaluate(
    config: Path = typer.Option(
        "configs/eval/default.yaml", help="Evaluation config YAML"
    ),
):
    """Run evaluation on the test set."""
    from domain_llm_studio.config import EvalConfig, load_config
    from domain_llm_studio.evaluation.runner import run_evaluation

    cfg = load_config(EvalConfig, config)
    console.print("[bold green]Running evaluation...[/bold green]")
    run_evaluation(cfg)


@app.command()
def compare(
    config: Path = typer.Option(
        "configs/eval/default.yaml", help="Evaluation config YAML"
    ),
    output: Path = typer.Option(
        "experiments/comparison", help="Output directory for comparison report"
    ),
):
    """Generate base vs prompt-only vs tuned comparison report."""
    from domain_llm_studio.config import EvalConfig, load_config
    from domain_llm_studio.evaluation.comparator import run_comparison

    cfg = load_config(EvalConfig, config)
    console.print("[bold green]Generating comparison report...[/bold green]")
    run_comparison(cfg, output)


@app.command()
def serve(
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Base model HF name or path"
    ),
    adapter_path: Optional[str] = typer.Option(
        None, help="Path to LoRA adapter"
    ),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
):
    """Launch FastAPI inference server."""
    import uvicorn

    from domain_llm_studio.inference.server import create_app

    app_instance = create_app(model_path, adapter_path)
    uvicorn.run(app_instance, host=host, port=port)


@app.command()
def web(
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Base model HF name or path"
    ),
    adapter_path: Optional[str] = typer.Option(
        None, help="Path to LoRA adapter"
    ),
    port: int = typer.Option(7860, help="Gradio server port"),
    share: bool = typer.Option(False, help="Create public Gradio link"),
):
    """Launch Gradio web demo."""
    from domain_llm_studio.web.app import launch_demo

    launch_demo(model_path=model_path, adapter_path=adapter_path, port=port, share=share)


@app.command()
def demo(
    model_path: str = typer.Option(
        "Qwen/Qwen2.5-1.5B-Instruct", help="Base model HF name or path"
    ),
    adapter_path: Optional[str] = typer.Option(
        None, help="Path to LoRA adapter"
    ),
):
    """Run preset examples through all model variants and print results."""
    from domain_llm_studio.inference.predictor import DomainPredictor
    from domain_llm_studio.web.app import PRESET_EXAMPLES

    predictor = DomainPredictor(model_path, adapter_path)
    for example in PRESET_EXAMPLES:
        console.rule(f"[bold]{example['task']}[/bold] ({example.get('lang', 'en')})")
        console.print(f"[dim]Input:[/dim] {example['input_text'][:200]}...")
        for variant in ["base", "tuned"]:
            result = predictor.predict(
                task=example["task"],
                input_text=example["input_text"],
                model_type=variant,
                question=example.get("question"),
            )
            console.print(f"\n[bold cyan]{variant.upper()}:[/bold cyan]")
            console.print(result)
        console.print()


@app.command()
def inspect_results(
    results_dir: Path = typer.Option(
        "experiments", help="Experiments directory"
    ),
):
    """Show latest evaluation summary."""
    from domain_llm_studio.evaluation.report import print_latest_summary

    print_latest_summary(results_dir)


def main():
    app()


if __name__ == "__main__":
    main()
