"""Convenience script to download Qwen2.5 models from Hugging Face."""

from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5 models from Hugging Face")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory to save model (default: HF cache)",
    )
    args = parser.parse_args()

    print(f"Downloading model: {args.model}")
    print("This may take a few minutes depending on your connection speed.\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.output_dir:
        tokenizer.save_pretrained(args.output_dir)
        print(f"  Saved to {args.output_dir}")

    print("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype="auto"
    )
    if args.output_dir:
        model.save_pretrained(args.output_dir)
        print(f"  Saved to {args.output_dir}")

    print(f"\nModel '{args.model}' downloaded successfully.")
    print("You can now use it with: domain-llm-studio train / eval / serve / web")


if __name__ == "__main__":
    main()
