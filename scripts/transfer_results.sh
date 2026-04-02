#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    cat <<EOF
Usage:
  $0 pack <experiment_name>      Pack adapter + results into a tar.gz
  $0 unpack <tar_file>           Unpack results into experiments/

Examples:
  $0 pack lora_7b                → creates lora_7b_results.tar.gz
  $0 unpack lora_7b_results.tar.gz

What gets packed:
  - experiments/train/<name>/adapter/        (LoRA weights, ~200-400MB)
  - experiments/train/<name>/training_*.json (training logs)
  - experiments/train/<name>/train_config.json
  - experiments/eval/eval_tuned.json         (evaluation results)
  - experiments/comparison/                  (comparison reports)
EOF
    exit 1
}

cmd_pack() {
    local name="$1"
    local train_dir="$PROJECT_DIR/experiments/train/$name"

    if [ ! -d "$train_dir/adapter" ]; then
        echo "ERROR: No adapter found at $train_dir/adapter"
        echo "Did training complete? Check experiments/train/$name/"
        exit 1
    fi

    local out="$PROJECT_DIR/${name}_results.tar.gz"

    echo "Packing results for experiment: $name"
    echo "  Adapter:    $train_dir/adapter/"
    echo "  Eval:       $PROJECT_DIR/experiments/eval/"
    echo "  Comparison: $PROJECT_DIR/experiments/comparison/"

    cd "$PROJECT_DIR"

    local files=()
    files+=("experiments/train/$name/adapter/")

    for f in "training_log.json" "training_summary.json" "train_config.json" "README.md"; do
        [ -f "experiments/train/$name/$f" ] && files+=("experiments/train/$name/$f")
    done

    [ -d "experiments/eval" ] && files+=("experiments/eval/")
    [ -d "experiments/comparison" ] && files+=("experiments/comparison/")

    tar -czf "$out" "${files[@]}"

    local size
    size=$(du -h "$out" | cut -f1)
    echo ""
    echo "Packed: $out ($size)"
    echo ""
    echo "Transfer to local machine:"
    echo "  scp <cloud>:$(pwd)/$out ."
    echo "  bash scripts/transfer_results.sh unpack ${name}_results.tar.gz"
}

cmd_unpack() {
    local tarfile="$1"

    if [ ! -f "$tarfile" ]; then
        echo "ERROR: File not found: $tarfile"
        exit 1
    fi

    echo "Unpacking: $tarfile"
    echo "  Target: $PROJECT_DIR/"

    cd "$PROJECT_DIR"
    tar -xzf "$tarfile"

    echo ""
    echo "Unpacked successfully. Contents:"
    tar -tzf "$tarfile" | head -20
    echo "..."
    echo ""
    echo "Next steps:"
    echo "  uv run domain-llm-studio compare     # regenerate comparison with new results"
    echo "  uv run domain-llm-studio web --adapter-path experiments/train/<name>/adapter"
}

if [ $# -lt 2 ]; then
    usage
fi

case "$1" in
    pack)   cmd_pack "$2" ;;
    unpack) cmd_unpack "$2" ;;
    *)      usage ;;
esac
