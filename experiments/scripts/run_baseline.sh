#!/bin/bash
# Baseline: vanilla ZeRO-3 training
# Usage: bash run_baseline.sh [model_size] [num_steps]

MODEL_SIZE=${1:-small}
NUM_STEPS=${2:-50}
NUM_GPUS=${3:-2}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs"

echo "======================================"
echo "Running BASELINE ZeRO-3 (no caching)"
echo "  Model: GPT-2 $MODEL_SIZE"
echo "  Steps: $NUM_STEPS"
echo "  GPUs:  $NUM_GPUS"
echo "======================================"

deepspeed --num_gpus=$NUM_GPUS \
    "$SCRIPT_DIR/train_gpt2.py" \
    --model_size $MODEL_SIZE \
    --num_steps $NUM_STEPS \
    --deepspeed_config "$CONFIG_DIR/ds_config_baseline.json"
