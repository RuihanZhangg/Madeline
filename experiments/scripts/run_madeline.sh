#!/bin/bash
# Madeline: ZeRO-3 with forward-pass parameter caching
# Usage: bash run_madeline.sh [model_size] [num_steps]

MODEL_SIZE=${1:-small}
NUM_STEPS=${2:-50}
NUM_GPUS=${3:-2}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs"

echo "======================================"
echo "Running MADELINE ZeRO-3 (with caching)"
echo "  Model: GPT-2 $MODEL_SIZE"
echo "  Steps: $NUM_STEPS"
echo "  GPUs:  $NUM_GPUS"
echo "======================================"

deepspeed --num_gpus=$NUM_GPUS \
    "$SCRIPT_DIR/train_gpt2.py" \
    --model_size $MODEL_SIZE \
    --num_steps $NUM_STEPS \
    --deepspeed_config "$CONFIG_DIR/ds_config_madeline.json"
