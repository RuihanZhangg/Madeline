#!/bin/bash
# Madeline: ZeRO-3 with forward-pass parameter caching for LLaMA-3
# Usage: bash run_madeline_llama3.sh [model_size] [num_steps] [num_gpus]
#   model_size: 7b | 13b | 30b | 70b  (default: 7b)
#   num_steps:  number of training steps (default: 50)
#   num_gpus:   number of GPUs to use   (default: 2)

MODEL_SIZE=${1:-7b}
NUM_STEPS=${2:-50}
NUM_GPUS=${3:-2}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../configs"

echo "======================================"
echo "Running MADELINE ZeRO-3 (with caching)"
echo "  Model: LLaMA-3 $MODEL_SIZE"
echo "  Steps: $NUM_STEPS"
echo "  GPUs:  $NUM_GPUS"
echo "======================================"

deepspeed --num_gpus=$NUM_GPUS \
    "$SCRIPT_DIR/train_llama3.py" \
    --model_size $MODEL_SIZE \
    --num_steps $NUM_STEPS \
    --deepspeed_config "$CONFIG_DIR/ds_config_madeline_llama3.json"
