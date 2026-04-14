#!/bin/bash
# ============================================================
# Madeline: Server Setup Script
# 一键搭建 Linux 服务器上的开发环境
# 
# Usage:
#   bash setup_server.sh
#
# Prerequisites:
#   - NVIDIA GPU driver installed (nvidia-smi works)
#   - conda installed (Anaconda or Miniconda)
#   - git configured with SSH access to GitHub
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Configuration ----
# Fork 仓库地址（修改为你自己 fork 的 DeepSpeed 地址）
DEEPSPEED_FORK_REPO="git@github.com:RuihanZhangg/DeepSpeed.git"
DEEPSPEED_BRANCH="madeline"   # 包含 Madeline 修改的分支
DEEPSPEED_DIR="$SCRIPT_DIR/_deepspeed_ref"

echo "=========================================="
echo "  Madeline Server Setup"
echo "=========================================="

# ---- Step 1: 检查 NVIDIA 驱动和 CUDA ----
echo ""
echo "[Step 1] Checking NVIDIA GPU and CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CUDA version from nvidia-smi:"
nvidia-smi | grep "CUDA Version"

# ---- Step 2: 创建 conda 环境 ----
echo ""
echo "[Step 2] Creating conda environment 'madeline'..."
if conda info --envs | grep -q "madeline"; then
    echo "Environment 'madeline' already exists. Skipping creation."
else
    conda create -n madeline python=3.10 -y
fi

# 使用 conda run 来在目标环境中执行命令
CONDA_RUN="conda run -n madeline --no-capture-output"

# ---- Step 3: 安装 PyTorch (CUDA 12.1, 兼容 CUDA 12.2) ----
echo ""
echo "[Step 3] Installing PyTorch with CUDA 12.1 support..."
$CONDA_RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证 PyTorch CUDA
echo "Verifying PyTorch CUDA..."
$CONDA_RUN python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)')
"

# ---- Step 4: 克隆并安装修改后的 DeepSpeed ----
echo ""
echo "[Step 4] Setting up forked DeepSpeed..."

if [ -d "$DEEPSPEED_DIR" ]; then
    echo "DeepSpeed directory already exists at $DEEPSPEED_DIR"
    echo "Pulling latest changes..."
    cd "$DEEPSPEED_DIR"
    git fetch origin
    git checkout "$DEEPSPEED_BRANCH" 2>/dev/null || git checkout -b "$DEEPSPEED_BRANCH" "origin/$DEEPSPEED_BRANCH"
    git pull origin "$DEEPSPEED_BRANCH"
    cd "$SCRIPT_DIR"
else
    echo "Cloning DeepSpeed fork..."
    git clone --branch "$DEEPSPEED_BRANCH" --depth 50 "$DEEPSPEED_FORK_REPO" "$DEEPSPEED_DIR"
fi

# 安装 DeepSpeed（editable mode，跳过 C++ 编译加速安装）
cd "$DEEPSPEED_DIR"
$CONDA_RUN DS_BUILD_OPS=0 pip install -e .
cd "$SCRIPT_DIR"

echo "Verifying DeepSpeed installation..."
$CONDA_RUN python -c "
import deepspeed
print(f'  DeepSpeed version: {deepspeed.__version__}')
"

# ---- Step 5: 安装 Madeline 和实验依赖 ----
echo ""
echo "[Step 5] Installing Madeline package..."
cd "$SCRIPT_DIR"
$CONDA_RUN pip install -e ".[dev,experiment]"

echo "Verifying Madeline installation..."
$CONDA_RUN python -c "
from madeline import MadelineConfig, GainModel
print(f'  Madeline imported successfully')
config = MadelineConfig(enabled=True)
print(f'  Config: enabled={config.enabled}, auto_profile={config.auto_profile}')
"

# ---- Step 6: 运行单元测试 ----
echo ""
echo "[Step 6] Running unit tests..."
$CONDA_RUN python -m pytest tests/unit -v

# ---- Step 7: GPU 状态检查 ----
echo ""
echo "[Step 7] Quick GPU sanity check..."
$CONDA_RUN python -c "
import torch
import deepspeed

if torch.cuda.device_count() >= 2:
    print(f'  {torch.cuda.device_count()} GPUs detected - multi-GPU training is possible')
else:
    print(f'  WARNING: Only {torch.cuda.device_count()} GPU(s) detected')

print()
print('  Memory per GPU:')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'    GPU {i}: {props.name} - {props.total_mem / 1e9:.1f} GB')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Activate environment:  conda activate madeline"
echo ""
echo "  Run baseline:          bash experiments/scripts/run_baseline.sh small 50"
echo "  Run Madeline:          bash experiments/scripts/run_madeline.sh small 50"
echo "  Run tests:             python -m pytest tests/ -v"
echo "=========================================="
