# Madeline
本项目是对于Deepspeed框架的优化，主要是对ZeRO-3的缓存模块研究。

---

## 指南

#### 1. 服务器前置要求

```bash
# 确认 NVIDIA 驱动和 CUDA
nvidia-smi
# 期望至少：2 张 GPU，CUDA Version >= 12.1

# 确认 conda 已安装
conda --version

# 确认 GitHub SSH 已配置（用于拉取 DeepSpeed fork）
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com
# 期望：Hi RuihanZhangg! You've successfully authenticated...
```

#### 2. Clone Madeline 主仓库并执行脚本

```bash
git clone <你的 Madeline 仓库地址>  # 如 git@github.com:RuihanZhangg/Madeline.git
cd Madeline
bash setup_server.sh
```

脚本会自动完成：
- 创建 `conda` 环境 `madeline` (Python 3.10)
- 安装 PyTorch (CUDA 12.1)
- Clone 你的 DeepSpeed fork (`madeline` 分支) 到 `_deepspeed_ref/`
- 以 editable 模式安装修改后的 DeepSpeed
- 以 editable 模式安装 Madeline 及实验依赖
- 运行单元测试验证

#### 3. 运行实验对比

```bash
conda activate madeline

# Baseline（无缓存）
bash experiments/scripts/run_baseline.sh small 50

# Madeline（启用前向缓存）
bash experiments/scripts/run_madeline.sh small 50
```

两个脚本默认都使用 **2 张 GPU**。如需指定其他数量：
```bash
bash experiments/scripts/run_madeline.sh small 50 4   # 4 卡
```

---

### 已知注意事项

1. **`DS_BUILD_OPS=0`**：`setup_server.sh` 跳过 DeepSpeed C++ 扩展编译，安装更快。ZeRO-3 不依赖这些扩展，但如果你需要 fused Adam 等优化，可在安装后运行 `DS_BUILD_FUSED_ADAM=1 pip install -e .` 补编译。

2. **`_deepspeed_ref/` 目录**：脚本会将 DeepSpeed fork clone 到 Madeline 目录内的 `_deepspeed_ref/`。该目录已被 `.gitignore` 忽略，不会影响你 push Madeline 代码。

3. **首次运行 Madeline 的迭代 1**：`train_gpt2.py` 的 step 0 是 **RECORD** 阶段（无缓存），step 1 起进入 **COMPLETE** 阶段（缓存生效）。日志中你会看到：
   ```
   completed record trace of 28 sub modules
   [Madeline] Cache initialized: X modules cached, Y numel used / Z budget
   ```

4. **如果需要对比内存/吞吐**：两脚本都打印了 `GPU Mem Peak` 和 `Avg tokens/s/gpu`，直接对比最后一行即可。