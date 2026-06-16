# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""LLaMA-3 model training script for experiments.

Uses HuggingFace Transformers LLaMA-3 as the base model for training
experiments with DeepSpeed ZeRO-3.  Supports both loading a pretrained
checkpoint and constructing a fresh model from a named size config.

Model size mapping (approximate parameter counts):
  7b  → LLaMA-3-8B  architecture  (~8B)
  13b → LLaMA-3-13B architecture  (~13B)
  30b → LLaMA-3-30B architecture  (~30B, also called LLaMA-3-34B)
  70b → LLaMA-3-70B architecture  (~70B)

You may also pass a HuggingFace Hub model-id (e.g. "meta-llama/Meta-Llama-3-8B")
to load a pretrained checkpoint directly.
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader, Dataset

import deepspeed
from transformers import LlamaForCausalLM, LlamaConfig


# ---------------------------------------------------------------------------
# Size presets (vocab_size follows LLaMA-3 tokenizer: 128256)
# ---------------------------------------------------------------------------
_LLAMA3_CONFIGS = {
    "7b": LlamaConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,        # GQA
        max_position_embeddings=8192,
        vocab_size=128256,
        rms_norm_eps=1e-5,
    ),
    "13b": LlamaConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,        # GQA
        max_position_embeddings=8192,
        vocab_size=128256,
        rms_norm_eps=1e-5,
    ),
    "30b": LlamaConfig(
        hidden_size=7168,
        intermediate_size=20480,
        num_hidden_layers=48,
        num_attention_heads=56,
        num_key_value_heads=8,        # GQA
        max_position_embeddings=8192,
        vocab_size=128256,
        rms_norm_eps=1e-5,
    ),
    "70b": LlamaConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,        # GQA
        max_position_embeddings=8192,
        vocab_size=128256,
        rms_norm_eps=1e-5,
    ),
}


class RandomTokenDataset(Dataset):
    """Synthetic dataset generating random token sequences.

    Used for benchmarking to isolate training throughput from data loading.
    """

    def __init__(self, vocab_size: int, seq_length: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def get_model(model_name_or_size: str) -> LlamaForCausalLM:
    """Create a LLaMA-3 model.

    Args:
        model_name_or_size: One of '7b', '13b', '30b', '70b',
            or a HuggingFace model-id like 'meta-llama/Meta-Llama-3-8B'.
    """
    key = model_name_or_size.lower()
    if key in _LLAMA3_CONFIGS:
        config = _LLAMA3_CONFIGS[key]
        model = LlamaForCausalLM(config)
    else:
        # Load pretrained checkpoint from HuggingFace Hub or local path
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_size,
            torch_dtype=torch.float16,
        )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: LLaMA-3 {model_name_or_size}, Parameters: {num_params:,}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA-3 training with DeepSpeed ZeRO-3")
    parser.add_argument(
        "--model_size", type=str, default="7b",
        choices=list(_LLAMA3_CONFIGS.keys()),
        help="LLaMA-3 model size preset, or pass a HuggingFace model-id via --model_path",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="HuggingFace Hub model-id or local path (overrides --model_size)",
    )
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length for training")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of synthetic training samples")
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Number of training steps to run")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by DeepSpeed)")
    # DeepSpeed adds its own args
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize DeepSpeed distributed backend
    deepspeed.init_distributed()
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    # Create model — prefer explicit path over size preset
    model_id = args.model_path if args.model_path else args.model_size
    model = get_model(model_id)

    # Derive vocab_size for the synthetic dataset
    vocab_size = model.config.vocab_size
    dataset = RandomTokenDataset(vocab_size, args.seq_length, args.num_samples)

    # Initialize DeepSpeed
    model_engine, optimizer, dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
    )

    # Training loop
    device = model_engine.device
    total_tokens = 0
    start_time = time.time()
    step_times = []

    model_engine.train()

    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break

        step_start = time.time()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        model_engine.backward(loss)
        model_engine.step()

        step_time = time.time() - step_start
        step_times.append(step_time)

        batch_tokens = input_ids.numel()
        total_tokens += batch_tokens

        if step % 10 == 0 and local_rank == 0:
            tokens_per_sec = batch_tokens / step_time
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(
                f"Step {step:4d} | Loss: {loss.item():.4f} | "
                f"Time: {step_time:.3f}s | "
                f"Tokens/s: {tokens_per_sec:.0f} | "
                f"GPU Mem Peak: {gpu_mem:.2f} GB"
            )

    # Summary
    elapsed = time.time() - start_time
    if local_rank == 0:
        # Skip first 5 steps for warmup
        warmup = min(5, len(step_times))
        steady_times = step_times[warmup:]
        if steady_times:
            avg_step_time = sum(steady_times) / len(steady_times)
            avg_tokens_per_sec = (
                args.seq_length * model_engine.train_micro_batch_size_per_gpu()
            ) / avg_step_time
            print("\n" + "=" * 60)
            print(f"Training Summary (LLaMA-3 {model_id})")
            print(f"  Total steps:        {len(step_times)}")
            print(f"  Total time:         {elapsed:.2f}s")
            print(f"  Avg step time:      {avg_step_time:.4f}s (excluding {warmup} warmup steps)")
            print(f"  Avg tokens/s/gpu:   {avg_tokens_per_sec:.0f}")
            print(f"  GPU mem peak:       {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
            print("=" * 60)


if __name__ == "__main__":
    main()
