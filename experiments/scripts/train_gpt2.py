# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""GPT-2 model definition for experiments.

Uses HuggingFace Transformers GPT-2 as the base model for training
experiments with DeepSpeed ZeRO-3.
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader, Dataset

import deepspeed
from transformers import GPT2LMHeadModel, GPT2Config


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


def get_model(model_name_or_size: str) -> GPT2LMHeadModel:
    """Create a GPT-2 model.

    Args:
        model_name_or_size: One of 'small', 'medium', 'large', 'xl',
            or a HuggingFace model name like 'gpt2', 'gpt2-medium', etc.
    """
    config_map = {
        "small": GPT2Config(n_layer=12, n_head=12, n_embd=768),    # 124M
        "medium": GPT2Config(n_layer=24, n_head=16, n_embd=1024),  # 355M
        "large": GPT2Config(n_layer=36, n_head=20, n_embd=1280),   # 774M
        "xl": GPT2Config(n_layer=48, n_head=25, n_embd=1600),      # 1.5B
    }

    if model_name_or_size in config_map:
        config = config_map[model_name_or_size]
        model = GPT2LMHeadModel(config)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name_or_size)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT-2 {model_name_or_size}, Parameters: {num_params:,}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="GPT-2 training with DeepSpeed ZeRO-3")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["small", "medium", "large", "xl"],
                        help="GPT-2 model size")
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

    # Create model
    model = get_model(args.model_size)

    # Create synthetic dataset
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
            avg_tokens_per_sec = (args.seq_length * model_engine.train_micro_batch_size_per_gpu()) / avg_step_time
            print("\n" + "=" * 60)
            print(f"Training Summary ({args.model_size})")
            print(f"  Total steps:        {len(step_times)}")
            print(f"  Total time:         {elapsed:.2f}s")
            print(f"  Avg step time:      {avg_step_time:.4f}s (excluding {warmup} warmup steps)")
            print(f"  Avg tokens/s/gpu:   {avg_tokens_per_sec:.0f}")
            print(f"  GPU mem peak:       {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
            print("=" * 60)


if __name__ == "__main__":
    main()
