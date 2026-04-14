# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Madeline: Forward-Pass Parameter Caching for DeepSpeed ZeRO-3.

This package implements a caching mechanism that retains selected sub-module
parameters in GPU memory during the forward pass, eliminating redundant
all-gather communication in the backward pass.
"""

__version__ = "0.1.0"

from madeline.config import MadelineConfig
from madeline.gain_model import GainModel

# Lazy imports for components that depend on torch/deepspeed,
# so that pure-Python modules (config, gain_model) can be used
# without a full CUDA environment.


def __getattr__(name):
    if name == "ForwardCacheManager":
        from madeline.cache_manager import ForwardCacheManager
        return ForwardCacheManager
    if name == "MemoryProfiler":
        from madeline.memory_profiler import MemoryProfiler
        return MemoryProfiler
    raise AttributeError(f"module 'madeline' has no attribute {name!r}")
