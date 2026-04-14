# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Memory profiler for estimating the GPU memory budget available for caching.

The profiler is invoked once after the first training iteration (when the
DeepSpeed trace transitions from RECORD to COMPLETE). It uses the peak memory
watermark from that iteration to estimate how much surplus GPU memory can be
allocated to parameter caching.
"""

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Estimates the GPU memory budget available for forward-pass caching.

    The profiling approach:
    1. After the first forward+backward pass, read ``torch.cuda.max_memory_allocated()``
       to obtain the peak memory usage under vanilla ZeRO-3.
    2. Compute the available surplus as:
       ``budget = total_gpu_mem - peak_usage - safety_margin``
    3. Convert the byte budget to a numel budget based on parameter dtype.

    Attributes:
        reserved_memory_ratio: Fraction of total GPU memory reserved as safety margin.
        device: The CUDA device to profile.
    """

    def __init__(self, reserved_memory_ratio: float = 0.1, device: int = 0):
        self.reserved_memory_ratio = reserved_memory_ratio
        self.device = device
        self._peak_memory: int = 0
        self._total_memory: int = 0
        self._profiled: bool = False

    def capture_peak(self) -> None:
        """Capture peak memory usage.  Call this after the first iteration."""
        self._peak_memory = torch.cuda.max_memory_allocated(self.device)
        self._total_memory = torch.cuda.get_device_properties(self.device).total_mem
        self._profiled = True
        logger.info(
            f"[Madeline MemoryProfiler] peak_memory={self._peak_memory / 1e9:.2f} GB, "
            f"total_memory={self._total_memory / 1e9:.2f} GB"
        )

    def compute_budget_bytes(self) -> int:
        """Return the memory budget for caching in bytes."""
        if not self._profiled:
            raise RuntimeError("MemoryProfiler.capture_peak() must be called first")
        safety_margin = int(self._total_memory * self.reserved_memory_ratio)
        budget = self._total_memory - self._peak_memory - safety_margin
        budget = max(0, budget)
        logger.info(
            f"[Madeline MemoryProfiler] cache budget={budget / 1e9:.2f} GB "
            f"(safety_margin={safety_margin / 1e9:.2f} GB)"
        )
        return budget

    def compute_budget_numel(self, bytes_per_element: int = 2) -> int:
        """Return the memory budget for caching in number of elements.

        Args:
            bytes_per_element: Bytes per parameter element.
                2 for fp16/bf16, 4 for fp32.
        """
        return self.compute_budget_bytes() // bytes_per_element

    @staticmethod
    def collect_submodule_sizes(
        submodule_order: List,
    ) -> Dict[int, int]:
        """Collect the full-gathered parameter size (in numel) for each sub-module.

        Args:
            submodule_order: The recorded trace of sub-modules from DeepSpeed's
                PartitionedParameterCoordinator (``__submodule_order``).

        Returns:
            A dict mapping ``sub_module.ds_id`` to the total numel of its
            parameters when fully gathered.
        """
        from deepspeed.runtime.zero.partitioned_param_coordinator import iter_params
        from deepspeed.utils import z3_leaf_module

        sizes: Dict[int, int] = {}
        for module in submodule_order:
            total_numel = sum(
                p.ds_numel
                for p in iter_params(module, recurse=z3_leaf_module(module))
            )
            sizes[module.ds_id] = total_numel
        return sizes

    @staticmethod
    def collect_submodule_partition_sizes(
        submodule_order: List,
    ) -> Dict[int, int]:
        """Collect the partitioned parameter size (communication volume) for each sub-module.

        This represents the actual bytes that would be communicated during an
        all-gather, which equals ``ds_numel`` (the full parameter size) because
        each rank sends its partition (``ds_numel / world_size``) and the result
        is the full tensor.

        Args:
            submodule_order: The recorded trace of sub-modules.

        Returns:
            A dict mapping ``sub_module.ds_id`` to the total numel that would
            be all-gathered for it.
        """
        from deepspeed.runtime.zero.partitioned_param_coordinator import iter_params
        from deepspeed.utils import z3_leaf_module

        sizes: Dict[int, int] = {}
        for module in submodule_order:
            total_numel = sum(
                p.partition_numel()
                for p in iter_params(module, recurse=z3_leaf_module(module))
            )
            sizes[module.ds_id] = total_numel
        return sizes
