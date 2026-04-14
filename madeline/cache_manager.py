# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Forward Cache Manager: orchestrates parameter caching during training.

The ``ForwardCacheManager`` is the central coordination component that:
1. Tracks whether the current execution is in the forward or backward phase.
2. Holds the cache set (which sub-modules should be cached).
3. Provides query methods used by the modified ``PartitionedParameterCoordinator``
   to decide whether to skip parameter release (forward) or skip all-gather
   (backward -- handled automatically by DeepSpeed's existing fetch_numel check).
4. Integrates with ``MemoryProfiler`` and ``GainModel`` to initialise the cache
   set after the first trace is recorded.
"""

import logging
from typing import Dict, Optional, Set

from madeline.config import MadelineConfig
from madeline.gain_model import GainModel

logger = logging.getLogger(__name__)


class ForwardCacheManager:
    """Manages the lifecycle of forward-pass parameter caching.

    This object is created by ``DeepSpeedZeRoOffload`` and passed into
    ``PartitionedParameterCoordinator``.  The coordinator consults it during
    ``release_sub_module`` to decide whether a sub-module's parameters should
    be retained in GPU memory.

    Lifecycle per iteration:
        1. ``set_forward_phase(True)`` -- called at the start of forward.
        2. During forward, ``should_cache(ds_id)`` returns True for cached modules.
           The coordinator skips parameter release for these modules.
        3. ``set_forward_phase(False)`` -- called when backward begins.
        4. During backward, cached modules' params are still AVAILABLE, so
           ``fetch_sub_module`` naturally skips all-gather (fetch_numel == 0).
        5. After backward, all params are released normally.
        6. ``on_step_end()`` -- resets per-iteration state.

    Attributes:
        config: The Madeline configuration.
        cache_set: Set of ds_id values whose parameters should be cached.
        is_forward_phase: Whether we are currently in the forward pass.
        is_active: Whether caching is active (disabled during trace recording).
    """

    def __init__(self, config: MadelineConfig, device: int = 0):
        self.config = config
        self.device = device

        # State
        self.cache_set: Set[int] = set()
        self.is_forward_phase: bool = True
        self.is_active: bool = False
        self._initialized: bool = False

        # Components (created lazily)
        self._profiler = None  # Optional[MemoryProfiler]
        self._gain_model: Optional[GainModel] = None

        # Statistics (per iteration, for logging)
        self._stats_cached_numel: int = 0
        self._stats_allgather_skipped: int = 0

    def should_cache(self, ds_id: int) -> bool:
        """Return True if the given sub-module's parameters should be kept cached.

        This is only meaningful during the forward phase.  During backward,
        all parameters are released after use regardless.

        Args:
            ds_id: The ``ds_id`` attribute of the sub-module.
        """
        return (
            self.is_active
            and self.is_forward_phase
            and ds_id in self.cache_set
        )

    def set_forward_phase(self, forward: bool) -> None:
        """Set the current execution phase."""
        self.is_forward_phase = forward

    def initialize(
        self,
        submodule_order,
        bytes_per_element: int = 2,
    ) -> None:
        """Initialize the cache set using profiling data and the gain model.

        Called once from ``PartitionedParameterCoordinator.reset_step()`` when
        the trace transitions from RECORD to COMPLETE.

        Args:
            submodule_order: The frozen sub-module trace from the coordinator.
            bytes_per_element: Bytes per parameter element (2 for fp16/bf16).
        """
        if self._initialized:
            return

        # Determine memory budget
        if self.config.memory_budget_numel is not None:
            budget_numel = self.config.memory_budget_numel
            logger.info(
                f"[Madeline] Using explicit memory budget: {budget_numel} numel"
            )
        elif self.config.auto_profile:
            from madeline.memory_profiler import MemoryProfiler
            self._profiler = MemoryProfiler(
                reserved_memory_ratio=self.config.reserved_memory_ratio,
                device=self.device,
            )
            self._profiler.capture_peak()
            budget_numel = self._profiler.compute_budget_numel(bytes_per_element)
        else:
            logger.warning(
                "[Madeline] No memory budget specified and auto_profile is disabled. "
                "Caching will not be active."
            )
            self._initialized = True
            return

        if budget_numel <= 0:
            logger.warning(
                f"[Madeline] Computed cache budget is {budget_numel} numel (<= 0). "
                "No parameters will be cached."
            )
            self._initialized = True
            return

        # Collect sub-module sizes
        from madeline.memory_profiler import MemoryProfiler
        submodule_sizes = MemoryProfiler.collect_submodule_sizes(submodule_order)

        # Compute gains and select cache set
        gw = self.config.gain_weights
        self._gain_model = GainModel(alpha=gw.position, beta=gw.efficiency)
        gains = self._gain_model.compute_gains(submodule_order, submodule_sizes)
        self.cache_set = self._gain_model.select_cache_set(gains, budget_numel)

        self.is_active = len(self.cache_set) > 0
        self._initialized = True

        if self.config.verbose and self.is_active:
            total_cached_numel = sum(
                submodule_sizes[ds_id]
                for ds_id in self.cache_set
                if ds_id in submodule_sizes
            )
            logger.info(
                f"[Madeline] Cache set initialized: {len(self.cache_set)} modules, "
                f"{total_cached_numel} numel cached out of {budget_numel} budget"
            )
            for info in gains:
                cached_marker = "*" if info.ds_id in self.cache_set else " "
                logger.info(
                    f"  [{cached_marker}] ds_id={info.ds_id:4d}  "
                    f"numel={info.numel:12d}  "
                    f"fwd_idx={info.forward_index:4d}  "
                    f"gain={info.gain_score:.4e}"
                )

    def get_cached_numel(self, submodule_sizes: Dict[int, int]) -> int:
        """Return the total numel currently designated for caching.

        Useful for adjusting ``__max_n_available_params`` in the coordinator
        so that cached params don't throttle prefetching.
        """
        return sum(
            submodule_sizes.get(ds_id, 0)
            for ds_id in self.cache_set
        )

    def on_step_end(self) -> None:
        """Reset per-iteration state.  Called from ``reset_step()``."""
        self.is_forward_phase = True
        if self.config.verbose and self.is_active:
            logger.info(
                f"[Madeline] Step complete: "
                f"allgather_skipped={self._stats_allgather_skipped}"
            )
        self._stats_cached_numel = 0
        self._stats_allgather_skipped = 0

    def record_allgather_skip(self, ds_id: int, numel: int) -> None:
        """Record that an all-gather was skipped for a cached module."""
        self._stats_allgather_skipped += 1
        self._stats_cached_numel += numel
