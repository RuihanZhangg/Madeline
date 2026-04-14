# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Gain Model: three-factor scoring for sub-module caching decisions.

The Gain Model quantifies the "caching value" of each sub-module by combining:
1. Module size (S) -- larger modules save more all-gather communication.
2. Relative position (R) -- modules closer to the output layer are computed
   first in backward, enabling better computation-communication overlap.
3. Cache efficiency (E) -- modules closer to the output layer have a shorter
   cache lifetime (forward-use to backward-use), yielding better memory
   cost-effectiveness.

The combined gain score is:
    Gain(m) = S(m) * (alpha * R(m) + beta * E(m))

Modules are selected greedily by descending gain until the memory budget is
exhausted (a 0/1 knapsack via greedy approximation).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModuleGainInfo:
    """Per-module information used by the Gain Model."""
    ds_id: int
    numel: int          # full gathered parameter size in elements
    forward_index: int  # index in forward execution order (0 = first computed)
    gain_score: float = 0.0


class GainModel:
    """Three-factor scoring model for forward-cache sub-module selection.

    Args:
        alpha: Weight for the relative position factor.
        beta: Weight for the cache efficiency factor.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta

    def compute_gains(
        self,
        submodule_order: List,
        submodule_sizes: Dict[int, int],
    ) -> List[ModuleGainInfo]:
        """Score every sub-module that appears in the forward portion of the trace.

        The ``submodule_order`` from DeepSpeed includes both the forward and
        backward traversal.  For a standard model the forward portion is the
        first half and the backward portion is the second half (reversed).

        Only the forward-portion sub-modules are candidates for caching,
        because caching means "keep params after forward, reuse in backward".

        We deduplicate modules that appear multiple times in the forward portion
        (e.g., shared parameters) and use their first occurrence index.

        Args:
            submodule_order: Full forward+backward trace from
                ``PartitionedParameterCoordinator.__submodule_order``.
            submodule_sizes: Mapping of ``ds_id`` -> numel from
                ``MemoryProfiler.collect_submodule_sizes``.

        Returns:
            List of ``ModuleGainInfo`` for each unique forward sub-module,
            sorted by gain score descending.
        """
        # Determine the forward portion of the trace.
        # In a standard training loop, the trace contains N forward steps
        # followed by N backward steps.  We detect the split by finding the
        # first ds_id that repeats -- the second occurrence marks the start
        # of the backward portion.
        forward_modules = self._extract_forward_modules(submodule_order)
        n_forward = len(forward_modules)

        if n_forward == 0:
            return []

        gains: List[ModuleGainInfo] = []
        for idx, module in enumerate(forward_modules):
            ds_id = module.ds_id
            numel = submodule_sizes.get(ds_id, 0)
            if numel == 0:
                continue

            s = numel  # Factor 1: module size
            r = idx / n_forward  # Factor 2: relative position (0=input, ~1=output)

            # Factor 3: cache efficiency (inverse of cache lifetime)
            # Lifetime = 2 * (N - idx) steps (forward use -> backward use).
            # Modules near output have short lifetime -> high efficiency.
            lifetime = max(1, 2 * (n_forward - idx))
            e = 1.0 / lifetime

            gain_score = s * (self.alpha * r + self.beta * e)

            gains.append(ModuleGainInfo(
                ds_id=ds_id,
                numel=numel,
                forward_index=idx,
                gain_score=gain_score,
            ))

        gains.sort(key=lambda g: g.gain_score, reverse=True)
        return gains

    def select_cache_set(
        self,
        gains: List[ModuleGainInfo],
        memory_budget_numel: int,
    ) -> Set[int]:
        """Greedily select modules to cache within the given memory budget.

        Args:
            gains: Scored modules from ``compute_gains``, sorted by gain descending.
            memory_budget_numel: Maximum total numel that can be cached.

        Returns:
            Set of ``ds_id`` values for the selected cache set.
        """
        selected: Set[int] = set()
        remaining_budget = memory_budget_numel

        for info in gains:
            if info.numel <= remaining_budget:
                selected.add(info.ds_id)
                remaining_budget -= info.numel

        logger.info(
            f"[Madeline GainModel] selected {len(selected)} modules for caching, "
            f"using {memory_budget_numel - remaining_budget} / {memory_budget_numel} numel "
            f"({len(gains)} candidates evaluated)"
        )
        return selected

    @staticmethod
    def _extract_forward_modules(submodule_order) -> List:
        """Extract the forward-only portion from the full forward+backward trace.

        Strategy: Walk the trace and track seen ds_ids.  The first time a ds_id
        is seen for the second time, all modules from that point onward belong
        to the backward pass.  This works because backward visits modules in
        roughly reverse order.
        """
        seen_ids: Set[int] = set()
        forward_modules = []

        for module in submodule_order:
            if module.ds_id in seen_ids:
                # This ds_id was already seen in forward; we've entered backward.
                break
            seen_ids.add(module.ds_id)
            forward_modules.append(module)

        return forward_modules
