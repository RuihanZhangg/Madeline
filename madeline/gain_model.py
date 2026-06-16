# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Gain Model: per-module scoring and 0/1 knapsack selection for caching decisions.

Each sub-module's caching gain is quantified by three additive terms (Eq. 3 in paper):

    G(u) = S(u)  +  (k_inv * D / bs)^alpha  +  (1 - pos(u) / n)^beta

where:
  S(u)          -- Bandwidth Gain: full-gathered numel of module u. Represents the
                   deterministic reduction in AllGather communication volume.

  (k_inv * D / bs)^alpha  -- Lifespan Gain (Inter-Stage / Global): k is the 1-based
                   index of the prefetch-bucket containing u (k=1 input-side, k=K
                   output-side). k_inv = K - k + 1 so that input-side modules score
                   higher. D is total model numel; bs is prefetch_bucket_size (numel).
                   Input-side modules are held in memory from forward to the end of
                   backprop -- a higher "time-space integral" = higher gain.

  (1 - pos(u)/n)^beta  -- Latency Gain (Intra-Stage / Local Dependency): pos(u) is the
                   1-based execution index of u within its bucket in *backward* order
                   (pos=1 = tail, first executed in backward; pos=n = head, last
                   executed). Tail modules unblock the pipeline and therefore gain
                   more; head modules are penalised (latency gain → 0).

Bucket construction:
  Modules in forward order are accumulated until their cumulative numel exceeds
  prefetch_bucket_size (bs), then a new bucket starts.  This mirrors DeepSpeed's
  own prefetch grouping logic in PartitionedParameterCoordinator.

Solver:
  Module selection is solved as a 0/1 Knapsack problem via Dynamic Programming
  (exact solution), with discretised capacity in units of `capacity_granularity`
  numel to keep the DP table tractable.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# Granularity for DP table discretisation (numel units).
# Smaller = more accurate but larger table.  1M numel ~ 2MB for fp16.
DEFAULT_CAPACITY_GRANULARITY = 1_000_000  # 1M numel


@dataclass
class BucketInfo:
    """Describes a single prefetch bucket (mirrors DeepSpeed prefetch grouping)."""
    bucket_idx: int       # 1-based, 1 = input-side
    modules: List[int]    # ds_ids in backward execution order (tail first)
    total_numel: int      # sum of module sizes in this bucket


@dataclass
class ModuleGainInfo:
    """Per-module information produced by the Gain Model."""
    ds_id: int
    numel: int            # full-gathered parameter size in elements
    bucket_idx: int       # 1-based bucket index (1 = input-side)
    pos_in_bucket: int    # 1-based position in backward execution order within bucket
    bucket_size: int      # number of modules in this bucket
    gain_score: float = 0.0


class GainModel:
    """Three-term additive gain model + exact 0/1 knapsack DP solver.

    Args:
        alpha: Exponent for the Lifespan (inter-stage) gain term.
        beta: Exponent for the Latency (intra-stage) gain term.
        prefetch_bucket_size: Prefetch bucket size in numel.  Should match
            DeepSpeed ``zero_optimization.stage3_prefetch_bucket_size``
            (default 50M in DeepSpeed).  Used to partition forward modules
            into buckets for gain scoring.
        model_total_numel: Total model parameter count D (numel).  Used in
            the lifespan gain numerator.
        capacity_granularity: DP table discretisation unit in numel.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        prefetch_bucket_size: int = 50_000_000,
        model_total_numel: int = 1,
        capacity_granularity: int = DEFAULT_CAPACITY_GRANULARITY,
    ):
        self.alpha = alpha
        self.beta = beta
        self.prefetch_bucket_size = max(1, prefetch_bucket_size)
        self.model_total_numel = max(1, model_total_numel)
        self.capacity_granularity = max(1, capacity_granularity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_gains(
        self,
        submodule_order: List,
        submodule_sizes: Dict[int, int],
    ) -> List[ModuleGainInfo]:
        """Score every forward sub-module according to Eq. 3.

        Args:
            submodule_order: Full forward+backward trace from
                ``PartitionedParameterCoordinator.__submodule_order``.
            submodule_sizes: Mapping ds_id -> full-gathered numel from
                ``MemoryProfiler.collect_submodule_sizes``.

        Returns:
            List of ``ModuleGainInfo``, sorted by gain score descending.
        """
        forward_modules = _extract_forward_modules(submodule_order)
        if not forward_modules:
            return []

        buckets = self._build_buckets(forward_modules, submodule_sizes)
        K = len(buckets)  # total number of buckets

        gains: List[ModuleGainInfo] = []
        bs = self.prefetch_bucket_size
        D = self.model_total_numel

        for bucket in buckets:
            k = bucket.bucket_idx        # 1-based, 1=input-side
            k_inv = K - k + 1            # inverted so input-side → large value
            n = len(bucket.modules)      # modules in this bucket

            for pos, ds_id in enumerate(bucket.modules, start=1):
                numel = submodule_sizes.get(ds_id, 0)
                if numel == 0:
                    continue

                # Term 1: Bandwidth gain
                bandwidth_gain = float(numel)

                # Term 2: Lifespan gain (inter-stage)
                # k_inv * D / bs: input-side modules (large k_inv) score higher.
                lifespan_gain = (k_inv * D / bs) ** self.alpha

                # Term 3: Latency gain (intra-stage / local dependency)
                # pos=1 is tail (backward-first): (1 - 1/n) close to 1.
                # pos=n is head (backward-last):  (1 - n/n) = 0.
                latency_gain = (1.0 - pos / n) ** self.beta if n > 1 else 0.0

                gain_score = bandwidth_gain + lifespan_gain + latency_gain

                gains.append(ModuleGainInfo(
                    ds_id=ds_id,
                    numel=numel,
                    bucket_idx=k,
                    pos_in_bucket=pos,
                    bucket_size=n,
                    gain_score=gain_score,
                ))

        gains.sort(key=lambda g: g.gain_score, reverse=True)
        return gains

    def select_cache_set(
        self,
        gains: List[ModuleGainInfo],
        memory_budget_numel: int,
    ) -> Set[int]:
        """Solve the 0/1 Knapsack problem via DP to select the optimal cache set.

        Args:
            gains: Scored modules from ``compute_gains``.
            memory_budget_numel: Total numel available for caching (capacity W).

        Returns:
            Set of ``ds_id`` values for the optimal cache set.
        """
        if not gains or memory_budget_numel <= 0:
            return set()

        g = self.capacity_granularity

        # Discretise sizes and capacity
        sizes_disc = [max(1, math.ceil(info.numel / g)) for info in gains]
        W = memory_budget_numel // g  # discretised capacity

        if W <= 0:
            return set()

        N = len(gains)

        # Guard: fall back to greedy if DP table would be too large
        dp_table_size = N * (W + 1)
        if dp_table_size > 50_000_000:
            logger.warning(
                f"[Madeline GainModel] DP table too large ({dp_table_size} cells), "
                f"falling back to greedy approximation."
            )
            return self._greedy_select(gains, memory_budget_numel)

        # Standard 0/1 knapsack DP with backtracking table
        # dp[m] = best gain achievable with exactly m discretised capacity units
        dp = [0.0] * (W + 1)
        # kept[i][m] = True if item i was selected when dp[m] was set
        kept = [[False] * (W + 1) for _ in range(N)]

        for i, info in enumerate(gains):
            wi = sizes_disc[i]
            gi = info.gain_score
            # Iterate in reverse to ensure each item is used at most once
            for m in range(W, wi - 1, -1):
                candidate = dp[m - wi] + gi
                if candidate > dp[m]:
                    dp[m] = candidate
                    kept[i][m] = True

        # Backtrack to find which items were selected
        selected: Set[int] = set()
        m = W
        for i in range(N - 1, -1, -1):
            if kept[i][m]:
                selected.add(gains[i].ds_id)
                m -= sizes_disc[i]
                if m <= 0:
                    break

        used_numel = sum(gains[i].numel for i in range(N) if gains[i].ds_id in selected)
        logger.info(
            f"[Madeline GainModel] DP solver: selected {len(selected)} modules, "
            f"using {used_numel:,} / {memory_budget_numel:,} numel "
            f"({N} candidates, W={W} discretised units)"
        )
        return selected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_buckets(
        self,
        forward_modules: List,
        submodule_sizes: Dict[int, int],
    ) -> List[BucketInfo]:
        """Partition forward modules into prefetch buckets.

        Mirrors DeepSpeed's prefetch logic: modules are accumulated in forward
        order until the running total numel exceeds ``prefetch_bucket_size``,
        then a new bucket starts.

        Within each BucketInfo, ``modules`` is stored in *backward execution
        order* (tail first = last forward module first), to match the ``pos``
        semantics in the gain formula.

        Bucket index k is 1-based, with k=1 being the input-side bucket.
        """
        bs = self.prefetch_bucket_size
        buckets: List[BucketInfo] = []
        current_ids: List[int] = []
        current_numel: int = 0
        bucket_idx: int = 1

        for module in forward_modules:
            ds_id = module.ds_id
            numel = submodule_sizes.get(ds_id, 0)

            if current_numel + numel > bs and current_ids:
                # Flush current bucket; backward order = reverse of forward
                buckets.append(BucketInfo(
                    bucket_idx=bucket_idx,
                    modules=list(reversed(current_ids)),
                    total_numel=current_numel,
                ))
                bucket_idx += 1
                current_ids = []
                current_numel = 0

            current_ids.append(ds_id)
            current_numel += numel

        # Flush the last bucket
        if current_ids:
            buckets.append(BucketInfo(
                bucket_idx=bucket_idx,
                modules=list(reversed(current_ids)),
                total_numel=current_numel,
            ))

        logger.debug(
            f"[Madeline GainModel] Built {len(buckets)} buckets from "
            f"{len(forward_modules)} forward modules "
            f"(prefetch_bucket_size={bs:,})"
        )
        return buckets

    def _greedy_select(
        self,
        gains: List[ModuleGainInfo],
        memory_budget_numel: int,
    ) -> Set[int]:
        """Greedy fallback: select modules by descending gain score."""
        selected: Set[int] = set()
        remaining = memory_budget_numel
        for info in gains:
            if info.numel <= remaining:
                selected.add(info.ds_id)
                remaining -= info.numel
        return selected


# ------------------------------------------------------------------
# Module-level helpers (no GainModel state required)
# ------------------------------------------------------------------

def _extract_forward_modules(submodule_order) -> List:
    """Extract the forward-only portion from the full forward+backward trace.

    Walk the trace and track seen ds_ids.  The first repetition of a ds_id
    signals the start of the backward pass.
    """
    seen_ids: Set[int] = set()
    forward_modules = []
    for module in submodule_order:
        if module.ds_id in seen_ids:
            break
        seen_ids.add(module.ds_id)
        forward_modules.append(module)
    return forward_modules
