# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Madeline configuration dataclass.

Defines the configuration schema for forward-pass parameter caching.
These settings are embedded under ``zero_optimization.forward_cache``
in the DeepSpeed JSON config.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GainWeights:
    """Exponents for the Lifespan and Latency terms in the Gain formula (Eq. 3).

    Attributes:
        alpha: Exponent for the Lifespan (inter-stage) gain term
               ``(k_inv * D / bs)^alpha``.
        beta:  Exponent for the Latency (intra-stage) gain term
               ``(1 - pos(u)/n)^beta``.
    """
    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self):
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("Gain exponents (alpha, beta) must be non-negative")


@dataclass
class MadelineConfig:
    """Configuration for Madeline forward-pass parameter caching.

    Attributes:
        enabled: Master toggle for forward caching.
        auto_profile: If True, run a profiling pass on the first iteration
            to determine the memory budget automatically.
        memory_budget_numel: Explicit memory budget in number of elements.
            If None and auto_profile is True, the budget is determined
            automatically.  If both are set, this value overrides auto_profile.
        reserved_memory_ratio: Fraction of total GPU memory to reserve as a
            safety margin when computing the cache budget (only with auto_profile).
        prefetch_bucket_size: Prefetch bucket size in numel.  Should match
            DeepSpeed ``zero_optimization.stage3_prefetch_bucket_size``.
            Used by the Gain Model to partition modules into buckets for the
            Lifespan and Latency gain terms.  Default 50M matches DeepSpeed's
            own default (``prefetch_bucket_size`` in stage3 config).
        gain_weights: Exponents (alpha, beta) for the Gain formula.
        capacity_granularity: DP table discretisation unit in numel.
            Smaller = more precise but larger table (memory cost).
        verbose: If True, log detailed caching decisions and statistics.
    """
    enabled: bool = False
    auto_profile: bool = True
    memory_budget_numel: Optional[int] = None
    reserved_memory_ratio: float = 0.1
    prefetch_bucket_size: int = 50_000_000
    gain_weights: GainWeights = field(default_factory=GainWeights)
    capacity_granularity: int = 1_000_000
    verbose: bool = False

    def __post_init__(self):
        if not 0.0 <= self.reserved_memory_ratio < 1.0:
            raise ValueError(
                f"reserved_memory_ratio must be in [0, 1), got {self.reserved_memory_ratio}"
            )
        if self.prefetch_bucket_size <= 0:
            raise ValueError(
                f"prefetch_bucket_size must be positive, got {self.prefetch_bucket_size}"
            )
        if isinstance(self.gain_weights, dict):
            self.gain_weights = GainWeights(**self.gain_weights)

    @classmethod
    def from_dict(cls, d: dict) -> "MadelineConfig":
        """Create a MadelineConfig from a dictionary (e.g., parsed from JSON)."""
        if d is None:
            return cls()
        d = dict(d)
        if "gain_weights" in d and isinstance(d["gain_weights"], dict):
            d["gain_weights"] = GainWeights(**d["gain_weights"])
        return cls(**d)
