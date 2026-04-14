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
    """Weights for the three factors in the Gain Model scoring function.

    Attributes:
        position: Weight for relative position factor (overlap opportunity).
        efficiency: Weight for absolute position factor (cache lifetime efficiency).
    """
    position: float = 0.5
    efficiency: float = 0.5

    def __post_init__(self):
        if self.position < 0 or self.efficiency < 0:
            raise ValueError("Gain weights must be non-negative")


@dataclass
class MadelineConfig:
    """Configuration for Madeline forward-pass parameter caching.

    Attributes:
        enabled: Master toggle for forward caching.
        auto_profile: If True, run a profiling pass on the first iteration
            to determine the memory budget automatically.
        memory_budget_numel: Explicit memory budget in number of elements.
            If None and auto_profile is True, the budget is determined
            automatically. If both are set, this value overrides auto_profile.
        reserved_memory_ratio: Fraction of total GPU memory to reserve as a
            safety margin when computing the cache budget. Only used with
            auto_profile.
        gain_weights: Weights for the Gain Model scoring factors.
        verbose: If True, log detailed caching decisions and statistics.
    """
    enabled: bool = False
    auto_profile: bool = True
    memory_budget_numel: Optional[int] = None
    reserved_memory_ratio: float = 0.1
    gain_weights: GainWeights = field(default_factory=GainWeights)
    verbose: bool = False

    def __post_init__(self):
        if not 0.0 <= self.reserved_memory_ratio < 1.0:
            raise ValueError(
                f"reserved_memory_ratio must be in [0, 1), got {self.reserved_memory_ratio}"
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
