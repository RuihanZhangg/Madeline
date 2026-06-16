# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ForwardCacheManager and MadelineConfig."""

import pytest
from unittest.mock import MagicMock
from madeline.cache_manager import ForwardCacheManager
from madeline.config import MadelineConfig, GainWeights


# ──────────────────────────────────────────────────────────────────────────────
# ForwardCacheManager – lifecycle and decision logic
# ──────────────────────────────────────────────────────────────────────────────

class TestForwardCacheManager:

    def _make_config(self, **kwargs):
        return MadelineConfig(enabled=True, **kwargs)

    def test_not_active_before_initialize(self):
        mgr = ForwardCacheManager(self._make_config())
        assert not mgr.is_active
        assert not mgr.should_cache(42)

    def test_should_cache_only_in_forward_phase(self):
        mgr = ForwardCacheManager(self._make_config())
        mgr.is_active = True
        mgr.cache_set = {1, 2, 3}

        mgr.set_forward_phase(True)
        assert mgr.should_cache(1)
        assert not mgr.should_cache(99)  # not in cache_set

        mgr.set_forward_phase(False)
        assert not mgr.should_cache(1)  # backward phase → no caching

    def test_phase_starts_as_forward(self):
        mgr = ForwardCacheManager(self._make_config())
        assert mgr.is_forward_phase

    def test_set_forward_phase_toggles(self):
        mgr = ForwardCacheManager(self._make_config())
        mgr.set_forward_phase(False)
        assert not mgr.is_forward_phase
        mgr.set_forward_phase(True)
        assert mgr.is_forward_phase

    def test_on_step_end_resets_phase(self):
        mgr = ForwardCacheManager(self._make_config())
        mgr.set_forward_phase(False)
        mgr.on_step_end()
        assert mgr.is_forward_phase

    def test_on_step_end_resets_stats(self):
        mgr = ForwardCacheManager(self._make_config())
        mgr.record_allgather_skip(ds_id=1)
        mgr.record_allgather_skip(ds_id=2)
        assert mgr._stats_allgather_skipped == 2
        mgr.on_step_end()
        assert mgr._stats_allgather_skipped == 0

    def test_get_cached_numel(self):
        mgr = ForwardCacheManager(self._make_config())
        mgr.cache_set = {1, 3}
        sizes = {0: 100, 1: 200, 2: 300, 3: 400}
        assert mgr.get_cached_numel(sizes) == 600

    def test_get_cached_numel_empty_set(self):
        mgr = ForwardCacheManager(self._make_config())
        assert mgr.get_cached_numel({0: 100}) == 0

    def test_initialize_no_double_init(self):
        """initialize() must be idempotent."""
        mgr = ForwardCacheManager(
            MadelineConfig(enabled=True, auto_profile=False, memory_budget_numel=0)
        )
        mgr.initialize(submodule_order=[], bytes_per_element=2)
        mgr.initialize(submodule_order=[], bytes_per_element=2)
        # Should not raise and should only run once
        assert mgr._initialized

    def test_initialize_explicit_budget_zero(self):
        """Zero budget → no modules cached."""
        mgr = ForwardCacheManager(
            MadelineConfig(enabled=True, auto_profile=False, memory_budget_numel=0)
        )
        mgr.initialize(submodule_order=[], bytes_per_element=2)
        assert not mgr.is_active

    def test_initialize_no_profile_no_budget(self):
        """auto_profile=False and no memory_budget_numel → not active."""
        mgr = ForwardCacheManager(
            MadelineConfig(enabled=True, auto_profile=False)
        )
        mgr.initialize(submodule_order=[], bytes_per_element=2)
        assert not mgr.is_active


# ──────────────────────────────────────────────────────────────────────────────
# MadelineConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestMadelineConfig:

    def test_defaults(self):
        cfg = MadelineConfig()
        assert not cfg.enabled
        assert cfg.auto_profile
        assert cfg.reserved_memory_ratio == 0.1
        assert cfg.prefetch_bucket_size == 50_000_000
        assert cfg.capacity_granularity == 1_000_000
        assert cfg.gain_weights.alpha == 1.0
        assert cfg.gain_weights.beta == 1.0

    def test_from_dict_full(self):
        d = {
            "enabled": True,
            "auto_profile": False,
            "reserved_memory_ratio": 0.15,
            "prefetch_bucket_size": 200_000_000,
            "gain_weights": {"alpha": 2.0, "beta": 0.5},
            "capacity_granularity": 500_000,
            "verbose": True,
        }
        cfg = MadelineConfig.from_dict(d)
        assert cfg.enabled
        assert not cfg.auto_profile
        assert cfg.reserved_memory_ratio == 0.15
        assert cfg.prefetch_bucket_size == 200_000_000
        assert cfg.gain_weights.alpha == 2.0
        assert cfg.gain_weights.beta == 0.5
        assert cfg.capacity_granularity == 500_000
        assert cfg.verbose

    def test_from_dict_none(self):
        cfg = MadelineConfig.from_dict(None)
        assert not cfg.enabled

    def test_invalid_reserved_ratio(self):
        with pytest.raises(ValueError):
            MadelineConfig(reserved_memory_ratio=1.0)
        with pytest.raises(ValueError):
            MadelineConfig(reserved_memory_ratio=-0.1)

    def test_invalid_bucket_size(self):
        with pytest.raises(ValueError):
            MadelineConfig(prefetch_bucket_size=0)

    def test_gain_weights_from_dict(self):
        cfg = MadelineConfig(gain_weights={"alpha": 3.0, "beta": 0.0})
        assert cfg.gain_weights.alpha == 3.0
        assert cfg.gain_weights.beta == 0.0

    def test_negative_gain_exponents_raise(self):
        with pytest.raises(ValueError):
            GainWeights(alpha=-1.0, beta=0.5)
        with pytest.raises(ValueError):
            GainWeights(alpha=0.5, beta=-1.0)
