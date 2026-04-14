# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Forward Cache Manager."""

import pytest
from unittest.mock import MagicMock, patch
from madeline.cache_manager import ForwardCacheManager
from madeline.config import MadelineConfig, GainWeights


class TestForwardCacheManager:
    """Tests for ForwardCacheManager lifecycle and decision logic."""

    def _make_config(self, enabled=True, **kwargs):
        return MadelineConfig(enabled=enabled, **kwargs)

    def test_should_cache_inactive(self):
        """When not active, should_cache always returns False."""
        config = self._make_config()
        mgr = ForwardCacheManager(config)

        # Not initialized yet -> not active
        assert not mgr.is_active
        assert not mgr.should_cache(42)

    def test_should_cache_forward_phase(self):
        """should_cache returns True only during forward phase for cached modules."""
        config = self._make_config()
        mgr = ForwardCacheManager(config)

        # Manually activate and set cache
        mgr.is_active = True
        mgr.cache_set = {1, 2, 3}

        # Forward phase: cached module -> True
        mgr.set_forward_phase(True)
        assert mgr.should_cache(1)
        assert mgr.should_cache(2)
        assert mgr.should_cache(3)

        # Forward phase: non-cached module -> False
        assert not mgr.should_cache(99)

        # Backward phase: cached module -> False (backward always releases)
        mgr.set_forward_phase(False)
        assert not mgr.should_cache(1)
        assert not mgr.should_cache(2)

    def test_phase_tracking(self):
        """Phase tracking correctly toggles between forward and backward."""
        config = self._make_config()
        mgr = ForwardCacheManager(config)

        assert mgr.is_forward_phase  # default is True

        mgr.set_forward_phase(False)
        assert not mgr.is_forward_phase

        mgr.set_forward_phase(True)
        assert mgr.is_forward_phase

    def test_on_step_end_resets_phase(self):
        """on_step_end should reset phase to forward."""
        config = self._make_config()
        mgr = ForwardCacheManager(config)

        mgr.set_forward_phase(False)
        mgr.on_step_end()
        assert mgr.is_forward_phase

    def test_get_cached_numel(self):
        """get_cached_numel correctly sums cached module sizes."""
        config = self._make_config()
        mgr = ForwardCacheManager(config)
        mgr.cache_set = {1, 3}

        sizes = {0: 100, 1: 200, 2: 300, 3: 400}
        assert mgr.get_cached_numel(sizes) == 600  # 200 + 400

    def test_get_cached_numel_empty(self):
        """Empty cache set returns 0."""
        config = self._make_config()
        mgr = ForwardCacheManager(config)
        mgr.cache_set = set()

        assert mgr.get_cached_numel({0: 100}) == 0

    def test_record_allgather_skip(self):
        """Statistics recording works correctly."""
        config = self._make_config()
        mgr = ForwardCacheManager(config)

        mgr.record_allgather_skip(ds_id=1, numel=1000)
        mgr.record_allgather_skip(ds_id=2, numel=2000)

        assert mgr._stats_allgather_skipped == 2
        assert mgr._stats_cached_numel == 3000

        # Reset on step end
        mgr.on_step_end()
        assert mgr._stats_allgather_skipped == 0
        assert mgr._stats_cached_numel == 0

    def test_disabled_config(self):
        """Disabled config should not activate caching."""
        config = self._make_config(enabled=False)
        mgr = ForwardCacheManager(config)
        assert not mgr.is_active


class TestMadelineConfig:
    """Tests for MadelineConfig."""

    def test_default_config(self):
        """Default config has caching disabled."""
        config = MadelineConfig()
        assert not config.enabled
        assert config.auto_profile
        assert config.reserved_memory_ratio == 0.1

    def test_from_dict(self):
        """from_dict correctly parses a JSON-like dict."""
        d = {
            "enabled": True,
            "auto_profile": True,
            "reserved_memory_ratio": 0.2,
            "gain_weights": {"position": 0.7, "efficiency": 0.3},
            "verbose": True,
        }
        config = MadelineConfig.from_dict(d)
        assert config.enabled
        assert config.reserved_memory_ratio == 0.2
        assert config.gain_weights.position == 0.7
        assert config.gain_weights.efficiency == 0.3
        assert config.verbose

    def test_from_dict_none(self):
        """from_dict with None returns default config."""
        config = MadelineConfig.from_dict(None)
        assert not config.enabled

    def test_invalid_reserved_ratio(self):
        """Invalid reserved_memory_ratio should raise."""
        with pytest.raises(ValueError):
            MadelineConfig(reserved_memory_ratio=1.5)
        with pytest.raises(ValueError):
            MadelineConfig(reserved_memory_ratio=-0.1)

    def test_gain_weights_from_dict(self):
        """gain_weights can be passed as a dict."""
        config = MadelineConfig(gain_weights={"position": 0.8, "efficiency": 0.2})
        assert config.gain_weights.position == 0.8
        assert config.gain_weights.efficiency == 0.2

    def test_negative_gain_weights(self):
        """Negative gain weights should raise."""
        with pytest.raises(ValueError):
            GainWeights(position=-1.0, efficiency=0.5)
