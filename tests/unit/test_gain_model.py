# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Gain Model."""

import pytest
from unittest.mock import MagicMock
from madeline.gain_model import GainModel, ModuleGainInfo


def _make_mock_module(ds_id: int):
    """Create a mock sub-module with a given ds_id."""
    m = MagicMock()
    m.ds_id = ds_id
    return m


class TestGainModel:
    """Tests for GainModel scoring and selection."""

    def test_compute_gains_basic(self):
        """Modules closer to output should have higher gain scores."""
        # Create a trace: 4 forward modules + 4 backward modules (reversed)
        forward_modules = [_make_mock_module(i) for i in range(4)]
        backward_modules = list(reversed(forward_modules))
        submodule_order = forward_modules + backward_modules

        # All modules have equal size
        sizes = {i: 1000 for i in range(4)}

        model = GainModel(alpha=0.5, beta=0.5)
        gains = model.compute_gains(submodule_order, sizes)

        assert len(gains) == 4
        # Gains should be sorted descending
        for i in range(len(gains) - 1):
            assert gains[i].gain_score >= gains[i + 1].gain_score

        # Module 3 (closest to output) should have highest gain
        top_module = gains[0]
        assert top_module.ds_id == 3

    def test_compute_gains_size_matters(self):
        """Larger modules should have higher gain when position is equal."""
        forward_modules = [_make_mock_module(i) for i in range(2)]
        backward_modules = list(reversed(forward_modules))
        submodule_order = forward_modules + backward_modules

        # Module 0: small, Module 1: large
        sizes = {0: 100, 1: 10000}

        model = GainModel(alpha=0.5, beta=0.5)
        gains = model.compute_gains(submodule_order, sizes)

        # Module 1 should have higher gain (larger AND closer to output)
        assert gains[0].ds_id == 1

    def test_compute_gains_empty_trace(self):
        """Empty trace should return empty gains."""
        model = GainModel()
        gains = model.compute_gains([], {})
        assert len(gains) == 0

    def test_compute_gains_single_module(self):
        """Single module trace should work."""
        m = _make_mock_module(0)
        submodule_order = [m, m]  # forward + backward
        sizes = {0: 1000}

        model = GainModel()
        gains = model.compute_gains(submodule_order, sizes)
        assert len(gains) == 1
        assert gains[0].ds_id == 0

    def test_extract_forward_modules(self):
        """Forward/backward split detection should work correctly."""
        modules = [_make_mock_module(i) for i in range(5)]
        # Forward: 0,1,2,3,4 | Backward: 4,3,2,1,0
        trace = modules + list(reversed(modules))

        forward = GainModel._extract_forward_modules(trace)
        assert len(forward) == 5
        assert [m.ds_id for m in forward] == [0, 1, 2, 3, 4]

    def test_select_cache_set_budget_constraint(self):
        """Selection should respect memory budget."""
        gains = [
            ModuleGainInfo(ds_id=3, numel=500, forward_index=3, gain_score=10.0),
            ModuleGainInfo(ds_id=2, numel=500, forward_index=2, gain_score=8.0),
            ModuleGainInfo(ds_id=1, numel=500, forward_index=1, gain_score=5.0),
            ModuleGainInfo(ds_id=0, numel=500, forward_index=0, gain_score=2.0),
        ]

        model = GainModel()

        # Budget fits 2 modules
        selected = model.select_cache_set(gains, 1000)
        assert selected == {3, 2}

        # Budget fits 1 module
        selected = model.select_cache_set(gains, 500)
        assert selected == {3}

        # Zero budget
        selected = model.select_cache_set(gains, 0)
        assert len(selected) == 0

    def test_select_cache_set_large_budget(self):
        """If budget is large enough, all modules are cached."""
        gains = [
            ModuleGainInfo(ds_id=i, numel=100, forward_index=i, gain_score=float(i))
            for i in range(5)
        ]

        model = GainModel()
        selected = model.select_cache_set(gains, 10000)
        assert selected == {0, 1, 2, 3, 4}

    def test_select_cache_set_skip_large_module(self):
        """If a high-gain module doesn't fit, skip it and try the next."""
        gains = [
            ModuleGainInfo(ds_id=0, numel=800, forward_index=0, gain_score=10.0),
            ModuleGainInfo(ds_id=1, numel=200, forward_index=1, gain_score=5.0),
            ModuleGainInfo(ds_id=2, numel=200, forward_index=2, gain_score=3.0),
        ]

        model = GainModel()
        # Budget=400: module 0 doesn't fit (800), but modules 1 and 2 do (200+200)
        selected = model.select_cache_set(gains, 400)
        assert selected == {1, 2}

    def test_gain_weights_customization(self):
        """Different alpha/beta weights should change relative ordering."""
        forward_modules = [_make_mock_module(i) for i in range(4)]
        backward_modules = list(reversed(forward_modules))
        submodule_order = forward_modules + backward_modules
        sizes = {i: 1000 for i in range(4)}

        # Position-only: modules near output should dominate
        model_pos = GainModel(alpha=1.0, beta=0.0)
        gains_pos = model_pos.compute_gains(submodule_order, sizes)

        # Efficiency-only
        model_eff = GainModel(alpha=0.0, beta=1.0)
        gains_eff = model_eff.compute_gains(submodule_order, sizes)

        # Both should rank module 3 highest (it benefits from both factors)
        assert gains_pos[0].ds_id == 3
        assert gains_eff[0].ds_id == 3
