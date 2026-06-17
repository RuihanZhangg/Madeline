# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Gain Model (new additive formula + DP solver)."""

import math
import pytest
from unittest.mock import MagicMock
from madeline.gain_model import GainModel, ModuleGainInfo, BucketInfo, _extract_forward_modules


def _make_module(ds_id: int):
    m = MagicMock()
    m.ds_id = ds_id
    return m


def _make_trace(n_forward: int):
    """Build a canonical fwd+bwd trace for n_forward modules."""
    fwd = [_make_module(i) for i in range(n_forward)]
    bwd = list(reversed(fwd))
    return fwd + bwd, {i: 1000 for i in range(n_forward)}


# ──────────────────────────────────────────────────────────────────────────────
# _extract_forward_modules
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractForwardModules:

    def test_basic_split(self):
        modules = [_make_module(i) for i in range(5)]
        trace = modules + list(reversed(modules))
        fwd = _extract_forward_modules(trace)
        assert [m.ds_id for m in fwd] == [0, 1, 2, 3, 4]

    def test_empty(self):
        assert _extract_forward_modules([]) == []

    def test_single_module_appears_twice(self):
        m = _make_module(0)
        fwd = _extract_forward_modules([m, m])
        assert len(fwd) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Bucket construction
# ──────────────────────────────────────────────────────────────────────────────

class TestBucketConstruction:

    def _model(self, bs):
        return GainModel(prefetch_bucket_size=bs, model_total_numel=10000)

    def test_single_bucket_when_all_fit(self):
        """All modules fit in one bucket when total numel <= bucket_size."""
        trace, sizes = _make_trace(4)  # 4 modules * 1000 = 4000 numel
        model = self._model(bs=5000)
        fwd = _extract_forward_modules(trace)
        buckets = model._build_buckets(fwd, sizes)
        assert len(buckets) == 1
        assert buckets[0].bucket_idx == 1
        assert len(buckets[0].modules) == 4

    def test_two_buckets_split_evenly(self):
        """4 modules * 1000 numel, bucket_size=2000 → 2 buckets of 2."""
        trace, sizes = _make_trace(4)
        model = self._model(bs=2000)
        fwd = _extract_forward_modules(trace)
        buckets = model._build_buckets(fwd, sizes)
        assert len(buckets) == 2
        # Each bucket should have 2 modules
        assert len(buckets[0].modules) == 2
        assert len(buckets[1].modules) == 2

    def test_bucket_idx_is_one_based(self):
        trace, sizes = _make_trace(6)
        model = self._model(bs=2000)
        fwd = _extract_forward_modules(trace)
        buckets = model._build_buckets(fwd, sizes)
        for i, b in enumerate(buckets, start=1):
            assert b.bucket_idx == i

    def test_backward_order_within_bucket(self):
        """modules list inside a bucket should be in reverse-forward (backward) order."""
        trace, sizes = _make_trace(3)  # ids 0,1,2 in forward; bucket_size large
        model = self._model(bs=99999)
        fwd = _extract_forward_modules(trace)
        buckets = model._build_buckets(fwd, sizes)
        # Forward order is [0,1,2]; backward order should be [2,1,0]
        assert buckets[0].modules == [2, 1, 0]


# ──────────────────────────────────────────────────────────────────────────────
# Gain formula correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestGainFormula:

    def test_gain_three_terms_additive(self):
        """G(u) = S(u) + lifespan + latency -- verify manual calculation."""
        # 4 modules in 1 bucket (bs large), equal size 1000
        # K=1, k=1
        # D=4000, bs=99999
        # Module ids 0..3, forward order [0,1,2,3]
        # Backward order in bucket: [3,2,1,0] → pos=1..4
        trace, sizes = _make_trace(4)
        model = GainModel(
            alpha=1.0, beta=1.0,
            prefetch_bucket_size=99999,
            model_total_numel=4000,
        )
        gains = {g.ds_id: g for g in model.compute_gains(trace, sizes)}

        bs = 99999
        D = 4000
        n = 4

        for ds_id in range(4):
            info = gains[ds_id]
            k = 1  # only one bucket
            # pos in backward order: module 3 → pos=1, module 0 → pos=4
            expected_pos = n - ds_id  # module 3 → pos=1, module 0 → pos=4
            bw_gain = 1000.0
            lifespan = (k * bs / D) ** 1.0
            latency = (expected_pos / n) ** 1.0
            expected = bw_gain + lifespan + latency
            assert abs(info.gain_score - expected) < 1e-6, (
                f"ds_id={ds_id}: got {info.gain_score}, expected {expected}"
            )

    def test_head_module_higher_gain_than_tail(self):
        """Head module (last in backward, pos=n) should have higher latency gain than tail."""
        trace, sizes = _make_trace(4)
        model = GainModel(alpha=0.0, beta=1.0,
                          prefetch_bucket_size=99999, model_total_numel=4000)
        gains = {g.ds_id: g for g in model.compute_gains(trace, sizes)}
        # module 0 is head (pos=4, pos/n=1.0), module 3 is tail (pos=1, pos/n=0.25)
        assert gains[0].gain_score > gains[3].gain_score

    def test_output_side_module_higher_lifespan(self):
        """Output-side modules (larger bucket idx k) should have higher lifespan gain."""
        # 6 modules, bucket_size=2000 → 3 buckets of 2
        # bucket 1 (input-side, k=1) vs bucket 3 (output-side, k=3)
        trace, sizes = _make_trace(6)
        model = GainModel(alpha=1.0, beta=0.0,
                          prefetch_bucket_size=2000, model_total_numel=6000)
        gains = {g.ds_id: g for g in model.compute_gains(trace, sizes)}

        # Modules 0,1 are in bucket 1 (input-side, k=1); modules 4,5 in bucket 3 (output-side, k=3)
        # Lifespan of bucket-3 modules should be higher (k=3 > k=1)
        assert gains[5].gain_score > gains[0].gain_score
        assert gains[4].gain_score > gains[1].gain_score

    def test_larger_module_higher_bandwidth_gain(self):
        """Larger module size → higher gain when other factors are equal."""
        fwd = [_make_module(0), _make_module(1)]
        bwd = list(reversed(fwd))
        trace = fwd + bwd
        sizes = {0: 100, 1: 100000}

        model = GainModel(alpha=0.0, beta=0.0,
                          prefetch_bucket_size=99999, model_total_numel=100100)
        gains = {g.ds_id: g for g in model.compute_gains(trace, sizes)}
        # alpha=beta=0 → only bandwidth gain; module 1 is much larger
        assert gains[1].gain_score > gains[0].gain_score

    def test_empty_trace_returns_empty(self):
        model = GainModel()
        assert model.compute_gains([], {}) == []

    def test_module_with_zero_size_excluded(self):
        """Modules with numel=0 should not appear in gains."""
        trace, _ = _make_trace(3)
        sizes = {0: 0, 1: 1000, 2: 1000}  # module 0 has no params
        model = GainModel(prefetch_bucket_size=99999, model_total_numel=2000)
        gains = model.compute_gains(trace, sizes)
        assert all(g.ds_id != 0 for g in gains)


# ──────────────────────────────────────────────────────────────────────────────
# DP Solver (select_cache_set)
# ──────────────────────────────────────────────────────────────────────────────

class TestDPSolver:

    def _model(self):
        # Use granularity=1 for exact small-scale tests
        return GainModel(capacity_granularity=1)

    def _make_gains(self, items):
        """items: list of (ds_id, numel, gain_score)"""
        return [
            ModuleGainInfo(
                ds_id=ds_id, numel=numel,
                bucket_idx=1, pos_in_bucket=1, bucket_size=len(items),
                gain_score=gs,
            )
            for ds_id, numel, gs in items
        ]

    def test_empty_gains(self):
        model = self._model()
        assert model.select_cache_set([], 1000) == set()

    def test_zero_budget(self):
        model = self._model()
        gains = self._make_gains([(0, 100, 5.0)])
        assert model.select_cache_set(gains, 0) == set()

    def test_all_fit(self):
        """When budget > total size, select all."""
        model = self._model()
        gains = self._make_gains([(0, 100, 1.0), (1, 200, 2.0), (2, 300, 3.0)])
        selected = model.select_cache_set(gains, 10000)
        assert selected == {0, 1, 2}

    def test_exact_budget(self):
        """Budget exactly fits two modules; greedy and DP should agree."""
        model = self._model()
        gains = self._make_gains([
            (0, 600, 10.0),
            (1, 400, 8.0),
            (2, 400, 6.0),
        ])
        # Budget=800: greedy takes 0 (600) → only 200 left, can't fit 1 or 2
        # DP optimal: take 1+2 (400+400=800, gain=14) > taking 0 alone (gain=10)
        selected = model.select_cache_set(gains, 800)
        assert selected == {1, 2}

    def test_dp_vs_greedy_difference(self):
        """DP finds optimal solution that greedy misses."""
        # Classic knapsack counter-example
        # Items: (id, size, gain)
        #   A: size=5, gain=10  ← greedy picks first (best ratio)
        #   B: size=3, gain=7
        #   C: size=3, gain=7
        # Capacity=6: greedy picks A (gain=10), DP picks B+C (gain=14)
        model = self._model()
        gains = self._make_gains([
            (0, 5, 10.0),   # A
            (1, 3, 7.0),    # B
            (2, 3, 7.0),    # C
        ])
        selected = model.select_cache_set(gains, 6)
        assert selected == {1, 2}
        # verify total gain is 14 (better than greedy's 10)

    def test_single_item_fits(self):
        model = self._model()
        gains = self._make_gains([(0, 100, 5.0)])
        assert model.select_cache_set(gains, 100) == {0}

    def test_single_item_does_not_fit(self):
        model = self._model()
        gains = self._make_gains([(0, 200, 5.0)])
        assert model.select_cache_set(gains, 100) == set()

    def test_respects_capacity_granularity(self):
        """With granularity=1000, items < 1000 numel round up to 1 unit."""
        model = GainModel(capacity_granularity=1000)
        # numel=500 rounds up to 1 unit; budget=1000 → 1 unit
        gains = self._make_gains([(0, 500, 5.0), (1, 500, 3.0)])
        # Both items = 1 unit each; budget = 1 unit → only 1 fits
        selected = model.select_cache_set(gains, 1000)
        assert len(selected) == 1
        assert 0 in selected  # higher gain selected
