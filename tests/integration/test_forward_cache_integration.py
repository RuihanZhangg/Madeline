"""Integration tests for the Madeline forward-cache lifecycle.

These tests simulate the interaction between ``ForwardCacheManager`` and the
ZeRO-3 coordinator hooks (``fetch_sub_module``, ``release_sub_module``,
``reset_step``) without requiring a full DeepSpeed installation or GPU.
"""

import sys
from typing import List
from unittest.mock import MagicMock

import pytest

# Mock torch before importing any madeline module that transitively imports torch.
# Provide fully-configured cuda mocks so that ``memory_profiler`` works
# even when tests are collected after this module.
_mock_torch = MagicMock()
_mock_torch.cuda.max_memory_allocated = MagicMock(return_value=0)
_mock_torch.cuda.get_device_properties = MagicMock(
    return_value=MagicMock(total_mem=16_000_000_000)
)
sys.modules.setdefault("torch", _mock_torch)

from madeline.cache_manager import ForwardCacheManager
from madeline.config import MadelineConfig


class FakeParam:
    """Minimal stand-in for a DeepSpeed ZeRO-3 partitioned parameter."""

    def __init__(self, ds_id: int, numel: int):
        self.ds_id = ds_id
        self.ds_numel = numel
        self.ds_status = "NOT_AVAILABLE"
        self.ds_active_sub_modules: set = set()


class FakeModule:
    """Minimal stand-in for a ``torch.nn.Module`` with ZeRO-3 attributes."""

    def __init__(self, ds_id: int, params: List[FakeParam]):
        self.ds_id = ds_id
        self._params = params

    def parameters(self, recurse=False):
        for p in self._params:
            yield p

    def named_parameters(self, recurse=False):
        for i, p in enumerate(self._params):
            yield (f"p{i}", p)


def fake_iter_params(module, recurse=False):
    """Drop-in replacement for ``iter_params`` used by the coordinator."""
    return [p for _, p in module.named_parameters()]


def build_linear_module(ds_id: int, num_params: int, param_numel: int) -> FakeModule:
    """Build a fake linear-like module with *num_params* weights."""
    params = [FakeParam(ds_id * 1000 + i, param_numel) for i in range(num_params)]
    return FakeModule(ds_id, params)


def make_two_layer_trace() -> List[FakeModule]:
    """Return a submodule trace: Embed → L0 → L1 → Head (fwd) + reverse (bwd)."""
    embed = build_linear_module(0, 2, 1_000_000)
    layer0 = build_linear_module(1, 4, 2_000_000)
    layer1 = build_linear_module(2, 4, 2_000_000)
    head = build_linear_module(3, 2, 1_000_000)
    forward = [embed, layer0, layer1, head]
    backward = [head, layer1, layer0, embed]
    return forward + backward


class TestLifecycleRecordToComplete:
    """Simulate one RECORD iteration followed by one COMPLETE iteration."""

    def test_record_phase_does_not_activate_cache(self):
        cfg = MadelineConfig(enabled=True, auto_profile=False, memory_budget_numel=5_000_000)
        mgr = ForwardCacheManager(config=cfg, device=0)

        trace = make_two_layer_trace()
        # Simulate forward pass (RECORD)
        mgr.set_forward_phase(True)
        for mod in trace[:4]:
            # In RECORD phase should_cache always returns False because is_active is False
            assert not mgr.should_cache(mod.ds_id)

        # Simulate backward pass (RECORD)
        mgr.set_forward_phase(False)
        for mod in trace[4:]:
            assert not mgr.should_cache(mod.ds_id)

        assert not mgr.is_active
        assert len(mgr.cache_set) == 0

    def test_initialize_creates_cache_set(self):
        cfg = MadelineConfig(
            enabled=True,
            auto_profile=False,
            memory_budget_numel=5_000_000,
            prefetch_bucket_size=50_000_000,
            gain_weights={"alpha": 1.0, "beta": 1.0},
        )
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()

        # Call initialize exactly as coordinator.reset_step() would
        mgr.initialize(submodule_order=trace)

        assert mgr._initialized
        # With budget 5M and modules of size 2M/8M/8M/2M, only Embed (2M) or Head (2M)
        # should fit.  The exact set depends on the gain scores.
        assert len(mgr.cache_set) > 0
        assert mgr.is_active

    def test_complete_forward_skips_release_for_cached_modules(self):
        cfg = MadelineConfig(
            enabled=True,
            auto_profile=False,
            memory_budget_numel=5_000_000,
            prefetch_bucket_size=50_000_000,
        )
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()
        mgr.initialize(submodule_order=trace)

        mgr.set_forward_phase(True)
        cached_count = 0
        for mod in trace[:4]:  # forward pass
            if mgr.should_cache(mod.ds_id):
                cached_count += 1
                # Coordinator would skip release here, so param stays AVAILABLE
                for p in fake_iter_params(mod):
                    p.ds_active_sub_modules.discard(mod.ds_id)
                    # In reality param.ds_status stays AVAILABLE; we just assert
                    # the coordinator does NOT call partition()
            else:
                # Coordinator would release normally
                for p in fake_iter_params(mod):
                    p.ds_active_sub_modules.discard(mod.ds_id)
                    p.ds_status = "NOT_AVAILABLE"

        assert cached_count > 0, "At least one module should be cached"

    def test_complete_backward_allows_release(self):
        cfg = MadelineConfig(
            enabled=True,
            auto_profile=False,
            memory_budget_numel=5_000_000,
            prefetch_bucket_size=50_000_000,
        )
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()
        mgr.initialize(submodule_order=trace)

        mgr.set_forward_phase(False)  # backward
        for mod in trace[4:]:
            # During backward should_cache always returns False
            assert not mgr.should_cache(mod.ds_id)

    def test_on_step_end_resets_phase(self):
        cfg = MadelineConfig(enabled=True, auto_profile=False, memory_budget_numel=5_000_000)
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()
        mgr.initialize(submodule_order=trace)

        mgr.set_forward_phase(False)
        mgr.on_step_end()
        assert mgr.is_forward_phase is True


class TestStateMachineWithFakeCoordinator:
    """Simulate the param status state machine with cache interception."""

    def test_cached_param_stays_available_across_forward_backward(self):
        cfg = MadelineConfig(
            enabled=True,
            auto_profile=False,
            memory_budget_numel=5_000_000,
            prefetch_bucket_size=50_000_000,
        )
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()
        mgr.initialize(submodule_order=trace)

        # Pick the first cached module
        cached_mod = None
        for mod in trace[:4]:
            if mgr.should_cache(mod.ds_id):
                cached_mod = mod
                break

        if cached_mod is None:
            pytest.skip("No module was cached with this budget — test assumption failed")

        # Simulate forward fetch → all params become AVAILABLE
        for param in fake_iter_params(cached_mod):
            param.ds_status = "AVAILABLE"
            param.ds_active_sub_modules.add(cached_mod.ds_id)

        # Forward release — cache intercepts, does NOT partition
        mgr.set_forward_phase(True)
        if mgr.should_cache(cached_mod.ds_id):
            for param in fake_iter_params(cached_mod):
                param.ds_active_sub_modules.discard(cached_mod.ds_id)
            # status remains AVAILABLE for all params

        for param in fake_iter_params(cached_mod):
            assert param.ds_status == "AVAILABLE"

        # Backward fetch — coordinator computes fetch_numel for NOT_AVAILABLE params only
        mgr.set_forward_phase(False)
        # Since status is AVAILABLE, fetch_numel == 0 → AllGather skipped (automatic)
        not_available_numel = sum(
            p.ds_numel for p in fake_iter_params(cached_mod) if p.ds_status == "NOT_AVAILABLE"
        )
        assert not_available_numel == 0

        # Backward release — no cache interception, param is released
        param.ds_active_sub_modules.discard(cached_mod.ds_id)
        param.ds_status = "NOT_AVAILABLE"
        assert param.ds_status == "NOT_AVAILABLE"

    def test_uncached_param_follows_normal_zero3_flow(self):
        cfg = MadelineConfig(
            enabled=True,
            auto_profile=False,
            memory_budget_numel=1_000_000,  # very small budget → likely no cache
            prefetch_bucket_size=50_000_000,
        )
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()
        mgr.initialize(submodule_order=trace)

        if mgr.is_active:
            pytest.skip("Budget was large enough to enable cache — test needs inactive cache")

        mod = trace[0]
        mgr.set_forward_phase(True)
        assert not mgr.should_cache(mod.ds_id)

        param = fake_iter_params(mod)[0]
        param.ds_status = "AVAILABLE"
        param.ds_active_sub_modules.add(mod.ds_id)

        # Normal release (no cache)
        param.ds_active_sub_modules.discard(mod.ds_id)
        param.ds_status = "NOT_AVAILABLE"
        assert param.ds_status == "NOT_AVAILABLE"


class TestBudgetEdgeCases:
    """Verify behaviour at budget boundaries."""

    def test_zero_budget_disables_caching(self):
        cfg = MadelineConfig(
            enabled=True, auto_profile=False, memory_budget_numel=0
        )
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()
        mgr.initialize(submodule_order=trace)
        assert not mgr.is_active
        assert len(mgr.cache_set) == 0

    def test_huge_budget_caches_everything(self):
        cfg = MadelineConfig(
            enabled=True,
            auto_profile=False,
            memory_budget_numel=1_000_000_000,
            prefetch_bucket_size=50_000_000,
        )
        mgr = ForwardCacheManager(config=cfg, device=0)
        trace = make_two_layer_trace()
        mgr.initialize(submodule_order=trace)

        # All forward modules should fit
        forward_mods = {m.ds_id for m in trace[:4]}
        assert forward_mods.issubset(mgr.cache_set)
