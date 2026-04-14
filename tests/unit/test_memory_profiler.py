# Copyright (c) Madeline Project Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Memory Profiler.

These tests mock torch.cuda so they can run without a GPU environment.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock torch before importing memory_profiler, since it imports torch at module level.
_mock_torch = MagicMock()
_mock_torch.cuda.max_memory_allocated = MagicMock(return_value=0)
_mock_torch.cuda.get_device_properties = MagicMock()
sys.modules.setdefault("torch", _mock_torch)

from madeline.memory_profiler import MemoryProfiler


class TestMemoryProfiler:
    """Tests for MemoryProfiler budget calculation."""

    def test_compute_budget_basic(self):
        """Basic budget computation with known values."""
        # 12 GB total, 6 GB peak, 10% safety -> budget = 12 - 6 - 1.2 = 4.8 GB
        _mock_torch.cuda.max_memory_allocated.return_value = 6_000_000_000
        _mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_mem=12_000_000_000
        )

        profiler = MemoryProfiler(reserved_memory_ratio=0.1, device=0)
        profiler.capture_peak()

        budget = profiler.compute_budget_bytes()
        expected = 12_000_000_000 - 6_000_000_000 - 1_200_000_000  # 4.8 GB
        assert budget == expected

    def test_compute_budget_numel_fp16(self):
        """Budget in numel for fp16 (2 bytes per element)."""
        _mock_torch.cuda.max_memory_allocated.return_value = 6_000_000_000
        _mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_mem=12_000_000_000
        )

        profiler = MemoryProfiler(reserved_memory_ratio=0.1, device=0)
        profiler.capture_peak()

        budget_numel = profiler.compute_budget_numel(bytes_per_element=2)
        expected_bytes = 12_000_000_000 - 6_000_000_000 - 1_200_000_000
        assert budget_numel == expected_bytes // 2

    def test_compute_budget_near_full(self):
        """When nearly all memory is used, budget should be 0."""
        _mock_torch.cuda.max_memory_allocated.return_value = 11_500_000_000
        _mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_mem=12_000_000_000
        )

        profiler = MemoryProfiler(reserved_memory_ratio=0.1, device=0)
        profiler.capture_peak()

        budget = profiler.compute_budget_bytes()
        # 12 - 11.5 - 1.2 = -0.7 -> clamped to 0
        assert budget == 0

    def test_compute_budget_before_capture(self):
        """Calling compute_budget before capture should raise."""
        profiler = MemoryProfiler()
        with pytest.raises(RuntimeError):
            profiler.compute_budget_bytes()

    def test_zero_safety_margin(self):
        """Zero safety margin should use all available memory."""
        _mock_torch.cuda.max_memory_allocated.return_value = 4_000_000_000
        _mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_mem=12_000_000_000
        )

        profiler = MemoryProfiler(reserved_memory_ratio=0.0, device=0)
        profiler.capture_peak()

        budget = profiler.compute_budget_bytes()
        assert budget == 8_000_000_000  # 12 - 4 - 0
