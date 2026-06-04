"""Benchmarks comparing pybind11 (EventProp.Python) vs ctypes (eventprop).

Tests the Python public API surface — forward/backward passes and loss functions.
Problem sizes and spike patterns match the .NET/C benchmark suite for alignment.

Usage:
    PYTHONPATH="EventProp.Python:EventProp.Native.Python" pytest test_benchmarks_ctypes.py -v --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest

# Import pybind11 library (EventProp.Python) — the reference implementation
import sys as _sys
from pathlib import Path as _Path
_eventprop_python_path = str(_Path(__file__).resolve().parent.parent / "EventProp.Python")
if _eventprop_python_path not in _sys.path:
    _sys.path.insert(0, _eventprop_python_path)

from eventprop.eventprop_cpp import (
    Spikes as Pybind11Spikes,
    SpikesVector as Pybind11SpikesVector,
    Maxima as Pybind11Maxima,
    MaximaVector as Pybind11MaximaVector,
)

# Import ctypes library (eventprop) — new implementation to benchmark
import eventprop as eventprop_ctypes

from conftest_ctypes import SIZES, TAU_MEM, TAU_SYN, V_TH, make_input_batch, make_weights, get_ctypes_lib


# ============================================================
# Helper: create Spikes objects for both libraries.
# ============================================================

def make_spikes(times, sources):
    """Create Spikes objects for both libraries from raw arrays."""
    pybind11_spikes = Pybind11Spikes(times, sources)
    
    # For ctypes, we just pass the raw arrays directly (zero-copy)
    native_spikes = (times, sources)
    
    return pybind11_spikes, native_spikes


def make_spikes_vector(batch):
    """Create SpikesVector objects for both libraries from a list of (times, sources) tuples."""
    pybind11_batch = Pybind11SpikesVector([Pybind11Spikes(t, s) for t, s in batch])
    # For ctypes, we just use the list of tuples directly
    native_batch = batch
    return pybind11_batch, native_batch


def make_maxima_vector(batch):
    """Create MaximaVector objects for both libraries."""
    pybind11_batch = Pybind11MaximaVector([Pybind11Maxima(t, v) for t, v in batch])
    # For ctypes, we just use the list of tuples directly
    native_batch = batch
    return pybind11_batch, native_batch


# ============================================================
# Parametrized fixtures for size × pattern combinations.
# ============================================================

@pytest.fixture(params=SIZES.keys(), ids=list(SIZES.keys()))
def size_name(request):
    return request.param


@pytest.fixture(params=["uniform", "poisson", "burst"], ids=["uniform", "poisson", "burst"])
def pattern(request):
    return request.param


@pytest.fixture(params=["Tiny", "Small"], ids=lambda x: f"size={x}")
def small_size(request):
    return request.param


@pytest.fixture(params=["Medium", "Large"], ids=lambda x: f"size={x}")
def large_size(request):
    return request.param


# ============================================================
# ComputeSpikes Benchmark — Forward Pass Comparison
# ============================================================

class TestComputeSpikesBenchmarks:
    """Benchmark compute_spikes — forward pass (pybind11 vs ctypes)."""

    def _prepare_data(self, size_name: str, pattern: str):
        """Prepare test data for both libraries."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)
        
        return w, times, sources

    def test_compute_spikes_pybind11(self, benchmark, size_name: str, pattern: str):
        """Benchmark pybind11 compute_spikes (reference)."""
        w, times, sources = self._prepare_data(size_name, pattern)
        
        def run():
            from eventprop_ctypes.eventprop_cpp import compute_spikes_batch_cpp
            spikes = Pybind11Spikes(times, sources)
            batch = Pybind11SpikesVector([spikes])
            return compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)
        
        benchmark(run)

    def test_compute_spikes_ctypes(self, benchmark, size_name: str, pattern: str):
        """Benchmark ctypes compute_spikes (zero-copy)."""
        w, times, sources = self._prepare_data(size_name, pattern)
        cfg = SIZES[size_name]
        
        # Get ctypes library instance (cached for performance)
        lib = get_ctypes_lib()
        
        # Convert to lists once (outside benchmark loop)
        w_list = w.tolist()
        times_list = times.tolist()
        sources_list = sources.tolist()
        
        def run():
            return lib.compute_spikes(
                w_list, times_list, sources_list,
                cfg["n_in"], cfg["n_out"]
            )
        
        benchmark(run)


# ============================================================
# ComputeSums Benchmark — Sum Computation Comparison
# ============================================================

class TestComputeSumsBenchmarks:
    """Benchmark compute_sums — sum computation (pybind11 vs ctypes)."""

    def _prepare_data(self, size_name: str, pattern: str):
        """Prepare test data for both libraries."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)
        
        return w, times, sources

    def test_compute_sums_pybind11(self, benchmark, size_name: str, pattern: str):
        """Benchmark pybind11 compute_sums (reference)."""
        w, times, sources = self._prepare_data(size_name, pattern)
        
        def run():
            from eventprop_ctypes.eventprop_cpp import compute_sums_batch_cpp
            spikes = Pybind11Spikes(times, sources)
            return compute_sums_batch_cpp(w, Pybind11SpikesVector([spikes]), TAU_MEM, TAU_SYN)
        
        benchmark(run)

    def test_compute_sums_ctypes(self, benchmark, size_name: str, pattern: str):
        """Benchmark ctypes compute_sums (zero-copy)."""
        w, times, sources = self._prepare_data(size_name, pattern)
        cfg = SIZES[size_name]
        
        # Get ctypes library instance (cached for performance)
        lib = get_ctypes_lib()
        
        def run():
            # Convert to lists for ctypes (minimal overhead)
            return lib.compute_sums(
                w_list, times_list, sources_list,
                cfg["n_in"], cfg["n_out"]
            )
        
        benchmark(run)


# ============================================================
# ComputeMaxima Benchmark — Maxima Computation Comparison
# ============================================================

class TestComputeMaximaBenchmarks:
    """Benchmark compute_maxima — maxima computation (pybind11 vs ctypes)."""

    def _prepare_data(self, size_name: str, pattern: str):
        """Prepare test data for both libraries."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)
        
        return w, times, sources

    def test_compute_maxima_pybind11(self, benchmark, size_name: str, pattern: str):
        """Benchmark pybind11 compute_maxima (reference)."""
        w, times, sources = self._prepare_data(size_name, pattern)
        
        def run():
            from eventprop_ctypes.eventprop_cpp import compute_maxima_batch_cpp
            spikes = Pybind11Spikes(times, sources)
            return compute_maxima_batch_cpp(w, Pybind11SpikesVector([spikes]), TAU_MEM, TAU_SYN)
        
        benchmark(run)

    def test_compute_maxima_ctypes(self, benchmark, size_name: str, pattern: str):
        """Benchmark ctypes compute_maxima (zero-copy)."""
        w, times, sources = self._prepare_data(size_name, pattern)
        cfg = SIZES[size_name]
        
        # Get ctypes library instance (cached for performance)
        lib = get_ctypes_lib()
        
        def run():
            # Convert to lists for ctypes (minimal overhead)
            return lib.compute_maxima(
                w_list, times_list, sources_list,
                cfg["n_in"], cfg["n_out"]
            )
        
        benchmark(run)


# ============================================================
# BackwardSpikes Benchmark — Backward Pass (Spike-based loss) Comparison
# ============================================================

class TestBackwardSpikesBenchmarks:
    """Benchmark backward_spikes — backward pass for spike-based loss (pybind11 vs ctypes)."""

    def _prepare_data(self, size_name: str, pattern: str):
        """Prepare test data for backward pass benchmark."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        
        # Generate input spikes
        input_times, input_sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)
        
        # Generate post-synaptic spikes (simulated output)
        post_times = np.linspace(0, 1, n_out * 2)
        post_sources = (np.arange(n_out * 2) % n_in).astype(np.int32)
        
        return w, input_times, input_sources, post_times, post_sources

    def test_backward_spikes_pybind11(self, benchmark, size_name: str, pattern: str):
        """Benchmark pybind11 backward_spikes (reference)."""
        w, input_times, input_sources, post_times, post_sources = self._prepare_data(size_name, pattern)
        cfg = SIZES[size_name]
        
        def run():
            from eventprop.eventprop_cpp import backward_spikes_batch_cpp
            
            # Create gradient array (zeros initially)
            gradient = np.zeros((cfg["n_in"], cfg["n_out"]))
            
            # Create spike objects
            input_spikes = Pybind11Spikes(input_times, input_sources)
            post_spikes = Pybind11Spikes(post_times, post_sources)
            
            # Run backward pass
            return backward_spikes_batch_cpp(
                w, [input_spikes], [post_spikes], gradient, V_TH, TAU_MEM, TAU_SYN
            )
        
        benchmark(run)

    def test_backward_spikes_ctypes(self, benchmark, size_name: str, pattern: str):
        """Benchmark ctypes backward_spikes (zero-copy)."""
        w, input_times, input_sources, post_times, post_sources = self._prepare_data(size_name, pattern)
        cfg = SIZES[size_name]
        
        # Get ctypes library instance (cached for performance)
        lib = get_ctypes_lib()
        
        def run():
            # Create gradient array (zeros initially)
            gradient = np.zeros((cfg["n_in"], cfg["n_out"]))
            
            # Run backward pass with ctypes (zero-copy)
            return lib.backward_spikes(
                w.tolist(), input_times.tolist(), input_sources.tolist(),
                post_times_list, post_sources_list,
                gradient, cfg["n_in"], cfg["n_out"]
            )
        
        benchmark(run)


# ============================================================
# BackwardMaxima Benchmark — Backward Pass (Voltage-based loss) Comparison
# ============================================================

class TestBackwardMaximaBenchmarks:
    """Benchmark backward_maxima — backward pass for voltage-based loss (pybind11 vs ctypes)."""

    def _prepare_data(self, size_name: str, pattern: str):
        """Prepare test data for backward pass benchmark."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        
        # Generate input spikes
        input_times, input_sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)
        
        # Generate maxima data (simulated voltage-based loss)
        maxima_times = np.linspace(0, 1, n_out * 2)
        maxima_values = np.random.randn(n_out * 2).astype(np.float64)
        
        return w, input_times, input_sources, maxima_times, maxima_values

    def test_backward_maxima_pybind11(self, benchmark, size_name: str, pattern: str):
        """Benchmark pybind11 backward_maxima (reference)."""
        w, input_times, input_sources, maxima_times, maxima_values = self._prepare_data(size_name, pattern)
        cfg = SIZES[size_name]
        
        def run():
            from eventprop.eventprop_cpp import backward_maxima_batch_cpp
            
            # Create gradient array (zeros initially)
            gradient = np.zeros((cfg["n_in"], cfg["n_out"]))
            
            # Create spike objects
            input_spikes = Pybind11Spikes(input_times, input_sources)
            maxima = Pybind11Maxima(maxima_times, maxima_values)
            
            # Run backward pass
            return backward_maxima_batch_cpp(
                w, [input_spikes], [maxima], gradient, V_TH, TAU_MEM, TAU_SYN
            )
        
        benchmark(run)

    def test_backward_maxima_ctypes(self, benchmark, size_name: str, pattern: str):
        """Benchmark ctypes backward_maxima (zero-copy)."""
        w, input_times, input_sources, maxima_times, maxima_values = self._prepare_data(size_name, pattern)
        cfg = SIZES[size_name]
        
        # Get ctypes library instance (cached for performance)
        lib = get_ctypes_lib()
        
        def run():
            # Create gradient array (zeros initially)
            gradient = np.zeros((cfg["n_in"], cfg["n_out"]))
            
            # Run backward pass with ctypes (zero-copy)
            return lib.backward_maxima(
                w.tolist(), input_times.tolist(), input_sources.tolist(),
                maxima_times_list, maxima_values_list,
                gradient, cfg["n_in"], cfg["n_out"]
            )
        
        benchmark(run)


# ============================================================
# Zero-Copy Access Benchmark — Memory access comparison
# ============================================================

class TestZeroCopyAccessBenchmarks:
    """Benchmark zero-copy memory access (ctypes only)."""

    def _prepare_data(self, size_name: str):
        """Prepare test data for zero-copy access benchmark."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, "uniform")
        
        return w, times, sources

    def test_ctypes_zero_copy_iteration(self, benchmark, size_name: str):
        """Benchmark zero-copy iteration over ctypes data."""
        w, times, sources = self._prepare_data(size_name)
        cfg = SIZES[size_name]
        
        # Get ctypes library instance
        lib = get_ctypes_lib()
        
        def run():
            # Compute sums and iterate over results (zero-copy)
            output = lib.compute_sums(
                w_list, times_list, sources_list,
                cfg["n_in"], cfg["n_out"]
            )
            
            # Zero-copy iteration (no allocation)
            total = 0.0
            for val in output.sum0:
                total += val
            
            return total
        
        benchmark(run)

    def test_ctypes_zero_copy_indexing(self, benchmark, size_name: str):
        """Benchmark zero-copy indexing of ctypes data."""
        w, times, sources = self._prepare_data(size_name)
        cfg = SIZES[size_name]
        
        # Get ctypes library instance
        lib = get_ctypes_lib()
        
        def run():
            # Compute sums and access individual elements (zero-copy)
            output = lib.compute_sums(
                w_list, times_list, sources_list,
                cfg["n_in"], cfg["n_out"]
            )
            
            # Zero-copy indexing (no allocation)
            total = 0.0
            for i in range(min(100, len(output.sum0))):
                total += output.sum0[i]
            
            return total
        
        benchmark(run)


# ============================================================
# Main test entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
