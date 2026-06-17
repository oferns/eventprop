"""Benchmarks comparing pybind11 (EventProp.Python) vs ctypes (eventprop_c).

Tests the Python public API surface — layer forward/backward passes and loss functions.
Problem sizes, spike patterns, batch sizes, and spike rates match the .NET/C benchmark suite.

Usage:
    PYTHONPATH="../EventProp.Python:../EventProp.Native.Python" pytest test_benchmarks_ctypes.py -v --benchmark-only
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
)
from eventprop.lif_layer import LIFLayer, LIFLayerParameters

# Import ctypes library (eventprop_c) — C library via ctypes
_eventprop_native_path = str(_Path(__file__).resolve().parent.parent / "EventProp.Native.Python")
if _eventprop_native_path not in _sys.path:
    _sys.path.insert(0, _eventprop_native_path)

from eventprop_c import (
    Spikes as CtypesSpikes, LIFLayer as CtypesLIFLayer,
    LIFLayerParameters as CtypesLIFLayerParameters,
)

from conftest_ctypes import SIZES, TAU_MEM, TAU_SYN, V_TH, make_input_batch, make_weights

# Additional batch sizes to test (1 is default)
BATCH_SIZES = [1, 4, 8, 16]

# Different spike rates (multiplier of spikes_per_neuron)
SPIKE_RATES = [0.5, 1.0, 2.0]  # Sparse, normal, dense


# ============================================================
# Helper: create Spikes objects for both libraries.
# ============================================================

def make_spikes(times, sources):
    """Create Spikes objects for both libraries from raw arrays."""
    pybind11_spikes = Pybind11Spikes(times, sources)
    native_spikes = CtypesSpikes(times, sources)
    
    return pybind11_spikes, native_spikes


# ============================================================
# Parametrized fixtures for size × pattern combinations.
# ============================================================

@pytest.fixture(params=SIZES.keys(), ids=list(SIZES.keys()))
def size_name(request):
    return request.param


@pytest.fixture(params=["uniform", "poisson", "burst"], ids=["uniform", "poisson", "burst"])
def pattern(request):
    return request.param


# ============================================================
# LIF Layer — Forward & Backward Benchmarks (Standard)
# ============================================================

class TestLIFForward:
    """Benchmark LIFLayer.forward() — the core forward pass."""

    def _make_layer_and_input(self, size_name, pattern):
        """Create LIF layer and input batch for both libraries."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Pybind11Spikes(times, sources)
        native_spikes = CtypesSpikes(times, sources)

        pybind11_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_params = CtypesLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)

        pybind11_layer = LIFLayer(pybind11_params, w_in=w)
        native_layer = CtypesLIFLayer(native_params, w_in=w)

        return (pybind11_layer, native_layer), pybind11_spikes, native_spikes

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, spikes, _ = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[0].forward, Pybind11SpikesVector([spikes]))

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_ctypes(self, benchmark, size_name, pattern):
        layer_pair, _, spikes = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[1].forward, [spikes])


class TestLIFBackward:
    """Benchmark LIFLayer.backward() — the core backward pass."""

    def _make_layer_and_batch(self, size_name, pattern):
        """Create LIF layer with pre-computed forward pass for both libraries."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_input = Pybind11Spikes(times, sources)
        native_input = CtypesSpikes(times, sources)

        pybind11_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_params = CtypesLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)

        pybind11_layer = LIFLayer(pybind11_params, w_in=w)
        native_layer = CtypesLIFLayer(native_params, w_in=w)

        # Run forward pass to generate post-synaptic spikes
        pybind11_layer.forward(Pybind11SpikesVector([pybind11_input]))
        native_layer.forward([native_input])

        return (pybind11_layer, native_layer), pybind11_input, native_input

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, post_spikes, _ = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[0].backward)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_ctypes(self, benchmark, size_name, pattern):
        layer_pair, _, post_spikes = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[1].backward)


# ============================================================
# Batch Size Benchmarks — Test performance at different batch sizes
# ============================================================

class TestBatchSizeForward:
    """Benchmark forward pass at different batch sizes."""

    def _make_layer_and_batch(self, size_name, pattern, batch_size):
        """Create LIF layer and input batch with specified batch size."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)

        # Generate multiple samples for batch
        pybind11_spikes_list = []
        native_spikes_list = []

        for i in range(batch_size):
            # Use different random seeds for each sample
            times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)
            pybind11_spikes_list.append(Pybind11Spikes(times, sources))
            native_spikes_list.append(CtypesSpikes(times, sources))

        pybind11_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_params = CtypesLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)

        pybind11_layer = LIFLayer(pybind11_params, w_in=w)
        native_layer = CtypesLIFLayer(native_params, w_in=w)

        return (pybind11_layer, native_layer), pybind11_spikes_list, native_spikes_list

    @pytest.mark.parametrize("size_name", ["Small", "Medium"])
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_pybind11(self, benchmark, size_name, batch_size):
        layer_pair, spikes_list, _ = self._make_layer_and_batch(size_name, "poisson", batch_size)
        benchmark(layer_pair[0].forward, Pybind11SpikesVector(spikes_list))

    @pytest.mark.parametrize("size_name", ["Small", "Medium"])
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_ctypes(self, benchmark, size_name, batch_size):
        layer_pair, _, spikes_list = self._make_layer_and_batch(size_name, "poisson", batch_size)
        benchmark(layer_pair[1].forward, spikes_list)


class TestBatchSizeBackward:
    """Benchmark backward pass at different batch sizes."""

    def _make_layer_and_batch(self, size_name, pattern, batch_size):
        """Create LIF layer with forward pass done for backward."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)

        # Generate multiple samples for batch
        pybind11_input_list = []
        native_input_list = []

        for i in range(batch_size):
            times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)
            pybind11_input_list.append(Pybind11Spikes(times, sources))
            native_input_list.append(CtypesSpikes(times, sources))

        pybind11_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_params = CtypesLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)

        pybind11_layer = LIFLayer(pybind11_params, w_in=w)
        native_layer = CtypesLIFLayer(native_params, w_in=w)

        # Run forward pass to generate post-synaptic spikes
        pybind11_layer.forward(Pybind11SpikesVector(pybind11_input_list))
        native_layer.forward(native_input_list)

        return (pybind11_layer, native_layer), pybind11_input_list, native_input_list

    @pytest.mark.parametrize("size_name", ["Small", "Medium"])
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_pybind11(self, benchmark, size_name, batch_size):
        layer_pair, input_list, _ = self._make_layer_and_batch(size_name, "poisson", batch_size)
        benchmark(layer_pair[0].backward)

    @pytest.mark.parametrize("size_name", ["Small", "Medium"])
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_ctypes(self, benchmark, size_name, batch_size):
        layer_pair, _, input_list = self._make_layer_and_batch(size_name, "poisson", batch_size)
        benchmark(layer_pair[1].backward)


# ============================================================
# Spike Rate Benchmarks — Test at different spike densities
# ============================================================

class TestSpikeRateForward:
    """Benchmark forward pass at different spike rates."""

    def _make_layer_and_input(self, size_name, rate):
        """Create LIF layer with modified spike count."""
        cfg = SIZES[size_name]
        n_in, n_out = cfg["n_in"], cfg["n_out"]
        spikes_per_neuron = int(cfg["spikes_per_neuron"] * rate)

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, "poisson")

        pybind11_spikes = Pybind11Spikes(times, sources)
        native_spikes = CtypesSpikes(times, sources)

        pybind11_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_params = CtypesLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)

        pybind11_layer = LIFLayer(pybind11_params, w_in=w)
        native_layer = CtypesLIFLayer(native_params, w_in=w)

        return (pybind11_layer, native_layer), pybind11_spikes, native_spikes

    @pytest.mark.parametrize("size_name", ["Small"])
    @pytest.mark.parametrize("rate", SPIKE_RATES)
    def test_pybind11(self, benchmark, size_name, rate):
        layer_pair, spikes, _ = self._make_layer_and_input(size_name, rate)
        benchmark(layer_pair[0].forward, Pybind11SpikesVector([spikes]))

    @pytest.mark.parametrize("size_name", ["Small"])
    @pytest.mark.parametrize("rate", SPIKE_RATES)
    def test_ctypes(self, benchmark, size_name, rate):
        layer_pair, _, spikes = self._make_layer_and_input(size_name, rate)
        benchmark(layer_pair[1].forward, [spikes])


# ============================================================
# Main test entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
