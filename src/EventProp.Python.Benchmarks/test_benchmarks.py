"""Benchmarks comparing pybind11 (EventProp.Python) vs CFFI (eventprop_cffi).

Tests the Python public API surface — layer forward/backward passes and loss functions.
Problem sizes and spike patterns match the .NET/C benchmark suite for alignment.

Usage:
    PYTHONPATH="EventProp.Python:EventProp.Native.Python" pytest test_benchmarks.py -v --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest

# Import both libraries — they coexist because eventprop_cffi was renamed.
from eventprop.eventprop_cpp import (
    Spikes, SpikesVector, Maxima, MaximaVector,
)

from eventprop_cffi.spikes import (
    Spikes as NativeSpikes, SpikesVector as NativeSpikesVector,
    Maxima as NativeMaxima, MaximaVector as NativeMaximaVector,
)

from eventprop.lif_layer import LIFLayer, LIFLayerParameters
from eventprop.li_layer import LILayer, LILayerParameters as LiLIFLayerParameters
from eventprop.loss_layer import TTFSCrossEntropyLoss, VMaxCrossEntropyLoss

from eventprop_cffi.lif_layer import LIFLayer as NativeLIFLayer, LIFLayerParameters as NativeLIFLayerParameters
from eventprop_cffi.li_layer import LILayer as NativeLILayer, LILayerParameters as NativeLiLIFLayerParameters
from eventprop_cffi.loss_layer import TTFSCrossEntropyLoss as NativeTTFSCrossEntropyLoss, VMaxCrossEntropyLoss as NativeVMaxCrossEntropyLoss

from conftest import SIZES, TAU_MEM, TAU_SYN, V_TH, make_input_batch, make_weights


# ============================================================
# Helper: create Spikes objects for both libraries.
# ============================================================

def make_spikes(times, sources):
    """Create Spikes objects for both libraries from raw arrays."""
    return (Spikes(times, sources), NativeSpikes(times, sources))


def make_spikes_vector(batch):
    """Create SpikesVector objects for both libraries from a list of (times, sources) tuples."""
    pybind11_batch = SpikesVector([Spikes(t, s) for t, s in batch])
    native_batch = NativeSpikesVector([NativeSpikes(t, s) for t, s in batch])
    return pybind11_batch, native_batch


def make_maxima_vector(batch):
    """Create MaximaVector objects for both libraries."""
    pybind11_batch = MaximaVector([Maxima(t, v) for t, v in batch])
    native_batch = NativeMaximaVector([NativeMaxima(t, v) for t, v in batch])
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
# LIF Layer — Forward & Backward Benchmarks
# ============================================================

class TestLIFForward:
    """Benchmark LIFLayer.forward() — the core forward pass."""

    def _make_layer_and_input(self, size_name, pattern):
        """Create LIF layer and input batch for both libraries."""
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        pybind11_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_params = NativeLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)

        pybind11_layer = LIFLayer(pybind11_params, w_in=w)
        native_layer = NativeLIFLayer(native_params, w_in=w)

        return (pybind11_layer, native_layer), pybind11_spikes, native_spikes

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, spikes, _ = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[0].forward, SpikesVector([spikes]))

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
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

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        pybind11_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_params = NativeLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)

        pybind11_layer = LIFLayer(pybind11_params, w_in=w)
        native_layer = NativeLIFLayer(native_params, w_in=w)

        # Run forward pass for both
        pybind11_post = pybind11_layer.forward(SpikesVector([pybind11_spikes]))
        native_post = native_layer.forward([native_spikes])

        # Set error=1.0 on all post-synaptic spikes
        for i in range(pybind11_post[0].n_spikes):
            pybind11_post[0].set_error(i, 1.0)
        for i in range(native_post[0].n_spikes):
            native_post[0].set_error(i, 1.0)

        return (pybind11_layer, native_layer), pybind11_post, native_post

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, post_spikes, _ = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[0].backward)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
        layer_pair, _, post_spikes = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[1].backward)


# ============================================================
# LI Layer — Forward & Backward Benchmarks (maxima-based)
# ============================================================

class TestLIForward:
    """Benchmark LILayer.forward() — membrane potential maxima computation."""

    def _make_layer_and_input(self, size_name, pattern):
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        pybind11_params = LiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)
        native_params = NativeLiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)

        pybind11_layer = LILayer(pybind11_params, w_in=w)
        native_layer = NativeLILayer(native_params, w_in=w)

        return (pybind11_layer, native_layer), pybind11_spikes, native_spikes

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, spikes, _ = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[0].forward, SpikesVector([spikes]))

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
        layer_pair, _, spikes = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[1].forward, [spikes])


class TestLIBackward:
    """Benchmark LILayer.backward() — maxima-based backward pass."""

    def _make_layer_and_batch(self, size_name, pattern):
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        pybind11_params = LiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)
        native_params = NativeLiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)

        pybind11_layer = LILayer(pybind11_params, w_in=w)
        native_layer = NativeLILayer(native_params, w_in=w)

        # Run forward pass for both
        pybind11_maxima = pybind11_layer.forward(SpikesVector([pybind11_spikes]))
        native_maxima = native_layer.forward([native_spikes])

        # Set error=1.0 on all maxima neurons
        for i in range(len(pybind11_maxima[0].values)):
            pybind11_maxima[0].set_error(i, 1.0)
        for i in range(len(native_maxima[0].values)):
            native_maxima[0].set_error(i, 1.0)

        return (pybind11_layer, native_layer), pybind11_maxima, native_maxima

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, maxima, _ = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[0].backward)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
        layer_pair, _, maxima = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[1].backward)


# ============================================================
# Loss Layer — Forward & Backward Benchmarks
# ============================================================

class TestTTFSForward:
    """Benchmark TTFSCrossEntropyLoss.forward() — includes first spike extraction."""

    def _make_layer_and_input(self, size_name, pattern):
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        lif_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        loss_params = type('TTFSCrossEntropyLossParameters', (), {'lif_parameters': lif_params, 'alpha': 1e-2})()

        native_lif_params = NativeLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_loss_params = type('TTFSCrossEntropyLossParameters', (), {'lif_parameters': native_lif_params, 'alpha': 1e-2})()

        pybind11_layer = TTFSCrossEntropyLoss(loss_params, w_in=w)
        native_layer = NativeTTFSCrossEntropyLoss(native_loss_params, w_in=w)

        return (pybind11_layer, native_layer), pybind11_spikes, native_spikes

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, spikes, _ = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[0].forward, SpikesVector([spikes]))

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
        layer_pair, _, spikes = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[1].forward, [spikes])


class TestTTFSBackward:
    """Benchmark TTFSCrossEntropyLoss.backward() — includes loss gradient setup."""

    def _make_layer_and_batch(self, size_name, pattern):
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        lif_params = LIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        loss_params = type('TTFSCrossEntropyLossParameters', (), {'lif_parameters': lif_params, 'alpha': 1e-2})()

        native_lif_params = NativeLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        native_loss_params = type('TTFSCrossEntropyLossParameters', (), {'lif_parameters': native_lif_params, 'alpha': 1e-2})()

        pybind11_layer = TTFSCrossEntropyLoss(loss_params, w_in=w)
        native_layer = NativeTTFSCrossEntropyLoss(native_loss_params, w_in=w)

        # Run forward pass for both
        pybind11_layer.forward(SpikesVector([pybind11_spikes]))
        native_layer.forward([native_spikes])

        # Set labels for backward pass
        batch_size = 1
        labels = np.array([0])

        return (pybind11_layer, native_layer), labels

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, labels = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[0].backward, labels)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
        layer_pair, labels = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[1].backward, labels)


class TestVMaxForward:
    """Benchmark VMaxCrossEntropyLoss.forward() — includes sum0 computation."""

    def _make_layer_and_input(self, size_name, pattern):
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        lif_params = LiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)
        native_lif_params = NativeLiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)

        pybind11_layer = VMaxCrossEntropyLoss(lif_params, w_in=w)
        native_layer = NativeVMaxCrossEntropyLoss(native_lif_params, w_in=w)

        return (pybind11_layer, native_layer), pybind11_spikes, native_spikes

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, spikes, _ = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[0].forward, SpikesVector([spikes]))

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
        layer_pair, _, spikes = self._make_layer_and_input(size_name, pattern)
        benchmark(layer_pair[1].forward, [spikes])


class TestVMaxBackward:
    """Benchmark VMaxCrossEntropyLoss.backward() — includes error setup."""

    def _make_layer_and_batch(self, size_name, pattern):
        cfg = SIZES[size_name]
        n_in, n_out, spikes_per_neuron = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]

        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spikes_per_neuron, pattern)

        pybind11_spikes = Spikes(times, sources)
        native_spikes = NativeSpikes(times, sources)

        lif_params = LiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)
        native_lif_params = NativeLiLIFLayerParameters(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN)

        pybind11_layer = VMaxCrossEntropyLoss(lif_params, w_in=w)
        native_layer = NativeVMaxCrossEntropyLoss(native_lif_params, w_in=w)

        # Run forward pass for both
        pybind11_layer.forward(SpikesVector([pybind11_spikes]))
        native_layer.forward([native_spikes])

        # Set labels for backward pass
        batch_size = 1
        labels = np.array([0])

        return (pybind11_layer, native_layer), labels

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_pybind11(self, benchmark, size_name, pattern):
        layer_pair, labels = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[0].backward, labels)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    def test_cffi(self, benchmark, size_name, pattern):
        layer_pair, labels = self._make_layer_and_batch(size_name, pattern)
        benchmark(layer_pair[1].backward, labels)
