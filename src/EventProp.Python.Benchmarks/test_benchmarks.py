"""Benchmarks comparing pybind11 (EventProp.Python) vs Cython (eventprop_c).

Tests the Python public API surface — layer forward/backward passes, loss layers,
full training loops, and utility functions. Problem sizes match the C/C++ benchmark suite.

Usage:
    cd src/EventProp.Python.Benchmarks
    source .venv/bin/activate
    DYLD_LIBRARY_PATH=../../EventProp.C/build pytest test_benchmarks.py -v --benchmark-only

    # Save baseline for comparison:
    pytest test_benchmarks.py --benchmark-only --benchmark-save=baseline

    # Compare against saved baseline:
    pytest test_benchmarks.py --benchmark-only --benchmark-compare=baseline
"""

from __future__ import annotations

import numpy as np
import pytest

# pybind11 (reference)
from eventprop.eventprop_cpp import Spikes as PySpikes, SpikesVector as PySpikesVector
from eventprop.lif_layer import LIFLayer as PyLIFLayer, LIFLayerParameters as PyLIFParams
from eventprop.li_layer import LILayer as PyLILayer, LILayerParameters as PyLIParams
from eventprop.loss_layer import (
    TTFSCrossEntropyLoss as PyTTFS,
    VMaxCrossEntropyLoss as PyVMax,
)

# Cython (under test)
from eventprop_c.spikes import Spikes as CySpikes, SpikesVector as CySpikesVector
from eventprop_c.lif_layer import LIFLayer as CyLIFLayer, LIFLayerParameters as CyLIFParams
from eventprop_c.li_layer import LILayer as CyLILayer, LILayerParameters as CyLIParams
from eventprop_c.loss_layer import (
    TTFSCrossEntropyLoss as CyTTFS,
    VMaxCrossEntropyLoss as CyVMax,
)

from conftest import SIZES, TAU_MEM, TAU_SYN, V_TH, make_input_batch, make_weights


# ============================================================
# LIF Layer — Forward & Backward Benchmarks
# ============================================================

# Burst patterns at Medium+ are extremely slow (root-finding iterations).
# Skip them to keep benchmark runs tractable — matches C/C++ suite behaviour.
SLOW_BURST = {("Medium", "burst"), ("Large", "burst")}


class TestLIF_Forward:
    """Benchmark LIFLayer.forward() — core forward pass."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_lif_forward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = PyLIFLayer(PyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])

        benchmark(layer.forward, batch)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_lif_forward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = CyLIFLayer(CyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])

        benchmark(layer.forward, batch)


class TestLIF_Backward:
    """Benchmark LIFLayer.backward() — core backward pass."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_lif_backward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = PyLIFLayer(PyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])
        post = layer.forward(batch)

        for i in range(post[0].n_spikes):
            post[0].set_error(i, 1.0)

        benchmark(layer.backward)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_lif_backward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = CyLIFLayer(CyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])
        post = layer.forward(batch)

        # Set errors on output spikes — matches pybind11 benchmark exactly
        for i in range(post[0].n_spikes):
            post[0].set_error(i, 1.0)

        benchmark(layer.backward)


# ============================================================
# LI Layer — Forward & Backward Benchmarks (maxima-based)
# ============================================================

class TestLI_Forward:
    """Benchmark LILayer.forward() — membrane potential maxima."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_li_forward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = PyLILayer(PyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])

        benchmark(layer.forward, batch)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_li_forward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = CyLILayer(CyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])

        benchmark(layer.forward, batch)


class TestLI_Backward:
    """Benchmark LILayer.backward() — maxima-based backward pass."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_li_backward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = PyLILayer(PyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])
        maxima = layer.forward(batch)

        for i in range(len(maxima[0].values)):
            maxima[0].set_error(i, 1.0)

        benchmark(layer.backward)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_li_backward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = CyLILayer(CyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])
        maxima = layer.forward(batch)

        for i in range(maxima[0].n_post):
            maxima[0].set_error(i, 1.0)

        benchmark(layer.backward)


# ============================================================
# Loss Layer — Forward & Backward Benchmarks (batch=1 for simplicity)
# ============================================================

class TestTTFSForward:
    """Benchmark TTFSCrossEntropyLoss.forward() — includes first spike extraction."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_ttfs_forward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        from eventprop.loss_layer import TTFSCrossEntropyLossParameters
        lif_params = PyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        loss_params = TTFSCrossEntropyLossParameters(lif_parameters=lif_params, alpha=1e-2)

        layer = PyTTFS(loss_params, w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])

        benchmark(layer.forward, batch)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_ttfs_forward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        from eventprop_c.loss_layer import TTFSCrossEntropyLossParameters
        lif_params = CyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        loss_params = TTFSCrossEntropyLossParameters(lif_parameters=lif_params, alpha=1e-2)

        layer = CyTTFS(loss_params, w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])

        benchmark(layer.forward, batch)


class TestTTFSBackward:
    """Benchmark TTFSCrossEntropyLoss.backward() — loss gradient + chain backward."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_ttfs_backward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        from eventprop.loss_layer import TTFSCrossEntropyLossParameters
        lif_params = PyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        loss_params = TTFSCrossEntropyLossParameters(lif_parameters=lif_params, alpha=1e-2)

        layer = PyTTFS(loss_params, w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])
        layer.forward(batch)

        benchmark(layer.backward, np.array([0]))

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_ttfs_backward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        from eventprop_c.loss_layer import TTFSCrossEntropyLossParameters
        lif_params = CyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH)
        loss_params = TTFSCrossEntropyLossParameters(lif_parameters=lif_params, alpha=1e-2)

        layer = CyTTFS(loss_params, w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])
        layer.forward(batch)

        benchmark(layer.backward, np.array([0]))


class TestVMaxForward:
    """Benchmark VMaxCrossEntropyLoss.forward() — includes sum0 computation."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_vmax_forward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = PyVMax(PyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])

        benchmark(layer.forward, batch)

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_vmax_forward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = CyVMax(CyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])

        benchmark(layer.forward, batch)


class TestVMaxBackward:
    """Benchmark VMaxCrossEntropyLoss.backward() — error setup + chain backward."""

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_vmax_backward_pybind11(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = PyVMax(PyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = PySpikesVector([PySpikes(times, sources)])
        layer.forward(batch)

        benchmark(layer.backward, np.array([0]))

    @pytest.mark.parametrize("size_name", SIZES.keys(), ids=list(SIZES.keys()))
    @pytest.mark.parametrize("pattern", ["uniform", "poisson", "burst"])
    def test_vmax_backward_cython(self, benchmark, size_name, pattern):
        if (size_name, pattern) in SLOW_BURST:
            pytest.skip("Medium+/burst root-finding too slow")
        cfg = SIZES[size_name]
        n_in, n_out, spn = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
        w = make_weights(n_in, n_out)
        times, sources = make_input_batch(n_in, n_out, spn, pattern)

        layer = CyVMax(CyLIParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w)
        batch = CySpikesVector([CySpikes(times=times, sources=sources)])
        layer.forward(batch)

        benchmark(layer.backward, np.array([0]))


# ============================================================
# Batch benchmarks — realistic batch sizes (4, 8, 16)
# ============================================================

class TestBatchForward:
    """Benchmark LIFLayer.forward() with realistic batch sizes."""

    @pytest.mark.parametrize("batch_size", [4, 8, 16])
    def test_pybind11_batch(self, benchmark, batch_size):
        n_in, n_out = 50, 20
        w = make_weights(n_in, n_out)

        layer = PyLIFLayer(PyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w)

        rng = np.random.default_rng(42)
        spikes = [PySpikes(
            np.sort(rng.uniform(0.001, 0.1, size=20)),
            rng.integers(0, n_in, 20, dtype=np.int32),
        ) for _ in range(batch_size)]

        benchmark(layer.forward, PySpikesVector(spikes))

    @pytest.mark.parametrize("batch_size", [4, 8, 16])
    def test_Cython_batch(self, benchmark, batch_size):
        n_in, n_out = 50, 20
        w = make_weights(n_in, n_out)

        layer = CyLIFLayer(CyLIFParams(n=n_out, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w)

        rng = np.random.default_rng(42)
        spikes = [CySpikes(
            times=np.sort(rng.uniform(0.001, 0.1, size=20)),
            sources=rng.integers(0, n_in, 20, dtype=np.int32),
        ) for _ in range(batch_size)]

        benchmark(layer.forward, CySpikesVector(spikes))


# ============================================================
# Dropout benchmark — pure NumPy (Cython) vs C++ (pybind11)
# ============================================================

class TestDropout:
    """Benchmark dropout_spikes — pure NumPy (Cython) vs C++ (pybind11)."""

    @pytest.mark.parametrize("batch_size", [4, 8, 16])
    def test_pybind11_dropout(self, benchmark, batch_size):
        from eventprop.training import dropout_spikes_cpp

        rng = np.random.default_rng(42)
        n_in, n_spikes_per_sample = 10, 30
        spikes = [PySpikes(
            np.sort(rng.uniform(0.001, 0.1, size=n_spikes_per_sample)),
            rng.integers(0, n_in, n_spikes_per_sample, dtype=np.int32),
        ) for _ in range(batch_size)]

        benchmark(dropout_spikes_cpp, PySpikesVector(spikes), 0.3, 42)

    @pytest.mark.parametrize("batch_size", [4, 8, 16])
    def test_Cython_dropout(self, benchmark, batch_size):
        from eventprop_c.training import dropout_spikes

        rng = np.random.default_rng(42)
        n_in, n_spikes_per_sample = 10, 30
        spikes = [CySpikes(
            times=np.sort(rng.uniform(0.001, 0.1, size=n_spikes_per_sample)),
            sources=rng.integers(0, n_in, n_spikes_per_sample, dtype=np.int32),
        ) for _ in range(batch_size)]

        benchmark(dropout_spikes, CySpikesVector(spikes), 0.3, 42)


# ============================================================
# Full training step — forward + backward chain (2-layer TTFS/VMax)
# ============================================================

class TestTrainingStep:
    """Benchmark a full training step: hidden.forward -> loss.forward -> loss.backward."""

    @pytest.mark.parametrize("batch_size", [4, 8])
    def test_pybind11_ttfs_step(self, benchmark, batch_size):
        n_in, n_hidden, n_classes = 10, 8, 3
        w1 = make_weights(n_in, n_hidden)
        w2 = make_weights(n_hidden, n_classes)

        from eventprop.loss_layer import TTFSCrossEntropyLossParameters
        hidden = PyLIFLayer(PyLIFParams(n=n_hidden, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w1)
        loss = PyTTFS(
            TTFSCrossEntropyLossParameters(lif_parameters=PyLIFParams(n=n_classes, n_in=n_hidden, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), alpha=1e-2),
            w_in=w2,
        )

        rng = np.random.default_rng(42)
        spikes = [PySpikes(
            np.sort(rng.uniform(0.001, 0.05, size=15)),
            rng.integers(0, n_in, 15, dtype=np.int32),
        ) for _ in range(batch_size)]

        def step():
            hidden.forward(PySpikesVector(spikes))
            loss.forward(hidden.post_batch)
            loss.backward(np.array([i % n_classes for i in range(batch_size)]))

        benchmark(step)

    @pytest.mark.parametrize("batch_size", [4, 8])
    def test_Cython_ttfs_step(self, benchmark, batch_size):
        n_in, n_hidden, n_classes = 10, 8, 3
        w1 = make_weights(n_in, n_hidden)
        w2 = make_weights(n_hidden, n_classes)

        from eventprop_c.loss_layer import TTFSCrossEntropyLossParameters
        hidden = CyLIFLayer(CyLIFParams(n=n_hidden, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w1)
        loss = CyTTFS(
            TTFSCrossEntropyLossParameters(lif_parameters=CyLIFParams(n=n_classes, n_in=n_hidden, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), alpha=1e-2),
            w_in=w2,
        )

        rng = np.random.default_rng(42)
        spikes = [CySpikes(
            times=np.sort(rng.uniform(0.001, 0.05, size=15)),
            sources=rng.integers(0, n_in, 15, dtype=np.int32),
        ) for _ in range(batch_size)]

        def step():
            hidden.forward(CySpikesVector(spikes))
            loss.forward(hidden.post_batch)
            loss.backward(np.array([i % n_classes for i in range(batch_size)]))

        benchmark(step)

    @pytest.mark.parametrize("batch_size", [4, 8])
    def test_pybind11_vmax_step(self, benchmark, batch_size):
        n_in, n_hidden, n_classes = 10, 8, 3
        w1 = make_weights(n_in, n_hidden)
        w2 = make_weights(n_hidden, n_classes)

        hidden = PyLIFLayer(PyLIFParams(n=n_hidden, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w1)
        loss = PyVMax(PyLIParams(n=n_classes, n_in=n_hidden, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w2)

        rng = np.random.default_rng(42)
        spikes = [PySpikes(
            np.sort(rng.uniform(0.001, 0.05, size=15)),
            rng.integers(0, n_in, 15, dtype=np.int32),
        ) for _ in range(batch_size)]

        def step():
            hidden.forward(PySpikesVector(spikes))
            loss.forward(hidden.post_batch)
            loss.backward(np.array([i % n_classes for i in range(batch_size)]))

        benchmark(step)

    @pytest.mark.parametrize("batch_size", [4, 8])
    def test_Cython_vmax_step(self, benchmark, batch_size):
        n_in, n_hidden, n_classes = 10, 8, 3
        w1 = make_weights(n_in, n_hidden)
        w2 = make_weights(n_hidden, n_classes)

        hidden = CyLIFLayer(CyLIFParams(n=n_hidden, n_in=n_in, tau_mem=TAU_MEM, tau_syn=TAU_SYN, v_th=V_TH), w_in=w1)
        loss = CyVMax(CyLIParams(n=n_classes, n_in=n_hidden, tau_mem=TAU_MEM, tau_syn=TAU_SYN), w_in=w2)

        rng = np.random.default_rng(42)
        spikes = [CySpikes(
            times=np.sort(rng.uniform(0.001, 0.05, size=15)),
            sources=rng.integers(0, n_in, 15, dtype=np.int32),
        ) for _ in range(batch_size)]

        def step():
            hidden.forward(CySpikesVector(spikes))
            loss.forward(hidden.post_batch)
            loss.backward(np.array([i % n_classes for i in range(batch_size)]))

        benchmark(step)
