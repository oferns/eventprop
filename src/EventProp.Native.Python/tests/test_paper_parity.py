"""Paper reproducibility parity tests — pybind11 vs Cython (eventprop_c).

Loads pre-trained checkpoints from EventProp.Python/paper/ and verifies that
both backends produce identical forward/backward results on real datasets.

Usage:
    cd src/EventProp.Native.Python
    "C:/Program Files/Python311/python.exe" -m pytest tests/test_paper_parity.py -v
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
_src = _repo_root / "src"
sys.path.insert(0, str(_src / "EventProp.Python"))
sys.path.insert(0, str(_src / "EventProp.Native.Python"))

from eventprop.eventprop_cpp import Spikes as PySpikes, SpikesVector as PySpikesVector
from eventprop.lif_layer import LIFLayerParameters as PyLIFParams
from eventprop.li_layer import LILayerParameters as PyLIParams
from eventprop.loss_layer import (
    TTFSCrossEntropyLoss as PyTTFS,
    VMaxCrossEntropyLoss as PyVMax,
)

from eventprop_c.lif_layer import LIFLayerParameters as CyLIFParams
from eventprop_c.spikes import SpikesVector as CySpikesVector
from eventprop_c.li_layer import LILayerParameters as CyLIParams
from eventprop_c.loss_layer import (
    TTFSCrossEntropyLoss as CyTTFS,
    VMaxCrossEntropyLoss as CyVMax,
)

PAPER_DIR = _src / "EventProp.Python" / "paper"
YINYANG_DATA = (_src / "EventProp.Python" / "eventprop" / "yin_yang_data_set" / "publication_data")


def _make_spikes_py(samples):
    spikes = []
    for s in samples:
        times = np.array([x * 30e-3 for x in s] + [0.0], dtype=np.float64)
        sources = np.arange(len(s) + 1, dtype=np.int32)
        idx = np.argsort(times)
        spikes.append(PySpikes(times[idx], sources[idx]))
    return spikes


def _make_spikes_cy(samples):
    from eventprop_c.spikes import Spikes as CySpikes
    spikes = []
    for s in samples:
        times = np.array([x * 30e-3 for x in s] + [0.0], dtype=np.float64)
        sources = np.arange(len(s) + 1, dtype=np.int32)
        idx = np.argsort(times)
        spikes.append(CySpikes(times[idx], sources[idx]))
    return spikes


def _assert_close(actual, expected, name):
    nan_mask = np.isnan(expected)
    if np.any(nan_mask):
        assert np.all(np.isnan(actual[nan_mask])), f"{name}: NaN mismatch"
    valid = ~nan_mask
    if np.any(valid):
        diff = np.abs(actual[valid] - expected[valid])
        max_abs = diff.max()
        rel = diff / (np.abs(expected[valid]) + 1e-30)
        assert max_abs < 1e-8 or rel.max() < 1e-7, f"{name}: abs={max_abs:.2e} rel={rel.max():.2e}"


# ====================================================================
# YinYang TTFS Parity Tests (figure2.py checkpoint)
# ====================================================================

class TestYinYangTTFS:
    LIF = {"tau_mem": 20e-3, "tau_syn": 5e-3, "v_th": 1.0}
    TTFS = {"alpha": 0.003, "tau0": 0.0005, "tau1": 0.0064}

    @pytest.fixture(scope="class")
    def checkpoint(self):
        with open(PAPER_DIR / "yinyang_0.pkl", "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def test_data(self):
        return np.load(YINYANG_DATA / "test_samples.npy"), np.load(YINYANG_DATA / "test_labels.npy")

    @pytest.fixture(scope="class")
    def weights(self, checkpoint):
        return checkpoint[8][-1]  # final trained weights: [hidden (5×200), output (200×3)]

    def test_forward_first_spike_times(self, test_data, weights):
        samples, labels = test_data
        samples = samples[:64]
        w_hidden, w_output = weights

        from eventprop.lif_layer import LIFLayer as PyLIFLayer
        from eventprop.loss_layer import TTFSCrossEntropyLossParameters as PyTTFSParams
        from eventprop_c.lif_layer import LIFLayer as CyLIFLayer
        from eventprop_c.loss_layer import TTFSCrossEntropyLossParameters as CyTTFSParams

        py_batch = PySpikesVector(_make_spikes_py(samples))
        cy_batch = CySpikesVector(_make_spikes_cy(samples))

        py_hidden = PyLIFLayer(PyLIFParams(n=200, n_in=5, **self.LIF), w_in=w_hidden)
        cy_hidden = CyLIFLayer(CyLIFParams(n=200, n_in=5, **self.LIF), w_in=w_hidden)

        py_loss = PyTTFS(PyTTFSParams(lif_parameters=PyLIFParams(n=3, n_in=200, **self.LIF), **self.TTFS), w_in=w_output)
        cy_loss = CyTTFS(CyTTFSParams(lif_parameters=CyLIFParams(n=3, n_in=200, **self.LIF), **self.TTFS), w_in=w_output)

        py_hidden.forward(py_batch); py_loss.forward(py_hidden.post_batch)
        cy_hidden.forward(cy_batch); cy_loss.forward(cy_hidden.post_batch)

        _assert_close(np.asarray(cy_loss.first_spike_times), np.asarray(py_loss.first_spike_times), "YinYang first_spike_times")

    def test_forward_predictions(self, test_data, weights):
        samples, labels = test_data
        samples = samples[:64]
        w_hidden, w_output = weights

        from eventprop.lif_layer import LIFLayer as PyLIFLayer
        from eventprop.loss_layer import TTFSCrossEntropyLossParameters as PyTTFSParams
        from eventprop_c.lif_layer import LIFLayer as CyLIFLayer
        from eventprop_c.loss_layer import TTFSCrossEntropyLossParameters as CyTTFSParams

        py_batch = PySpikesVector(_make_spikes_py(samples))
        cy_batch = CySpikesVector(_make_spikes_cy(samples))

        py_hidden = PyLIFLayer(PyLIFParams(n=200, n_in=5, **self.LIF), w_in=w_hidden)
        cy_hidden = CyLIFLayer(CyLIFParams(n=200, n_in=5, **self.LIF), w_in=w_hidden)

        py_loss = PyTTFS(PyTTFSParams(lif_parameters=PyLIFParams(n=3, n_in=200, **self.LIF), **self.TTFS), w_in=w_output)
        cy_loss = CyTTFS(CyTTFSParams(lif_parameters=CyLIFParams(n=3, n_in=200, **self.LIF), **self.TTFS), w_in=w_output)

        py_hidden.forward(py_batch); py_loss.forward(py_hidden.post_batch)
        cy_hidden.forward(cy_batch); cy_loss.forward(cy_hidden.post_batch)

        # np.nanargmin raises "All-NaN slice encountered" when no output neuron
        # fires for a sample — which the reference itself produces (1 of 64 samples
        # in this fixture). Predictions are only defined where a spike exists, so
        # compare rows that both backends fired and require the same all-NaN mask.
        def _preds(fst):
            masked = np.where(np.isnan(fst), np.inf, fst)
            out = np.argmin(masked, axis=1)
            out[np.all(np.isnan(fst), axis=1)] = -1
            return out

        py_preds = _preds(py_loss.first_spike_times)
        cy_preds = _preds(cy_loss.first_spike_times)
        assert np.array_equal(py_preds, cy_preds), "YinYang predictions mismatch"

    def test_backward_gradients(self, test_data, weights):
        """Both backends produce identical gradients on the same forward pass."""
        all_samples, all_labels = test_data
        samples, labels = all_samples[:32], all_labels[:32]
        w_hidden, w_output = weights

        from eventprop.lif_layer import LIFLayer as PyLIFLayer
        from eventprop.loss_layer import TTFSCrossEntropyLossParameters as PyTTFSParams
        from eventprop_c.lif_layer import LIFLayer as CyLIFLayer
        from eventprop_c.loss_layer import TTFSCrossEntropyLossParameters as CyTTFSParams

        py_batch = PySpikesVector(_make_spikes_py(samples))
        cy_batch = CySpikesVector(_make_spikes_cy(samples))

        py_hidden = PyLIFLayer(PyLIFParams(n=200, n_in=5, **self.LIF), w_in=w_hidden)
        cy_hidden = CyLIFLayer(CyLIFParams(n=200, n_in=5, **self.LIF), w_in=w_hidden)

        py_loss = PyTTFS(PyTTFSParams(lif_parameters=PyLIFParams(n=3, n_in=200, **self.LIF), **self.TTFS), w_in=w_output)
        cy_loss = CyTTFS(CyTTFSParams(lif_parameters=CyLIFParams(n=3, n_in=200, **self.LIF), **self.TTFS), w_in=w_output)

        py_hidden.forward(py_batch); py_loss.forward(py_hidden.post_batch)
        cy_hidden.forward(cy_batch); cy_loss.forward(cy_hidden.post_batch)

        # Backward pass
        py_loss.backward(labels); py_hidden.backward()
        cy_loss.backward(labels); cy_hidden.backward()

        _assert_close(cy_hidden.gradient, py_hidden.gradient, "YinYang hidden gradients")
        _assert_close(cy_loss.gradient, py_loss.gradient, "YinYang output gradients")

    def test_full_test_set_accuracy(self, test_data, weights):
        """Verify accuracy on full test set matches paper's reported ~98%."""
        samples, labels = test_data  # all 1000 test samples
        w_hidden, w_output = weights

        from eventprop_c.lif_layer import LIFLayer as CyLIFLayer
        from eventprop_c.loss_layer import TTFSCrossEntropyLossParameters as CyTTFSParams

        cy_batch = CySpikesVector(_make_spikes_cy(samples))
        cy_hidden = CyLIFLayer(CyLIFParams(n=200, n_in=5, **self.LIF), w_in=w_hidden)
        cy_loss = CyTTFS(
            CyTTFSParams(lif_parameters=CyLIFParams(n=3, n_in=200, **self.LIF), **self.TTFS),
            w_in=w_output,
        )
        cy_hidden.forward(cy_batch)
        cy_loss.forward(cy_hidden.post_batch)

        accuracy = cy_loss.get_accuracy(labels)
        print(f"YinYang test accuracy: {accuracy*100:.2f}%")
        assert accuracy > 0.95, f"Accuracy {accuracy:.4f} too low (expected ~0.98)"


# ====================================================================
# MNIST VMax Parity Tests (figure3.py checkpoint)
# ====================================================================

class TestMNISTVMax:
    """Load mnist_vmax_0.pkl weights, run forward/backward on test data with both backends."""

    LIF = {"tau_mem": 20e-3, "tau_syn": 5e-3}

    @pytest.fixture(scope="class")
    def checkpoint(self):
        with open(PAPER_DIR / "mnist_vmax_0.pkl", "rb") as f:
            return pickle.load(f)

    @pytest.fixture(scope="class")
    def weights(self, checkpoint):
        return checkpoint[6][0]  # final trained weights: [hidden (784×350), output (350×10)]

    def test_forward_vmax_values(self, weights):
        """Both backends produce identical VMax values on a sample input."""
        w_hidden, w_output = weights

        # Create synthetic spikes mimicking MNIST encoding (784 pixels → 350 hidden)
        np.random.seed(42)
        n_samples = 16
        py_spikes_list, cy_spikes_list = [], []

        from eventprop_c.spikes import Spikes as CySpikes

        for _ in range(n_samples):
            times = np.sort(np.random.uniform(0, 1.0, size=200))
            sources = np.random.choice(784, size=200, replace=False).astype(np.int32)
            py_spikes_list.append(PySpikes(times, sources))
            cy_spikes_list.append(CySpikes(times, sources))

        from eventprop.lif_layer import LIFLayer as PyLIFLayer
        from eventprop_c.lif_layer import LIFLayer as CyLIFLayer

        py_batch = PySpikesVector(py_spikes_list)
        cy_batch = CySpikesVector(cy_spikes_list)

        py_hidden = PyLIFLayer(PyLIFParams(n=350, n_in=784, **self.LIF), w_in=w_hidden)
        cy_hidden = CyLIFLayer(CyLIFParams(n=350, n_in=784, **self.LIF), w_in=w_hidden)

        py_loss = PyVMax(PyLIParams(n=10, n_in=350, **self.LIF), w_in=w_output)
        cy_loss = CyVMax(CyLIParams(n=10, n_in=350, **self.LIF), w_in=w_output)

        py_hidden.forward(py_batch); py_loss.forward(py_hidden.post_batch)
        cy_hidden.forward(cy_batch); cy_loss.forward(cy_hidden.post_batch)

        # Compare VMax values (batch × 10 neurons).
        # The pybind11 loss exposes maxima per sample as maxima_batch[i].values;
        # the Cython loss exposes the same numbers as values_batch[i].
        for i in range(n_samples):
            _assert_close(
                np.asarray(cy_loss.values_batch[i]),
                np.asarray(py_loss.maxima_batch[i].values),
                f"MNIST VMax sample {i}",
            )

    def test_forward_predictions(self, weights):
        """Both backends produce identical predictions."""
        w_hidden, w_output = weights
        np.random.seed(42)
        n_samples = 16

        from eventprop_c.spikes import Spikes as CySpikes
        py_spikes_list, cy_spikes_list = [], []
        for _ in range(n_samples):
            times = np.sort(np.random.uniform(0, 1.0, size=200))
            sources = np.random.choice(784, size=200, replace=False).astype(np.int32)
            py_spikes_list.append(PySpikes(times, sources))
            cy_spikes_list.append(CySpikes(times, sources))

        from eventprop.lif_layer import LIFLayer as PyLIFLayer
        from eventprop_c.lif_layer import LIFLayer as CyLIFLayer

        py_batch = PySpikesVector(py_spikes_list)
        cy_batch = CySpikesVector(cy_spikes_list)

        py_hidden = PyLIFLayer(PyLIFParams(n=350, n_in=784, **self.LIF), w_in=w_hidden)
        cy_hidden = CyLIFLayer(CyLIFParams(n=350, n_in=784, **self.LIF), w_in=w_hidden)

        py_loss = PyVMax(PyLIParams(n=10, n_in=350, **self.LIF), w_in=w_output)
        cy_loss = CyVMax(CyLIParams(n=10, n_in=350, **self.LIF), w_in=w_output)

        py_hidden.forward(py_batch); py_loss.forward(py_hidden.post_batch)
        cy_hidden.forward(cy_batch); cy_loss.forward(cy_hidden.post_batch)

        assert np.array_equal(
            py_loss.get_predictions(),
            cy_loss.get_predictions(),
        ), "MNIST predictions mismatch"

    def test_backward_gradients(self, weights):
        """Both backends produce identical gradients."""
        w_hidden, w_output = weights
        np.random.seed(42)
        n_samples = 16
        labels = np.random.randint(0, 10, size=n_samples)

        from eventprop_c.spikes import Spikes as CySpikes
        py_spikes_list, cy_spikes_list = [], []
        for _ in range(n_samples):
            times = np.sort(np.random.uniform(0, 1.0, size=200))
            sources = np.random.choice(784, size=200, replace=False).astype(np.int32)
            py_spikes_list.append(PySpikes(times, sources))
            cy_spikes_list.append(CySpikes(times, sources))

        from eventprop.lif_layer import LIFLayer as PyLIFLayer
        from eventprop_c.lif_layer import LIFLayer as CyLIFLayer

        py_batch = PySpikesVector(py_spikes_list)
        cy_batch = CySpikesVector(cy_spikes_list)

        py_hidden = PyLIFLayer(PyLIFParams(n=350, n_in=784, **self.LIF), w_in=w_hidden)
        cy_hidden = CyLIFLayer(CyLIFParams(n=350, n_in=784, **self.LIF), w_in=w_hidden)

        py_loss = PyVMax(PyLIParams(n=10, n_in=350, **self.LIF), w_in=w_output)
        cy_loss = CyVMax(CyLIParams(n=10, n_in=350, **self.LIF), w_in=w_output)

        py_hidden.forward(py_batch); py_loss.forward(py_hidden.post_batch)
        cy_hidden.forward(cy_batch); cy_loss.forward(cy_hidden.post_batch)

        # Backward pass
        py_loss.backward(labels); py_hidden.backward()
        cy_loss.backward(labels); cy_hidden.backward()

        _assert_close(cy_hidden.gradient, py_hidden.gradient, "MNIST hidden gradients")
        _assert_close(cy_loss.gradient, py_loss.gradient, "MNIST output gradients")