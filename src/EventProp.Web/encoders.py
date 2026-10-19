"""Spike encoding helpers for both backends."""
import numpy as np
from config import CySpikes, CySpikesVector, PySpikes, PySpikesVector

def encode_cy(sample):
    """Encode a single sample into a Cython Spikes object."""
    t = np.array([v * 30e-3 for v in list(sample) + [0.0]], dtype=np.float64)
    s = np.arange(len(sample) + 1, dtype=np.int32)
    i = np.argsort(t)
    return CySpikes(t[i], s[i])

def encode_py(sample):
    """Encode a single sample into a Pybind11 Spikes object."""
    t = np.array([v * 30e-3 for v in list(sample) + [0.0]], dtype=np.float64).reshape(-1, 1)
    s = np.arange(len(sample) + 1, dtype=np.int32).reshape(-1, 1)
    i = np.argsort(t.ravel())
    return PySpikes(t[i], s[i])

def encode_batch_cy(samples):
    """Encode a batch of samples into a Cython SpikesVector."""
    return CySpikesVector([encode_cy(x) for x in samples])

def encode_batch_py(samples):
    """Encode a batch of samples into a Pybind11 SpikesVector."""
    return PySpikesVector([encode_py(x) for x in samples])
