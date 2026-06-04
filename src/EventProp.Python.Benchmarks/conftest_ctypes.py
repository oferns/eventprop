"""Shared fixtures and helpers for EventProp Python benchmarks (ctypes version).

Compares pybind11 (EventProp.Python) vs ctypes (eventprop).
Problem sizes and spike patterns match the .NET/C benchmark suite for alignment.

Usage:
    PYTHONPATH="EventProp.Python:EventProp.Native.Python" pytest test_benchmarks_ctypes.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add both library source directories to path so they can coexist.
_here = Path(__file__).resolve().parent
_src = _here.parent

# EventProp.Python (pybind11) — the "truth" library
sys.path.insert(0, str(_src / "EventProp.Python"))

# eventprop (ctypes) — new ctypes implementation
sys.path.insert(0, str(_src / "EventProp.Native.Python"))

import numpy as np
import pytest


# ============================================================
# Problem sizes — match .NET/C benchmarks exactly.
# ============================================================

SIZES = {
    "Tiny":   {"n_in": 10,   "n_out": 5,   "spikes_per_neuron": 20},
    "Small":  {"n_in": 50,   "n_out": 20,  "spikes_per_neuron": 100},
    "Medium": {"n_in": 200,  "n_out": 80,  "spikes_per_neuron": 500},
    "Large":  {"n_in": 500,  "n_out": 200, "spikes_per_neuron": 1000},
}

# Biophysical parameters (same across all tests)
TAU_MEM = 0.02   # 20 ms
TAU_SYN = 0.005  # 5 ms
V_TH = 1.0


# ============================================================
# Spike pattern generation — matches .NET/C benchmarks exactly.
# Seed 42, rate-based Poisson @200Hz, burst clusters.
# ============================================================

def generate_uniform_spikes(n_in: int, n_out: int, spikes_per_neuron: int, rng) -> tuple:
    """Uniformly spaced spike times across [0, 1)."""
    n_spikes = spikes_per_neuron
    times = np.empty(n_spikes, dtype=np.float64)
    sources = np.empty(n_spikes, dtype=np.int32)
    for i in range(n_spikes):
        times[i] = (i + 1) / (n_spikes + 1)
        sources[i] = rng.integers(0, n_in)
    return times, sources


def generate_poisson_spikes(n_in: int, n_out: int, spikes_per_neuron: int, rng) -> tuple:
    """Rate-based Poisson @200Hz with strictly increasing times."""
    n_spikes = spikes_per_neuron
    rate = 200.0
    intervals = rng.exponential(1.0 / rate, size=n_spikes)

    times = np.empty(n_spikes, dtype=np.float64)
    sources = np.empty(n_spikes, dtype=np.int32)
    times[0] = intervals[0]
    for i in range(1, n_spikes):
        times[i] = times[i - 1] + intervals[i]

    # Clamp to [0, 1)
    times = np.clip(times, 0.0, 0.999)

    for i in range(n_spikes):
        sources[i] = rng.integers(0, n_in)

    # Sort by time (should already be sorted, but ensure it)
    sort_idx = np.argsort(times)
    return times[sort_idx], sources[sort_idx]


def generate_burst_spikes(n_in: int, n_out: int, spikes_per_neuron: int, rng) -> tuple:
    """Burst clusters of 5 spikes with 10ms gaps between bursts.
    Matches C/C++ bench_main.cpp generation exactly."""
    n_spikes = spikes_per_neuron
    times = np.empty(n_spikes, dtype=np.float64)
    sources = np.empty(n_spikes, dtype=np.int32)

    spike_idx = 0
    for nrn in range(n_in):
        if spike_idx >= n_spikes:
            break
        burst_start = 0.05 + nrn * 0.1
        while burst_start < 1.0 and spike_idx < n_spikes:
            for s in range(5):
                if spike_idx >= n_spikes:
                    break
                times[spike_idx] = burst_start + s * 0.01
                sources[spike_idx] = nrn
                spike_idx += 1
            burst_start += 0.05

    # Sort by time
    sort_idx = np.argsort(times[:spike_idx])
    return times[sort_idx][:spike_idx], sources[sort_idx][:spike_idx]


def make_input_batch(n_in: int, n_out: int, spikes_per_neuron: int, pattern: str) -> tuple:
    """Create input batch with given pattern."""
    rng = np.random.default_rng(42)

    if pattern == "uniform":
        return generate_uniform_spikes(n_in, n_out, spikes_per_neuron, rng)
    elif pattern == "poisson":
        return generate_poisson_spikes(n_in, n_out, spikes_per_neuron, rng)
    elif pattern == "burst":
        return generate_burst_spikes(n_in, n_out, spikes_per_neuron, rng)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def make_weights(n_in: int, n_out: int) -> np.ndarray:
    """Create weight matrix with random values."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1.0, (n_in, n_out)).astype(np.float64)


# ============================================================
# Helper functions for ctypes library access
# ============================================================

def get_ctypes_lib():
    """Get or create a ctypes library instance."""
    import eventprop
    from pathlib import Path
    
    # Try to find the library relative to project root
    _project_root = Path(__file__).resolve().parent.parent.parent  # eventprop.net root
    
    lib_paths = [
        _project_root / "src" / "EventProp.C" / "build" / "libeventprop_c_api.dylib",
        _project_root / "src" / "EventProp.C" / "build" / "libeventprop_c_api.so",
        Path.cwd() / "libeventprop_c_api.dylib",
        Path.cwd() / "libeventprop_c_api.so",
    ]
    
    for path in lib_paths:
        if path.exists():
            return eventprop.EventPropCtypes(str(path))
    
    raise FileNotFoundError(
        "Could not find libeventprop_c_api.dylib/.so. Run ./build.sh c first, or "
        "provide lib_path explicitly to EventPropCtypes."
    )


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
