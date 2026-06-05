"""Shared fixtures and helpers for EventProp Python benchmarks.

Compares pybind11 (EventProp.Python) vs ctypes (eventprop).
Problem sizes and spike patterns match the .NET/C benchmark suite for alignment.

Usage:
    PYTHONPATH="EventProp.Python:EventProp.Native.Python" pytest test_benchmarks.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add both library source directories to path so they can coexist.
_here = Path(__file__).resolve().parent  # EventProp.Python.Benchmarks/
_src = _here.parent                       # src/ (parent of directory)

# EventProp.Python (pybind11) — the "truth" library
sys.path.insert(0, str(_src / "EventProp.Python"))

# eventprop_c (ctypes) — new ctypes implementation
sys.path.insert(0, str(_src / "EventProp.Native.Python"))

# Initialize ctypes library at module load time
import eventprop_c as native_lib
_project_root = _src.parent  # eventprop.net/ (parent of src/)
lib_paths = [
    _project_root / "src" / "EventProp.C" / "build" / "libeventprop_c_api.dylib",
    _project_root / "src" / "EventProp.C" / "build" / "libeventprop_c_api.so",
]
for path in lib_paths:
    if path.exists():
        _ = native_lib.NativeApi(str(path))  # Initialize global library
        break

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
                times[spike_idx] = burst_start + s * 0.002
                sources[spike_idx] = nrn
                spike_idx += 1
            burst_start += 0.12

    # Fill remaining spikes with uniform spacing
    for idx in range(spike_idx, n_spikes):
        times[idx] = (idx + 1) / (n_spikes + 1)
        sources[idx] = rng.integers(0, n_in)

    return times, sources


PATTERN_GENERATORS = {
    "uniform": generate_uniform_spikes,
    "poisson": generate_poisson_spikes,
    "burst":   generate_burst_spikes,
}


# ============================================================
# Fixture helpers — create input data for a given size/pattern.
# Returns (times, sources) numpy arrays ready to wrap in Spikes objects.
# ============================================================

def make_input_batch(n_in: int, n_out: int, spikes_per_neuron: int,
                     pattern: str) -> tuple:
    """Create spike times and sources for the given parameters."""
    rng = np.random.default_rng(42)
    return PATTERN_GENERATORS[pattern](n_in, n_out, spikes_per_neuron, rng)


def make_weights(n_in: int, n_out: int) -> np.ndarray:
    """Create weight matrix with same distribution as .NET benchmarks."""
    rng = np.random.default_rng(42)
    return 0.5 + rng.random((n_in, n_out))
