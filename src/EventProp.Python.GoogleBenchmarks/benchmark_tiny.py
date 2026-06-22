"""Benchmark compute_spikes_batch at tiny scales (0–10 spikes).

Tests the contention that Cython is much slower than pybind11 at very small input sizes.
Both backends receive identical numpy arrays; input constructed once outside the loop.

Usage:
    cd src/EventProp.Python.GoogleBenchmarks
    python benchmark_tiny.py --benchmark_min_time=0.5s
"""

import numpy as np
import google_benchmark as benchmark


# ── Problem config (fixed, vary only spike count) ───────────────────────

N_PRE     = 10   # small pre-synaptic layer
N_POST    = 4    # small post-synaptic layer
TAU_MEM   = 0.02
TAU_SYN   = 0.005
V_TH      = 1.0


def make_input(n_spikes, seed=42):
    """Generate input data for a given spike count."""
    rng = np.random.default_rng(seed)

    # Weights: uniform [2.5, 10.0] to ensure threshold crossings when spikes exist
    w = rng.uniform(2.5, 10.0, size=N_PRE * N_POST).reshape(N_PRE, N_POST)

    if n_spikes == 0:
        # Empty arrays (dtype must match what each backend expects)
        times = np.array([], dtype=np.float64)
        sources = np.array([], dtype=np.int32)
    else:
        times = np.linspace(0.1, 0.9, n_spikes, dtype=np.float64)
        sources = rng.integers(0, N_PRE, size=n_spikes, dtype=np.int32)

    return times, sources, w


# ── pybind11 benchmarks ────────────────────────────────────────────────

from eventprop.eventprop_cpp import Spikes as PySpikes, SpikesVector as PySpikesVector
from eventprop.eventprop_cpp import compute_spikes_batch_cpp


@benchmark.register
def bm_pybind11_0sp(state):
    times, sources, w = make_input(0)
    batch = PySpikesVector([PySpikes(times, sources)])
    while state:
        compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)


@benchmark.register
def bm_pybind11_1sp(state):
    times, sources, w = make_input(1)
    batch = PySpikesVector([PySpikes(times, sources)])
    while state:
        compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)


@benchmark.register
def bm_pybind11_5sp(state):
    times, sources, w = make_input(5)
    batch = PySpikesVector([PySpikes(times, sources)])
    while state:
        compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)


@benchmark.register
def bm_pybind11_10sp(state):
    times, sources, w = make_input(10)
    batch = PySpikesVector([PySpikes(times, sources)])
    while state:
        compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)


# ── Cython benchmarks ──────────────────────────────────────────────────

from eventprop_c._native import compute_spikes_batch as cy_compute_spikes_batch


@benchmark.register
def bm_cython_0sp(state):
    times, sources, w = make_input(0)
    n_per_sample = np.array([0], dtype=np.int32)
    while state:
        cy_compute_spikes_batch(
            times, sources, n_per_sample, w, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH,
        )


@benchmark.register
def bm_cython_1sp(state):
    times, sources, w = make_input(1)
    n_per_sample = np.array([1], dtype=np.int32)
    while state:
        cy_compute_spikes_batch(
            times, sources, n_per_sample, w, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH,
        )


@benchmark.register
def bm_cython_5sp(state):
    times, sources, w = make_input(5)
    n_per_sample = np.array([5], dtype=np.int32)
    while state:
        cy_compute_spikes_batch(
            times, sources, n_per_sample, w, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH,
        )


@benchmark.register
def bm_cython_10sp(state):
    times, sources, w = make_input(10)
    n_per_sample = np.array([10], dtype=np.int32)
    while state:
        cy_compute_spikes_batch(
            times, sources, n_per_sample, w, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH,
        )


# ── Sanity check ───────────────────────────────────────────────────────

def _sanity_check():
    """Both backends must produce identical output for each spike count."""
    for n in [0, 1, 5, 10]:
        times, sources, w = make_input(n)

        # pybind11
        batch = PySpikesVector([PySpikes(times, sources)])
        py_out = compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)
        py_count = py_out[0].n_spikes if n > 0 else 0

        # Cython
        n_per_sample = np.array([n], dtype=np.int32)
        cy_out = cy_compute_spikes_batch(
            times, sources, n_per_sample, w, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH,
        )
        cy_times, _, _ = cy_out[0]
        cy_count = len(cy_times)

        assert py_count == cy_count, \
            f"n={n}: spike count mismatch pybind11={py_count} vs Cython={cy_count}"

        if py_count > 0:
            py_times = np.array(py_out[0].times).flatten()
            assert np.allclose(cy_times, py_times, rtol=1e-9), \
                f"n={n}: spike times differ"

    print("Sanity check passed for all spike counts (0, 1, 5, 10).")


if __name__ == "__main__":
    _sanity_check()

    print("""
=== Tiny-Scale Benchmark ===
Compares compute_spikes_batch at very small input sizes.
Tests the contention that Cython is much slower than pybind11 near zero spikes.
""")
    benchmark.main()
