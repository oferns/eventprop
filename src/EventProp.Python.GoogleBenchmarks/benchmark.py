"""Benchmark matching bench_main.cpp BM_ComputeSpikes — Small/uniform.

Calls the lowest-level API in each library (bypassing LIFLayer wrapper classes).
Measures only the core compute_spikes call per iteration.

Mirrors bench_main.cpp data generator:
  - Same seed (42), same time distribution, same source randomness
  - Same problem size: Small = n_pre=50, n_post=20, n_spikes=100
  - Weights scaled to U[2.5, 10.0] so threshold crossings actually occur
    (raw U[0.5, 2.0] from bench_main.cpp produces 0 output spikes for this config)
  - Input constructed once outside the loop
  - Output freed each iteration

Usage:
    cd src/EventProp.Python.GoogleBenchmarks
    python benchmark.py --benchmark_min_time=1s
"""

import numpy as np
import google_benchmark as benchmark


# ── Data generation — mirrors GenerateUniformData from bench_main.cpp ───

def generate_uniform_data(n_pre, n_post, n_spikes, seed=42):
    """Same logic as bench_main.cpp GenerateUniformData.

    Weights scaled by 5x so that threshold crossings occur with this
    problem size and time distribution. (Raw U[0.5,2.0] → 0 output spikes.)
    """
    rng = np.random.default_rng(seed)

    # Weights: uniform [2.5, 10.0], row-major (n_pre × n_post)
    w = rng.uniform(2.5, 10.0, size=n_pre * n_post).reshape(n_pre, n_post)

    # Times: evenly spaced (i+1)/(n_spikes+1), same as C++ benchmark
    times = np.array([(i + 1) / (n_spikes + 1) for i in range(n_spikes)], dtype=np.float64)

    # Sources: uniform random integers [0, n_pre-1]
    sources = rng.integers(0, n_pre, size=n_spikes, dtype=np.int32)

    return times, sources, w


# ── Problem size (Small from bench_main.cpp kSizes[]) ───────────────────

N_PRE     = 50
N_POST    = 20
N_SPIKES  = 100
TAU_MEM   = 0.02
TAU_SYN   = 0.005
V_TH      = 1.0


# ── pybind11 (EventProp.Python — reference) ─────────────────────────────

from eventprop.eventprop_cpp import Spikes as PySpikes, SpikesVector as PySpikesVector
from eventprop.eventprop_cpp import compute_spikes_batch_cpp


@benchmark.register
def bm_forward_pybind11(state):
    """ComputeSpikes/Small/uniform — pybind11 (reference).

    Mirrors bench_main.cpp BM_ComputeSpikes with USE_CPP_WRAPPER:
      - Eigen::Map + Spikes constructed once outside loop
      - compute_spikes() called each iteration, output freed by RAII
    """
    times, sources, w = generate_uniform_data(N_PRE, N_POST, N_SPIKES)

    # Construct input once (matches C++ pre-constructing Eigen objects)
    batch = PySpikesVector([PySpikes(times, sources)])

    while state:
        output = compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)


# ── Cython (EventProp.Native.Python — C library via ctypes) ─────────────

from eventprop_c._native import compute_spikes_batch as cy_compute_spikes_batch


@benchmark.register
def bm_forward_cython(state):
    """ComputeSpikes/Small/uniform — Cython → C library.

    Mirrors bench_main.cpp BM_ComputeSpikes without USE_CPP_WRAPPER:
      - EventPropInput struct + flat arrays, constructed once outside loop
      - eventprop_compute_spikes() called each iteration, output freed manually
    """
    times, sources, w = generate_uniform_data(N_PRE, N_POST, N_SPIKES)

    # n_spikes_per_sample: single sample → [100]
    n_spikes_per_sample = np.array([N_SPIKES], dtype=np.int32)

    while state:
        output = cy_compute_spikes_batch(
            times, sources, n_spikes_per_sample,
            w, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH
        )


# ── Sanity check ────────────────────────────────────────────────────────

def _sanity_check():
    """Both libraries must produce identical spike counts on same input."""
    times, sources, w = generate_uniform_data(N_PRE, N_POST, N_SPIKES)

    # pybind11
    batch = PySpikesVector([PySpikes(times, sources)])
    py_output = compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)

    # Cython → C
    n_per_sample = np.array([N_SPIKES], dtype=np.int32)
    cy_output = cy_compute_spikes_batch(
        times, sources, n_per_sample, w, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH
    )

    py_count = py_output[0].n_spikes
    cy_times, _, _ = cy_output[0]
    cy_count = len(cy_times)

    assert py_count == cy_count, \
        f"Spike count mismatch: pybind11={py_count} vs Cython={cy_count}"

    # Also check spike times match (rtol=1e-9)
    if py_count > 0:
        py_times = np.array(py_output[0].times).flatten()
        assert np.allclose(cy_times, py_times, rtol=1e-9), \
            "Spike times differ between backends"

    print(f"Sanity check passed: both produced {py_count} output spikes.")


if __name__ == "__main__":
    _sanity_check()

    print("""
=== Benchmark Summary ===
This benchmark calls the LOWEST-LEVEL API in each library:
  - pybind11: compute_spikes_batch_cpp(w, SpikesVector, ...) -> Eigen core
  - Cython:   compute_spikes_batch(times, sources, n_per, w, ...) -> C library

Both receive identical numpy arrays. Input constructed once outside loop.
Output freed each iteration (C via __dealloc__, pybind11 via RAII).
""")
    benchmark.main()
