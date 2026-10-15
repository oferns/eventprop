"""Shared fixtures and helpers for EventProp Python benchmarks.

Compares pybind11 (EventProp.Python) vs Cython (eventprop).
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

# eventprop_c (Cython) — Cython bindings to C library
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
#
# Contract every generator upholds, because the C library enforces it
# (EVENTPROP_UNSORTED_SPIKE_TIMES = 60) and the Eigen reference silently
# returns garbage without it:
#   * spike times are non-decreasing
#   * every spike lands inside [0, 1)
#   * exactly `spikes_per_neuron` spikes are produced, none left unwritten
# ============================================================

def _sort_times_sources(times: np.ndarray, sources: np.ndarray) -> tuple:
    """Stable-sort spike times, carrying sources along (mirrors SortTimesSources
    in the native benchmarks)."""
    if times.size <= 1:
        return times, sources
    sort_idx = np.argsort(times, kind="stable")
    return times[sort_idx], sources[sort_idx]

def generate_uniform_spikes(n_in: int, n_out: int, spikes_per_neuron: int, rng) -> tuple:
    """Uniformly spaced spike times across [0, 1)."""
    n_spikes = spikes_per_neuron
    times = np.empty(n_spikes, dtype=np.float64)
    sources = np.empty(n_spikes, dtype=np.int32)
    for i in range(n_spikes):
        times[i] = (i + 1) / (n_spikes + 1)
        sources[i] = rng.integers(0, n_in)
    return _sort_times_sources(times, sources)


def generate_poisson_spikes(n_in: int, n_out: int, spikes_per_neuron: int, rng) -> tuple:
    """"Poisson" pattern — the shared cross-suite key is `poisson`, but this is NOT a
    homogeneous Poisson process over [0, 1). Each spike time is an independent
    exponential order statistic -log(1-u)/rate, so the train concentrates in the first
    few tens of ms: effective rate ~1.8 kHz (Tiny, 20 spikes) to ~25 kHz (Large, 1000),
    not 200 Hz. Matching native bench_main.cpp and .NET SpikePatterns.cs.

    Each sample is drawn independently as -log(1-u)/rate and then pushed forward
    to keep the series strictly increasing. The previous version summed
    exponential inter-arrival intervals and clipped to [0, 0.999]: a genuine
    Poisson process, but a different spike density from the other two suites,
    and the clip compressed the tail. Cross-suite numbers need the same
    distribution, so the inverse-CDF form is the one to keep. See AGENTS.md.
    """
    n_spikes = spikes_per_neuron
    rate = 200.0
    dt = 1.0 / (rate * n_spikes) if n_spikes > 0 else 0.0

    times = np.empty(n_spikes, dtype=np.float64)
    sources = np.empty(n_spikes, dtype=np.int32)

    for i in range(n_spikes):
        u = rng.random()
        times[i] = -np.log(1.0 - u) / rate
        sources[i] = rng.integers(0, n_in)
        if i > 0 and times[i] <= times[i - 1]:
            times[i] = times[i - 1] + dt

    # Already non-decreasing by construction; sort defensively so the shape is
    # identical to the other generators.
    return _sort_times_sources(times, sources)


def generate_burst_spikes(n_in: int, n_out: int, spikes_per_neuron: int, rng) -> tuple:
    """Canonical burst pattern — matches native bench_main.cpp and .NET BenchPatterns.cs.

    Each bursting neuron fires 5 bursts of 5 spikes; spikes within a burst are
    2 ms apart and consecutive bursts are 200 ms apart (> 5 * tau_mem, so the
    membrane decays between bursts). Neurons are staggered by a computed offset
    so exactly spikes_per_neuron spikes land inside [0, 1) at every size.

    The stagger is computed, not hardcoded: the previous fixed 0.1 s per-neuron
    offset saturated at ~10 bursting neurons, leaving most of the array at the
    uniform filler times and producing descending pairs.
    """
    n_spikes = spikes_per_neuron
    times = np.empty(n_spikes, dtype=np.float64)
    sources = np.empty(n_spikes, dtype=np.int32)

    SPIKES_PER_BURST = 5
    BURSTS_PER_NEURON = 5
    BURST_GAP = 0.2          # > 5 * tau_mem
    SPIKE_STEP = 0.002
    START = 0.05
    WINDOW_END = 1.0
    EDGE_EPS = 1e-6          # keep the last spike strictly inside [0, 1)

    if n_spikes <= 0:
        return times, sources

    per_neuron = SPIKES_PER_BURST * BURSTS_PER_NEURON            # 25
    n_bursting = (n_spikes + per_neuron - 1) // per_neuron       # ceil
    burst_span = (BURSTS_PER_NEURON - 1) * BURST_GAP             # 0.8
    last_in_burst = (SPIKES_PER_BURST - 1) * SPIKE_STEP          # 0.008
    room = WINDOW_END - EDGE_EPS - START - burst_span - last_in_burst
    stagger = room / (n_bursting - 1) if n_bursting > 1 else 0.0

    spike_idx = 0
    for j in range(n_bursting):
        if spike_idx >= n_spikes:
            break
        base = START + j * stagger
        src = j % n_in
        for b in range(BURSTS_PER_NEURON):
            if spike_idx >= n_spikes:
                break
            for s in range(SPIKES_PER_BURST):
                if spike_idx >= n_spikes:
                    break
                times[spike_idx] = base + b * BURST_GAP + s * SPIKE_STEP
                sources[spike_idx] = src
                spike_idx += 1

    # Filled neuron by neuron, so the bursts interleave. The C library requires
    # non-decreasing times (EVENTPROP_UNSORTED_SPIKE_TIMES = 60) and the Cython
    # layer reports the error without raising, so an unsorted array silently
    # measures a validation short-circuit instead of the algorithm.
    return _sort_times_sources(times, sources)


PATTERN_GENERATORS = {
    "uniform": generate_uniform_spikes,
    "poisson": generate_poisson_spikes,
    "burst":   generate_burst_spikes,
}


# ============================================================
# Fixture helpers — create input data for a given size/pattern.
# Returns (times, sources) numpy arrays ready to wrap in Spikes objects.
# ============================================================

def validate_spike_data(times: np.ndarray, sources: np.ndarray, n_in: int,
                        label: str = "") -> None:
    """Assert the spike-data contract, at the point of generation.

    The C library enforces non-decreasing times and rejects inputs whose
    max_time / min_tau would overflow exp (EVENTPROP_UNSORTED_SPIKE_TIMES = 60,
    EVENTPROP_TIMES_TAU_MISMATCH). The Cython layer reports error_no without
    raising, so a rejected input silently measures a validation short-circuit
    while the pybind11 side does real work — which is exactly how the burst
    rows read as ~10x faster than they are.
    """
    n = times.size
    if n == 0:
        return

    diffs = np.diff(times)
    bad = np.nonzero(diffs < 0)[0]
    if bad.size:
        raise AssertionError(
            f"{label}: spike times not non-decreasing — {bad.size} descending pair(s), "
            f"first at index {int(bad[0])}: t[{int(bad[0])}]={times[int(bad[0])]:.6f} > "
            f"t[{int(bad[0]) + 1}]={times[int(bad[0]) + 1]:.6f}"
        )

    if times[0] < 0.0:
        raise AssertionError(f"{label}: negative spike time {times[0]:.6f}")

    if times[-1] >= 1.0:
        ratio = times[-1] / TAU_SYN
        raise AssertionError(
            f"{label}: max spike time {times[-1]:.6f} is outside [0, 1); "
            f"max_time / tau_syn = {ratio:.1f} vs the C library's overflow guard at "
            f"{700.0:.0f} — the native side rejects this input"
        )

    unwritten = int(np.count_nonzero(times == 0.0))
    if unwritten:
        raise AssertionError(
            f"{label}: {unwritten} spike time(s) are exactly 0.0 — the generator left "
            f"slots unwritten, which is a simultaneous-spike storm, not the intended pattern"
        )

    if sources.size and (int(sources.min()) < 0 or int(sources.max()) >= n_in):
        raise AssertionError(
            f"{label}: source index out of [0, {n_in}) — "
            f"min={int(sources.min())} max={int(sources.max())}"
        )


def make_input_batch(n_in: int, n_out: int, spikes_per_neuron: int,
                     pattern: str) -> tuple:
    """Create spike times and sources for the given parameters, then assert the
    data contract so a bad generator fails here rather than showing up as a
    suspiciously fast benchmark."""
    rng = np.random.default_rng(42)
    times, sources = PATTERN_GENERATORS[pattern](n_in, n_out, spikes_per_neuron, rng)
    validate_spike_data(times, sources, n_in, label=f"{pattern}")
    return times, sources


def make_weights(n_in: int, n_out: int) -> np.ndarray:
    """Create weight matrix with same distribution as .NET benchmarks.

    U[0.5, 2.0): matches `w_dist(0.5, 2.0)` in bench_main.cpp and
    `0.5 + rng.NextDouble() * 1.5` in SpikePatterns.cs. The previous form,
    `0.5 + rng.random(...)`, was U[0.5, 1.5) — half the intended range, which
    cut Medium output spikes from 10308 to 7768 (-24.6%) and made Python-vs-native
    ratios compare different workloads.
    """
    rng = np.random.default_rng(42)
    return 0.5 + 1.5 * rng.random((n_in, n_out))
