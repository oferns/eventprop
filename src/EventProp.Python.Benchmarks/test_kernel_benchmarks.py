"""Core kernel benchmarks — pybind11 vs Cython, NO layer involved.

The layer benchmarks in test_benchmarks.py fuse two costs: the C kernel and the
binding/marshalling. This file isolates the second. Both bindings call the same
C library, so a difference here is binding overhead and nothing else — which is
the axis BENCHMARK_SCHEMA.md calls scenario 2 and which was previously
unmeasurable in Python (all 28 tests in test_benchmarks.py are layer/loss level).

It also emits impl:"C" records for the kernel families, which is what closes the
cross-suite chain: Eigen <-> C <-> Managed <-> pybind11/Cython in one table.

Usage:
    cd src/EventProp.Python.Benchmarks
    PYTHONPATH=../EventProp.Python:../EventProp.Native.Python \
      pytest test_kernel_benchmarks.py --benchmark-only \
      --benchmark-json=results_kernels.json

The two bindings take structurally different arguments on purpose:
  pybind11: compute_spikes_batch_cpp(w, SpikesVector([Spikes(times, sources)]), v_th, tau_mem, tau_syn)
  Cython:   compute_spikes_batch(times, sources, n_per_sample, w, n_pre, n_post, tau_mem, tau_syn, v_th)
That difference IS the thing under measurement — do not "equalize" it.
"""

from __future__ import annotations

import numpy as np
import pytest

from eventprop.eventprop_cpp import (
    Spikes as PySpikes,
    SpikesVector as PySpikesVector,
    compute_spikes_batch_cpp,
    compute_maxima_batch_cpp,
)
from eventprop_c._native import (
    compute_spikes_batch,
    compute_maxima_batch,
)

from conftest import SIZES, TAU_MEM, TAU_SYN, V_TH, make_input_batch, make_weights


# Burst at Medium+ is extremely slow (root-finding), matching the layer suite.
SLOW_BURST = {("Medium", "burst"), ("Large", "burst")}

PATTERNS = ["uniform", "poisson", "burst"]
SIZE_NAMES = list(SIZES)


def _params(size_name: str, pattern: str):
    if (size_name, pattern) in SLOW_BURST:
        pytest.skip(f"{size_name}/{pattern} burst is too slow (root-finding)")
    cfg = SIZES[size_name]
    n_in, n_out, spk = cfg["n_in"], cfg["n_out"], cfg["spikes_per_neuron"]
    times, sources = make_input_batch(n_in, n_out, spk, pattern)
    w = make_weights(n_in, n_out)
    return n_in, n_out, w, times, sources


# ============================================================
# ComputeSpikes — forward pass kernel
# ============================================================

class Test_Kernel_ComputeSpikes:

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_pybind11(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        batch = PySpikesVector([PySpikes(times, sources)])

        def call():
            return compute_spikes_batch_cpp(w, batch, V_TH, TAU_MEM, TAU_SYN)

        result = benchmark(call)
        # A kernel that returns nothing meaningful measures a validation
        # short-circuit, not the algorithm.
        assert len(result) == 1

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_cython(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        n_per_sample = np.array([times.size], dtype=np.int32)

        def call():
            return compute_spikes_batch(
                times, sources, n_per_sample, w, n_in, n_out,
                TAU_MEM, TAU_SYN, V_TH)

        result = benchmark(call)
        assert result.batch_size == 1


# ============================================================
# ComputeMaxima — voltage-maxima kernel (voltage-based loss path)
# ============================================================

class Test_Kernel_ComputeMaxima:

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_pybind11(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        batch = PySpikesVector([PySpikes(times, sources)])

        def call():
            return compute_maxima_batch_cpp(w, batch, TAU_MEM, TAU_SYN)

        result = benchmark(call)
        assert len(result) == 1

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_cython(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        n_per_sample = np.array([times.size], dtype=np.int32)

        def call():
            # v_th is passed as +inf to match LILayer: the maxima path never reads the
            # threshold (the C validator has a matching no-v_th path), but the binding
            # signature requires the argument.
            return compute_maxima_batch(
                times, sources, n_per_sample, w, n_in, n_out,
                TAU_MEM, TAU_SYN, float("inf"))

        result = benchmark(call)
        assert result.batch_size == 1
