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
    backward_spikes_batch_cpp,
    backward_maxima_batch_cpp,
)
from eventprop_c._native import (
    compute_spikes_batch,
    compute_maxima_batch,
    backward_spikes_batch_accumulate,
    backward_maxima_batch_accumulate,
)
from eventprop_c.spikes import SpikesVector as CySpikesVector

from conftest import SIZES, TAU_MEM, TAU_SYN, V_TH, make_input_batch, make_weights


# Burst at Medium+ is extremely slow (root-finding), matching the layer suite.
# XL/uniform and XL/poisson are NOT skipped: measured on the native suite, an XL
# uniform BackwardSpikes iteration is ~8 s and ComputeSpikes ~4 s, and
# pytest-benchmark budgets by TIME (fixture.py: rounds = ceil(max_time/duration)),
# so an 8 s call gets exactly 1 round. The suite self-limits; it does not run 6
# iterations of an 8 s workload the way BenchmarkDotNet's fixed IterationCount=3
# + WarmupCount=3 does. That budget-vs-fixed-count difference is why the .NET
# suite costs ~22 min and this one under 2.
SLOW_BURST = {("Medium", "burst"), ("Large", "burst"), ("XL", "burst")}

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


def _post_spike_count_pybind(post_batch) -> int:
    """Total output spikes across a pybind11 SpikesVector.

    Used to distinguish "gradient is zero because the kernel is broken" from "gradient
    is zero because the forward pass produced no spikes" - Tiny/Small uniform is the
    latter (documented zero-spike regime), and a blanket non-zero assertion fails on a
    correct kernel there.
    """
    return sum(post_batch[s].n_spikes for s in range(len(post_batch)))


def _post_spike_count_cython(fwd) -> int:
    """Total output spikes across a Cython _SpikesBatchOutput.

    The Cython output exposes no per-sample spike count; new_errors(i) allocates one
    entry per post spike, so its length is the count. Allocating it here is harmless -
    the benchmarked call allocates its own.
    """
    return sum(fwd.new_errors(i).size for i in range(fwd.batch_size))


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


# ============================================================
# BackwardSpikes — adjoint backward pass kernel (spike-loss path)
# ============================================================
# The backward pass is the expensive half of a training step and had NO core-kernel
# benchmark in Python at all: the cross-suite report's BackwardSpikes rows were
# native-vs-.NET only, with no Python peer. Both bindings call the same C library,
# so the pybind11/Cython difference here is binding cost and nothing else.
#
# The two bindings differ structurally on purpose (see module docstring):
#   pybind11 accumulates into a caller-owned `gradient` array it is handed;
#   Cython returns an _AccumulatedBackwardResult that owns the summed gradient.

class Test_Kernel_BackwardSpikes:

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_pybind11(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        in_batch = PySpikesVector([PySpikes(times, sources)])
        # Post-synaptic spikes are the forward output; the backward sweep walks them.
        post_batch = compute_spikes_batch_cpp(w, in_batch, V_TH, TAU_MEM, TAU_SYN)
        # The post spikes must carry the loss derivative. The C forward pass does not
        # produce it, so the caller sets it - exactly as the layer benchmarks do
        # (test_benchmarks.py: post[0].set_error(i, 1.0)). Without it the backward
        # has no loss signal and the gradient is legitimately all zero, which is what
        # the assertion below caught.
        for s in range(len(post_batch)):
            for i in range(post_batch[s].n_spikes):
                post_batch[s].set_error(i, 1.0)
        gradient = np.zeros((n_in, n_out), dtype=np.float64)

        def call():
            # backward accumulates into `gradient` (never zeroes it), matching the
            # Python reference. Re-zeroing inside the timed call would measure the
            # memset, so the accumulation is left in place; the assertion below
            # checks the kernel did real work.
            backward_spikes_batch_cpp(in_batch, post_batch, w, gradient,
                                      V_TH, TAU_MEM, TAU_SYN)

        benchmark(call)
        # A non-zero gradient is only expected where the forward pass actually fires.
        # Tiny/Small uniform produce ZERO output spikes (weight span U[0.5,2.0) is far
        # below v_th=1.0), so the backward has no spikes to sweep and the gradient is
        # legitimately all zero - the same regime merge.py marks zero_output_spikes.
        # Asserting non-zero there would fail on a correct kernel.
        assert np.any(gradient) or _post_spike_count_pybind(post_batch) == 0

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_cython(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        n_per_sample = np.array([times.size], dtype=np.int32)
        fwd = compute_spikes_batch(
            times, sources, n_per_sample, w, n_in, n_out, TAU_MEM, TAU_SYN, V_TH)
        # post_spike_errors is caller-owned: the C forward pass does not carry it.
        # new_errors(i) allocates a writable array sized to sample i's POST spike
        # count - the validator rejects a length mismatch ("must have one entry per
        # spike"), so it must not be derived from the input spike count.
        errors = [fwd.new_errors(i) for i in range(fwd.batch_size)]
        for e in errors:
            e[:] = 1.0

        def call():
            return backward_spikes_batch_accumulate(fwd, errors)

        result = benchmark(call)
        assert result.grad.size == n_in * n_out
        # See the pybind11 side: an all-zero gradient is correct when the forward
        # produced no spikes (Tiny/Small uniform).
        assert np.any(result.grad) or _post_spike_count_cython(fwd) == 0
        # The result owns native buffers the caller may still be reading; keep it
        # alive for the lifetime of the benchmark, per the zero-copy lifetime rule.
        assert result.batch_size == 1


# ============================================================
# BackwardMaxima — adjoint backward pass kernel (voltage-loss path)
# ============================================================

class Test_Kernel_BackwardMaxima:

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_pybind11(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        in_batch = PySpikesVector([PySpikes(times, sources)])
        maxima = compute_maxima_batch_cpp(w, in_batch, TAU_MEM, TAU_SYN)
        # The maxima batch carries the loss derivative; it starts at zero and
        # backward_maxima reads it, so an unset batch yields a legitimately all-zero
        # gradient. The pybind11 `errors` property is READ-ONLY - set it through
        # set_error(), as the voltage loss layer does.
        for s in range(len(maxima)):
            for i in range(len(maxima[s].values)):
                maxima[s].set_error(i, 1.0)
        gradient = np.zeros((n_in, n_out), dtype=np.float64)

        def call():
            backward_maxima_batch_cpp(in_batch, maxima, w, gradient,
                                      TAU_MEM, TAU_SYN)

        benchmark(call)
        assert np.any(gradient), "backward_maxima produced an all-zero gradient"

    @pytest.mark.parametrize("pattern", PATTERNS)
    @pytest.mark.parametrize("size_name", SIZE_NAMES)
    def test_cython(self, benchmark, size_name, pattern):
        n_in, n_out, w, times, sources = _params(size_name, pattern)
        n_per_sample = np.array([times.size], dtype=np.int32)
        maxima = compute_maxima_batch(
            times, sources, n_per_sample, w, n_in, n_out,
            TAU_MEM, TAU_SYN, float("inf"))
        # Same as the pybind11 side: get_maxima_errors(i) is a WRITABLE zero-copy view
        # into the library-allocated errors array, and it starts at zero. Writing 1.0
        # through it is what gives the backward pass a loss signal; leaving it zero
        # makes the gradient legitimately all zero.
        for i in range(maxima.batch_size):
            maxima.get_maxima_errors(i)[:] = 1.0

        def call():
            return backward_maxima_batch_accumulate(maxima)

        result = benchmark(call)
        assert result.grad.size == n_in * n_out
        assert np.any(result.grad), "backward_maxima produced an all-zero gradient"
