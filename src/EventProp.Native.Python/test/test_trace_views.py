"""Traces are views over the library's buffers, not copies of them.

`compute_voltage_trace` and `compute_lambda_i_trace` used to memcpy both arrays out of C memory
and free the buffer in the same call. The copy is `2 · n_t · 8` bytes where `n_t` is the
caller's own `dt` choice — `dt=2.5e-4` over 0.3 s is 1 200 points (~19 KB), `dt=1e-5` is 30 000
points (~470 KB) — so a diagnostic path had a per-call cost bounded only by a slider. The
buffers are now owned by a `_VoltageTraceOutput` / `_LambdaITraceOutput` and handed out as
views; the copy is gone and the memory is released when the last view dies.

Two things are easy to get wrong here, so they are asserted rather than assumed:

* A view with no name holding its owner must still be readable — that is the whole lifetime
  argument for the owner object (and, before `set_array_base` existed in this binding, it was
  a way to read freed heap; see `test_view_ownership.py`).
* Agreement checks must be evaluated where the value is NOT zero. A trace is mostly zeros: an
  assertion comparing a grid sample against a pointwise query at a silent time passes with both
  sides broken, so the non-vacuity check comes first.
"""

import gc
import resource
import sys

import numpy as np
import pytest

from eventprop_c import _native as native

TAU_MEM, TAU_SYN, V_TH = 0.02, 0.005, 1.0
N_PRE, N_POST = 3, 2
T_MAX, DT = 0.3, 2.5e-4

W = np.ascontiguousarray(np.array([[6.0, 5.0], [4.0, 3.5], [2.5, 5.5]], dtype=np.float64))
TIMES = np.array([0.010, 0.018, 0.026, 0.033], dtype=np.float64)
SOURCES = np.array([0, 1, 2, 0], dtype=np.int32)
N_PER_SAMPLE = np.array([4], dtype=np.int32)


def _forward():
    """This sample's own output spikes — a trace built from a second forward pass describes a
    network state the layer does not have."""
    spikes = native.compute_spikes_batch(TIMES, SOURCES, N_PER_SAMPLE, W, N_PRE, N_POST,
                                         TAU_MEM, TAU_SYN, V_TH)
    times = np.ascontiguousarray(np.array(spikes[0][0], dtype=np.float64))
    sources = np.ascontiguousarray(np.array(spikes[0][1], dtype=np.int32))
    assert times.size > 0, "fixture stopped spiking; every trace below would be vacuous"
    return times, sources


FWD_T, FWD_S = _forward()
ZERO_ERRORS = np.zeros(FWD_T.shape[0], dtype=np.float64)
UNIT_ERRORS = np.ones(FWD_T.shape[0], dtype=np.float64)


def _voltage_trace(t_max=T_MAX, dt=DT):
    return native.compute_voltage_trace(TIMES, SOURCES, W, N_PRE, N_POST,
                                        TAU_MEM, TAU_SYN, V_TH,
                                        FWD_T, FWD_S, t_max, dt, 0)


def _lambda_trace(errors=UNIT_ERRORS, t_max=T_MAX, dt=DT):
    return native.compute_lambda_i_trace(TIMES, SOURCES, W, N_PRE, N_POST,
                                         TAU_MEM, TAU_SYN, V_TH,
                                         FWD_T, FWD_S, errors, t_max, dt, 0)


def _churn(blocks=2000, size=4096):
    return [bytearray(b"\xa5" * size) for _ in range(blocks)]


class TestTracesAreViews:
    def test_voltage_trace_arrays_are_views_not_copies(self):
        ts, v = _voltage_trace()
        assert ts.base is not None and v.base is not None, "a copy came back"
        assert ts.base is v.base, "the two arrays do not share one owner"
        assert type(ts) is np.ndarray and type(v) is np.ndarray
        assert ts.dtype == np.float64 and v.dtype == np.float64
        assert v.flags["C_CONTIGUOUS"]

    def test_lambda_trace_arrays_are_views_not_copies(self):
        ts, lam = _lambda_trace()
        assert ts.base is not None and lam.base is not None, "a copy came back"
        assert ts.base is lam.base
        assert type(lam) is np.ndarray

    def test_length_is_the_grid_that_was_asked_for(self):
        """The claim behind removing the copy: cost scales with n_t, and n_t is the caller's."""
        for t_max, dt in [(0.3, 2.5e-4), (0.05, 1e-3), (0.3, 1e-5)]:
            ts, v = _voltage_trace(t_max, dt)
            assert ts.shape == v.shape
            assert abs(ts.shape[0] * dt - t_max) <= dt, (
                f"n_t={ts.shape[0]} does not describe {t_max}/{dt}")
        big_ts, big_v = _voltage_trace(0.3, 1e-5)
        small_ts, small_v = _voltage_trace(0.3, 2.5e-4)
        assert big_v.nbytes > 10 * small_v.nbytes

    def test_views_reach_the_same_memory_as_numpy_ops(self):
        _, v = _voltage_trace()
        assert np.may_share_memory(v, np.asarray(v))
        assert np.may_share_memory(v, v[::10])
        assert np.isfinite(v).all()


class TestLifetime:
    def test_one_view_alone_keeps_both_buffers_valid(self):
        """Neither name is special: the owner must survive either view alone."""
        ts, v = _voltage_trace()
        expected_t, expected_v = ts.copy(), v.copy()

        del ts
        _churn()
        gc.collect()
        assert np.array_equal(v, expected_v)

        ts2, v2 = _voltage_trace()
        expected_t2, expected_v2 = ts2.copy(), v2.copy()
        del v2
        _churn()
        gc.collect()
        assert np.array_equal(ts2, expected_t2)

    def test_owner_is_released_with_the_last_view(self):
        ts, v = _voltage_trace()
        owner = ts.base
        # Guard first: with a copy in the path `base` is None, and getrefcount(None) reports a
        # meaningless number rather than failing the way this test actually wants to.
        assert owner is not None, "a copy came back; there is no owner to release"
        both = sys.getrefcount(owner)
        del v
        one = sys.getrefcount(owner)
        assert one == both - 1, "dropping a view did not release its reference"

        del ts          # only the local `owner` name refers to it now
        gc.collect()
        assert sys.getrefcount(owner) < one, (
            "the owner outlived every view: its buffers would never be freed")

    def test_many_trace_calls_do_not_grow_the_heap(self):
        """A coarse but honest leak check: the owner object is what replaced the copy, so if it
        were never collected this would climb without bound."""
        def rss_kb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        _voltage_trace()
        baseline = rss_kb()
        for _ in range(5000):
            _voltage_trace(0.05, 1e-3)
        gc.collect()
        growth = rss_kb() - baseline
        # macOS reports bytes; allow a generous 32 MB of allocator/arena growth either way.
        limit = 32 * 1024 * 1024
        assert growth < limit, f"RSS grew by {growth} bytes over 5000 trace calls"


class TestValuesStillRight:
    def test_zero_errors_give_a_zero_adjoint_and_real_errors_do_not(self):
        """The documented reason the backward pass must run before a trace is interesting."""
        _, quiet = _lambda_trace(errors=ZERO_ERRORS)
        _, loud = _lambda_trace(errors=UNIT_ERRORS)
        assert np.all(quiet == 0.0)
        assert np.abs(loud).max() > 0.0

    def test_pointwise_agrees_with_the_grid_at_a_nonzero_point(self):
        ts, lam = _lambda_trace()
        peak = int(np.argmax(np.abs(lam)))
        assert abs(lam[peak]) > 1e-6, (
            "the trace never departs from zero; this comparison would prove nothing")

        value = native.compute_lambda_i_at_time(
            TIMES, SOURCES, W, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH,
            FWD_T, FWD_S, UNIT_ERRORS, float(ts[peak]), 0)

        assert abs(value - lam[peak]) < 1e-12

    def test_voltage_trace_rises_to_threshold_for_a_neuron_that_fired(self):
        _, v = _voltage_trace()
        assert v.max() >= V_TH * 0.99, (
            f"traced neuron never reached threshold (peak {v.max()})")
        assert np.isfinite(v).all()


class TestErrorPaths:
    def test_unsorted_output_spikes_are_refused(self):
        scrambled = FWD_T[::-1].copy()
        with pytest.raises(RuntimeError):
            native.compute_voltage_trace(TIMES, SOURCES, W, N_PRE, N_POST,
                                         TAU_MEM, TAU_SYN, V_TH,
                                         scrambled, FWD_S, T_MAX, DT, 0)

    def test_out_of_range_forward_source_is_refused(self):
        bad_sources = FWD_S.copy()
        bad_sources[0] = N_POST
        with pytest.raises(RuntimeError):
            native.compute_voltage_trace(TIMES, SOURCES, W, N_PRE, N_POST,
                                         TAU_MEM, TAU_SYN, V_TH,
                                         FWD_T, bad_sources, T_MAX, DT, 0)

    def test_failures_do_not_poison_later_success(self):
        """Every refused call frees its own output; a leak or a double free on that path would
        show up here rather than in the exception."""
        scrambled = FWD_T[::-1].copy()
        for _ in range(200):
            with pytest.raises(RuntimeError):
                native.compute_voltage_trace(TIMES, SOURCES, W, N_PRE, N_POST,
                                             TAU_MEM, TAU_SYN, V_TH,
                                             scrambled, FWD_S, T_MAX, DT, 0)
            _churn(64)
        ts, v = _voltage_trace()
        assert ts.shape[0] > 0 and v.max() >= V_TH * 0.99

    def test_non_finite_pointwise_query_time_is_refused(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(RuntimeError):
                native.compute_lambda_i_at_time(
                    TIMES, SOURCES, W, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH,
                    FWD_T, FWD_S, UNIT_ERRORS, bad, 0)
