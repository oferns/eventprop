"""Zero-copy views must keep the C memory they point at alive.

The view helpers build arrays with `PyArray_SimpleNewFromData`, which returns an array whose
base object is NULL: the buffer is tied to nothing. The wrapper classes free that buffer in
`__dealloc__`, so a view could outlive its owner and read freed heap. That does not segfault —
it returns plausible numbers, which is worse. Measured before this was fixed, from the
completely natural one-liner `np.array(backward_maxima_batch_accumulate(b).grad)`::

    [-3.6013585e+178, 7.1007115e-320, 0.0, 0.0, 0.0, 0.0]

where the answer is all zeros. Reading freed memory is undefined behaviour, so these tests are
detectors rather than proofs: this one reported garbage on the first iteration before the fix
and reads clean for thousands after it.

The fix is `cnp.set_array_base(arr, owner)` in `_native.pyx` — the same base-object mechanism
`np.frombuffer` uses — so the tests below check three things and only three things:

* the view keeps its owner alive (and the value stays correct after the owner's name is gone),
* the reference is released when the view dies (retention, not a leak),
* the view is still a *view* — writing through it still reaches C. That is the guard against a
  "fix" that quietly copies, which would break every caller that writes errors into a buffer.
"""

import gc
import sys

import numpy as np
import pytest

from eventprop_c import _native as native

TAU_MEM, TAU_SYN, V_TH = 0.02, 0.005, 1.0
N_PRE, N_POST = 3, 2

W = np.ascontiguousarray(np.array([[6.0, 5.0], [4.0, 3.5], [2.5, 5.5]], dtype=np.float64))
TIMES = np.array([0.010, 0.018, 0.026, 0.033, 0.012, 0.021, 0.029, 0.040], dtype=np.float64)
SOURCES = np.array([0, 1, 2, 0, 0, 1, 2, 0], dtype=np.int32)
N_PER_SAMPLE = np.array([4, 4], dtype=np.int32)
DECODER = np.ascontiguousarray(np.array([0.7, -1.3], dtype=np.float64))


def _spikes():
    return native.compute_spikes_batch(TIMES, SOURCES, N_PER_SAMPLE, W, N_PRE, N_POST,
                                       TAU_MEM, TAU_SYN, V_TH)


def _maxima():
    return native.compute_maxima_batch(TIMES, SOURCES, N_PER_SAMPLE, W, N_PRE, N_POST,
                                       TAU_MEM, TAU_SYN, V_TH)


def _dropout():
    return native.dropout_spikes_batch(TIMES, SOURCES, N_PER_SAMPLE, 0.3, 7)


def _accumulated():
    maxima = _maxima()
    maxima.regress_sample(0, DECODER, 0.25, 0.9, float("inf"))
    return native.backward_maxima_batch_accumulate(maxima)


def _churn(rounds=64):
    """Hand the allocator a stream of same-ish sized blocks, so memory released by a freed
    wrapper is quickly overwritten with something else."""
    keep = []
    for _ in range(rounds):
        keep.append(bytearray(b"\xa5" * 512))
    gc.collect()
    return len(keep)


class TestViewKeepsItsOwnerAlive:
    def test_dropped_temporary_result_still_reads_zero(self):
        """The exact reported shape: no name ever holds the result object."""
        maxima = _maxima()  # errors untouched -> the true gradient is exactly zero
        for _ in range(500):
            grad = np.array(native.backward_maxima_batch_accumulate(maxima).grad,
                            dtype=np.float64)
            assert np.all(grad == 0.0), f"read freed memory: {grad}"
            _churn(8)

    def test_dropped_temporary_result_matches_the_held_one(self):
        maxima = _maxima()
        maxima.regress_sample(0, DECODER, 0.25, 0.9, float("inf"))
        held = _accumulated()
        expected = np.array(held.grad, dtype=np.float64).copy()
        assert np.any(expected != 0.0), "fixture should produce a nonzero gradient"

        for _ in range(300):
            grad = np.array(native.backward_maxima_batch_accumulate(maxima).grad)
            assert np.array_equal(grad, expected)
            _churn(8)

    @pytest.mark.parametrize("factory,access", [
        (_spikes, lambda o: o[0][0]),
        (_spikes, lambda o: o[0][1]),
        (_maxima, lambda o: o[0][0]),
        (_maxima, lambda o: o.get_values(0)),
        (_maxima, lambda o: o.get_times(1)),
        (_maxima, lambda o: o.get_maxima_errors(0)),
        (_dropout, lambda o: o[0][0]),
        (_dropout, lambda o: o[0][1]),
        (_accumulated, lambda o: o.grad),
        (_accumulated, lambda o: o.get_grad(0)),
        (_accumulated, lambda o: o.get_spike_errors(0)),
    ])
    def test_view_survives_losing_the_owners_name(self, factory, access):
        owner = factory()
        view = access(owner)
        assert view is not None
        expected = np.array(view, dtype=view.dtype).copy()
        assert expected.size > 0, "fixture produced an empty view; pick a busier sample"

        del owner          # __dealloc__ would free the C buffer here without retention
        gc.collect()
        _churn(256)

        assert np.array_equal(np.array(view, dtype=view.dtype), expected), (
            "the view outlived the object that owns its memory")

    def test_view_base_points_at_the_owner(self):
        """The mechanism itself. A `.copy()` anywhere in a view path makes base None again."""
        spikes = _spikes()
        maxima = _maxima()
        accumulated = _accumulated()
        dropout = _dropout()
        for view, owner in [
            (spikes[0][0], spikes), (spikes[0][1], spikes),
            (maxima[0][0], maxima), (maxima[0][1], maxima),
            (maxima.get_values(0), maxima), (maxima.get_times(0), maxima),
            (maxima.get_maxima_errors(0), maxima),
            (dropout[0][0], dropout), (dropout[0][1], dropout),
            (accumulated.grad, accumulated), (accumulated.get_grad(0), accumulated),
            (accumulated.get_spike_errors(0), accumulated),
        ]:
            assert view.base is owner, f"{type(owner).__name__} view has base {view.base!r}"


class TestRetentionDoesNotLeak:
    def test_reference_is_acquired_and_released(self):
        maxima = _maxima()
        before = sys.getrefcount(maxima)
        view = maxima.get_values(0)
        assert sys.getrefcount(maxima) == before + 1, "the view did not retain its owner"
        del view
        gc.collect()
        assert sys.getrefcount(maxima) == before, "the view leaked a reference to its owner"

    def test_last_view_releasing_frees_the_owner(self):
        """Everything in one expression: the wrapper's only reference ends up held by the view."""
        values = np.array(_maxima().get_values(0), dtype=np.float64)
        gc.collect()
        _churn(128)
        assert values.shape == (N_POST,)
        assert np.all(np.isfinite(values))


class TestStillZeroCopy:
    def test_writing_through_a_view_still_reaches_c(self):
        """The anti-copy guard.

        If retention were implemented by copying — which would satisfy every test above and
        silently break the binding — writes to `errors` would stop reaching the adjoint.
        """
        untouched = _maxima()
        clean = np.array(native.backward_maxima_batch_accumulate(untouched).grad,
                         dtype=np.float64)
        assert np.all(clean == 0.0)

        scored = _maxima()
        scored.get_maxima_errors(0)[0] = 3.0
        grad = np.array(native.backward_maxima_batch_accumulate(scored).grad, dtype=np.float64)
        assert np.any(grad != 0.0), (
            "a write through the errors view no longer reaches C: the view became a copy")

    def test_views_are_plain_ndarrays(self):
        """A subclass-based retention hack would change type() for every consumer."""
        maxima = _maxima()
        for view in (maxima[0][0], maxima.get_values(0), maxima.get_maxima_errors(0),
                     _accumulated().grad, _dropout()[0][1], _spikes()[0][1]):
            assert type(view) is np.ndarray, f"view is a {type(view).__name__}"

    def test_view_flags_are_unchanged(self):
        maxima = _maxima()
        values = maxima.get_values(0)
        assert values.flags["WRITEABLE"] and values.flags["C_CONTIGUOUS"]
        assert values.dtype == np.float64
        sources = _spikes()[0][1]
        assert sources.dtype == np.int32 and sources.flags["WRITEABLE"]

    def test_two_views_of_one_sample_share_the_buffer(self):
        maxima = _maxima()
        a, b = maxima[0][1], maxima.get_values(0)
        assert np.may_share_memory(a, b)
        a[0] = -12.5
        assert b[0] == -12.5

    def test_documented_copy_out_arrays_are_independent(self):
        """`maxima_regression` returns copies on purpose (n_post is small; an owner object
        would cost more than the memcpy it saves). Copies must stay copies, and independent
        ones: two calls must not hand back the same buffer."""
        maxima = _maxima()
        values = np.ascontiguousarray(np.array(maxima.get_values(0), dtype=np.float64))
        first = native.maxima_regression(values, DECODER, 0.25, 0.9, float("inf"))
        second = native.maxima_regression(values, DECODER, 0.25, 0.9, float("inf"))
        for key in ("errors", "decoder_grad"):
            assert first[key].base is None, f"{key} is now a view; this test guards the copy"
            assert not np.may_share_memory(first[key], second[key]), f"{key} buffers are shared"
            assert not np.array_equal(first[key], np.zeros_like(first[key]))
