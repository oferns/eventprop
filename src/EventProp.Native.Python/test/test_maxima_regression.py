"""The per-sample in-place regression readout (`_MaximaBatchOutput.regress_sample`).

`eventprop_maxima_regression_into` is the one entry point in `eventprop_loss.h` that writes
into memory the library already owns rather than returning fresh arrays, so the interesting
questions are not "is the arithmetic right" (the C suite finite-differences that, and the
managed↔native parity tests in `EventProp.Tests` pin the formulas) but:

* does the write land in the buffer the **backward pass reads**? A binding method that
  computed the right numbers and stored them in the wrong place would still return a
  convincing dict.
* does it touch only the named sample? That isolation is the whole reason to call the
  per-sample entry point instead of the batch one.
* does it agree with the two other ways of asking the same question?

The zero-gradient baseline in `test_written_errors_drive_the_backward_pass` is what makes the
first question non-vacuous: `eventprop_compute_maxima_batch` calloc's the error buffers, so an
untouched batch must produce an exactly-zero weight gradient, and only a write that reached
the right memory can make it nonzero.
"""

import math

import numpy as np
import pytest

from eventprop_c import _native as native

TAU_MEM, TAU_SYN, V_TH = 0.02, 0.005, 1.0
N_PRE, N_POST = 3, 2

# Column-major-ish weights: every post-synaptic neuron must actually spike, otherwise the
# maxima are 0.0 and the readout arithmetic below degenerates.
W = np.ascontiguousarray(np.array([[6.0, 5.0], [4.0, 3.5], [2.5, 5.5]], dtype=np.float64))
TIMES = np.array([0.010, 0.018, 0.026, 0.033, 0.012, 0.021, 0.029, 0.040], dtype=np.float64)
SOURCES = np.array([0, 1, 2, 0, 0, 1, 2, 0], dtype=np.int32)
N_PER_SAMPLE = np.array([4, 4], dtype=np.int32)
BATCH = 2

DECODER = np.ascontiguousarray(np.array([0.7, -1.3], dtype=np.float64))
BIAS = 0.25
TARGETS = np.array([0.9, -0.4], dtype=np.float64)


def _batch():
    return native.compute_maxima_batch(
        TIMES, SOURCES, N_PER_SAMPLE, W, N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH)


def _values(batch, idx):
    return np.array(batch[idx][1], dtype=np.float64)


def _errors(batch, idx):
    return np.array(batch.get_maxima_errors(idx), dtype=np.float64)


class TestRegressSample:
    def test_matches_the_batch_and_allocating_paths(self):
        """Three ways to ask for one sample's readout; they must not disagree."""
        batch = _batch()
        batch_dict = batch.regress(DECODER, BIAS, TARGETS, math.inf)

        for i in range(BATCH):
            one = batch.regress_sample(i, DECODER, BIAS, TARGETS[i], math.inf)
            alloc = native.maxima_regression(
                np.ascontiguousarray(_values(batch, i)), DECODER, BIAS, TARGETS[i], math.inf)

            assert one["prediction"] == pytest.approx(batch_dict["predictions"][i], rel=1e-12)
            assert one["loss"] == pytest.approx(batch_dict["losses"][i], rel=1e-12)
            assert one["prediction"] == pytest.approx(alloc["prediction"], rel=1e-12)
            assert one["loss"] == pytest.approx(alloc["loss"], rel=1e-12)
            assert one["residual_derivative"] == pytest.approx(
                alloc["residual_derivative"], rel=1e-12)
            assert one["n_skipped"] == 0

    def test_errors_are_the_chain_rule(self):
        """dL/dmaxima[n] == (y_hat - target) * a[n] for least squares — asserted against
        arithmetic done here, not against another call into the library."""
        batch = _batch()
        for i in range(BATCH):
            values = _values(batch, i)
            expected_y = BIAS + float(np.dot(DECODER, values))
            expected_err = (expected_y - TARGETS[i]) * DECODER

            one = batch.regress_sample(i, DECODER, BIAS, TARGETS[i], math.inf)

            assert one["prediction"] == pytest.approx(expected_y, rel=1e-12)
            assert one["residual_derivative"] == pytest.approx(expected_y - TARGETS[i],
                                                               rel=1e-12, abs=1e-15)
            assert one["loss"] == pytest.approx(0.5 * (expected_y - TARGETS[i]) ** 2, rel=1e-9)
            np.testing.assert_allclose(_errors(batch, i), expected_err, rtol=1e-12, atol=1e-15)

    def test_writes_only_the_named_sample(self):
        """Isolation is the reason this entry point exists. Sample 1's buffer is calloc'd and
        must still be all zeros after sample 0 has been scored."""
        batch = _batch()
        assert np.all(_errors(batch, 0) == 0.0)
        assert np.all(_errors(batch, 1) == 0.0)

        batch.regress_sample(0, DECODER, BIAS, TARGETS[0], math.inf)

        assert np.any(_errors(batch, 0) != 0.0), "nothing was written to the named sample"
        assert np.all(_errors(batch, 1) == 0.0), "the other sample was written to as well"

    def test_written_errors_drive_the_backward_pass(self):
        """The buffer written here is the buffer the adjoint reads.

        An untouched batch must give an exactly-zero gradient (calloc'd errors), so any
        nonzero gradient can only have come through `regress_sample`. Comparing against a
        backward call that reads a different buffer would pass even if this method stored its
        results somewhere nothing looks at.
        """
        untouched = _batch()
        # The result object must stay bound while the view is read: `.grad` is a zero-copy
        # view of a buffer that object owns, and the zero-copy views this binding hands out do
        # not retain their owner. Dropping it into a temporary reads freed memory (measured:
        # -3.6e+178 and a denormal where the zeros should be).
        untouched_result = native.backward_maxima_batch_accumulate(untouched)
        baseline = np.array(untouched_result.grad, dtype=np.float64)
        assert np.all(baseline == 0.0), "an untouched batch should contribute no gradient"

        scored = _batch()
        scored.regress_sample(0, DECODER, BIAS, TARGETS[0], math.inf)
        scored_result = native.backward_maxima_batch_accumulate(scored)
        gradient = np.array(scored_result.grad, dtype=np.float64)

        assert gradient.shape == baseline.shape
        assert np.any(gradient != 0.0), (
            "regress_sample wrote errors the backward pass never consumed")
        assert not np.allclose(gradient, baseline, atol=0.0)

    def test_huber_saturates_the_residual_derivative(self):
        """The smooth Huber derivative approaches ±delta without ever reaching it.

        It is `r / sqrt(1 + (r/delta)²)`, not a clamp: the property that makes the loss robust
        is that |dL/dy_hat| stays under delta while the squared loss's grows without bound.
        Asserting an exact ±delta would be asserting the non-smooth variant, which this is
        not — and the deficit is the O(delta³/r²) tail, not a bug.
        """
        batch = _batch()
        far_target = TARGETS[0] + 50.0
        delta = 0.5

        one = batch.regress_sample(0, DECODER, BIAS, far_target, delta)
        squared = batch.regress_sample(0, DECODER, BIAS, far_target, math.inf)

        assert abs(one["residual_derivative"]) < delta
        assert abs(one["residual_derivative"]) > 0.999 * delta
        assert abs(squared["residual_derivative"]) > 100.0 * delta
        assert one["loss"] < squared["loss"], "Huber must cost less than squared error out here"

        # Bigger outlier => closer to the cap, from below.
        nearer = batch.regress_sample(0, DECODER, BIAS, TARGETS[0] + 5.0, delta)
        assert abs(one["residual_derivative"]) > abs(nearer["residual_derivative"])

    def test_a_repeated_call_replaces_rather_than_accumulates_the_errors(self):
        """Errors are overwritten per neuron, so a reused batch cannot carry a previous step's
        error forward. The gradients accumulate; the errors do not."""
        batch = _batch()
        batch.regress_sample(0, DECODER, BIAS, TARGETS[0], math.inf)
        first = _errors(batch, 0)

        batch.regress_sample(0, DECODER, BIAS, TARGETS[1], math.inf)
        second = _errors(batch, 0)

        assert not np.allclose(first, second)
        values = _values(batch, 0)
        expected = (BIAS + float(np.dot(DECODER, values)) - TARGETS[1]) * DECODER
        np.testing.assert_allclose(second, expected, rtol=1e-12, atol=1e-15)

    def test_rejects_a_bad_sample_index(self):
        batch = _batch()
        for idx in (-1, BATCH, BATCH + 5):
            with pytest.raises(IndexError):
                batch.regress_sample(idx, DECODER, BIAS, TARGETS[0], math.inf)

    def test_rejects_a_decoder_of_the_wrong_length(self):
        """Caught here rather than in C: the C entry point takes only a pointer and trusts the
        caller, so a short array here would read past the end of the buffer."""
        batch = _batch()
        with pytest.raises(ValueError):
            batch.regress_sample(0, np.zeros(1), BIAS, TARGETS[0], math.inf)
        with pytest.raises(ValueError):
            batch.regress_sample(0, np.zeros(N_POST + 1), BIAS, TARGETS[0], math.inf)
        with pytest.raises(ValueError):
            batch.regress_sample(0, np.zeros(0), BIAS, TARGETS[0], math.inf)

    def test_library_refusals_surface_as_runtime_errors(self):
        """The C validation codes have to arrive as exceptions, not as a plausible-looking
        result struct."""
        batch = _batch()
        nan_decoder = np.ascontiguousarray(np.array([float("nan"), -1.3]))
        cases = [
            ("zero huber scale", DECODER, BIAS, TARGETS[0], 0.0),
            ("NaN huber scale", DECODER, BIAS, TARGETS[0], float("nan")),
            ("negative huber scale", DECODER, BIAS, TARGETS[0], -1.0),
            ("NaN target", DECODER, BIAS, float("nan"), math.inf),
            ("infinite target", DECODER, BIAS, float("inf"), math.inf),
            ("NaN decoder weight", nan_decoder, BIAS, TARGETS[0], math.inf),
            ("infinite bias", DECODER, float("inf"), TARGETS[0], math.inf),
        ]
        for what, dec, bias, target, delta in cases:
            with pytest.raises(RuntimeError):
                batch.regress_sample(0, np.ascontiguousarray(dec, dtype=np.float64),
                                     bias, target, delta)
