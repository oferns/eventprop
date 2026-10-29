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

from numpy.testing import assert_allclose

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


def _grad_buffers():
    return np.zeros(N_POST), np.zeros(1)


class TestBatchRegress:
    """`_MaximaBatchOutput.regress` — the entry a training loop uses."""

    def test_matches_the_sum_of_per_sample_calls(self):
        """The batch readout must be the per-sample one, summed, in both directions."""
        batch = _batch()
        dgrad, bgrad = _grad_buffers()
        step = batch.regress(DECODER, BIAS, TARGETS, math.inf,
                             decoder_grad=dgrad, bias_grad=bgrad)

        per_sample = [native.maxima_regression(_values(batch, i), DECODER, BIAS,
                                               float(TARGETS[i]), math.inf)
                      for i in range(BATCH)]
        assert len(step["predictions"]) == len(step["losses"]) == BATCH
        for i, exp in enumerate(per_sample):
            assert_allclose(step["predictions"][i], exp["prediction"], rtol=0, atol=0)
            assert_allclose(step["losses"][i], exp["loss"], rtol=0, atol=0)
            assert_allclose(_errors(batch, i), exp["errors"], rtol=0, atol=0)

        assert_allclose(dgrad, sum(p["decoder_grad"] for p in per_sample), rtol=1e-12, atol=0)
        assert_allclose(bgrad, [sum(float(p["residual_derivative"]) for p in per_sample)],
                        rtol=1e-12, atol=0)

    def test_readout_buffers_accumulate_and_are_caller_owned(self):
        batch = _batch()
        dgrad, bgrad = _grad_buffers()
        batch.regress(DECODER, BIAS, TARGETS, math.inf, decoder_grad=dgrad, bias_grad=bgrad)
        first_d, first_b = dgrad.copy(), bgrad.copy()
        assert np.abs(first_d).max() > 0.0, "the readout produced no decoder gradient at all"

        batch.regress(DECODER, BIAS, TARGETS, math.inf, decoder_grad=dgrad, bias_grad=bgrad)
        assert_allclose(dgrad, 2 * first_d, rtol=1e-12, atol=0)
        assert_allclose(bgrad, 2 * first_b, rtol=1e-12, atol=0)

        # The returned gradients are a view of the buffers passed in, not a snapshot of them:
        # writing through one is visible in the other. `is` would be the wrong assertion — the
        # binding wraps the memoryview, so the object differs and the memory does not.
        fresh_d, fresh_b = _grad_buffers()
        out = batch.regress(DECODER, BIAS, TARGETS, math.inf,
                            decoder_grad=fresh_d, bias_grad=fresh_b)
        for returned, passed in ((out["decoder_grad"], fresh_d), (out["bias_grad"], fresh_b)):
            assert returned is not passed
            assert returned.__array_interface__["data"][0] == passed.__array_interface__["data"][0]
            returned *= 0.0
            assert np.abs(passed).max() == 0.0

    def test_written_errors_drive_the_backward_pass(self):
        """Prove the batch write reaches the memory the backward pass actually reads."""
        clean = _batch()
        assert np.asarray(native.backward_maxima_batch_accumulate(clean).grad).max() == 0.0

        batch = _batch()
        batch.regress(DECODER, BIAS, TARGETS, math.inf)
        grad = np.asarray(native.backward_maxima_batch_accumulate(batch).grad)
        assert grad.size == N_PRE * N_POST          # flat; callers reshape to (n_pre, n_post)
        assert np.abs(grad).max() > 0.0


class TestAllocatingSingleSample:
    """`native.maxima_regression` — the entry a notebook reaches for."""

    def test_gradients_match_central_differences(self):
        batch = _batch()
        values = _values(batch, 0)
        target, eps = 0.4, 1e-5

        def loss(dec, bias):
            return float(native.maxima_regression(values, np.ascontiguousarray(dec),
                                                  float(bias), target, math.inf)["loss"])

        dec0 = np.ascontiguousarray(DECODER.copy())
        out = native.maxima_regression(values, dec0, BIAS, target, math.inf)
        assert float(out["n_skipped"]) == 0.0, \
            "a silent neuron would zero the gradient this check is about"
        assert abs(float(out["loss"])) > 0.0

        for n in range(N_POST):
            dp, dm = dec0.copy(), dec0.copy()
            dp[n] += eps
            dm[n] -= eps
            fd = (loss(dp, BIAS) - loss(dm, BIAS)) / (2 * eps)
            assert abs(fd - out["decoder_grad"][n]) < 1e-4 * abs(fd) + 1e-8, \
                f"decoder[{n}]: fd={fd} analytic={out['decoder_grad'][n]}"

        fd_b = (loss(dec0, BIAS + eps) - loss(dec0, BIAS - eps)) / (2 * eps)
        assert abs(fd_b - float(out["residual_derivative"])) < 1e-4 * abs(fd_b) + 1e-8

    def test_the_allocating_path_does_not_accumulate_across_calls(self):
        """The opposite contract from `regress`: fresh arrays, nothing carried over."""
        batch = _batch()
        values = _values(batch, 0)
        first = native.maxima_regression(values, DECODER, BIAS, TARGETS[0], math.inf)
        second = native.maxima_regression(values, DECODER, BIAS, TARGETS[0], math.inf)
        assert_allclose(second["decoder_grad"], first["decoder_grad"], rtol=0, atol=0)
        assert first["decoder_grad"] is not second["decoder_grad"]


class TestRefusals:
    """Validation has to arrive as an exception, on every entry point."""

    def test_huber_scale_rules(self):
        batch = _batch()
        for delta in (0.0, -1.0, float("nan")):
            with pytest.raises(RuntimeError):
                batch.regress(DECODER, BIAS, TARGETS, delta)
            with pytest.raises(RuntimeError):
                native.maxima_regression(_values(batch, 0), DECODER, BIAS, TARGETS[0], delta)
        # non-finite and finite-positive are both accepted, and mean different losses
        ls = batch.regress(DECODER, BIAS, TARGETS, math.inf)["losses"].copy()
        hub = batch.regress(DECODER, BIAS, TARGETS, 1e-3)["losses"].copy()
        assert np.abs(ls - hub).max() > 0.0

    def test_non_finite_readout_parameters_are_refused_even_for_silent_neurons(self):
        """A poisoned weight must not be able to hide behind a neuron that never fires."""
        zero_w = np.zeros((N_PRE, N_POST))
        silent = native.compute_maxima_batch(TIMES, SOURCES, N_PER_SAMPLE, zero_w,
                                             N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH)
        nan_decoder = np.ascontiguousarray(np.array([float("nan"), -1.3]))
        with pytest.raises(RuntimeError):
            silent.regress(nan_decoder, BIAS, TARGETS, math.inf)

    def test_length_mismatch_is_refused(self):
        batch = _batch()
        with pytest.raises(ValueError):
            batch.regress(np.zeros(N_POST + 1), BIAS, TARGETS, math.inf)
        with pytest.raises(ValueError):
            batch.regress(DECODER, BIAS, np.zeros(BATCH + 1), math.inf)

    def test_a_population_at_rest_is_a_valid_sample(self):
        """Zero weights: every neuron's maximum is exactly 0, which is a real value.

        Measured rather than assumed — this is the case that *looks* like the skip path and is
        not: the maxima are finite, so nothing is skipped, the readout collapses to the bias, and
        the decoder gradient is legitimately zero because dL/ddecoder = dL/dy_hat * maxima.
        """
        zero_w = np.zeros((N_PRE, N_POST))
        at_rest = native.compute_maxima_batch(TIMES, SOURCES, N_PER_SAMPLE, zero_w,
                                              N_PRE, N_POST, TAU_MEM, TAU_SYN, V_TH)
        assert np.isfinite(np.array(at_rest[0][1])).all()
        step = at_rest.regress(DECODER, 0.3, TARGETS, math.inf)
        np.testing.assert_array_equal(np.asarray(step["n_skipped"]), [0] * BATCH)
        assert_allclose(step["predictions"], [0.3] * BATCH, rtol=0, atol=0)
        assert_allclose(step["losses"], 0.5 * (0.3 - TARGETS) ** 2, rtol=1e-12, atol=0)
        assert np.abs(np.asarray(step["decoder_grad"])).max() == 0.0

    def test_a_neuron_that_never_peaked_is_excluded_and_counted(self):
        """The skip path itself, driven directly: a non-finite maximum is the library's signal
        for "this neuron never peaked", so it must not enter the readout or the gradient."""
        values = np.array([float("nan"), 0.5])
        out = native.maxima_regression(values, np.ones(2), 0.1, 0.7, math.inf)
        assert out["n_skipped"] == 1
        assert_allclose(out["prediction"], 0.1 + 0.5, rtol=1e-12, atol=0)
        assert_allclose(np.asarray(out["errors"])[0], 0.0, rtol=0, atol=0)

        all_silent = np.full(2, float("nan"))
        out = native.maxima_regression(all_silent, np.ones(2), 0.3, 0.7, math.inf)
        assert out["n_skipped"] == 2
        assert_allclose(out["prediction"], 0.3, rtol=0, atol=0)
        assert_allclose(out["loss"], 0.5 * (0.3 - 0.7) ** 2, rtol=1e-12, atol=0)
        assert np.abs(np.asarray(out["errors"])).max() == 0.0
        assert np.abs(np.asarray(out["decoder_grad"])).max() == 0.0


class TestLayerEntryPoints:
    """`LILayer.regression` and `MaximaVector.regress` are the shapes callers reach for.

    The claim they have to earn is not that they forward arguments — it is that the errors they
    write are the errors the *backward pass* reads, so that forward → regression → backward
    produces dL/dw_in. That is checked against finite differences of the loss, which involves
    every layer of the binding at once and cannot be satisfied by a method that writes into a copy.
    """

    @staticmethod
    def _params():
        from eventprop_c import LILayerParameters
        return LILayerParameters(n=N_POST, n_in=N_PRE, tau_mem=TAU_MEM, tau_syn=TAU_SYN,
                                 plastic_weights=False)

    @classmethod
    def _spikes(cls):
        from eventprop_c import Spikes, SpikesVector
        out = []
        for i in range(BATCH):
            lo, hi = int(np.sum(N_PER_SAMPLE[:i])), int(np.sum(N_PER_SAMPLE[:i + 1]))
            out.append(Spikes(times=TIMES[lo:hi].copy(), sources=SOURCES[lo:hi].copy()))
        return SpikesVector(out)

    @classmethod
    def _layer(cls):
        from eventprop_c import LILayer
        return LILayer(cls._params(), w_in=W.copy())

    def test_regression_requires_a_prior_forward(self):
        with pytest.raises(RuntimeError, match="prior forward"):
            self._layer().regression(DECODER, BIAS, TARGETS)

    def test_matches_an_explicit_batch_call(self):
        """The convenience method must not recompute, reorder, or re-seed anything."""
        spikes = self._spikes()

        layer = self._layer()
        layer.forward(spikes)
        via_layer = layer.regression(DECODER, BIAS, TARGETS)

        # huber_delta is positional and required on the raw entry point — only the layer wrapper
        # defaults it, so the low-level call never picks a loss implicitly.
        manual = native.compute_maxima_batch(TIMES, SOURCES, N_PER_SAMPLE, W,
                                             N_PRE, N_POST, TAU_MEM, TAU_SYN, float("inf")) \
            .regress(DECODER, BIAS, TARGETS, math.inf)

        assert_allclose(np.asarray(via_layer["predictions"]), np.asarray(manual["predictions"]),
                        rtol=0, atol=0)
        assert_allclose(np.asarray(via_layer["losses"]), np.asarray(manual["losses"]),
                        rtol=0, atol=0)

    def test_maxima_vector_entry_points_agree_with_the_layer(self):
        spikes = self._spikes()
        layer = self._layer()
        maxima = layer.forward(spikes)

        via_vector = maxima.regress(DECODER, BIAS, TARGETS)
        via_layer = layer.regression(DECODER, BIAS, TARGETS)
        assert_allclose(np.asarray(via_vector["losses"]), np.asarray(via_layer["losses"]),
                        rtol=0, atol=0)

        one = maxima.regress_sample(0, DECODER, BIAS, float(TARGETS[0]))
        assert_allclose(one["prediction"], float(np.asarray(via_layer["predictions"])[0]),
                        rtol=0, atol=0)
        assert_allclose(one["loss"], float(np.asarray(via_layer["losses"])[0]), rtol=0, atol=0)

    def test_the_whole_chain_matches_finite_differences(self):
        """forward → regression → backward, against finite differences of the same loss."""
        decoder = np.ascontiguousarray(np.array([0.6, -0.3]))
        bias, targets = 0.4, np.array([0.8, -0.2])
        spikes = self._spikes()

        layer = self._layer()
        layer.forward(spikes)
        layer.regression(decoder, bias, targets,
                         decoder_grad=np.zeros(N_POST), bias_grad=np.zeros(1))
        layer.backward()
        analytic = layer.gradient.copy()
        assert np.abs(analytic).max() > 1e-9, "the scenario produced no weight gradient at all"

        def total_loss(w):
            probe = LILayer_for_fd(w)
            probe.forward(spikes)
            out = probe.regression(decoder, bias, targets,
                                   decoder_grad=np.zeros(N_POST), bias_grad=np.zeros(1))
            return float(np.sum(np.asarray(out["losses"])))

        w0 = layer.w_in.copy()
        # Not a tautology guard: if the loss does not decrease downhill, "matches finite
        # differences" could be matching a scenario where nothing at all happens.
        assert total_loss(w0 - 1e-5 * analytic) < total_loss(w0), "a step downhill raised the loss"

        eps, scale = 1e-5, np.abs(analytic).max()
        for i in range(N_PRE):
            for j in range(N_POST):
                wp, wm = w0.copy(), w0.copy()
                wp[i, j] += eps
                wm[i, j] -= eps
                fd = (total_loss(wp) - total_loss(wm)) / (2 * eps)
                assert abs(fd - analytic[i, j]) < 1e-3 * scale + 1e-9, \
                    f"dW[{i},{j}]: fd={fd} analytic={analytic[i, j]}"

    def test_a_training_step_reduces_the_loss(self):
        """Three plain-SGD steps through the layer wrapper. Not a benchmark, a direction."""
        decoder = np.ascontiguousarray(np.array([0.0, 0.0]))
        bias, targets = 0.0, np.array([0.8, -0.2])
        spikes = self._spikes()
        layer = self._layer()

        # A step size for this fixture, chosen so the assertion is about direction rather than
        # about a divergent optimizer: the readout gradient here is O(10) against maxima of O(0.1).
        lr_w, lr_readout = 0.02, 0.02
        losses = []
        for _ in range(4):
            layer.zero_grad()
            layer.forward(spikes)
            step = layer.regression(decoder, bias, targets,
                                    decoder_grad=np.zeros(N_POST), bias_grad=np.zeros(1))
            layer.backward()
            losses.append(float(np.sum(np.asarray(step["losses"]))))
            layer.w_in = layer.w_in - lr_w * layer.gradient
            decoder = decoder - lr_readout * np.asarray(step["decoder_grad"])
            bias = bias - lr_readout * float(np.asarray(step["bias_grad"])[0])

        assert losses[0] > 0.0
        assert all(b < a for a, b in zip(losses, losses[1:])), f"not monotone: {losses}"


def LILayer_for_fd(w):
    """Tiny indirection so the FD helper above reads as one expression."""
    from eventprop_c import LILayer
    return LILayer(TestLayerEntryPoints._params(), w_in=w.copy())


class TestPythonBuiltMaximaRefuses:
    """The readout lives in the C library, so a maxima vector built in Python has nowhere to
    write its errors — and must say so rather than compute into a detached copy."""

    def test_regress_and_regress_sample_refuse(self):
        from eventprop_c import Maxima, MaximaVector
        v = MaximaVector([Maxima(times=np.zeros(0), values=np.zeros(N_POST))])
        with pytest.raises(RuntimeError, match="C-owned"):
            v.regress(DECODER, BIAS, TARGETS)
        with pytest.raises(RuntimeError, match="C-owned"):
            v.regress_sample(0, DECODER, BIAS, float(TARGETS[0]))
