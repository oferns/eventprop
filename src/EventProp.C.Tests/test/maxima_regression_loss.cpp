/**
 * @file maxima_regression_loss.cpp
 * @brief Tests for the maxima readout and regression losses (eventprop_loss.h).
 *
 * The headline test is `MatchesCentralFiniteDifferenceForWeightGradients`. It is the first
 * finite-difference-versus-analytic comparison in this C suite — the others live in the
 * Python and .NET layers — and it is here because this feature fixes a *convention* as much
 * as a formula: `maxima_errors` is documented as dL/d(maxima value), and nothing in the
 * library checks that the caller meant that. Feeding the errors this module produces into
 * `eventprop_backward_maxima` and comparing the resulting weight gradient against a central
 * difference of the loss is the only test that notices if the two sides disagree about what
 * the array holds. A wrong sign, a missing factor, or an off-by-one neuron all pass a test
 * that merely re-implements the same arithmetic.
 *
 * Everything else here pins a stated policy: which neurons are skipped and why, that a
 * silent population is not an error, that the readout gradients accumulate rather than
 * overwrite, that the single-sample and batch paths cannot disagree, and that a refused call
 * hands back NULL arrays rather than a half-written output.
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_loss.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr double kTauMem = 0.02;
constexpr double kTauSyn = 0.005;
constexpr double kVTh    = 1.0;
/** +Inf built at runtime rather than from the INFINITY macro: the deliberate /wd4756
 *  suppression lives on the library target, so a constant-folded infinity in a test would be
 *  an error under MSVC. The volatile keeps the division out of the constant folder. */
double least_squares_scale() {
    volatile double zero = 0.0;
    return 1.0 / zero;
}

/* A non-finite huber_delta means plain least squares; see eventprop_loss.h. */
const double kMse = least_squares_scale();

/* Named distinctly from make_input() in the neighbouring test files: those are `inline`
 * definitions of the same name in other translation units, and reusing the name for a
 * different signature would be an ODR violation the linker is under no obligation to
 * mention. */
EventPropInput loss_input(const std::vector<double>& w, int n_pre, int n_post,
                          const std::vector<double>& times,
                          const std::vector<int>& sources) {
    return EventPropInput{
        w.data(), n_pre, n_post,
        times.data(), sources.data(), (int)times.size(),
        kTauMem, kTauSyn, kVTh
    };
}

/** A drive strong enough that both output neurons reach a genuine peak. */
struct Scenario {
    std::vector<double> w      = {6.0, 5.0, 4.0, 3.5};
    std::vector<double> times  = {0.01, 0.02, 0.03};
    std::vector<int>    sources = {0, 1, 0};
    static constexpr int n_pre  = 2;
    static constexpr int n_post = 2;

    EventPropInput input() const {
        return loss_input(w, n_pre, n_post, times, sources);
    }
};

/** Every maxima value must exist before a gradient means anything: comparing two
 *  derivatives of a readout that skipped a neuron is comparing a different function. */
void ExpectAllPeaksPresent(const MaximaOutput& m) {
    ASSERT_EQ(m.error_no, EVENTPROP_OK);
    ASSERT_EQ(m.n_post, Scenario::n_post);
    for (int n = 0; n < m.n_post; ++n) {
        ASSERT_TRUE(std::isfinite(m.values[n])) << "neuron " << n << " never peaked";
    }
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────────────
// The convention, measured rather than asserted
// ──────────────────────────────────────────────────────────────────────────────

TEST(MaximaRegressionLoss, MatchesCentralFiniteDifferenceForWeightGradients) {
    Scenario s;
    const std::vector<double> decoder = {0.7, -1.3};
    const double bias = 0.25;
    const double target = 1.5;

    MaximaOutput maxima = eventprop_compute_maxima(s.input());
    ExpectAllPeaksPresent(maxima);

    MaximaRegressionOutput loss = eventprop_maxima_regression(
        {maxima.values, decoder.data(), bias, target, kMse, Scenario::n_post});
    ASSERT_EQ(loss.error_no, EVENTPROP_OK);
    ASSERT_EQ(loss.n_skipped, 0);

    BackwardOutput bwd =
        eventprop_backward_maxima(s.input(), maxima.times, loss.errors);
    ASSERT_EQ(bwd.error_no, EVENTPROP_OK);

    /* Loss as a function of one weight, recomputed through the public forward path. */
    const double h = 1e-6;
    auto loss_at = [&](int k, double delta) {
        std::vector<double> w = s.w;
        w[k] += delta;
        EventPropInput in = loss_input(w, Scenario::n_pre, Scenario::n_post,
                                       s.times, s.sources);
        MaximaOutput m = eventprop_compute_maxima(in);
        MaximaRegressionOutput l = eventprop_maxima_regression(
            {m.values, decoder.data(), bias, target, kMse, Scenario::n_post});
        EXPECT_EQ(l.error_no, EVENTPROP_OK);
        const double out = l.loss;
        eventprop_free_maxima_output(m);
        eventprop_free_maxima_regression_output(l);
        return out;
    };

    for (int k = 0; k < Scenario::n_pre * Scenario::n_post; ++k) {
        const double fd = (loss_at(k, h) - loss_at(k, -h)) / (2.0 * h);
        const double analytic = bwd.grad[k];
        /* Non-vacuity first: two NaNs or two zeros would otherwise agree. */
        ASSERT_TRUE(std::isfinite(fd)) << "weight " << k;
        ASSERT_TRUE(std::isfinite(analytic)) << "weight " << k;
        EXPECT_NE(analytic, 0.0) << "weight " << k << " — a zero analytic gradient means the "
                                 "adjoint never saw this synapse, so the comparison is vacuous";
        EXPECT_NEAR(fd, analytic, 1e-5 * (1.0 + std::fabs(analytic)))
            << "weight " << k << " fd=" << fd << " analytic=" << analytic;
    }

    eventprop_free_maxima_output(maxima);
    eventprop_free_maxima_regression_output(loss);
    eventprop_free_backward_output(bwd);
}

TEST(MaximaRegressionLoss, DecoderAndBiasGradientsMatchFiniteDifferences) {
    Scenario s;
    const std::vector<double> decoder = {0.7, -1.3};
    const double bias = 0.25;
    const double target = 1.5;

    MaximaOutput maxima = eventprop_compute_maxima(s.input());
    ExpectAllPeaksPresent(maxima);

    MaximaRegressionOutput loss = eventprop_maxima_regression(
        {maxima.values, decoder.data(), bias, target, kMse, Scenario::n_post});
    ASSERT_EQ(loss.error_no, EVENTPROP_OK);

    const double h = 1e-6;
    for (int n = 0; n < Scenario::n_post; ++n) {
        std::vector<double> plus = decoder, minus = decoder;
        plus[n] += h;
        minus[n] -= h;
        MaximaRegressionOutput lp = eventprop_maxima_regression(
            {maxima.values, plus.data(), bias, target, kMse, Scenario::n_post});
        MaximaRegressionOutput lm = eventprop_maxima_regression(
            {maxima.values, minus.data(), bias, target, kMse, Scenario::n_post});
        ASSERT_EQ(lp.error_no, EVENTPROP_OK);
        ASSERT_EQ(lm.error_no, EVENTPROP_OK);
        const double fd = (lp.loss - lm.loss) / (2.0 * h);
        EXPECT_NEAR(fd, loss.decoder_grad[n], 1e-6 * (1.0 + std::fabs(fd)))
            << "decoder entry " << n;
        eventprop_free_maxima_regression_output(lp);
        eventprop_free_maxima_regression_output(lm);
    }

    MaximaRegressionOutput bp = eventprop_maxima_regression(
        {maxima.values, decoder.data(), bias + h, target, kMse, Scenario::n_post});
    MaximaRegressionOutput bm = eventprop_maxima_regression(
        {maxima.values, decoder.data(), bias - h, target, kMse, Scenario::n_post});
    ASSERT_EQ(bp.error_no, EVENTPROP_OK);
    ASSERT_EQ(bm.error_no, EVENTPROP_OK);
    EXPECT_NEAR((bp.loss - bm.loss) / (2.0 * h), loss.bias_grad,
                1e-6 * (1.0 + std::fabs(loss.bias_grad)));
    eventprop_free_maxima_regression_output(bp);
    eventprop_free_maxima_regression_output(bm);

    eventprop_free_maxima_output(maxima);
    eventprop_free_maxima_regression_output(loss);
}

// ──────────────────────────────────────────────────────────────────────────────
// Stated policies
// ──────────────────────────────────────────────────────────────────────────────

TEST(MaximaRegressionLoss, ZeroResidualGivesZeroErrorsButStillReportsPrediction) {
    Scenario s;
    const std::vector<double> decoder = {0.7, -1.3};
    MaximaOutput maxima = eventprop_compute_maxima(s.input());
    ExpectAllPeaksPresent(maxima);

    MaximaRegressionOutput probe = eventprop_maxima_regression(
        {maxima.values, decoder.data(), 0.25, 0.0, kMse, Scenario::n_post});
    ASSERT_EQ(probe.error_no, EVENTPROP_OK);

    /* Asking for the prediction as the target is how a caller reaches zero residual. */
    MaximaRegressionOutput zero = eventprop_maxima_regression(
        {maxima.values, decoder.data(), 0.25, probe.prediction, kMse, Scenario::n_post});
    EXPECT_EQ(zero.error_no, EVENTPROP_OK);
    EXPECT_DOUBLE_EQ(zero.loss, 0.0);
    EXPECT_DOUBLE_EQ(zero.residual_derivative, 0.0);
    EXPECT_DOUBLE_EQ(zero.bias_grad, 0.0);
    for (int n = 0; n < Scenario::n_post; ++n) {
        EXPECT_DOUBLE_EQ(zero.errors[n], 0.0);
        EXPECT_DOUBLE_EQ(zero.decoder_grad[n], 0.0);
    }

    eventprop_free_maxima_output(maxima);
    eventprop_free_maxima_regression_output(probe);
    eventprop_free_maxima_regression_output(zero);
}

TEST(MaximaRegressionLoss, ErrorsScaleLinearlyWithResidual) {
    Scenario s;
    const std::vector<double> decoder = {0.7, -1.3};
    const double bias = 0.25;
    MaximaOutput maxima = eventprop_compute_maxima(s.input());
    ExpectAllPeaksPresent(maxima);

    MaximaRegressionOutput one = eventprop_maxima_regression(
        {maxima.values, decoder.data(), bias, 1.5, kMse, Scenario::n_post});
    ASSERT_EQ(one.error_no, EVENTPROP_OK);
    const double residual = one.prediction - 1.5;
    ASSERT_NE(residual, 0.0);

    /* Doubling the residual means asking for the target that far on the other side. */
    MaximaRegressionOutput two = eventprop_maxima_regression(
        {maxima.values, decoder.data(), bias, one.prediction - 2.0 * residual, kMse,
         Scenario::n_post});
    ASSERT_EQ(two.error_no, EVENTPROP_OK);

    for (int n = 0; n < Scenario::n_post; ++n) {
        EXPECT_NEAR(two.errors[n], 2.0 * one.errors[n], 1e-12);
        EXPECT_NEAR(two.decoder_grad[n], 2.0 * one.decoder_grad[n], 1e-12);
    }
    EXPECT_NEAR(two.bias_grad, 2.0 * one.bias_grad, 1e-12);

    eventprop_free_maxima_output(maxima);
    eventprop_free_maxima_regression_output(one);
    eventprop_free_maxima_regression_output(two);
}

TEST(MaximaRegressionLoss, NonFiniteMaximaAreSkippedNotFatal) {
    const std::vector<double> values = {2.0, NAN, 3.0};
    const std::vector<double> decoder = {0.5, 0.9, -0.5};

    MaximaRegressionOutput out = eventprop_maxima_regression(
        {values.data(), decoder.data(), 0.1, 1.0, kMse, 3});
    EXPECT_EQ(out.error_no, EVENTPROP_OK);
    EXPECT_EQ(out.n_skipped, 1);
    /* The silent neuron contributes nothing: y_hat is over the two that peaked, and its
     * error is written as 0 rather than left to whatever the buffer held. */
    EXPECT_DOUBLE_EQ(out.prediction, 0.1 + 0.5 * 2.0 + (-0.5) * 3.0);
    EXPECT_DOUBLE_EQ(out.errors[1], 0.0);
    EXPECT_NE(out.errors[0], 0.0);
    EXPECT_DOUBLE_EQ(out.decoder_grad[1], 0.0);

    eventprop_free_maxima_regression_output(out);
}

TEST(MaximaRegressionLoss, EntirelySilentPopulationIsAValidZeroGradient) {
    const std::vector<double> values = {NAN, NAN};
    const std::vector<double> decoder = {0.5, 0.9};

    MaximaRegressionOutput out = eventprop_maxima_regression(
        {values.data(), decoder.data(), 0.4, 1.0, kMse, 2});
    EXPECT_EQ(out.error_no, EVENTPROP_OK)
        << "a sample whose neurons never peaked is data, not a programming error";
    EXPECT_EQ(out.n_skipped, 2);
    EXPECT_DOUBLE_EQ(out.prediction, 0.4);          /* bias only */
    EXPECT_DOUBLE_EQ(out.loss, 0.5 * (0.4 - 1.0) * (0.4 - 1.0));
    EXPECT_DOUBLE_EQ(out.errors[0], 0.0);
    EXPECT_DOUBLE_EQ(out.errors[1], 0.0);
    EXPECT_DOUBLE_EQ(out.decoder_grad[0], 0.0);
    EXPECT_DOUBLE_EQ(out.bias_grad, out.residual_derivative);

    eventprop_free_maxima_regression_output(out);
}

// ──────────────────────────────────────────────────────────────────────────────
// The two losses, and why one expression covers both
// ──────────────────────────────────────────────────────────────────────────────

TEST(MaximaRegressionLoss, HuberSaturatesWhereLeastSquaresDoesNot) {
    const std::vector<double> values = {100.0};
    const std::vector<double> decoder = {1.0};

    MaximaRegressionOutput ls = eventprop_maxima_regression(
        {values.data(), decoder.data(), 0.0, 0.0, kMse, 1});
    MaximaRegressionOutput hb = eventprop_maxima_regression(
        {values.data(), decoder.data(), 0.0, 0.0, 0.01, 1});
    ASSERT_EQ(ls.error_no, EVENTPROP_OK);
    ASSERT_EQ(hb.error_no, EVENTPROP_OK);

    /* Same residual, so least squares drives a gradient of ~100 and Huber of ~delta. */
    EXPECT_NEAR(ls.residual_derivative, 100.0, 1e-9);
    EXPECT_NEAR(hb.residual_derivative, 0.01, 1e-9);
    EXPECT_LT(std::fabs(hb.residual_derivative), std::fabs(ls.residual_derivative));
    /* Saturation is the point: an outlier must not dominate the step. */
    EXPECT_LT(hb.errors[0], ls.errors[0]);

    eventprop_free_maxima_regression_output(ls);
    eventprop_free_maxima_regression_output(hb);
}

TEST(MaximaRegressionLoss, HuberApproachesLeastSquaresAsScaleGrows) {
    const std::vector<double> values = {1.7, -0.4};
    const std::vector<double> decoder = {0.6, 1.1};

    MaximaRegressionOutput ls = eventprop_maxima_regression(
        {values.data(), decoder.data(), 0.2, 0.9, kMse, 2});
    MaximaRegressionOutput wide = eventprop_maxima_regression(
        {values.data(), decoder.data(), 0.2, 0.9, 1e9, 2});
    ASSERT_EQ(ls.error_no, EVENTPROP_OK);
    ASSERT_EQ(wide.error_no, EVENTPROP_OK);

    EXPECT_NEAR(wide.loss, ls.loss, 1e-12 * (1.0 + ls.loss));
    EXPECT_NEAR(wide.residual_derivative, ls.residual_derivative, 1e-9);
    /* Non-vacuity: if both came out NaN these comparisons would be false, not true, but say
     * so explicitly rather than letting a reader wonder which failure they'd see. */
    EXPECT_TRUE(std::isfinite(ls.loss));
    EXPECT_TRUE(std::isfinite(wide.loss));

    eventprop_free_maxima_regression_output(ls);
    eventprop_free_maxima_regression_output(wide);
}

TEST(MaximaRegressionLoss, SmallResidualKeepsItsSignificantDigits) {
    /* delta^2 (sqrt(1 + r^2/delta^2) - 1) is exactly 0 in floating point once r is small
     * relative to delta, and a loss of exactly 0 with a nonzero gradient tells a training
     * loop it has converged when it has not. The r^2/(sqrt(1+q^2)+1) spelling does not. */
    const std::vector<double> values = {1.0 + 1e-8, 1.0};
    const std::vector<double> decoder = {1.0, 1.0};

    MaximaRegressionOutput out = eventprop_maxima_regression(
        {values.data(), decoder.data(), 0.0, 2.0, 1.0, 2});
    ASSERT_EQ(out.error_no, EVENTPROP_OK);
    EXPECT_GT(out.loss, 0.0) << "loss collapsed to zero by cancellation";
    /* Expected is r^2/2 for the residual the readout actually produced — not a literal 1e-8,
     * because 1 + 1e-8 is not exactly representable and the difference is larger than the
     * 1e-8 relative band this assertion is checking. */
    const double r = out.prediction - 2.0;
    const double expected = 0.5 * r * r;
    EXPECT_GT(r, 0.0);
    EXPECT_NEAR(out.loss, expected, 1e-8 * expected);
    EXPECT_GT(out.residual_derivative, 0.0);

    eventprop_free_maxima_regression_output(out);
}

// ──────────────────────────────────────────────────────────────────────────────
// Refusals: the arrays must come back NULL, not half-written
// ──────────────────────────────────────────────────────────────────────────────

TEST(MaximaRegressionLoss, RefusalsReturnNullArrays) {
    const std::vector<double> values = {1.0, 2.0};
    const std::vector<double> decoder = {0.5, 0.5};
    const std::vector<double> nan_decoder = {NAN, 0.5};
    const std::vector<double> huge = {1e308, 1e308};
    const std::vector<double> huge_values = {1e308, 1e308};

    struct Case {
        const char* what;
        MaximaRegressionInput in;
        EventPropError want;
    };
    const double inf = least_squares_scale();
    const Case cases[] = {
        {"NULL values",      {nullptr, decoder.data(), 0.0, 1.0, kMse, 2},       EVENTPROP_NULL_POINTER},
        {"NULL decoder",     {values.data(), nullptr,  0.0, 1.0, kMse, 2},       EVENTPROP_NULL_POINTER},
        {"n_post == 0",      {values.data(), decoder.data(), 0.0, 1.0, kMse, 0}, EVENTPROP_INVALID_DIMENSION},
        {"negative n_post",  {values.data(), decoder.data(), 0.0, 1.0, kMse, -3},EVENTPROP_INVALID_DIMENSION},
        {"NaN target",       {values.data(), decoder.data(), 0.0, NAN, kMse, 2}, EVENTPROP_NONFINITE_TARGET},
        {"infinite target",  {values.data(), decoder.data(), 0.0, inf, kMse, 2}, EVENTPROP_NONFINITE_TARGET},
        {"zero Huber scale", {values.data(), decoder.data(), 0.0, 1.0, 0.0, 2},  EVENTPROP_INVALID_HUBER_DELTA},
        {"negative scale",   {values.data(), decoder.data(), 0.0, 1.0, -1.0, 2}, EVENTPROP_INVALID_HUBER_DELTA},
        {"NaN scale",        {values.data(), decoder.data(), 0.0, 1.0, NAN, 2},  EVENTPROP_INVALID_HUBER_DELTA},
        {"NaN bias",         {values.data(), decoder.data(), NAN, 1.0, kMse, 2}, EVENTPROP_NONFINITE_DECODER},
        {"NaN decoder",      {values.data(), nan_decoder.data(), 0.0, 1.0, kMse, 2}, EVENTPROP_NONFINITE_DECODER},
        {"overflow readout", {huge_values.data(), huge.data(), 0.0, 1.0, kMse, 2}, EVENTPROP_NONFINITE_PREDICTION},
    };

    for (const auto& c : cases) {
        MaximaRegressionOutput out = eventprop_maxima_regression(c.in);
        EXPECT_EQ(out.error_no, c.want) << c.what << " — got "
                                        << eventprop_error_string(out.error_no);
        EXPECT_EQ(out.errors, nullptr) << c.what << ": half-written errors would be read "
                                          "as a gradient by a caller that ignores error_no";
        EXPECT_EQ(out.decoder_grad, nullptr) << c.what;
        EXPECT_DOUBLE_EQ(out.n_post, 0);
        /* Freeing a refused output must be safe — every free function in this library is. */
        eventprop_free_maxima_regression_output(out);
    }
}

TEST(MaximaRegressionLoss, ErrorMessageIsSpecificNotGeneric) {
    /* The strings are part of the API for a caller debugging from a Python traceback. */
    EXPECT_NE(std::string(eventprop_error_string(EVENTPROP_NONFINITE_PREDICTION))
                  .find("overflow"), std::string::npos);
    EXPECT_NE(std::string(eventprop_error_string(EVENTPROP_INVALID_HUBER_DELTA))
                  .find("Huber"), std::string::npos);
}

// ──────────────────────────────────────────────────────────────────────────────
// In-place and batch paths must be the same computation, not a re-implementation
// ──────────────────────────────────────────────────────────────────────────────

TEST(MaximaRegressionLoss, InPlaceWritesTheSameErrorsAsTheAllocatingCall) {
    Scenario s;
    const std::vector<double> decoder = {0.7, -1.3};
    MaximaOutput maxima = eventprop_compute_maxima(s.input());
    ExpectAllPeaksPresent(maxima);

    MaximaRegressionOutput alloc = eventprop_maxima_regression(
        {maxima.values, decoder.data(), 0.25, 1.5, kMse, Scenario::n_post});
    ASSERT_EQ(alloc.error_no, EVENTPROP_OK);

    /* Pre-poison the library's errors array: a stale value surviving the in-place call would
     * otherwise look like agreement if the real error happened to be 0. */
    maxima.errors[0] = 12345.0;
    maxima.errors[1] = -12345.0;
    MaximaRegressionResult in_place =
        eventprop_maxima_regression_into(maxima, decoder.data(), 0.25, 1.5, kMse);
    ASSERT_EQ(in_place.error_no, EVENTPROP_OK);

    EXPECT_DOUBLE_EQ(in_place.prediction, alloc.prediction);
    EXPECT_DOUBLE_EQ(in_place.loss, alloc.loss);
    EXPECT_EQ(in_place.n_skipped, alloc.n_skipped);
    for (int n = 0; n < Scenario::n_post; ++n) {
        EXPECT_DOUBLE_EQ(maxima.errors[n], alloc.errors[n]) << "neuron " << n;
    }

    eventprop_free_maxima_output(maxima);
    eventprop_free_maxima_regression_output(alloc);
}

TEST(MaximaRegressionLoss, BatchMatchesPerSampleAndAccumulatesReadoutGradients) {
    constexpr int kBatch = 3;
    std::vector<Scenario> scenarios(kBatch);
    scenarios[1].w = {7.0, 4.0, 3.0, 5.0};
    scenarios[2].w = {5.0, 6.0, 2.5, 4.0};

    std::vector<EventPropInput> inputs;
    for (const auto& s : scenarios) inputs.push_back(s.input());

    MaximaBatch batch = eventprop_compute_maxima_batch(inputs.data(), kBatch);
    ASSERT_EQ(batch.error_no, EVENTPROP_OK);
    ASSERT_EQ(batch.batch_size, kBatch);

    const std::vector<double> decoder = {0.7, -1.3};
    const double bias = 0.25;
    const std::vector<double> targets = {1.5, -0.5, 2.0};

    std::vector<double> predictions(kBatch), losses(kBatch);
    std::vector<double> grad_a(Scenario::n_post, 0.0);
    std::vector<double> expected_grad_a(Scenario::n_post, 0.0);
    double grad_b = 0.0, expected_grad_b = 0.0;
    std::vector<int> skipped(kBatch, -1);

    ASSERT_EQ(eventprop_maxima_regression_batch(batch, decoder.data(), bias, targets.data(),
                                                kMse, predictions.data(), losses.data(),
                                                skipped.data(), grad_a.data(), &grad_b),
              EVENTPROP_OK);

    for (int i = 0; i < kBatch; ++i) {
        MaximaRegressionOutput one = eventprop_maxima_regression(
            {batch.outputs[i].values, decoder.data(), bias, targets[i], kMse,
             Scenario::n_post});
        ASSERT_EQ(one.error_no, EVENTPROP_OK);

        EXPECT_DOUBLE_EQ(predictions[i], one.prediction) << "sample " << i;
        EXPECT_DOUBLE_EQ(losses[i], one.loss) << "sample " << i;
        EXPECT_EQ(skipped[i], one.n_skipped) << "sample " << i;
        for (int n = 0; n < Scenario::n_post; ++n) {
            EXPECT_DOUBLE_EQ(batch.outputs[i].errors[n], one.errors[n])
                << "sample " << i << " neuron " << n;
            expected_grad_a[n] += one.decoder_grad[n];
        }
        expected_grad_b += one.bias_grad;
        eventprop_free_maxima_regression_output(one);
    }

    for (int n = 0; n < Scenario::n_post; ++n) {
        EXPECT_NEAR(grad_a[n], expected_grad_a[n], 1e-12 * (1.0 + std::fabs(grad_a[n])))
            << "decoder entry " << n;
    }
    EXPECT_NEAR(grad_b, expected_grad_b, 1e-12 * (1.0 + std::fabs(grad_b)));

    eventprop_free_maxima_batch(batch);
}

TEST(MaximaRegressionLoss, ReadoutGradientsAccumulateRatherThanOverwrite) {
    /* The layer contract this repo holds everywhere: Backward() adds, ZeroGrad() clears.
     * A readout that overwrote would silently drop every sample but the last. */
    constexpr int kBatch = 2;
    std::vector<Scenario> scenarios(kBatch);
    scenarios[1].w = {7.0, 4.0, 3.0, 5.0};
    std::vector<EventPropInput> inputs;
    for (const auto& s : scenarios) inputs.push_back(s.input());

    MaximaBatch batch = eventprop_compute_maxima_batch(inputs.data(), kBatch);
    ASSERT_EQ(batch.error_no, EVENTPROP_OK);

    const std::vector<double> decoder = {0.7, -1.3};
    const std::vector<double> targets = {1.5, -0.5};
    std::vector<double> grad_a(Scenario::n_post, 0.0);
    double grad_b = 0.0;

    ASSERT_EQ(eventprop_maxima_regression_batch(batch, decoder.data(), 0.25, targets.data(),
                                                kMse, nullptr, nullptr, nullptr,
                                                grad_a.data(), &grad_b),
              EVENTPROP_OK);
    ASSERT_NE(grad_a[0], 0.0);
    const std::vector<double> once = grad_a;
    const double b_once = grad_b;

    ASSERT_EQ(eventprop_maxima_regression_batch(batch, decoder.data(), 0.25, targets.data(),
                                                kMse, nullptr, nullptr, nullptr,
                                                grad_a.data(), &grad_b),
              EVENTPROP_OK);
    for (int n = 0; n < Scenario::n_post; ++n) {
        EXPECT_DOUBLE_EQ(grad_a[n], 2.0 * once[n]);
    }
    EXPECT_DOUBLE_EQ(grad_b, 2.0 * b_once);

    /* NULL accumulators are allowed: a caller evaluating a loss without training the readout
     * must still get predictions and errors. */
    std::vector<double> preds(kBatch, 0.0);
    ASSERT_EQ(eventprop_maxima_regression_batch(batch, decoder.data(), 0.25, targets.data(),
                                                kMse, preds.data(), nullptr, nullptr,
                                                nullptr, nullptr),
              EVENTPROP_OK);
    EXPECT_NE(preds[0], 0.0);

    eventprop_free_maxima_batch(batch);
}

TEST(MaximaRegressionLoss, BatchRejectsMismatchedWidthsBeforeReadingTheDecoder) {
    /* A short sample would make the shared decoder array look longer than it is; the width
     * check is what turns that into an error code instead of a buffer read. */
    std::vector<double> w2 = {6.0, 5.0};
    std::vector<double> w3 = {6.0, 5.0, 4.0, 3.5, 2.0, 1.5};
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};

    EventPropInput narrow = loss_input(w2, 2, 1, times, sources);
    EventPropInput wide   = loss_input(w3, 2, 3, times, sources);

    MaximaBatch narrow_batch = eventprop_compute_maxima_batch(&narrow, 1);
    MaximaBatch wide_batch   = eventprop_compute_maxima_batch(&wide, 1);
    ASSERT_EQ(narrow_batch.error_no, EVENTPROP_OK);
    ASSERT_EQ(wide_batch.error_no, EVENTPROP_OK);

    const std::vector<double> decoder = {0.7};
    const std::vector<double> target = {1.0};

    /* Hand-assemble a two-sample batch of differing widths. The array of outputs is ours for
     * the duration of the call, which is exactly the caller-owned shape the API documents. */
    MaximaOutput mixed[2];
    mixed[0] = narrow_batch.outputs[0];
    mixed[1] = wide_batch.outputs[0];
    MaximaBatch mismatched{EVENTPROP_OK, 2, 0, mixed};
    EXPECT_EQ(eventprop_maxima_regression_batch(mismatched, decoder.data(), 0.0,
                                                target.data(), kMse,
                                                nullptr, nullptr, nullptr, nullptr, nullptr),
              EVENTPROP_BATCH_DIMENSION_MISMATCH);

    /* Ownership: the mixed array borrowed both outputs, so clear the slots the batches
     * still own before freeing them. */
    narrow_batch.outputs[0] = MaximaOutput{};
    wide_batch.outputs[0]   = MaximaOutput{};
    eventprop_free_maxima_batch(narrow_batch);
    eventprop_free_maxima_batch(wide_batch);
}

TEST(MaximaRegressionLoss, BatchRefusesNullsAndAcceptsAnEmptyBatch) {
    Scenario s;
    const EventPropInput single = s.input();
    MaximaBatch batch = eventprop_compute_maxima_batch(&single, 1);
    ASSERT_EQ(batch.error_no, EVENTPROP_OK);
    const std::vector<double> decoder = {0.7, -1.3};
    const std::vector<double> target = {1.0};

    EXPECT_EQ(eventprop_maxima_regression_batch(batch, nullptr, 0.0, target.data(), kMse,
                                                nullptr, nullptr, nullptr, nullptr, nullptr),
              EVENTPROP_NULL_POINTER);
    EXPECT_EQ(eventprop_maxima_regression_batch(batch, decoder.data(), 0.0, nullptr, kMse,
                                                nullptr, nullptr, nullptr, nullptr, nullptr),
              EVENTPROP_NULL_POINTER);

    MaximaOutput none[1];
    MaximaBatch empty{EVENTPROP_OK, 0, 0, none};
    EXPECT_EQ(eventprop_maxima_regression_batch(empty, decoder.data(), 0.0, target.data(),
                                                kMse, nullptr, nullptr, nullptr, nullptr,
                                                nullptr),
              EVENTPROP_OK) << "an empty batch is nothing to do, not a failure";

    eventprop_free_maxima_batch(batch);
}

TEST(MaximaRegressionLoss, BatchReportsTheOffendingSampleWithoutCorruptingTheRest) {
    constexpr int kBatch = 2;
    std::vector<Scenario> scenarios(kBatch);
    scenarios[1].w = {7.0, 4.0, 3.0, 5.0};
    std::vector<EventPropInput> inputs;
    for (const auto& s : scenarios) inputs.push_back(s.input());

    MaximaBatch batch = eventprop_compute_maxima_batch(inputs.data(), kBatch);
    ASSERT_EQ(batch.error_no, EVENTPROP_OK);

    const std::vector<double> decoder = {0.7, -1.3};
    const std::vector<double> targets = {1.5, NAN};

    EXPECT_EQ(eventprop_maxima_regression_batch(batch, decoder.data(), 0.25, targets.data(),
                                                kMse, nullptr, nullptr, nullptr, nullptr,
                                                nullptr),
              EVENTPROP_NONFINITE_TARGET);
    /* The forward pass's own flag must not be rewritten by a mistake made in a loss call:
     * the batch it describes succeeded. */
    EXPECT_EQ(batch.has_errors, 0);

    eventprop_free_maxima_batch(batch);
}
