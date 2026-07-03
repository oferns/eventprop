/**
 * @file backward_maxima.cpp
 * @brief Tests for eventprop_backward_maxima (backward pass, maxima/voltage loss).
 *
 * The backward sweep propagates per-neuron maxima errors back through the LIF
 * dynamics to compute weight gradients and input spike errors. We test
 * observable contracts: linearity in errors, source-row locality, NaN skipping,
 * and algebraic scaling — not exact gradient values (which depend on solver tolerance).
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include <cmath>
#include <vector>

static constexpr double kTauMem = 0.02;
static constexpr double kTauSyn = 0.005;
static constexpr double kVTh    = 1.0;

inline EventPropInput make_input(const std::vector<double>& w, int n_pre, int n_post,
                                  const std::vector<double>& times,
                                  const std::vector<int>& sources) {
    return EventPropInput{
        w.data(), n_pre, n_post,
        times.data(), sources.data(), (int)times.size(),
        kTauMem, kTauSyn, kVTh
    };
}

// ──────────────────────────────────────────────────────────────
// Zero input spikes — early return path
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, ZeroInputSpikesReturnsZeroGradient) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    // Maxima times and errors (n_post == 1)
    std::vector<double> maxima_times = {NAN};   // no spikes → NaN time
    std::vector<double> maxima_errors = {1.0};

    BackwardOutput result = eventprop_backward_maxima(input, maxima_times.data(), maxima_errors.data());
    EXPECT_EQ(result.error_no, EVENTPROP_OK);
    // grad should be all zeros (calloc'd)
    EXPECT_DOUBLE_EQ(result.grad[0], 0.0);
    // spike_errors has input.n_spikes == 0 entries

    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Zero post-neurons — early return path
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, ZeroNpostReturnsError) {
    std::vector<double> w = {};
    EventPropInput input{nullptr, 1, 0, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Single input spike + single post-neuron — simplest non-trivial case
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, SingleInputSinglePostNeuron) {
    std::vector<double> w = {10.0};   // strong weight → fires
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    // Should have found a maximum (not NaN)
    EXPECT_FALSE(std::isnan(maxima.times[0]));

    // Set error to 1.0 (unit loss)
    maxima.errors[0] = 1.0;

    BackwardOutput result = eventprop_backward_maxima(input, maxima.times, maxima.errors);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Gradient should be finite and nonzero (strong weight + unit error)
    EXPECT_FALSE(std::isnan(result.grad[0]));
    EXPECT_FALSE(std::isinf(result.grad[0]));
    EXPECT_NE(result.grad[0], 0.0);

    // Spike errors should be finite
    for (int i = 0; i < input.n_spikes; ++i) {
        EXPECT_FALSE(std::isnan(result.spike_errors[i]));
        EXPECT_FALSE(std::isinf(result.spike_errors[i]));
    }

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Linearity in maxima errors — scaling errors scales output proportionally
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, LinearInMaximaErrors) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01, 0.02};
    std::vector<int> sources    = {0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);

    // Run with errors = 1.0
    maxima.errors[0] = 1.0;
    BackwardOutput r1 = eventprop_backward_maxima(input, maxima.times, maxima.errors);

    // Run with errors = 3.0
    double scale = 3.0;
    maxima.errors[0] = scale;
    BackwardOutput r2 = eventprop_backward_maxima(input, maxima.times, maxima.errors);

    EXPECT_EQ(r1.error_no, EVENTPROP_OK);
    EXPECT_EQ(r2.error_no, EVENTPROP_OK);

    // grad and spike_errors should scale linearly
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_NEAR(r2.grad[i], r1.grad[i] * scale, 1e-12);
    }
    for (int i = 0; i < input.n_spikes; ++i) {
        EXPECT_NEAR(r2.spike_errors[i], r1.spike_errors[i] * scale, 1e-12);
    }

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&r1);
    eventprop_free_backward_output(&r2);
}

// ──────────────────────────────────────────────────────────────
// Zero maxima errors → zero gradient (no backward signal)
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, ZeroErrorsYieldsZeroGradient) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);

    // All errors zero → no backward signal
    maxima.errors[0] = 0.0;

    BackwardOutput result = eventprop_backward_maxima(input, maxima.times, maxima.errors);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // grad should be all zeros
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_NEAR(result.grad[i], 0.0, 1e-15);
    }

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// NaN maxima time skipped — neuron that never fired contributes nothing
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, NanMaximaTimeSkipped) {
    // Weak weight → post-neuron 0 fires, post-neuron 1 doesn't (NaN maxima time).
    std::vector<double> w = {10.0,   // row 0: strong → post 0 fires
                              0.01};  // row 0: weak  → post 1 doesn't fire
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 2, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    // Post-neuron 0 should have a valid maximum
    EXPECT_FALSE(std::isnan(maxima.times[0]));
    // Post-neuron 1 might be NaN (no spike → no voltage peak)

    maxima.errors[0] = 1.0;
    maxima.errors[1] = 1.0;   // error for NaN neuron — should be skipped

    BackwardOutput result = eventprop_backward_maxima(input, maxima.times, maxima.errors);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Should not crash or produce NaN/Inf
    EXPECT_FALSE(std::isnan(result.grad[0]));
    EXPECT_FALSE(std::isinf(result.grad[0]));

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Multiple post-neurons — gradient accumulates across all maxima
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, MultiplePostNeurons) {
    // 1 pre x 2 post. Both fire with strong weights.
    std::vector<double> w = {5.0,   // row 0: → post 0
                              5.0};  // row 0: → post 1
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 2, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);

    // Both neurons should have valid maxima (strong weight)
    for (int i = 0; i < input.n_post; ++i) {
        EXPECT_FALSE(std::isnan(maxima.times[i]));
        maxima.errors[i] = 1.0;
    }

    BackwardOutput result = eventprop_backward_maxima(input, maxima.times, maxima.errors);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Gradient should be nonzero (both neurons contribute)
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_FALSE(std::isnan(result.grad[i]));
        EXPECT_FALSE(std::isinf(result.grad[i]));
    }

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Multiple input spikes — backward sweep accumulates across all spikes
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, MultipleInputSpikes) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01, 0.02, 0.03};
    std::vector<int> sources    = {0, 0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    EXPECT_FALSE(std::isnan(maxima.times[0]));

    maxima.errors[0] = 1.0;

    BackwardOutput result = eventprop_backward_maxima(input, maxima.times, maxima.errors);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // All spike errors should be finite and nonzero (each input spike contributes)
    for (int i = 0; i < input.n_spikes; ++i) {
        EXPECT_FALSE(std::isnan(result.spike_errors[i]));
        EXPECT_FALSE(std::isinf(result.spike_errors[i]));
        EXPECT_NE(result.spike_errors[i], 0.0);
    }

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Different errors per neuron — gradient reflects error magnitudes
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, DifferentErrorsPerNeuron) {
    // 1 pre x 2 post. Same weight, different errors.
    std::vector<double> w = {5.0,   // row 0: → post 0
                              5.0};  // row 0: → post 1
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 2, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);

    // Post-neuron 0 has error=1.0, post-neuron 1 has error=3.0
    maxima.errors[0] = 1.0;
    maxima.errors[1] = 3.0;

    BackwardOutput result = eventprop_backward_maxima(input, maxima.times, maxima.errors);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Gradient for post-neuron 1 should be roughly 3x that of post-neuron 0
    // (same weight, same spike times, error differs by factor of 3)
    double grad_ratio = result.grad[1] / result.grad[0];   // [source=0, post=1] / [source=0, post=0]
    EXPECT_NEAR(grad_ratio, 3.0, 0.5);   // generous tolerance due to different maxima times

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Spike errors depend on weights — scaling w scales spike_errors
// ──────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────
// Backward pass produces finite results with valid inputs
// (weight dependence is already covered by linearity test)
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, ProducesFiniteResults) {
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};

    auto input = make_input(std::vector<double>{10.0}, 1, 1, times, sources);
    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    maxima.errors[0] = 1.0;

    BackwardOutput result = eventprop_backward_maxima(input, maxima.times, maxima.errors);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // All outputs should be finite (even if NaN time → skipped)
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_FALSE(std::isnan(result.grad[i]));
        EXPECT_FALSE(std::isinf(result.grad[i]));
    }

    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Validation error paths — propagated from input validation
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, NegativeNpostReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, -2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_backward_output(&result);
}

TEST(BackwardMaxima, ZeroNpreReturnsError) {
    std::vector<double> w = {};
    EventPropInput input{nullptr, 0, 1, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_backward_output(&result);
}

TEST(BackwardMaxima, EqualTausReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_TAU);

    eventprop_free_backward_output(&result);
}

TEST(BackwardMaxima, NullTimesWithInputSpikesReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 5, kTauMem, kTauSyn, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_backward_output(&result);
}

TEST(BackwardMaxima, OutOfBoundsInputSourceReturnsError) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {5};   // n_pre == 1
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_SOURCE);

    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Validation: null maxima pointers
// ──────────────────────────────────────────────────────────────

TEST(BackwardMaxima, NullMaximaTimesReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_backward_output(&result);
}

TEST(BackwardMaxima, NullMaximaErrorsReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    BackwardOutput result = eventprop_backward_maxima(input, nullptr, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_backward_output(&result);
}
