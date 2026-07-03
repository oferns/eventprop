/**
 * @file backward_spikes.cpp
 * @brief Tests for eventprop_backward_spikes (backward pass, spike loss).
 *
 * The backward sweep propagates post-synaptic errors back through the LIF
 * dynamics to compute weight gradients and input spike errors. We test
 * observable contracts: linearity in errors, source-row locality, and
 * algebraic scaling — not exact gradient values (which depend on solver tolerance).
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
// Zero output spikes — early return path
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, ZeroOutputSpikesReturnsZeroGradient) {
    std::vector<double> w = {1.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    // Forward pass produces spikes, but we pass an empty SpikeOutput manually.
    SpikeOutput spike{};
    spike.n_spikes = 0;
    spike.times = nullptr;
    spike.sources = nullptr;
    spike.currents = nullptr;
    spike.errors = nullptr;

    BackwardOutput result = eventprop_backward_spikes(input, spike);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);
    // grad should be all zeros (calloc'd)
    EXPECT_DOUBLE_EQ(result.grad[0], 0.0);
    // spike_errors has input.n_spikes entries, all zero
    EXPECT_NEAR(result.spike_errors[0], 0.0, 1e-15);

    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Single input + single output spike — simplest non-trivial case
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, SingleInputSingleOutputSpike) {
    std::vector<double> w = {10.0};   // strong weight → fires
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    // Set post-synaptic error to 1.0 (unit loss)
    for (int i = 0; i < fwd.n_spikes; ++i) {
        fwd.errors[i] = 1.0;
    }

    BackwardOutput result = eventprop_backward_spikes(input, fwd);
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

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Linearity in post errors — scaling errors scales output proportionally
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, LinearInPostErrors) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01, 0.02};
    std::vector<int> sources    = {0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    // Run with errors = 1.0
    for (int i = 0; i < fwd.n_spikes; ++i) fwd.errors[i] = 1.0;
    BackwardOutput r1 = eventprop_backward_spikes(input, fwd);

    // Run with errors = 3.0
    double scale = 3.0;
    for (int i = 0; i < fwd.n_spikes; ++i) fwd.errors[i] = scale;
    BackwardOutput r2 = eventprop_backward_spikes(input, fwd);

    EXPECT_EQ(r1.error_no, EVENTPROP_OK);
    EXPECT_EQ(r2.error_no, EVENTPROP_OK);

    // grad and spike_errors should scale linearly
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_NEAR(r2.grad[i], r1.grad[i] * scale, 1e-12);
    }
    for (int i = 0; i < input.n_spikes; ++i) {
        EXPECT_NEAR(r2.spike_errors[i], r1.spike_errors[i] * scale, 1e-12);
    }

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&r1);
    eventprop_free_backward_output(&r2);
}

// ──────────────────────────────────────────────────────────────
// Zero post errors → zero gradient (no backward signal)
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, ZeroPostErrorsYieldsZeroGradient) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    // All errors zero → no backward signal
    for (int i = 0; i < fwd.n_spikes; ++i) fwd.errors[i] = 0.0;

    BackwardOutput result = eventprop_backward_spikes(input, fwd);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // grad should be all zeros
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_NEAR(result.grad[i], 0.0, 1e-15);
    }

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Only source rows touched — input spike from source k only affects row k
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, OnlySourceRowsTouched) {
    // 3 pre x 1 post. Input spikes come from sources 0 and 2 only.
    std::vector<double> w = {5.0,   // row 0: source 0 → post 0
                              5.0,   // row 1: source 1 → post 0 (unused)
                              5.0};  // row 2: source 2 → post 0
    std::vector<double> times   = {0.01, 0.03};
    std::vector<int> sources    = {0, 2};   // only rows 0 and 2 touched
    auto input = make_input(w, 3, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    for (int i = 0; i < fwd.n_spikes; ++i) fwd.errors[i] = 1.0;

    BackwardOutput result = eventprop_backward_spikes(input, fwd);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Row 1 was never a source → its gradient should be zero
    for (int col = 0; col < input.n_post; ++col) {
        EXPECT_NEAR(result.grad[1 * input.n_post + col], 0.0, 1e-15);
    }

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Spike errors depend on weights — scaling w scales spike_errors
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, SpikeErrorsDependOnWeights) {
    std::vector<double> w1 = {5.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};

    auto input1 = make_input(w1, 1, 1, times, sources);
    SpikeOutput fwd1 = eventprop_compute_spikes(input1);
    for (int i = 0; i < fwd1.n_spikes; ++i) fwd1.errors[i] = 1.0;

    std::vector<double> w2 = {10.0};   // double the weight
    auto input2 = make_input(w2, 1, 1, times, sources);
    SpikeOutput fwd2 = eventprop_compute_spikes(input2);
    for (int i = 0; i < fwd2.n_spikes; ++i) fwd2.errors[i] = 1.0;

    BackwardOutput r1 = eventprop_backward_spikes(input1, fwd1);
    BackwardOutput r2 = eventprop_backward_spikes(input2, fwd2);

    EXPECT_EQ(r1.error_no, EVENTPROP_OK);
    EXPECT_EQ(r2.error_no, EVENTPROP_OK);

    // Spike errors should differ (outbound signal depends on w)
    for (int i = 0; i < input1.n_spikes; ++i) {
        EXPECT_NE(r1.spike_errors[i], r2.spike_errors[i]);
    }

    eventprop_free_spike_output(&fwd1);
    eventprop_free_spike_output(&fwd2);
    eventprop_free_backward_output(&r1);
    eventprop_free_backward_output(&r2);
}

// ──────────────────────────────────────────────────────────────
// Multiple post spikes from same neuron — jump accumulation
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, MultiplePostSpikesSameNeuron) {
    // Strong weight → multiple output spikes from the same post-neuron.
    std::vector<double> w = {15.0};
    std::vector<double> times   = {0.01, 0.02, 0.03};
    std::vector<int> sources    = {0, 0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    // With strong weight and multiple inputs, should fire at least twice.
    EXPECT_GT(fwd.n_spikes, 1);

    for (int i = 0; i < fwd.n_spikes; ++i) {
        EXPECT_EQ(fwd.sources[i], 0);   // all from same post-neuron
        fwd.errors[i] = 1.0;
    }

    BackwardOutput result = eventprop_backward_spikes(input, fwd);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // All outputs should be finite and nonzero
    EXPECT_FALSE(std::isnan(result.grad[0]));
    EXPECT_FALSE(std::isinf(result.grad[0]));
    EXPECT_NE(result.grad[0], 0.0);

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Multiple post-neurons — independent backward sweeps per neuron
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, MultiplePostNeuronsIndependent) {
    // 1 pre x 2 post. Only post-neuron 0 fires (strong weight), post-neuron 1 doesn't (weak).
    std::vector<double> w = {10.0,   // row 0: strong → post 0 fires
                              0.01};  // row 0: weak  → post 1 doesn't fire
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 2, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);

    // Only post-neuron 0 should have fired.
    bool post0Fired = false;
    for (int i = 0; i < fwd.n_spikes; ++i) {
        if (fwd.sources[i] == 0) post0Fired = true;
        fwd.errors[i] = 1.0;
    }
    EXPECT_TRUE(post0Fired);

    BackwardOutput result = eventprop_backward_spikes(input, fwd);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Gradient for the firing neuron should be nonzero
    EXPECT_NE(result.grad[0], 0.0);   // grad[source=0, post=0]

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Validation error paths — propagated from input validation
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, NegativeNpostReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, -2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_backward_output(&result);
}

TEST(BackwardSpikes, ZeroNpreReturnsError) {
    std::vector<double> w = {};
    EventPropInput input{nullptr, 0, 1, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_backward_output(&result);
}

TEST(BackwardSpikes, EqualTausReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_TAU);

    eventprop_free_backward_output(&result);
}

TEST(BackwardSpikes, NullTimesWithInputSpikesReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 5, kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike);
    EXPECT_EQ(result.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_backward_output(&result);
}

TEST(BackwardSpikes, OutOfBoundsInputSourceReturnsError) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {5};   // n_pre == 1
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_SOURCE);

    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Validation: output spike source bounds (post-sources index post-neurons)
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, OutOfBoundsOutputSourceReturnsError) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    // Corrupt a source to be out of bounds (>= n_post)
    fwd.sources[0] = 5;   // n_post == 1

    BackwardOutput result = eventprop_backward_spikes(input, fwd);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_SOURCE);

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&result);
}

// ──────────────────────────────────────────────────────────────
// Validation: current equals v_th → division by zero guard
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, CurrentEqualsVthReturnsError) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    // Set current to exactly v_th → would cause div-by-zero in inv_ic
    for (int i = 0; i < fwd.n_spikes; ++i) {
        fwd.currents[i] = kVTh;   // == input.v_th
        fwd.errors[i] = 1.0;
    }

    BackwardOutput result = eventprop_backward_spikes(input, fwd);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_CURRENT);

    eventprop_free_spike_output(&fwd);
    eventprop_free_backward_output(&result);
}
