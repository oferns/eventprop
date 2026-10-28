/**
 * @file backward_spikes.cpp
 * @brief Tests for eventprop_backward_spikes (backward pass, spike loss).
 *
 * The backward sweep propagates post-synaptic errors back through the LIF
 * dynamics to compute weight gradients and input spike errors. We test
 * observable contracts: linearity in errors, source-row locality, and
 * algebraic scaling — not exact gradient values (which depend on solver
 * tolerance).
 *
 * Two arrays the pass used to carry inside SpikeOutput are caller-owned or
 * derived now: the post-spike derivatives are passed in as an argument, and
 * the synaptic current at a spike time is rebuilt inside the sweep from the
 * input spikes. Both changes are covered here.
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include <cmath>
#include <limits>
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

// The post-spike derivatives are caller-owned: the library no longer hands
// back an array that only the caller writes. Tests fill one of these.
static std::vector<double> make_post_errors(const SpikeOutput& fwd, double value) {
    return std::vector<double>((size_t)fwd.n_spikes, value);
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

    BackwardOutput result = eventprop_backward_spikes(input, spike, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_OK);
    // grad should be all zeros (calloc'd)
    EXPECT_DOUBLE_EQ(result.grad[0], 0.0);
    // spike_errors has input.n_spikes entries, all zero
    EXPECT_NEAR(result.spike_errors[0], 0.0, 1e-15);

    eventprop_free_backward_output(result);
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

    // Unit loss at every output spike
    const auto errs = make_post_errors(fwd, 1.0);

    BackwardOutput result = eventprop_backward_spikes(input, fwd, errs.data());
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

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(result);
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

    const auto errs1 = make_post_errors(fwd, 1.0);
    BackwardOutput r1 = eventprop_backward_spikes(input, fwd, errs1.data());

    // Run with errors = 3.0
    const double scale = 3.0;
    const auto errs2 = make_post_errors(fwd, scale);
    BackwardOutput r2 = eventprop_backward_spikes(input, fwd, errs2.data());

    EXPECT_EQ(r1.error_no, EVENTPROP_OK);
    EXPECT_EQ(r2.error_no, EVENTPROP_OK);

    // grad and spike_errors should scale linearly
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_NEAR(r2.grad[i], r1.grad[i] * scale, 1e-12);
    }
    for (int i = 0; i < input.n_spikes; ++i) {
        EXPECT_NEAR(r2.spike_errors[i], r1.spike_errors[i] * scale, 1e-12);
    }

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(r1);
    eventprop_free_backward_output(r2);
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
    const auto errs = make_post_errors(fwd, 0.0);

    BackwardOutput result = eventprop_backward_spikes(input, fwd, errs.data());
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // grad should be all zeros
    for (int i = 0; i < input.n_pre * input.n_post; ++i) {
        EXPECT_NEAR(result.grad[i], 0.0, 1e-15);
    }

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(result);
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

    const auto errs = make_post_errors(fwd, 1.0);

    BackwardOutput result = eventprop_backward_spikes(input, fwd, errs.data());
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Row 1 was never a source → its gradient should be zero
    for (int col = 0; col < input.n_post; ++col) {
        EXPECT_NEAR(result.grad[1 * input.n_post + col], 0.0, 1e-15);
    }

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(result);
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
    const auto errs1 = make_post_errors(fwd1, 1.0);

    std::vector<double> w2 = {10.0};   // double the weight
    auto input2 = make_input(w2, 1, 1, times, sources);
    SpikeOutput fwd2 = eventprop_compute_spikes(input2);
    const auto errs2 = make_post_errors(fwd2, 1.0);

    BackwardOutput r1 = eventprop_backward_spikes(input1, fwd1, errs1.data());
    BackwardOutput r2 = eventprop_backward_spikes(input2, fwd2, errs2.data());

    EXPECT_EQ(r1.error_no, EVENTPROP_OK);
    EXPECT_EQ(r2.error_no, EVENTPROP_OK);

    // Spike errors should differ (outbound signal depends on w)
    for (int i = 0; i < input1.n_spikes; ++i) {
        EXPECT_NE(r1.spike_errors[i], r2.spike_errors[i]);
    }

    eventprop_free_spike_output(fwd1);
    eventprop_free_spike_output(fwd2);
    eventprop_free_backward_output(r1);
    eventprop_free_backward_output(r2);
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

    const auto errs = make_post_errors(fwd, 1.0);
    for (int i = 0; i < fwd.n_spikes; ++i) {
        EXPECT_EQ(fwd.sources[i], 0);   // all from same post-neuron
    }

    BackwardOutput result = eventprop_backward_spikes(input, fwd, errs.data());
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // All outputs should be finite and nonzero
    EXPECT_FALSE(std::isnan(result.grad[0]));
    EXPECT_FALSE(std::isinf(result.grad[0]));
    EXPECT_NE(result.grad[0], 0.0);

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(result);
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
    }
    EXPECT_TRUE(post0Fired);

    const auto errs = make_post_errors(fwd, 1.0);

    BackwardOutput result = eventprop_backward_spikes(input, fwd, errs.data());
    EXPECT_EQ(result.error_no, EVENTPROP_OK);

    // Gradient for the firing neuron should be nonzero
    EXPECT_NE(result.grad[0], 0.0);   // grad[source=0, post=0]

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(result);
}

// ──────────────────────────────────────────────────────────────
// The synaptic current is derived, not stored: the same spikes handed back
// with different post-spike times must produce currents consistent with the
// input spikes. This fixture pins the derived value against a hand copy of
// the identity the sweep uses.
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, CurrentIsDerivedFromInputSpikes) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    // Rebuild the current the sweep derives: the input spike at 0.01 is the
    // only one at or before the post-spike time, so the current is
    // w * exp(t_in / tau_syn) * exp(-t_post / tau_syn).
    const double cur = (w[0] * std::exp(times[0] / kTauSyn)) *
                       std::exp(-fwd.times[0] / kTauSyn);
    EXPECT_GT(cur, kVTh);   // a fired spike is above threshold, so inv_ic is finite

    const auto errs = make_post_errors(fwd, 1.0);
    BackwardOutput result = eventprop_backward_spikes(input, fwd, errs.data());
    EXPECT_EQ(result.error_no, EVENTPROP_OK);
    EXPECT_FALSE(std::isnan(result.grad[0]));
    EXPECT_FALSE(std::isinf(result.grad[0]));

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(result);
}

// ──────────────────────────────────────────────────────────────
// Validation error paths — propagated from input validation
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, NegativeNpostReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, -2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_backward_output(result);
}

TEST(BackwardSpikes, ZeroNpreReturnsError) {
    std::vector<double> w = {};
    EventPropInput input{nullptr, 0, 1, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_backward_output(result);
}

TEST(BackwardSpikes, EqualTausReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_TAU);

    eventprop_free_backward_output(result);
}

TEST(BackwardSpikes, NullTimesWithInputSpikesReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 5, kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_backward_output(result);
}

TEST(BackwardSpikes, OutOfBoundsInputSourceReturnsError) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {5};   // n_pre == 1
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    SpikeOutput spike{};
    spike.n_spikes = 0;

    BackwardOutput result = eventprop_backward_spikes(input, spike, nullptr);
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_SOURCE);

    eventprop_free_backward_output(result);
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

    const auto errs = make_post_errors(fwd, 1.0);
    BackwardOutput result = eventprop_backward_spikes(input, fwd, errs.data());
    EXPECT_EQ(result.error_no, EVENTPROP_INVALID_SOURCE);

    eventprop_free_spike_output(fwd);
    eventprop_free_backward_output(result);
}

// ──────────────────────────────────────────────────────────────
// Validation: NULL output spike arrays with n_spikes > 0, including the
// caller-owned derivative array
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, NullOutputSpikeArraysReturnError) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput fwd = eventprop_compute_spikes(input);
    EXPECT_EQ(fwd.error_no, EVENTPROP_OK);
    EXPECT_GT(fwd.n_spikes, 0);

    const auto errs = make_post_errors(fwd, 1.0);

    // Each array must be non-NULL when n_spikes > 0 — NULL → EVENTPROP_NULL_POINTER,
    // not a segfault.
    double* saved_times   = fwd.times;
    int*    saved_sources = fwd.sources;

    fwd.times = nullptr;
    EXPECT_EQ(eventprop_backward_spikes(input, fwd, errs.data()).error_no, EVENTPROP_NULL_POINTER);
    fwd.times = saved_times;

    fwd.sources = nullptr;
    EXPECT_EQ(eventprop_backward_spikes(input, fwd, errs.data()).error_no, EVENTPROP_NULL_POINTER);
    fwd.sources = saved_sources;

    // The derivative array is the third thing that must not be NULL.
    EXPECT_EQ(eventprop_backward_spikes(input, fwd, nullptr).error_no, EVENTPROP_NULL_POINTER);

    // Sanity: with everything restored the call succeeds again.
    BackwardOutput ok = eventprop_backward_spikes(input, fwd, errs.data());
    EXPECT_EQ(ok.error_no, EVENTPROP_OK);
    eventprop_free_backward_output(ok);

    eventprop_free_spike_output(fwd);
}

// ──────────────────────────────────────────────────────────────
// Validation: post spike times must be non-decreasing too — the currents are
// rebuilt with one forward cursor over the post spikes, so their order is part
// of the contract now
// ──────────────────────────────────────────────────────────────

TEST(BackwardSpikes, UnsortedOutputSpikeTimesReturnsError) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    std::vector<double> post_times    = {0.02, 0.01};   // descending
    std::vector<int> post_sources     = {0, 0};
    std::vector<double> post_errors(2, 1.0);
    SpikeOutput fwd{EVENTPROP_OK, 2, post_times.data(), post_sources.data()};

    BackwardOutput result = eventprop_backward_spikes(input, fwd, post_errors.data());
    EXPECT_EQ(result.error_no, EVENTPROP_UNSORTED_SPIKE_TIMES);
    EXPECT_EQ(result.grad, nullptr);
    EXPECT_EQ(result.spike_errors, nullptr);
}

// The shared post-spike validation (eventprop_validate_post_spikes, also used by both diagnostic
// traces) carries two more rules this call site used to skip: a post-spike time may not be
// negative or non-finite. Both previously reached the adjoint sweep and came out the other side as
// a NaN gradient reported with EVENTPROP_OK — a wrong number, not a refusal.
//
// The negative time is first in the list so the ordering check is not what fires, and equal/normal
// times follow it.
TEST(BackwardSpikes, NegativeOutputSpikeTimeReturnsError) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    std::vector<double> post_times    = {-0.001, 0.02};   // ascending, sign is the violation
    std::vector<int> post_sources     = {0, 0};
    std::vector<double> post_errors(2, 1.0);
    SpikeOutput fwd{EVENTPROP_OK, 2, post_times.data(), post_sources.data()};

    BackwardOutput result = eventprop_backward_spikes(input, fwd, post_errors.data());
    EXPECT_EQ(result.error_no, EVENTPROP_NEGATIVE_SPIKE_TIME);
    EXPECT_EQ(result.grad, nullptr);
}

TEST(BackwardSpikes, NonFiniteOutputSpikeTimeReturnsError) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    std::vector<double> post_times    = {0.02, std::numeric_limits<double>::quiet_NaN()};
    std::vector<int> post_sources     = {0, 0};
    std::vector<double> post_errors(2, 1.0);
    SpikeOutput fwd{EVENTPROP_OK, 2, post_times.data(), post_sources.data()};

    BackwardOutput result = eventprop_backward_spikes(input, fwd, post_errors.data());
    EXPECT_EQ(result.error_no, EVENTPROP_NONFINITE_SPIKE_TIME);
    EXPECT_EQ(result.grad, nullptr);
}
