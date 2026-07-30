/**
 * @file compute_spikes.cpp
 * @brief Tests for eventprop_compute_spikes (forward pass).
 *
 * The LIF neuron fires when membrane potential crosses v_th.
 * We test observable contracts rather than exact spike times,
 * which depend on the root-finding solver tolerance.
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
// Zero-spikes edge case
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, ZeroInputSpikesReturnsZeroOutput) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    EXPECT_EQ(spikes.n_spikes, 0);
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// Single spike — strong weight fires, weak weight doesn't
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, StrongWeightProducesSpike) {
    std::vector<double> w = {10.0};   // strong weight
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    EXPECT_GT(spikes.n_spikes, 0);   // at least one spike fired
    for (int i = 0; i < spikes.n_spikes; ++i) {
        EXPECT_GE(spikes.times[i], 0.0);       // times are non-negative
        EXPECT_EQ(spikes.sources[i], 0);        // only post-neuron is 0
    }
    eventprop_free_spike_output(spikes);
}

TEST(ComputeSpikes, WeakWeightProducesNoSpike) {
    std::vector<double> w = {0.01};   // tiny weight — won't cross threshold
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    EXPECT_EQ(spikes.n_spikes, 0);
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// Multiple inputs accumulate to fire
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, SubThresholdInputsAccumulateToFire) {
    // Two spikes from same source — each alone is sub-threshold but together they fire.
    std::vector<double> w = {5.0};   // moderate weight
    std::vector<double> times   = {0.01, 0.02};
    std::vector<int> sources    = {0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    // With two inputs at moderate weight, should fire at least once.
    EXPECT_GT(spikes.n_spikes, 0);
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// Output spike times are sorted and positive
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, OutputTimesAreSortedAndPositive) {
    std::vector<double> w = {5.0};
    std::vector<double> times   = {0.01, 0.02, 0.03};
    std::vector<int> sources    = {0, 0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    for (int i = 0; i < spikes.n_spikes; ++i) {
        EXPECT_GT(spikes.times[i], 0.0);
        if (i > 0) {
            EXPECT_GT(spikes.times[i], spikes.times[i - 1]);   // strictly increasing
        }
    }
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// Output sources are valid post-neuron indices
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, OutputSourcesWithinBounds) {
    // 2 pre x 3 post — output sources should be in [0, 3).
    std::vector<double> w = {5.0, 5.0, 5.0,   // row 0: pre-neuron 0 → posts 0,1,2
                              5.0, 5.0, 5.0};  // row 1: pre-neuron 1 → posts 0,1,2
    std::vector<double> times   = {0.01, 0.02};
    std::vector<int> sources    = {0, 1};      // from both pre-neurons
    auto input = make_input(w, 2, 3, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    for (int i = 0; i < spikes.n_spikes; ++i) {
        EXPECT_GE(spikes.sources[i], 0);
        EXPECT_LT(spikes.sources[i], 3);   // n_post == 3
    }
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// Parameter sensitivity — higher threshold reduces spikes
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, HigherThresholdReducesSpikeCount) {
    std::vector<double> w = {5.0};
    std::vector<double> times   = {0.01, 0.02, 0.03};
    std::vector<int> sources    = {0, 0, 0};

    EventPropInput inputLow{w.data(), 1, 1, times.data(), sources.data(),
                            (int)times.size(), kTauMem, kTauSyn, 0.5};   // low threshold
    EventPropInput inputHigh{w.data(), 1, 1, times.data(), sources.data(),
                             (int)times.size(), kTauMem, kTauSyn, 2.0};  // high threshold

    SpikeOutput spikesLow = eventprop_compute_spikes(inputLow);
    SpikeOutput spikesHigh = eventprop_compute_spikes(inputHigh);

    EXPECT_EQ(spikesLow.error_no, EVENTPROP_OK);
    EXPECT_EQ(spikesHigh.error_no, EVENTPROP_OK);
    // Higher threshold should produce ≤ as many spikes (monotonicity)
    EXPECT_LE(spikesHigh.n_spikes, spikesLow.n_spikes);

    eventprop_free_spike_output(spikesLow);
    eventprop_free_spike_output(spikesHigh);
}

// ──────────────────────────────────────────────────────────────
// Single post-neuron works correctly
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, SinglePostNeuronWorks) {
    std::vector<double> w = {10.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    EXPECT_GT(spikes.n_spikes, 0);
    for (int i = 0; i < spikes.n_spikes; ++i) {
        EXPECT_EQ(spikes.sources[i], 0);   // only one post-neuron exists
    }
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// All input spikes from same source neuron
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, AllSpikesFromSameSource) {
    std::vector<double> w = {8.0};
    std::vector<double> times   = {0.01, 0.02, 0.03, 0.04};
    std::vector<int> sources    = {0, 0, 0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    EXPECT_GT(spikes.n_spikes, 0);
    for (int i = 0; i < spikes.n_spikes; ++i) {
        EXPECT_EQ(spikes.sources[i], 0);
    }
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// Validation error propagation from compute_sums
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, EqualTausPropagatesError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         kTauMem, kTauMem, kVTh};   // tau_mem == tau_syn

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_INVALID_TAU);
    EXPECT_EQ(spikes.n_spikes, 0);
    eventprop_free_spike_output(spikes);
}

TEST(ComputeSpikes, NegativeNpostPropagatesError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, -2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_INVALID_DIMENSION);
    EXPECT_EQ(spikes.n_spikes, 0);
    eventprop_free_spike_output(spikes);
}

TEST(ComputeSpikes, OutOfBoundsSourcePropagatesError) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {5};   // n_pre == 1
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_INVALID_SOURCE);
    EXPECT_EQ(spikes.n_spikes, 0);
    eventprop_free_spike_output(spikes);
}
