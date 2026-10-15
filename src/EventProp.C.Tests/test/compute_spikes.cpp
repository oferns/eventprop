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

// ──────────────────────────────────────────────────────────────
// Input spike times must be non-decreasing.
//
// The prefix sums are built in array order and then indexed by position as if
// by time (sum0[k][nrn] = "all input spikes up to times[k]"), so an unsorted
// array is not a different-but-valid input — it is a different network. The
// Eigen reference has the same requirement and does not check, so it returns
// plausible-looking garbage; the C library reports it instead.
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, UnsortedTimesRejected) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> times = {0.05, 0.02};   // descending
    std::vector<int> sources  = {0, 1};
    EventPropInput input{w.data(), 2, 2, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_UNSORTED_SPIKE_TIMES);
    EXPECT_EQ(spikes.n_spikes, 0);
    eventprop_free_spike_output(spikes);
}

TEST(ComputeSpikes, EqualTimesAccepted) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> times = {0.05, 0.05, 0.08};   // non-decreasing is valid
    std::vector<int> sources  = {0, 1, 0};
    EventPropInput input{w.data(), 2, 2, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    eventprop_free_spike_output(spikes);
}

// ──────────────────────────────────────────────────────────────
// Last-interval bracket floor
//
// sum0[i]/sum1[i] include every input spike up to times[i - 1], so the voltage
// model of interval i is only valid for t >= times[i - 1]. The bracket floor
// must therefore be max(times[i - 1], post spikes), including in the final
// interval. Without the times[i - 1] term the floor collapses to 0.0 when the
// neuron has not fired yet, the `t_before <= tmax` guard passes for a peak that
// lies behind the interval, and the solver brackets a region where the model is
// an extrapolation of a train that has not arrived — reporting roots at times
// where the real membrane potential sits well under v_th.
// ──────────────────────────────────────────────────────────────

TEST(ComputeSpikes, LastIntervalPeakBehindLastInputTimeNoSpuriousSpike) {
    // Trained YinYang hidden-layer column (figure2 checkpoint, neuron 177).
    // The analytical peak of the final interval is tmax = 0.017081371, which
    // lies behind times[4] = 0.022977101; the true voltage never exceeds
    // 0.960201051, so this neuron must not fire at all.
    std::vector<double> w       = {2.410547, 1.669236, -0.549479, 1.582348, 1.501614};
    std::vector<double> times   = {0.0, 0.007022899, 0.012051749, 0.017948251, 0.022977101};
    std::vector<int>    sources = {4, 0, 1, 3, 2};
    EventPropInput input = make_input(w, 5, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    EXPECT_EQ(spikes.n_spikes, 0)
        << "spurious last-interval root at t="
        << (spikes.n_spikes > 0 ? spikes.times[0] : -1.0);
    eventprop_free_spike_output(spikes);
}

TEST(ComputeSpikes, LastIntervalGenuineSpikeStillFound) {
    // Same input shape, weights scaled so the final interval genuinely crosses
    // v_th: the peak sits at 0.023057647, past times[4] = 0.022977101, and the
    // voltage reaches 1.001664. The bracket floor must not suppress it.
    std::vector<double> w       = {3.454698, 3.811084, 1.134205, 0.537015, 3.865112};
    std::vector<double> times   = {0.0, 0.007022899, 0.012051749, 0.017948251, 0.022977101};
    std::vector<int>    sources = {4, 0, 1, 3, 2};
    EventPropInput input = make_input(w, 5, 1, times, sources);

    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);

    bool found_last_interval_spike = false;
    for (int i = 0; i < spikes.n_spikes; i++) {
        if (spikes.times[i] > times.back()) found_last_interval_spike = true;
    }
    EXPECT_TRUE(found_last_interval_spike)
        << "n_spikes=" << spikes.n_spikes << ", last input time = " << times.back();
    eventprop_free_spike_output(spikes);
}
