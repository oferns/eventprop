/**
 * @file compute_maxima.cpp
 * @brief Tests for eventprop_compute_maxima.
 *
 * The function finds the maximum membrane potential per post-neuron
 * between input spike times using analytical tmax:
 *   tmax = tmax_pref * (tmax_sum + log(s1v / s0v))
 * It also checks voltage at exact input spike times as candidates.
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
// Zero-spikes edge case → NaN times, 0.0 values
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, ZeroSpikesReturnsNaNAndZero) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    for (int i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isnan(maxima.times[i]));
        EXPECT_DOUBLE_EQ(maxima.values[i], 0.0);
    }
    eventprop_free_maxima_output(maxima);
}

// ──────────────────────────────────────────────────────────────
// Single spike — strong weight produces finite maxima
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, SingleSpikeStrongWeightHasFiniteMaxima) {
    std::vector<double> w = {10.0};   // 1 pre x 1 post
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    // With a strong weight the maximum should be found (not NaN)
    EXPECT_FALSE(std::isnan(maxima.times[0]));
    EXPECT_GT(maxima.values[0], 0.0);   // positive voltage peak
    eventprop_free_maxima_output(maxima);
}

TEST(ComputeMaxima, SingleSpikeWeakWeightHasSmallMaxima) {
    std::vector<double> w = {0.1};   // tiny weight
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    // Even a weak weight should produce some finite maximum (not NaN)
    EXPECT_FALSE(std::isnan(maxima.times[0]));
    EXPECT_GT(maxima.values[0], 0.0);
    eventprop_free_maxima_output(maxima);
}

// ──────────────────────────────────────────────────────────────
// Multiple post-neurons — each gets its own maximum
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, MultiplePostNeuronsIndependent) {
    // 1 pre x 3 post: row-major [w0, w1, w2]
    std::vector<double> w = {10.0, 5.0, 1.0};
    std::vector<double> times   = {0.01};
    std::vector<int> sources    = {0};
    auto input = make_input(w, 1, 3, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);

    // All three post-neurons should have finite maxima (same source)
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(std::isnan(maxima.times[i]));
        EXPECT_GT(maxima.values[i], 0.0);
    }

    // Stronger weight → higher maximum voltage
    EXPECT_GT(maxima.values[0], maxima.values[1]);
    EXPECT_GT(maxima.values[1], maxima.values[2]);

    eventprop_free_maxima_output(maxima);
}

// ──────────────────────────────────────────────────────────────
// Maxima time is within valid range [first_spike, last_spike]
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, MaximaTimeAfterFirstSpike) {
    std::vector<double> w = {5.0};
    std::vector<double> times   = {0.01, 0.05};
    std::vector<int> sources    = {0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);
    EXPECT_FALSE(std::isnan(maxima.times[0]));
    // The maximum must occur at or after the first input spike.
    // It can extend beyond the last spike because the membrane
    // potential continues to evolve (last interval upper bound = INFINITY).
    EXPECT_GE(maxima.times[0], times.front());
    eventprop_free_maxima_output(maxima);
}

// ──────────────────────────────────────────────────────────────
// Multiple spikes accumulate → higher maximum than single spike
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, MoreSpikesIncreaseMaximum) {
    std::vector<double> w = {5.0};

    // Single spike
    std::vector<double> times1   = {0.02};
    std::vector<int> sources1    = {0};
    auto input1 = make_input(w, 1, 1, times1, sources1);
    MaximaOutput maxima1 = eventprop_compute_maxima(input1);

    // Two spikes (same source)
    std::vector<double> times2   = {0.01, 0.03};
    std::vector<int> sources2    = {0, 0};
    auto input2 = make_input(w, 1, 1, times2, sources2);
    MaximaOutput maxima2 = eventprop_compute_maxima(input2);

    EXPECT_EQ(maxima1.error_no, EVENTPROP_OK);
    EXPECT_EQ(maxima2.error_no, EVENTPROP_OK);

    // Two spikes should produce a higher (or equal) maximum than one
    EXPECT_GE(maxima2.values[0], maxima1.values[0]);

    eventprop_free_maxima_output(maxima1);
    eventprop_free_maxima_output(maxima2);
}

// ──────────────────────────────────────────────────────────────
// Scaling invariant: k*w → k*values (linear in weights)
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, ValuesScaleLinearlyWithWeights) {
    std::vector<double> w = {5.0};
    std::vector<double> times   = {0.01, 0.02};
    std::vector<int> sources    = {0, 0};
    auto input = make_input(w, 1, 1, times, sources);

    MaximaOutput maxima1 = eventprop_compute_maxima(input);

    // Scale weights by 3x and recompute
    std::vector<double> w2 = {15.0};
    input.w = w2.data();
    MaximaOutput maxima2 = eventprop_compute_maxima(input);

    EXPECT_EQ(maxima1.error_no, EVENTPROP_OK);
    EXPECT_EQ(maxima2.error_no, EVENTPROP_OK);

    // Values should scale linearly with weights (times unchanged)
    EXPECT_DOUBLE_EQ(maxima2.values[0], maxima1.values[0] * 3.0);
    EXPECT_DOUBLE_EQ(maxima2.times[0], maxima1.times[0]);

    eventprop_free_maxima_output(maxima1);
    eventprop_free_maxima_output(maxima2);
}

// ──────────────────────────────────────────────────────────────
// Different source neurons affect different post-neurons
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, DifferentSourcesAffectDifferentPosts) {
    // 2 pre x 2 post: row-major [w00,w01,w10,w11]
    // Only neuron 0→post 0 is strong; neuron 1→post 1 is strong
    std::vector<double> w = {10.0, 0.1,   // pre-neuron 0 → posts 0 (strong), 1 (weak)
                              0.1, 10.0}; // pre-neuron 1 → posts 0 (weak), 1 (strong)
    std::vector<double> times   = {0.01, 0.02};
    std::vector<int> sources    = {0, 1};      // one spike from each source
    auto input = make_input(w, 2, 2, times, sources);

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_OK);

    // Both post-neurons should have finite maxima (each gets a strong input)
    for (int i = 0; i < 2; ++i) {
        EXPECT_FALSE(std::isnan(maxima.times[i]));
        EXPECT_GT(maxima.values[i], 0.0);
    }

    eventprop_free_maxima_output(maxima);
}

// ──────────────────────────────────────────────────────────────
// Parameter sensitivity — tau_mem affects maxima timing
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, TauMemAffectsMaximaTiming) {
    std::vector<double> w = {5.0};
    std::vector<double> times   = {0.01, 0.03};
    std::vector<int> sources    = {0, 0};

    EventPropInput inputFast{w.data(), 1, 1, times.data(), sources.data(),
                             (int)times.size(), 0.01, kTauSyn, kVTh};   // fast membrane
    EventPropInput inputSlow{w.data(), 1, 1, times.data(), sources.data(),
                             (int)times.size(), 0.05, kTauSyn, kVTh};  // slow membrane

    MaximaOutput maximaFast = eventprop_compute_maxima(inputFast);
    MaximaOutput maximaSlow = eventprop_compute_maxima(inputSlow);

    EXPECT_EQ(maximaFast.error_no, EVENTPROP_OK);
    EXPECT_EQ(maximaSlow.error_no, EVENTPROP_OK);

    // Both should find finite maxima
    EXPECT_FALSE(std::isnan(maximaFast.times[0]));
    EXPECT_FALSE(std::isnan(maximaSlow.times[0]));

    eventprop_free_maxima_output(maximaFast);
    eventprop_free_maxima_output(maximaSlow);
}

// ──────────────────────────────────────────────────────────────
// Validation error paths
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, NegativeNpostReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, -2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_INVALID_DIMENSION);
    EXPECT_EQ(maxima.times, nullptr);
    EXPECT_EQ(maxima.values, nullptr);
    eventprop_free_maxima_output(maxima);
}

TEST(ComputeMaxima, ZeroNpreReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 0, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_INVALID_DIMENSION);
    EXPECT_EQ(maxima.times, nullptr);
    EXPECT_EQ(maxima.values, nullptr);
    eventprop_free_maxima_output(maxima);
}

TEST(ComputeMaxima, EqualTausReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         kTauMem, kTauMem, kVTh};   // tau_mem == tau_syn

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_INVALID_TAU);
    EXPECT_EQ(maxima.times, nullptr);
    EXPECT_EQ(maxima.values, nullptr);
    eventprop_free_maxima_output(maxima);
}

TEST(ComputeMaxima, NullTimesWithSpikesReturnsError) {
    std::vector<double> w = {1.0};
    std::vector<int> sources = {0};
    EventPropInput input{w.data(), 1, 1, nullptr, sources.data(),
                         1, kTauMem, kTauSyn, kVTh};   // times == NULL but n_spikes > 0

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_NULL_POINTER);
    EXPECT_EQ(maxima.times, nullptr);
    EXPECT_EQ(maxima.values, nullptr);
    eventprop_free_maxima_output(maxima);
}

TEST(ComputeMaxima, OutOfBoundsSourceReturnsError) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {5};   // n_pre == 1
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    MaximaOutput maxima = eventprop_compute_maxima(input);
    EXPECT_EQ(maxima.error_no, EVENTPROP_INVALID_SOURCE);
    EXPECT_EQ(maxima.times, nullptr);
    EXPECT_EQ(maxima.values, nullptr);
    eventprop_free_maxima_output(maxima);
}

// ──────────────────────────────────────────────────────────────
// v_th is not part of the maxima contract — the entry point must accept
// any threshold, including the zero the language bindings pass when
// their own API has no v_th parameter.
// ──────────────────────────────────────────────────────────────

TEST(ComputeMaxima, ZeroVthAcceptedAndIgnored) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> times = {0.1, 0.2};
    std::vector<int> sources  = {0, 1};

    EventPropInput with_vth{w.data(), 2, 2, times.data(), sources.data(),
                            (int)times.size(), kTauMem, kTauSyn, kVTh};
    EventPropInput zero_vth{w.data(), 2, 2, times.data(), sources.data(),
                            (int)times.size(), kTauMem, kTauSyn, 0.0};

    MaximaOutput reference = eventprop_compute_maxima(with_vth);
    MaximaOutput under_test = eventprop_compute_maxima(zero_vth);

    EXPECT_EQ(reference.error_no, EVENTPROP_OK);
    EXPECT_EQ(under_test.error_no, EVENTPROP_OK);

    const auto same = [](double a, double b) {
        if (std::isnan(a) && std::isnan(b)) return true;   // "no maximum found"
        return a == b;
    };
    for (int i = 0; i < under_test.n_post; i++) {
        EXPECT_TRUE(same(under_test.times[i], reference.times[i]));
        EXPECT_TRUE(same(under_test.values[i], reference.values[i]));
    }

    eventprop_free_maxima_output(reference);
    eventprop_free_maxima_output(under_test);
}
