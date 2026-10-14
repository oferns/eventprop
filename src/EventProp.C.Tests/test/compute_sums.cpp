/**
 * @file compute_sums.cpp
 * @brief Tests for eventprop_compute_sums.
 *
 * The function computes cumulative exponential-weighted sums:
 *   s0[i+1][j] = s0[i][j] + w[src_i][j] * exp(t_i / tau_syn)
 *   s1[i+1][j] = s1[i][j] + w[src_i][j] * exp(t_i / tau_mem)
 *
 * Weight matrix is row-major: w[row * n_post + col].
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

TEST(ComputeSums, ZeroSpikesReturnsSingleZeroRow) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SumsOutput sums = eventprop_compute_sums(input);
    EXPECT_EQ(sums.error_no, EVENTPROP_OK);
    EXPECT_EQ(sums.sum0[0], 0.0);
    EXPECT_EQ(sums.sum1[0], 0.0);
    eventprop_free_sums_output(sums);
}

// ──────────────────────────────────────────────────────────────
// Single spike — exact formula match
// ──────────────────────────────────────────────────────────────

TEST(ComputeSums, SingleSpikesExactFormula) {
    // 2 pre x 3 post weight matrix (row-major):
    //   row 0: [1, 2, 3]
    //   row 1: [4, 5, 6]
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int n_pre  = 2;
    int n_post = 3;

    std::vector<double> times   = {0.1};
    std::vector<int> sources    = {1};         // source neuron 1 -> weight row [4,5,6]
    auto input = make_input(w, n_pre, n_post, times, sources);

    SumsOutput sums = eventprop_compute_sums(input);

    // Row 0 is all zeros (initial state)
    for (int j = 0; j < n_post; ++j) {
        EXPECT_DOUBLE_EQ(sums.sum0[j], 0.0);
        EXPECT_DOUBLE_EQ(sums.sum1[j], 0.0);
    }

    // Row 1: w[src][j] * exp(t / tau), stride = n_post
    double e_syn = exp(0.1 / kTauSyn);
    double e_mem = exp(0.1 / kTauMem);
    for (int j = 0; j < n_post; ++j) {
        EXPECT_DOUBLE_EQ(sums.sum0[n_post + j], w[1 * n_post + j] * e_syn);
        EXPECT_DOUBLE_EQ(sums.sum1[n_post + j], w[1 * n_post + j] * e_mem);
    }

    eventprop_free_sums_output(sums);
}

// ──────────────────────────────────────────────────────────────
// Two spikes — additive accumulation
// ──────────────────────────────────────────────────────────────

TEST(ComputeSums, TwoSpikesDifferentSourcesAreAdditive) {
    // 2 pre x 2 post: row-major [w00,w01,w10,w11]
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0};
    int n_post = 2;

    std::vector<double> times   = {0.1, 0.2};
    std::vector<int> sources    = {0, 1};      // row 0 then row 1
    auto input = make_input(w, 2, n_post, times, sources);

    SumsOutput sums = eventprop_compute_sums(input);

    double e_syn_0 = exp(0.1 / kTauSyn), e_mem_0 = exp(0.1 / kTauMem);
    double e_syn_1 = exp(0.2 / kTauSyn), e_mem_1 = exp(0.2 / kTauMem);

    // Row 2 (final): sum of both contributions, stride = n_post
    for (int j = 0; j < n_post; ++j) {
        EXPECT_DOUBLE_EQ(sums.sum0[2 * n_post + j],
                         w[0 * n_post + j] * e_syn_0 + w[1 * n_post + j] * e_syn_1);
        EXPECT_DOUBLE_EQ(sums.sum1[2 * n_post + j],
                         w[0 * n_post + j] * e_mem_0 + w[1 * n_post + j] * e_mem_1);
    }

    eventprop_free_sums_output(sums);
}

TEST(ComputeSums, TwoSpikesSameSourceAreAdditive) {
    std::vector<double> w = {1.0, 2.0};
    int n_post = 2;

    std::vector<double> times   = {0.1, 0.3};
    std::vector<int> sources    = {0, 0};      // both from row 0
    auto input = make_input(w, 1, n_post, times, sources);

    SumsOutput sums = eventprop_compute_sums(input);

    double e_syn_0 = exp(0.1 / kTauSyn), e_mem_0 = exp(0.1 / kTauMem);
    double e_syn_1 = exp(0.3 / kTauSyn), e_mem_1 = exp(0.3 / kTauMem);

    for (int j = 0; j < n_post; ++j) {
        EXPECT_DOUBLE_EQ(sums.sum0[2 * n_post + j], w[j] * (e_syn_0 + e_syn_1));
        EXPECT_DOUBLE_EQ(sums.sum1[2 * n_post + j], w[j] * (e_mem_0 + e_mem_1));
    }

    eventprop_free_sums_output(sums);
}

// ──────────────────────────────────────────────────────────────
// Negative dimensions (validation catches these when enabled)
// ──────────────────────────────────────────────────────────────

TEST(ComputeSums, NegativeNpostDetected) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, -2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    SumsOutput sums = eventprop_compute_sums(input);
    EXPECT_EQ(sums.error_no, EVENTPROP_INVALID_DIMENSION);
    EXPECT_EQ(sums.sum0, nullptr);
    EXPECT_EQ(sums.sum1, nullptr);
    eventprop_free_sums_output(sums);  // safe even if null
}

TEST(ComputeSums, NegativeNspikesDetected) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, -3, kTauMem, kTauSyn, kVTh};

    SumsOutput sums = eventprop_compute_sums(input);
    EXPECT_EQ(sums.error_no, EVENTPROP_NEGATIVE_SPIKES);
    EXPECT_EQ(sums.sum0, nullptr);
    EXPECT_EQ(sums.sum1, nullptr);
    eventprop_free_sums_output(sums);
}

// ──────────────────────────────────────────────────────────────
// Scaling invariant: k*w -> k*sums
// ──────────────────────────────────────────────────────────────

TEST(ComputeSums, SumsScaleLinearlyWithWeights) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0};
    int n_post = 2;

    std::vector<double> times   = {0.1, 0.2};
    std::vector<int> sources    = {0, 1};
    auto input = make_input(w, 2, n_post, times, sources);

    SumsOutput sums1 = eventprop_compute_sums(input);

    // Scale weights by 3x and recompute
    std::vector<double> w2 = w;
    for (auto& v : w2) v *= 3.0;
    input.w = w2.data();
    SumsOutput sums2 = eventprop_compute_sums(input);

    int n_rows = 3; // n_spikes + 1
    for (int i = 0; i < n_rows * n_post; ++i) {
        EXPECT_DOUBLE_EQ(sums2.sum0[i], sums1.sum0[i] * 3.0);
        EXPECT_DOUBLE_EQ(sums2.sum1[i], sums1.sum1[i] * 3.0);
    }

    eventprop_free_sums_output(sums1);
    eventprop_free_sums_output(sums2);
}

// ──────────────────────────────────────────────────────────────
// Validation error paths (only active with EVENTPROP_VALIDATE_INPUTS)
// ──────────────────────────────────────────────────────────────

TEST(ComputeSums, EqualTausReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0,
                         kTauMem, kTauMem, kVTh};   // tau_mem == tau_syn

    SumsOutput sums = eventprop_compute_sums(input);
    EXPECT_EQ(sums.error_no, EVENTPROP_INVALID_TAU);
    EXPECT_EQ(sums.sum0, nullptr);
    EXPECT_EQ(sums.sum1, nullptr);
    eventprop_free_sums_output(sums);
}

TEST(ComputeSums, ZeroNpreReturnsError) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 0, 2, nullptr, nullptr, 0,
                         kTauMem, kTauSyn, kVTh};

    SumsOutput sums = eventprop_compute_sums(input);
    EXPECT_EQ(sums.error_no, EVENTPROP_INVALID_DIMENSION);
    EXPECT_EQ(sums.sum0, nullptr);
    EXPECT_EQ(sums.sum1, nullptr);
    eventprop_free_sums_output(sums);
}

TEST(ComputeSums, NullTimesWithSpikesReturnsError) {
    std::vector<double> w = {1.0};
    std::vector<int> sources = {0};
    EventPropInput input{w.data(), 1, 1, nullptr, sources.data(),
                         1, kTauMem, kTauSyn, kVTh};   // times == NULL but n_spikes > 0

    SumsOutput sums = eventprop_compute_sums(input);
    EXPECT_EQ(sums.error_no, EVENTPROP_NULL_POINTER);
    EXPECT_EQ(sums.sum0, nullptr);
    EXPECT_EQ(sums.sum1, nullptr);
    eventprop_free_sums_output(sums);
}

TEST(ComputeSums, OutOfBoundsSourceReturnsError) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {5};   // n_pre == 1
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    SumsOutput sums = eventprop_compute_sums(input);
    EXPECT_EQ(sums.error_no, EVENTPROP_INVALID_SOURCE);
    EXPECT_EQ(sums.sum0, nullptr);
    EXPECT_EQ(sums.sum1, nullptr);
    eventprop_free_sums_output(sums);
}

// ──────────────────────────────────────────────────────────────
// v_th is not part of the sums contract — the entry point must accept
// any threshold, including the zero the language bindings pass when
// their own API has no v_th parameter.
// ──────────────────────────────────────────────────────────────

TEST(ComputeSums, ZeroVthAcceptedAndIgnored) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> times = {0.1, 0.2};
    std::vector<int> sources  = {0, 1};

    EventPropInput with_vth{w.data(), 2, 2, times.data(), sources.data(),
                            (int)times.size(), kTauMem, kTauSyn, kVTh};
    EventPropInput zero_vth{w.data(), 2, 2, times.data(), sources.data(),
                            (int)times.size(), kTauMem, kTauSyn, 0.0};

    SumsOutput reference = eventprop_compute_sums(with_vth);
    SumsOutput under_test = eventprop_compute_sums(zero_vth);

    EXPECT_EQ(reference.error_no, EVENTPROP_OK);
    EXPECT_EQ(under_test.error_no, EVENTPROP_OK);

    const size_t n = (times.size() + 1) * 2;
    for (size_t i = 0; i < n; i++) {
        EXPECT_DOUBLE_EQ(under_test.sum0[i], reference.sum0[i]);
        EXPECT_DOUBLE_EQ(under_test.sum1[i], reference.sum1[i]);
    }

    eventprop_free_sums_output(reference);
    eventprop_free_sums_output(under_test);
}
