/**
 * @file validate_input.cpp
 * @brief Tests for eventprop_validate_input (the reusable validation function).
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include <vector>

static constexpr double kTauMem = 0.02;
static constexpr double kTauSyn = 0.005;
static constexpr double kVTh    = 1.0;

// ──────────────────────────────────────────────────────────────
// Valid input — always OK
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, ValidInputReturnsOk) {
    std::vector<double> w   = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> times = {0.1, 0.2};
    std::vector<int> sources  = {0, 1};

    EventPropInput input{w.data(), 2, 2, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};

    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_OK);
}

TEST(ValidateInput, ValidZeroSpikesReturnsOk) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_OK);
}

// ──────────────────────────────────────────────────────────────
// Dimension checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, ZeroNpreReturnsInvalidDimension) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 0, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_DIMENSION);
}

TEST(ValidateInput, ZeroNpostReturnsInvalidDimension) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 2, 0, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_DIMENSION);
}

TEST(ValidateInput, NegativeNpreReturnsInvalidDimension) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), -3, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_DIMENSION);
}

// ──────────────────────────────────────────────────────────────
// Spike count checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, NegativeNspikesReturnsNegativeSpikes) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, -1, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_NEGATIVE_SPIKES);
}

// ──────────────────────────────────────────────────────────────
// Tau checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, ZeroTauMemReturnsInvalidTau) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, 0.0, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_TAU);
}

TEST(ValidateInput, ZeroTauSynReturnsInvalidTau) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, 0.0, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_TAU);
}

TEST(ValidateInput, NegativeTauMemReturnsInvalidTau) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, -0.02, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_TAU);
}

TEST(ValidateInput, EqualTausReturnsInvalidTau) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_TAU);
}

// ──────────────────────────────────────────────────────────────
// Null pointer checks (only when n_spikes > 0)
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, NullWWithSpikesReturnsNullPointer) {
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {0};
    EventPropInput input{nullptr, 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_NULL_POINTER);
}

TEST(ValidateInput, NullTimesWithSpikesReturnsNullPointer) {
    std::vector<double> w = {1.0};
    std::vector<int> sources  = {0};
    EventPropInput input{w.data(), 1, 1, nullptr, sources.data(),
                         1, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_NULL_POINTER);
}

TEST(ValidateInput, NullSourcesWithSpikesReturnsNullPointer) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    EventPropInput input{w.data(), 1, 1, times.data(), nullptr,
                         1, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_NULL_POINTER);
}

// ──────────────────────────────────────────────────────────────
// Source index bounds checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, OutOfBoundsSourceReturnsInvalidSource) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {5};   // n_pre == 1, source 5 is out of bounds
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_SOURCE);
}

TEST(ValidateInput, NegativeSourceReturnsInvalidSource) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {0.1};
    std::vector<int> sources  = {-1};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_SOURCE);
}

// ──────────────────────────────────────────────────────────────
// Priority: first error wins (dimension checked before tau)
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, FirstErrorWinsDimensionBeforeTau) {
    std::vector<double> w = {1.0};
    // Both n_pre == 0 AND tau_mem == tau_syn — dimension should win
    EventPropInput input{w.data(), 0, 2, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_DIMENSION);
}

// ──────────────────────────────────────────────────────────────
// Dimension upper-bound checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, NpreTooLargeReturnsDimensionTooLarge) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), EVENTPROP_MAX_DIM + 1, 2,
                         nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_DIMENSION_TOO_LARGE);
}

TEST(ValidateInput, NpostTooLargeReturnsDimensionTooLarge) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 2, EVENTPROP_MAX_DIM + 1,
                         nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_DIMENSION_TOO_LARGE);
}

TEST(ValidateInput, NpreAtMaxIsOk) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), EVENTPROP_MAX_DIM, 2,
                         nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_OK);
}

// ──────────────────────────────────────────────────────────────
// Tau bounds checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, TauMemTooSmallReturnsTauTooSmall) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         EVENTPROP_MIN_TAU * 0.5, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TAU_TOO_SMALL);
}

TEST(ValidateInput, TauSynTooSmallReturnsTauTooSmall) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         kTauMem, EVENTPROP_MIN_TAU * 0.5, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TAU_TOO_SMALL);
}

TEST(ValidateInput, TauMemAtMinIsOk) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         EVENTPROP_MIN_TAU, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_OK);
}

TEST(ValidateInput, TauMemTooLargeReturnsTauTooLarge) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         EVENTPROP_MAX_TAU * 2.0, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TAU_TOO_LARGE);
}

TEST(ValidateInput, TauSynTooLargeReturnsTauTooLarge) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         kTauMem, EVENTPROP_MAX_TAU * 2.0, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TAU_TOO_LARGE);
}

// ──────────────────────────────────────────────────────────────
// Tau proximity check (k_prefactor blowup)
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, TausTooCloseReturnsTauTooClose) {
    std::vector<double> w = {1.0};
    // tau_mem and tau_syn differ by less than 1e-6 relative → k_prefactor blows up
    double tau_mem = 0.02;
    double tau_syn = 0.02 + 1e-10;   // rel_diff ≈ 5e-9 < 1e-6
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         tau_mem, tau_syn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TAU_TOO_CLOSE);
}

TEST(ValidateInput, TausJustAboveRelDiffMinIsOk) {
    std::vector<double> w = {1.0};
    // Clearly above the boundary: rel_diff == 2e-6 → should pass
    double tau_mem = 0.02;
    double tau_syn = tau_mem * (1.0 + 2.0 * EVENTPROP_TAU_REL_DIFF_MIN);
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         tau_mem, tau_syn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_OK);
}

// ──────────────────────────────────────────────────────────────
// Threshold voltage checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, ZeroVthReturnsInvalidVth) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         kTauMem, kTauSyn, 0.0};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_VTH);
}

TEST(ValidateInput, NegativeVthReturnsInvalidVth) {
    std::vector<double> w = {1.0};
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         kTauMem, kTauSyn, -1.0};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_VTH);
}

// ──────────────────────────────────────────────────────────────
// Spike time checks
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, NegativeSpikeTimeReturnsNegativeSpikeTime) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {-0.5};
    std::vector<int> sources  = {0};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_NEGATIVE_SPIKE_TIME);
}

TEST(ValidateInput, ExcessiveSpikeTimeReturnsExcessiveSpikeTime) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {EVENTPROP_MAX_SPIKE_TIME * 2.0};
    std::vector<int> sources  = {0};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_EXCESSIVE_SPIKE_TIME);
}

TEST(ValidateInput, SpikeTimeAtMaxIsOk) {
    std::vector<double> w = {1.0};
    std::vector<double> times = {EVENTPROP_MAX_SPIKE_TIME};
    std::vector<int> sources  = {0};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};
    // This should pass the individual time check but fail the cross-field check
    // because max_time / min_tau = 1e6 / 0.005 = 200000 >> 700
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TIMES_TAU_MISMATCH);
}

// ──────────────────────────────────────────────────────────────
// Cross-field: times vs tau overflow check
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, TimesTauMismatchReturnsTimesTauMismatch) {
    std::vector<double> w = {1.0};
    // max_time / min_tau >= 700 → exp() would overflow
    double tau_mem = 0.02;
    double tau_syn = 0.005;   // min_tau = 0.005
    std::vector<double> times = {4.0};   // 4.0 / 0.005 = 800 >= 700
    std::vector<int> sources  = {0};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), tau_mem, tau_syn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TIMES_TAU_MISMATCH);
}

TEST(ValidateInput, TimesTauJustBelowThresholdIsOk) {
    std::vector<double> w = {1.0};
    // max_time / min_tau just below 700 → should pass
    double tau_mem = 0.02;
    double tau_syn = 0.005;   // min_tau = 0.005
    std::vector<double> times = {3.49};   // 3.49 / 0.005 = 698 < 700
    std::vector<int> sources  = {0};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), tau_mem, tau_syn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_OK);
}

TEST(ValidateInput, TimesTauMismatchWithLargeTauMemSmallTauSyn) {
    std::vector<double> w = {1.0};
    // tau_mem is large but tau_syn is tiny → min_tau = tau_syn triggers mismatch
    double tau_mem = 100.0;
    double tau_syn = 1e-6;   // min_tau = 1e-6
    std::vector<double> times = {0.001};   // 0.001 / 1e-6 = 1000 >= 700
    std::vector<int> sources  = {0};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), tau_mem, tau_syn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_TIMES_TAU_MISMATCH);
}

// ──────────────────────────────────────────────────────────────
// Priority ordering: new checks respect existing precedence
// ──────────────────────────────────────────────────────────────

TEST(ValidateInput, DimensionCheckedBeforeTauBounds) {
    std::vector<double> w = {1.0};
    // n_pre == 0 AND tau_mem too small — dimension should win
    EventPropInput input{w.data(), 0, 2, nullptr, nullptr, 0,
                         EVENTPROP_MIN_TAU * 0.5, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_DIMENSION);
}

TEST(ValidateInput, TauCheckedBeforeVth) {
    std::vector<double> w = {1.0};
    // tau_mem == 0 AND v_th == -1 — tau should win (checked first)
    EventPropInput input{w.data(), 1, 1, nullptr, nullptr, 0,
                         0.0, kTauSyn, -1.0};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_INVALID_TAU);
}

TEST(ValidateInput, NegativeTimeCheckedBeforeExcessiveTime) {
    std::vector<double> w = {1.0};
    // First spike is negative → should return NEGATIVE_SPIKE_TIME (not EXCESSIVE)
    std::vector<double> times = {-1.0, EVENTPROP_MAX_SPIKE_TIME * 2.0};
    std::vector<int> sources  = {0, 0};
    EventPropInput input{w.data(), 1, 1, times.data(), sources.data(),
                         (int)times.size(), kTauMem, kTauSyn, kVTh};
    EXPECT_EQ(eventprop_validate_input(input), EVENTPROP_NEGATIVE_SPIKE_TIME);
}
