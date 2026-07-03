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
