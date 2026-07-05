/**
 * @file free_functions.cpp
 * @brief Tests for all eventprop_*_free functions.
 *
 * Properties under test:
 *   - Null safety: passing NULL does not crash
 *   - Double-free safety: calling twice on the same struct does not crash
 *     (internal pointers are nulled after first free)
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"
#include <cstring>

// ──────────────────────────────────────────────────────────────
// Single-sample free functions (from eventprop.h)
// ──────────────────────────────────────────────────────────────

TEST(FreeSpikeOutput, NullPointerIsSafe) {
    eventprop_free_spike_output(nullptr);
}

TEST(FreeSpikeOutput, DoubleFreeIsSafe) {
    SpikeOutput output;
    std::memset(&output, 0, sizeof(output));
    eventprop_free_spike_output(&output);
    eventprop_free_spike_output(&output);
}

TEST(FreeSumsOutput, NullPointerIsSafe) {
    eventprop_free_sums_output(nullptr);
}

TEST(FreeSumsOutput, DoubleFreeIsSafe) {
    SumsOutput output;
    std::memset(&output, 0, sizeof(output));
    eventprop_free_sums_output(&output);
    eventprop_free_sums_output(&output);
}

TEST(FreeMaximaOutput, NullPointerIsSafe) {
    eventprop_free_maxima_output(nullptr);
}

TEST(FreeMaximaOutput, DoubleFreeIsSafe) {
    MaximaOutput output;
    std::memset(&output, 0, sizeof(output));
    eventprop_free_maxima_output(&output);
    eventprop_free_maxima_output(&output);
}

TEST(FreeBackwardOutput, NullPointerIsSafe) {
    eventprop_free_backward_output(nullptr);
}

TEST(FreeBackwardOutput, DoubleFreeIsSafe) {
    BackwardOutput output;
    std::memset(&output, 0, sizeof(output));
    eventprop_free_backward_output(&output);
    eventprop_free_backward_output(&output);
}

// ──────────────────────────────────────────────────────────────
// Batch free functions (from eventprop_batch.h)
// ──────────────────────────────────────────────────────────────

TEST(FreeSpikesBatch, NullPointerIsSafe) {
    eventprop_free_spikes_batch(nullptr);
}

TEST(FreeMaximaBatch, NullPointerIsSafe) {
    eventprop_free_maxima_batch(nullptr);
}

TEST(FreeBackwardsBatch, NullPointerIsSafe) {
    eventprop_free_backwards_batch(nullptr);
}

TEST(FreeGradientOutput, NullPointerIsSafe) {
    eventprop_free_gradient_output(nullptr);
}

TEST(FreeDropoutOutput, NullPointerIsSafe) {
    eventprop_free_dropout_output(nullptr);
}

TEST(FreeDropoutBatch, NullPointerIsSafe) {
    eventprop_free_dropout_batch(nullptr);
}

TEST(FreeAccumulatedBackwardResult, NullPointerIsSafe) {
    eventprop_free_accumulated_backward_result(nullptr);
}

// ──────────────────────────────────────────────────────────────
// Diagnostic free functions (from eventprop_diag.h)
// ──────────────────────────────────────────────────────────────

TEST(FreeLambdaITraceOutput, NullPointerIsSafe) {
    eventprop_free_lambda_i_trace_output(nullptr);
}

TEST(FreeLambdaITraceOutput, DoubleFreeIsSafe) {
    LambdaITraceOutput output;
    std::memset(&output, 0, sizeof(output));
    eventprop_free_lambda_i_trace_output(&output);
    eventprop_free_lambda_i_trace_output(&output);
}

TEST(FreeVoltageTraceOutput, NullPointerIsSafe) {
    eventprop_free_voltage_trace_output(nullptr);
}

TEST(FreeVoltageTraceOutput, DoubleFreeIsSafe) {
    VoltageTraceOutput output;
    std::memset(&output, 0, sizeof(output));
    eventprop_free_voltage_trace_output(&output);
    eventprop_free_voltage_trace_output(&output);
}
