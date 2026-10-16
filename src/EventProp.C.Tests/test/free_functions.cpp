/**
 * @file free_functions.cpp
 * @brief Tests for all eventprop_*_free functions.
 *
 * Properties under test:
 *   - Null safety: passing a zero-initialized struct does not crash
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"

// ──────────────────────────────────────────────────────────────
// Single-sample free functions (from eventprop.h)
// ──────────────────────────────────────────────────────────────

TEST(FreeSpikeOutput, NullPointerIsSafe) {
    eventprop_free_spike_output(SpikeOutput{});
}

TEST(FreeSumsOutput, NullPointerIsSafe) {
    eventprop_free_sums_output(SumsOutput{});
}

TEST(FreeMaximaOutput, NullPointerIsSafe) {
    eventprop_free_maxima_output(MaximaOutput{});
}

TEST(FreeBackwardOutput, NullPointerIsSafe) {
    eventprop_free_backward_output(BackwardOutput{});
}

// ──────────────────────────────────────────────────────────────
// Batch free functions (from eventprop_batch.h)
// ──────────────────────────────────────────────────────────────

TEST(FreeSpikesBatch, NullPointerIsSafe) {
    eventprop_free_spikes_batch(SpikesBatch{});
}

TEST(FreeMaximaBatch, NullPointerIsSafe) {
    eventprop_free_maxima_batch(MaximaBatch{});
}

TEST(FreeBackwardsBatch, NullPointerIsSafe) {
    eventprop_free_backwards_batch(BackwardsBatch{});
}

TEST(FreeGradientOutput, NullPointerIsSafe) {
    eventprop_free_gradient_output(GradientOutput{});
}

TEST(FreeDropoutOutput, NullPointerIsSafe) {
    eventprop_free_dropout_output(DropoutOutput{});
}

TEST(FreeDropoutBatch, NullPointerIsSafe) {
    eventprop_free_dropout_batch(DropoutBatch{});
}

TEST(FreeAccumulatedBackwardResult, NullPointerIsSafe) {
    eventprop_free_accumulated_backward_result(AccumulatedBackwardResult{});
}

// ──────────────────────────────────────────────────────────────
// Diagnostic free functions (from eventprop_diag.h)
// ──────────────────────────────────────────────────────────────

TEST(FreeLambdaITraceOutput, NullPointerIsSafe) {
    eventprop_free_lambda_i_trace_output(LambdaITraceOutput{});
}

TEST(FreeVoltageTraceOutput, NullPointerIsSafe) {
    eventprop_free_voltage_trace_output(VoltageTraceOutput{});
}
