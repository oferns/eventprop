/**
 * @file diag_functions.cpp
 * @brief Tests for diagnostic / debugging helpers (voltage trace, lambda-I trace).
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"
#include <cmath>
#include <vector>

static constexpr double kTauMem = 0.02;
static constexpr double kTauSyn = 0.01;
static constexpr double kVTh    = 1.0;

// ──────────────────────────────────────────────────────────────
// compute_voltage_trace
// ──────────────────────────────────────────────────────────────

TEST(DiagVoltageTrace, BasicTraceHasCorrectLength) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};

    EventPropInput input{w.data(), 2, 3, times.data(), sources.data(), 2, kTauMem, kTauSyn, kVTh};

    // Forward pass to get spike output
    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);

    double t_max = 0.1;
    double dt = 0.001;

    VoltageTraceInput trace_input{
        &input,           // input
        0,                // target_nrn_idx (first post-neuron)
        spikes.times,     // fwd_times
        spikes.sources,   // fwd_sources
        spikes.n_spikes,  // n_post_spikes
        t_max,            // t_max
        dt                // dt
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);

    EXPECT_EQ(trace.error_no, EVENTPROP_OK);
    EXPECT_EQ(trace.n_t, (int)(t_max / dt));
    EXPECT_NE(trace.v_trace, nullptr);
    EXPECT_NE(trace.ts, nullptr);

    // Time array should be evenly spaced: 0, dt, 2*dt, ...
    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_DOUBLE_EQ(trace.ts[i], i * dt);
    }

    eventprop_free_voltage_trace_output(trace);
    eventprop_free_spike_output(spikes);
}

TEST(DiagVoltageTrace, VoltageStartsNearZero) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};

    EventPropInput input{w.data(), 2, 3, times.data(), sources.data(), 2, kTauMem, kTauSyn, kVTh};
    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);

    VoltageTraceInput trace_input{
        &input, 0, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);

    // At t=0 (before any input), voltage should be near zero
    EXPECT_NEAR(trace.v_trace[0], 0.0, 1e-6);

    eventprop_free_voltage_trace_output(trace);
    eventprop_free_spike_output(spikes);
}

TEST(DiagVoltageTrace, VoltageRisesAfterInputSpike) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};

    EventPropInput input{w.data(), 2, 3, times.data(), sources.data(), 2, kTauMem, kTauSyn, kVTh};
    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);

    VoltageTraceInput trace_input{
        &input, 0, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);

    // After input spike at t=0.01, voltage should rise above initial value
    int idx_after_spike = (int)(0.015 / 0.001); // t=0.015
    if (idx_after_spike < trace.n_t) {
        EXPECT_GT(trace.v_trace[idx_after_spike], trace.v_trace[0]);
    }

    eventprop_free_voltage_trace_output(trace);
    eventprop_free_spike_output(spikes);
}

TEST(DiagVoltageTrace, DifferentNeuronsProduceDifferentTraces) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3 (different weights per post-neuron)
    std::vector<double> times = {0.01, 0.02};
    std::vector<int> sources = {0, 1};

    EventPropInput input{w.data(), 2, 3, times.data(), sources.data(), 2, kTauMem, kTauSyn, kVTh};
    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);

    VoltageTraceInput trace_input0{
        &input, 0, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
    };
    VoltageTraceInput trace_input1{
        &input, 1, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
    };

    VoltageTraceOutput trace0 = eventprop_compute_voltage_trace(trace_input0);
    VoltageTraceOutput trace1 = eventprop_compute_voltage_trace(trace_input1);

    // Neurons with different weights should produce different traces
    bool different = false;
    for (int i = 0; i < trace0.n_t && !different; ++i) {
        if (std::abs(trace0.v_trace[i] - trace1.v_trace[i]) > 1e-9)
            different = true;
    }
    EXPECT_TRUE(different);

    eventprop_free_voltage_trace_output(trace0);
    eventprop_free_voltage_trace_output(trace1);
    eventprop_free_spike_output(spikes);
}

TEST(DiagVoltageTrace, ZeroInputSpikesProducesFlatTrace) {
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 2x3

    EventPropInput input{w.data(), 2, 3, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
    SpikeOutput spikes = eventprop_compute_spikes(input);
    EXPECT_EQ(spikes.error_no, EVENTPROP_OK);
    EXPECT_EQ(spikes.n_spikes, 0);

    VoltageTraceInput trace_input{
        &input, 0, nullptr, nullptr, 0, 0.1, 0.001
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);

    // With no input spikes and no forward spikes, voltage should stay at zero
    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_NEAR(trace.v_trace[i], 0.0, 1e-9);
    }

    eventprop_free_voltage_trace_output(trace);
    eventprop_free_spike_output(spikes);
}

TEST(DiagVoltageTrace, NullInputReturnsError) {
    VoltageTraceInput trace_input{
        nullptr, 0, nullptr, nullptr, 0, 0.1, 0.001
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_NULL_POINTER);
    EXPECT_EQ(trace.n_t, 0);
    EXPECT_EQ(trace.v_trace, nullptr);
    EXPECT_EQ(trace.ts, nullptr);

    eventprop_free_voltage_trace_output(trace);
}

TEST(DiagVoltageTrace, ZeroDtReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    VoltageTraceInput trace_input{
        &input, 0, nullptr, nullptr, 0, 0.1, 0.0
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_voltage_trace_output(trace);
}

TEST(DiagVoltageTrace, EqualTausReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh}; // tau_mem == tau_syn

    VoltageTraceInput trace_input{
        &input, 0, nullptr, nullptr, 0, 0.1, 0.001
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_TAU);

    eventprop_free_voltage_trace_output(trace);
}

TEST(DiagVoltageTrace, OutOfBoundsTargetNeuronReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    VoltageTraceInput trace_input{
        &input, 5, nullptr, nullptr, 0, 0.1, 0.001 // target_nrn_idx=5 >= n_post=2
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_SOURCE);

    eventprop_free_voltage_trace_output(trace);
}

// ──────────────────────────────────────────────────────────────
// compute_lambda_i_trace
// ──────────────────────────────────────────────────────────────

TEST(DiagLambdaITrace, BasicTraceHasCorrectLength) {
    std::vector<double> fwd_times = {0.015, 0.025};
    std::vector<int> fwd_sources = {0, 1};

    double t_max = 0.1;
    double dt = 0.001;

    LambdaITraceInput input{
        fwd_times.data(),   // fwd_times
        2,                  // n_post_spikes
        0,                  // target_nrn_idx
        kVTh,               // v_th
        kTauMem,            // tau_mem
        kTauSyn,            // tau_syn
        t_max,              // t_max
        dt                  // dt
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);

    EXPECT_EQ(trace.n_t, (int)(t_max / dt));
    EXPECT_NE(trace.lambda_trace, nullptr);
    EXPECT_NE(trace.ts, nullptr);

    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_DOUBLE_EQ(trace.ts[i], i * dt);
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, LambdaGrowsNearSpike) {
    std::vector<double> fwd_times = {0.05};
    std::vector<int> fwd_sources = {0};

    LambdaITraceInput input{
        fwd_times.data(), 1, 0, kVTh, kTauMem, kTauSyn, 0.1, 0.001
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);

    // The lambda-I trace computes adjoints from the last forward spike backward.
    // As t approaches the spike time, dt_jump → 0 and the exponential terms grow,
    // so lambda should be larger near the spike than far from it.
    double early_lambda = std::abs(trace.lambda_trace[10]);   // t=0.01
    double late_lambda  = std::abs(trace.lambda_trace[45]);   // t=0.045 (near spike)
    EXPECT_GT(late_lambda, early_lambda);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, LambdaIsFiniteEverywhere) {
    std::vector<double> fwd_times = {0.03, 0.06};
    std::vector<int> fwd_sources = {0, 1};

    LambdaITraceInput input{
        fwd_times.data(), 2, 0, kVTh, kTauMem, kTauSyn, 0.1, 0.001
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);

    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_TRUE(std::isfinite(trace.lambda_trace[i]))
            << "Non-finite at t=" << trace.ts[i];
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ZeroForwardSpikesProducesZeroTrace) {
    std::vector<double> fwd_times; // empty
    std::vector<int> fwd_sources;  // empty

    LambdaITraceInput input{
        nullptr, 0, 0, kVTh, kTauMem, kTauSyn, 0.1, 0.001
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);

    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_NEAR(trace.lambda_trace[i], 0.0, 1e-9);
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, MultipleForwardSpikesAccumulateEffect) {
    std::vector<double> fwd_times = {0.03, 0.06};
    std::vector<int> fwd_sources = {0, 1};

    LambdaITraceInput input{
        fwd_times.data(), 2, 0, kVTh, kTauMem, kTauSyn, 0.1, 0.001
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);

    // After both spikes, lambda should be nonzero and finite
    bool found_nonzero = false;
    for (int i = 0; i < trace.n_t; ++i) {
        if (!std::isfinite(trace.lambda_trace[i])) {
            GTEST_FAIL() << "Non-finite value at t=" << trace.ts[i];
        }
        if (std::abs(trace.lambda_trace[i]) > 1e-9) {
            found_nonzero = true;
        }
    }
    EXPECT_TRUE(found_nonzero);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ZeroDtReturnsError) {
    LambdaITraceInput input{
        nullptr, 0, 0, kVTh, kTauMem, kTauSyn, 0.1, 0.0
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_DIMENSION);
    EXPECT_EQ(trace.n_t, 0);
    EXPECT_EQ(trace.lambda_trace, nullptr);
    EXPECT_EQ(trace.ts, nullptr);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, EqualTausReturnsError) {
    LambdaITraceInput input{
        nullptr, 0, 0, kVTh, kTauMem, kTauMem, 0.1, 0.001 // tau_mem == tau_syn
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_TAU);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ZeroVThReturnsError) {
    LambdaITraceInput input{
        nullptr, 0, 0, 0.0, kTauMem, kTauSyn, 0.1, 0.001 // v_th = 0
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_DIMENSION);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, NullFwdTimesWithSpikesReturnsError) {
    LambdaITraceInput input{
        nullptr, 1, 0, kVTh, kTauMem, kTauSyn, 0.1, 0.001 // n_post_spikes=1 but fwd_times=NULL
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_lambda_i_trace_output(trace);
}
