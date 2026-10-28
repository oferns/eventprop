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
        input,            // input (value)
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
        input, 0, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
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
        input, 0, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
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
        input, 0, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
    };
    VoltageTraceInput trace_input1{
        input, 1, spikes.times, spikes.sources, spikes.n_spikes, 0.1, 0.001
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
        input, 0, nullptr, nullptr, 0, 0.1, 0.001
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);

    // With no input spikes and no forward spikes, voltage should stay at zero
    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_NEAR(trace.v_trace[i], 0.0, 1e-9);
    }

    eventprop_free_voltage_trace_output(trace);
    eventprop_free_spike_output(spikes);
}


TEST(DiagVoltageTrace, ZeroDtReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    VoltageTraceInput trace_input{
        input, 0, nullptr, nullptr, 0, 0.1, 0.0
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_DIAG_TIMESTEP_INVALID);

    eventprop_free_voltage_trace_output(trace);
}

TEST(DiagVoltageTrace, EqualTausReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh}; // tau_mem == tau_syn

    VoltageTraceInput trace_input{
        input, 0, nullptr, nullptr, 0, 0.1, 0.001
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_TAU);

    eventprop_free_voltage_trace_output(trace);
}

TEST(DiagVoltageTrace, OutOfBoundsTargetNeuronReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    VoltageTraceInput trace_input{
        input, 5, nullptr, nullptr, 0, 0.1, 0.001 // target_nrn_idx=5 >= n_post=2
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_DIAG_TARGET_OUT_OF_RANGE);

    eventprop_free_voltage_trace_output(trace);
}

// ──────────────────────────────────────────────────────────────
// compute_lambda_i_trace
//
// These traces are adjoints, so they are driven by the error signal: the jump at a
// post-synaptic spike is 1/(current - v_th) * (v_th * lambda_v + error). The tests below keep
// the forward pass empty (n_spikes = 0) wherever an exact number is asserted, because then the
// rebuilt post-spike current is exactly zero and the jump collapses to -error / v_th.
// Multi-spike recursion, where lambda_v carries from spike to spike, is pinned against the
// Eigen core by the LambdaITrace cases in src/EventProp.Native.ParityTests instead.
// ──────────────────────────────────────────────────────────────

/** Forward-pass input with no input spikes, so every rebuilt post-spike current is zero. */
static EventPropInput make_empty_forward(std::vector<double>& w) {
    w = {0.5, 0.5};
    return EventPropInput{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};
}

/** k_bwd of the reference backward kernel, spelled out so expectations are analytic. */
static double expected_k_bwd(double elapsed) {
    return kTauMem / (kTauMem - kTauSyn) *
           (std::exp(-elapsed / kTauMem) - std::exp(-elapsed / kTauSyn));
}

TEST(DiagLambdaITrace, BasicTraceHasCorrectLength) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.015, 0.025};
    std::vector<int>    fwd_sources = {0, 1};
    std::vector<double> post_errors = {1.0, 1.0};

    double t_max = 0.1;
    double dt = 0.001;

    LambdaITraceInput input{
        make_empty_forward(w),
        0,                    // target_nrn_idx
        fwd_times.data(),
        fwd_sources.data(),
        post_errors.data(),
        2,                    // n_post_spikes
        t_max,
        dt
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);

    EXPECT_EQ(trace.error_no, EVENTPROP_OK);
    EXPECT_EQ(trace.n_t, (int)(t_max / dt));
    EXPECT_NE(trace.lambda_trace, nullptr);
    EXPECT_NE(trace.ts, nullptr);

    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_DOUBLE_EQ(trace.ts[i], i * dt);
    }

    eventprop_free_lambda_i_trace_output(trace);
}

// The whole trace, computed from the formula rather than from the implementation: one spike of
// the target neuron at tau, no input spikes, so lambda(t) = -(error / v_th) * k_bwd(tau - t)
// before the spike and zero at and after it. This is what pins the jump amplitude, the kernel,
// and the fact that kernel elapsed time is measured from the query time.
TEST(DiagLambdaITrace, SingleSpikeMatchesAnalyticAdjoint) {
    std::vector<double> w;
    const double tau = 0.05;
    const double error = 2.0;
    std::vector<double> fwd_times = {tau};
    std::vector<int>    fwd_sources = {0};
    std::vector<double> post_errors = {error};

    const double dt = 0.001;
    LambdaITraceInput input{
        make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
        post_errors.data(), 1, 0.1, dt
    };

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    ASSERT_EQ(trace.error_no, EVENTPROP_OK);

    const double jump = -error / kVTh;
    for (int i = 0; i < trace.n_t; ++i) {
        const double t = i * dt;
        const double expected = (t < tau) ? jump * expected_k_bwd(tau - t) : 0.0;
        EXPECT_NEAR(trace.lambda_trace[i], expected, 1e-12) << "at t=" << t;
    }

    eventprop_free_lambda_i_trace_output(trace);
}

// The reason the diagnostic is loss-driven: doubling the error signal doubles the adjoint.
// An implementation that ignores post_errors cannot pass this, and passes the old suite.
TEST(DiagLambdaITrace, ScalesLinearlyWithPostSpikeError) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.03, 0.06};
    std::vector<int>    fwd_sources = {0, 1};
    std::vector<double> err_one = {1.0, 1.0};
    std::vector<double> err_two = {2.0, 2.0};

    LambdaITraceInput in1{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                          err_one.data(), 2, 0.1, 0.001};
    LambdaITraceInput in2{in1};
    in2.post_errors = err_two.data();

    LambdaITraceOutput t1 = eventprop_compute_lambda_i_trace(in1);
    LambdaITraceOutput t2 = eventprop_compute_lambda_i_trace(in2);
    ASSERT_EQ(t1.error_no, EVENTPROP_OK);
    ASSERT_EQ(t2.error_no, EVENTPROP_OK);

    bool any_nonzero = false;
    for (int i = 0; i < t1.n_t; ++i) {
        EXPECT_NEAR(t2.lambda_trace[i], 2.0 * t1.lambda_trace[i], 1e-12)
            << "at t=" << t1.ts[i];
        if (std::abs(t1.lambda_trace[i]) > 1e-9) any_nonzero = true;
    }
    // Without a nonzero reference the doubling above is a comparison of zeros.
    EXPECT_TRUE(any_nonzero);

    eventprop_free_lambda_i_trace_output(t1);
    eventprop_free_lambda_i_trace_output(t2);
}

// A silent target neuron has no jumps to sum, so its adjoint is exactly zero. This is the case
// the reference C++ core reads out of bounds on, and what the Visualizer guards against.
TEST(DiagLambdaITrace, SilentTargetNeuronYieldsZeroTrace) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.02, 0.05};
    std::vector<int>    fwd_sources = {1, 1};   // neuron 0 never fires
    std::vector<double> post_errors = {1.0, 1.0};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 2, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    ASSERT_EQ(trace.error_no, EVENTPROP_OK);

    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_DOUBLE_EQ(trace.lambda_trace[i], 0.0) << "at t=" << trace.ts[i];
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ZeroErrorsYieldZeroTrace) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.02, 0.05};
    std::vector<int>    fwd_sources = {0, 1};
    std::vector<double> post_errors = {0.0, 0.0};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 2, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    ASSERT_EQ(trace.error_no, EVENTPROP_OK);

    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_DOUBLE_EQ(trace.lambda_trace[i], 0.0);
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, LambdaIsFiniteEverywhere) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.03, 0.06};
    std::vector<int>    fwd_sources = {0, 1};
    std::vector<double> post_errors = {1.0, 0.5};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 2, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    ASSERT_EQ(trace.error_no, EVENTPROP_OK);

    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_TRUE(std::isfinite(trace.lambda_trace[i]))
            << "Non-finite at t=" << trace.ts[i];
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ZeroForwardSpikesProducesZeroTrace) {
    std::vector<double> w;
    EventPropInput fwd = make_empty_forward(w);

    LambdaITraceInput input{fwd, 0, nullptr, nullptr, nullptr, 0, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);

    // No spikes means no jumps, which is a legitimate empty trace, not an error.
    EXPECT_EQ(trace.error_no, EVENTPROP_OK);
    for (int i = 0; i < trace.n_t; ++i) {
        EXPECT_NEAR(trace.lambda_trace[i], 0.0, 1e-9);
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, MultipleForwardSpikesAccumulateEffect) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.03, 0.06};
    std::vector<int>    fwd_sources = {0, 0};   // both belong to the traced neuron
    std::vector<double> post_errors = {1.0, 1.0};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 2, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    ASSERT_EQ(trace.error_no, EVENTPROP_OK);

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

// The sweep walks the spike list backwards and stops at the first spike earlier than the query
// time, so an unsorted list silently under-counts. The Eigen reference has the same requirement
// and checks neither — C is the diagnostic.
TEST(DiagLambdaITrace, UnsortedPostSpikeTimesRejected) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.06, 0.03};
    std::vector<int>    fwd_sources = {0, 0};
    std::vector<double> post_errors = {1.0, 1.0};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 2, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_UNSORTED_SPIKE_TIMES);

    eventprop_free_lambda_i_trace_output(trace);
}

// Equal times are the legitimate simultaneous-spike case and must stay accepted.
TEST(DiagLambdaITrace, EqualPostSpikeTimesAccepted) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.04, 0.04};
    std::vector<int>    fwd_sources = {0, 0};
    std::vector<double> post_errors = {1.0, 1.0};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 2, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_OK);

    eventprop_free_lambda_i_trace_output(trace);
}

// fwd_sources indexes the per-neuron lambda_v array, so an out-of-range index would write past
// it rather than merely producing a wrong number.
TEST(DiagLambdaITrace, SourceOutOfRangeRejected) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.03};
    std::vector<int>    fwd_sources = {2};   // n_post is 2, so valid indices are 0 and 1
    std::vector<double> post_errors = {1.0};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 1, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_SOURCE);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, TargetOutOfRangeRejected) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.03};
    std::vector<int>    fwd_sources = {0};
    std::vector<double> post_errors = {1.0};

    LambdaITraceInput input{make_empty_forward(w), 5, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 1, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_DIAG_TARGET_OUT_OF_RANGE);

    eventprop_free_lambda_i_trace_output(trace);
}

// An empty spike list needs no spike arrays; a non-empty one needs all three.
TEST(DiagLambdaITrace, NullPostErrorsRejected) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.03};
    std::vector<int>    fwd_sources = {0};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            nullptr, 1, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_lambda_i_trace_output(trace);
}

// current == v_th zeroes the jump denominator. The backward pass rejects it with
// EVENTPROP_INVALID_CURRENT rather than emitting an infinite jump; the diagnostic matches.
// An input spike and the post spike both at t = 0 make the current exactly the weight, because
// exp(0) is exactly 1, so the equality is exact and not a tolerance question.
TEST(DiagLambdaITrace, CurrentEqualToVThRejected) {
    std::vector<double> w = {kVTh, 0.5};
    std::vector<double> in_times = {0.0};
    std::vector<int>    in_sources = {0};
    EventPropInput fwd{w.data(), 1, 2, in_times.data(), in_sources.data(), 1,
                       kTauMem, kTauSyn, kVTh};

    std::vector<double> post_times = {0.0};
    std::vector<int>    post_sources = {0};
    std::vector<double> post_errors = {1.0};

    LambdaITraceInput input{fwd, 0, post_times.data(), post_sources.data(),
                            post_errors.data(), 1, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_CURRENT);
    EXPECT_EQ(trace.lambda_trace, nullptr);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ZeroDtReturnsError) {
    std::vector<double> w;
    EventPropInput fwd = make_empty_forward(w);

    LambdaITraceInput input{fwd, 0, nullptr, nullptr, nullptr, 0, 0.1, 0.0};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_DIAG_TIMESTEP_INVALID);
    EXPECT_EQ(trace.n_t, 0);
    EXPECT_EQ(trace.lambda_trace, nullptr);
    EXPECT_EQ(trace.ts, nullptr);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, EqualTausReturnsError) {
    std::vector<double> w = {0.5, 0.5};
    EventPropInput fwd{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauMem, kVTh};

    LambdaITraceInput input{fwd, 0, nullptr, nullptr, nullptr, 0, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_TAU);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ZeroVThReturnsError) {
    std::vector<double> w = {0.5, 0.5};
    EventPropInput fwd{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, 0.0};

    LambdaITraceInput input{fwd, 0, nullptr, nullptr, nullptr, 0, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_INVALID_VTH);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, NullFwdTimesWithSpikesReturnsError) {
    std::vector<double> w;
    EventPropInput fwd = make_empty_forward(w);

    // n_post_spikes = 1 but the spike arrays are NULL
    LambdaITraceInput input{fwd, 0, nullptr, nullptr, nullptr, 1, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaITrace, ExcessiveSampleCountReturnsError) {
    std::vector<double> w;
    EventPropInput fwd = make_empty_forward(w);

    // t_max / dt = 1e9 samples — exceeds EVENTPROP_MAX_DIAG_SAMPLES
    LambdaITraceInput input{fwd, 0, nullptr, nullptr, nullptr, 0, 1e6, 1e-3};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    EXPECT_EQ(trace.error_no, EVENTPROP_DIAG_TRACE_TOO_LONG);
    EXPECT_EQ(trace.n_t, 0);
    EXPECT_EQ(trace.lambda_trace, nullptr);
    EXPECT_EQ(trace.ts, nullptr);

    eventprop_free_lambda_i_trace_output(trace);
}

// ──────────────────────────────────────────────────────────────
// compute_lambda_i_at_time — the pointwise query
//
// The grid trace and this entry point call the same sweep, so the interesting assertion is
// not "is it a number" but "is it the SAME number the grid produces". That equivalence is
// what keeps the two from drifting the way the trace drifted from the backward pass.
// ──────────────────────────────────────────────────────────────

TEST(DiagLambdaIAtTime, PointwiseMatchesTraceAtEverySample) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.02, 0.035, 0.06, 0.08};
    std::vector<int>    fwd_sources = {0, 1, 0, 1};
    std::vector<double> post_errors = {1.0, 0.5, 2.0, 1.5};
    const double dt = 0.001;

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 4, 0.1, dt};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    ASSERT_EQ(trace.error_no, EVENTPROP_OK);

    for (int i = 0; i < trace.n_t; ++i) {
        LambdaIAtTimeOutput one = eventprop_compute_lambda_i_at_time(input, i * dt);
        ASSERT_EQ(one.error_no, EVENTPROP_OK) << "at t=" << i * dt;
        // Identical by construction, not within a tolerance: same code path, same inputs.
        EXPECT_EQ(one.lambda_i, trace.lambda_trace[i]) << "at t=" << i * dt;
    }

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagLambdaIAtTime, PointwiseMatchesAnalyticAdjoint) {
    std::vector<double> w;
    const double tau = 0.05;
    const double error = 2.0;
    std::vector<double> fwd_times = {tau};
    std::vector<int>    fwd_sources = {0};
    std::vector<double> post_errors = {error};

    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 1, 0.1, 0.001};

    const double jump = -error / kVTh;
    for (const double t : {0.0, 0.01, 0.049}) {
        LambdaIAtTimeOutput one = eventprop_compute_lambda_i_at_time(input, t);
        ASSERT_EQ(one.error_no, EVENTPROP_OK);
        EXPECT_NEAR(one.lambda_i, jump * expected_k_bwd(tau - t), 1e-12) << "at t=" << t;
    }

    // At and after the spike there is nothing left to sum.
    LambdaIAtTimeOutput after = eventprop_compute_lambda_i_at_time(input, 0.07);
    ASSERT_EQ(after.error_no, EVENTPROP_OK);
    EXPECT_DOUBLE_EQ(after.lambda_i, 0.0);
}

TEST(DiagLambdaIAtTime, PointwiseZeroSpikesIsZeroAndOk) {
    std::vector<double> w;
    LambdaITraceInput input{make_empty_forward(w), 0, nullptr, nullptr, nullptr, 0, 0.1, 0.001};

    LambdaIAtTimeOutput one = eventprop_compute_lambda_i_at_time(input, 0.03);
    EXPECT_EQ(one.error_no, EVENTPROP_OK);
    EXPECT_DOUBLE_EQ(one.lambda_i, 0.0);
}

TEST(DiagLambdaIAtTime, PointwiseRejectsInvalidQueryTime) {
    std::vector<double> w;
    std::vector<double> fwd_times = {0.03};
    std::vector<int>    fwd_sources = {0};
    std::vector<double> post_errors = {1.0};
    LambdaITraceInput input{make_empty_forward(w), 0, fwd_times.data(), fwd_sources.data(),
                            post_errors.data(), 1, 0.1, 0.001};

    /* Built at runtime rather than from the INFINITY/NAN macros: the deliberate /wd4756
     * suppression lives on the library target, so a constant-folded infinity in a test would
     * be an error under MSVC. The volatile keeps both divisions out of the constant folder. */
    volatile double zero = 0.0;
    const double bad_times[] = {-0.001, -1.0, 1.0 / zero, zero / zero};
    for (const double t : bad_times) {
        LambdaIAtTimeOutput one = eventprop_compute_lambda_i_at_time(input, t);
        EXPECT_EQ(one.error_no, EVENTPROP_DIAG_QUERY_TIME_INVALID) << "t=" << t;
        EXPECT_DOUBLE_EQ(one.lambda_i, 0.0);
    }
    // t = 0 is a legitimate query, not a sentinel.
    EXPECT_EQ(eventprop_compute_lambda_i_at_time(input, 0.0).error_no, EVENTPROP_OK);
}

// The two entry points validate through one routine, so a rejected input reads identically on
// both. This is the drift the shared preparation exists to prevent.
TEST(DiagLambdaIAtTime, PointwiseSharesTraceValidation) {
    std::vector<double> w;
    std::vector<double> unsorted = {0.06, 0.03};
    std::vector<int>    sources = {0, 0};
    std::vector<double> errors = {1.0, 1.0};

    LambdaITraceInput input{make_empty_forward(w), 0, unsorted.data(), sources.data(),
                            errors.data(), 2, 0.1, 0.001};

    LambdaITraceOutput trace = eventprop_compute_lambda_i_trace(input);
    LambdaIAtTimeOutput one = eventprop_compute_lambda_i_at_time(input, 0.01);
    EXPECT_EQ(trace.error_no, EVENTPROP_UNSORTED_SPIKE_TIMES);
    EXPECT_EQ(one.error_no, trace.error_no);

    input.target_nrn_idx = 7;
    EXPECT_EQ(eventprop_compute_lambda_i_at_time(input, 0.01).error_no,
              EVENTPROP_DIAG_TARGET_OUT_OF_RANGE);
    input.target_nrn_idx = 0;
    input.post_errors = nullptr;
    EXPECT_EQ(eventprop_compute_lambda_i_at_time(input, 0.01).error_no, EVENTPROP_NULL_POINTER);

    eventprop_free_lambda_i_trace_output(trace);
}

TEST(DiagVoltageTrace, ExcessiveSampleCountReturnsError) {
    std::vector<double> w = {1.0, 2.0};
    EventPropInput input{w.data(), 1, 2, nullptr, nullptr, 0, kTauMem, kTauSyn, kVTh};

    VoltageTraceInput trace_input{
        input, 0, nullptr, nullptr, 0, 1e6, 1e-3   // 1e9 samples
    };

    VoltageTraceOutput trace = eventprop_compute_voltage_trace(trace_input);
    EXPECT_EQ(trace.error_no, EVENTPROP_DIAG_TRACE_TOO_LONG);
    EXPECT_EQ(trace.n_t, 0);

    eventprop_free_voltage_trace_output(trace);
}
