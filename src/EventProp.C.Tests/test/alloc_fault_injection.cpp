/**
 * @file alloc_fault_injection.cpp
 * @brief Exercises every allocation-failure path in the library.
 *
 * The library allocates through eventprop_try_malloc/calloc, which can be told
 * to fail the n-th allocation. Running each entry point with every failure
 * index asserts that:
 *   - the call returns either OK or a documented EVENTPROP_ALLOC_* code,
 *   - the returned struct is safe to hand to its free function (no double free,
 *     no NULL dereference),
 *   - no memory leaks on the error path (verified under ASan).
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"

#include <vector>

namespace {

constexpr double kTauMem = 0.03;
constexpr double kTauSyn = 0.002;
constexpr double kVTh    = 1.0;

bool IsAllocErrorCode(EventPropError e) {
    return e >= EVENTPROP_ALLOC_SUM0 && e <= EVENTPROP_ALLOC_DROPOUT_OFFSETS;
}

/** Small single-neuron input: deterministic allocation order, no OpenMP. */
struct SmallInput {
    std::vector<double> w     = { 1.5, -0.7 };
    std::vector<double> times = { 0.05, 0.10, 0.30 };
    std::vector<int>    src   = { 0, 1, 0 };
};

EventPropInput make_input(const SmallInput& d) {
    return EventPropInput{ d.w.data(), 2, 1, d.times.data(), d.src.data(),
                           (int)d.times.size(), kTauMem, kTauSyn, kVTh };
}

}  // anonymous namespace

class AllocFaultInjection : public ::testing::Test {
   protected:
    void SetUp() override {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(0);  // injection off by default
    }
    void TearDown() override {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(0);
    }
};

// compute_sums: 2 allocations (sum0, sum1)
TEST_F(AllocFaultInjection, ComputeSumsFailsCleanly) {
    SmallInput data;
    const EventPropInput input = make_input(data);

    for (int nth = 1; nth <= 4; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        SumsOutput out = eventprop_compute_sums(input);
        EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
        eventprop_free_sums_output(out);
    }
}

// compute_spikes: sums + per-neuron buffers + spike collection
TEST_F(AllocFaultInjection, ComputeSpikesFailsCleanly) {
    SmallInput data;
    const EventPropInput input = make_input(data);

    for (int nth = 1; nth <= 24; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        SpikeOutput out = eventprop_compute_spikes(input);
        EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
        eventprop_free_spike_output(out);
    }
}

// compute_maxima: per-neuron jump arrays + maxima output
TEST_F(AllocFaultInjection, ComputeMaximaFailsCleanly) {
    SmallInput data;
    const EventPropInput input = make_input(data);

    for (int nth = 1; nth <= 16; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        MaximaOutput out = eventprop_compute_maxima(input);
        EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
        eventprop_free_maxima_output(out);
    }
}

// backward_spikes: jump arrays + lambda buffers + gradient
TEST_F(AllocFaultInjection, BackwardSpikesFailsCleanly) {
    SmallInput data;
    const EventPropInput input = make_input(data);

    eventprop_test_alloc_reset();
    eventprop_test_alloc_fail_nth(0);
    SpikeOutput spikes = eventprop_compute_spikes(input);
    ASSERT_EQ(spikes.error_no, EVENTPROP_OK) << "fixture must produce spikes";

    for (int nth = 1; nth <= 24; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        BackwardOutput out = eventprop_backward_spikes(input, spikes);
        EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
        eventprop_free_backward_output(out);
    }
    eventprop_free_spike_output(spikes);
}

// Batch entry points (batch_size = 1 keeps them off the OpenMP path).
TEST_F(AllocFaultInjection, BatchEntryPointsFailCleanly) {
    SmallInput data;
    const EventPropInput input = make_input(data);

    for (int nth = 1; nth <= 32; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        SpikesBatch spikes = eventprop_compute_spikes_batch(&input, 1);
        EXPECT_TRUE(spikes.error_no == EVENTPROP_OK || IsAllocErrorCode(spikes.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(spikes.error_no);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        MaximaBatch maxima = eventprop_compute_maxima_batch(&input, 1);
        EXPECT_TRUE(maxima.error_no == EVENTPROP_OK || IsAllocErrorCode(maxima.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(maxima.error_no);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        DropoutBatch drop{EVENTPROP_OK, 0, 0, nullptr};
        std::vector<double> dt = {0.05};
        std::vector<int> dsrc  = {0};
        std::vector<int> dn    = {1};
        drop = eventprop_dropout_spikes_batch(dt.data(), dsrc.data(), dn.data(), 1, 0.5, 42);
        EXPECT_TRUE(drop.error_no == EVENTPROP_OK || IsAllocErrorCode(drop.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(drop.error_no);

        eventprop_free_spikes_batch(spikes);
        eventprop_free_maxima_batch(maxima);
        eventprop_free_dropout_batch(drop);
    }
}

// Diagnostic entry points
TEST_F(AllocFaultInjection, DiagEntryPointsFailCleanly) {
    SmallInput data;
    const EventPropInput input = make_input(data);

    eventprop_test_alloc_reset();
    eventprop_test_alloc_fail_nth(0);
    SpikeOutput fwd = eventprop_compute_spikes(input);
    VoltageTraceInput vin{input, 0, fwd.times, fwd.sources, fwd.n_spikes, 0.5, 0.05};
    LambdaITraceInput lin{fwd.times, fwd.n_spikes, 0, kVTh, kTauMem, kTauSyn, 0.5, 0.05};

    for (int nth = 1; nth <= 8; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        VoltageTraceOutput vt = eventprop_compute_voltage_trace(vin);
        EXPECT_TRUE(vt.error_no == EVENTPROP_OK || IsAllocErrorCode(vt.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(vt.error_no);
        eventprop_free_voltage_trace_output(vt);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        LambdaITraceOutput li = eventprop_compute_lambda_i_trace(lin);
        EXPECT_TRUE(li.error_no == EVENTPROP_OK || IsAllocErrorCode(li.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(li.error_no);
        eventprop_free_lambda_i_trace_output(li);
    }
    eventprop_free_spike_output(fwd);
}
