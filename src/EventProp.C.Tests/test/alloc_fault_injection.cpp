/**
 * @file alloc_fault_injection.cpp
 * @brief Exercises every allocation-failure path in the library.
 *
 * The library allocates through eventprop_try_malloc/calloc/realloc, which can be
 * told to fail the n-th allocation. Running each entry point with every failure
 * index asserts that:
 *   - the call returns either OK or a documented EVENTPROP_ALLOC_* code,
 *   - the returned struct is safe to hand to its free function (no double free,
 *     no NULL dereference, no dangling owner slot),
 *   - no memory leaks on the error path (verified under ASan),
 *   - and, via the reachability metric, that every allocation site belonging to
 *     that entry point was actually exercised.
 */

#include <gtest/gtest.h>
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"

#include <vector>
#include <set>
#include <sstream>
#include <iterator>

namespace {

constexpr double kTauMem = 0.02;
constexpr double kTauSyn = 0.005;
constexpr double kVTh    = 1.0;

bool IsAllocErrorCode(EventPropError e) {
    return e >= EVENTPROP_ALLOC_SUM0 && e <= EVENTPROP_ALLOC_BWD_CURRENTS;
}

/** Codes that are legitimate alongside allocation failures: a failed sample is
 *  reported through these, not through an allocation code. */
bool IsExpectedInjectionCode(EventPropError e) {
    return IsAllocErrorCode(e) || e == EVENTPROP_BATCH_PARTIAL_ACCUMULATION ||
           e == EVENTPROP_BATCHSIZE_ZERO_OR_NEG;  // batch_size nulled on internal failure
}

/** Fails with a listing of the allocation sites this entry point never reached. */
void ExpectCodesReached(const std::set<int>& observed,
                        const EventPropError* expected, size_t count,
                        const char* label)
{
    std::ostringstream missing;
    size_t hit = 0;
    for (size_t i = 0; i < count; ++i) {
        if (observed.count(expected[i])) ++hit;
        else missing << "\n  unreachable: " << eventprop_error_string(expected[i]);
    }
    EXPECT_EQ(hit, count) << label << " never reached these allocation sites:"
                          << missing.str();
}

/** Per-neuron buffers for the fixture: deterministic allocation order, no OpenMP.
 *  Fires twice, so the backward and collection paths are exercised. */
/** Same shape as SmallInput, but one neuron fires many times so the */
/* scratch-row growth (realloc) path is exercised. */
struct GrowthInput {
    std::vector<double> w     = { 15.0 };
    std::vector<double> times = { 0.01, 0.02, 0.03 };
    std::vector<int>    src   = { 0, 0, 0 };
};

struct SmallInput {
    std::vector<double> w     = { 10.0, -8.0, 10.0 };
    std::vector<double> times = { 0.05, 0.20, 0.40 };
    std::vector<int>    src   = { 0, 1, 2 };
};

template <typename T>
EventPropInput make_input(const T& d) {
    return EventPropInput{ d.w.data(), (int)d.w.size(), 1, d.times.data(), d.src.data(),
                           (int)d.times.size(), kTauMem, kTauSyn, kVTh };
}

}  // anonymous namespace

class AllocFaultInjection : public ::testing::Test {
   protected:
    void SetUp() override {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(0);  // injection off by default
        /* The forward sweep allocates scratch per thread, so the allocation */
        /* indices this suite walks shift with the OpenMP width. Pin it. */
        eventprop_test_set_max_threads(1);
    }
    void TearDown() override {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(0);
        eventprop_test_set_max_threads(0);  // restore the runtime default
    }
};

// ---------------------------------------------------------------------------
// compute_sums
// ---------------------------------------------------------------------------
TEST_F(AllocFaultInjection, ComputeSumsFailsCleanly) {
    static const EventPropError expected[] = {
        EVENTPROP_ALLOC_SUM0, EVENTPROP_ALLOC_SUM1,
        EVENTPROP_ALLOC_EXP_MEM, EVENTPROP_ALLOC_EXP_SYN,
    };
    std::set<int> observed;

    SmallInput data;
    const EventPropInput input = make_input(data);

    for (int nth = 1; nth <= 8; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        SumsOutput out = eventprop_compute_sums(input);
        EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
        observed.insert(out.error_no);
        eventprop_free_sums_output(out);
    }

    ExpectCodesReached(observed, expected, std::size(expected), "compute_sums");
}

// ---------------------------------------------------------------------------
// compute_spikes (includes the nested compute_sums allocations)
// ---------------------------------------------------------------------------
TEST_F(AllocFaultInjection, ComputeSpikesFailsCleanly) {
    static const EventPropError expected[] = {
        EVENTPROP_ALLOC_SUM0,         EVENTPROP_ALLOC_SUM1,
        EVENTPROP_ALLOC_EXP_MEM,      EVENTPROP_ALLOC_EXP_SYN,
        EVENTPROP_ALLOC_THREAD_SCRATCH,
        EVENTPROP_ALLOC_RESULT_TIMES,  EVENTPROP_ALLOC_RESULT_SOURCES,
    };
    std::set<int> observed;

    /* Two fixtures: the small one covers the entry-point allocations, the */
    /* strong-weight one fires the same neuron many times so the scratch row */
    /* growth path (realloc) is reached as well. */
    SmallInput small;
    const EventPropInput in_small = make_input(small);

    GrowthInput big;
    const EventPropInput in_big = make_input(big);

    const EventPropInput inputs[] = { in_small, in_big };
    for (const EventPropInput& input : inputs) {
        /* With the parallel width pinned in SetUp, the total allocation */
        /* count of a call is fixed; this bound only has to exceed it, and */
        /* covers the per-attempt temporary of the root-finding sweep. */
        for (int nth = 1; nth <= 4096; ++nth) {
            eventprop_test_alloc_reset();
            eventprop_test_alloc_fail_nth(nth);

            SpikeOutput out = eventprop_compute_spikes(input);
            EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
                << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
            observed.insert(out.error_no);
            eventprop_free_spike_output(out);
        }
    }

    ExpectCodesReached(observed, expected, std::size(expected), "compute_spikes");
}

// ---------------------------------------------------------------------------
// compute_maxima
// ---------------------------------------------------------------------------
TEST_F(AllocFaultInjection, ComputeMaximaFailsCleanly) {
    static const EventPropError expected[] = {
        EVENTPROP_ALLOC_SUM0,        EVENTPROP_ALLOC_SUM1,
        EVENTPROP_ALLOC_EXP_MEM,     EVENTPROP_ALLOC_EXP_SYN,
        EVENTPROP_ALLOC_MAXIMA_TIMES, EVENTPROP_ALLOC_MAXIMA_VALUES,
        EVENTPROP_ALLOC_MAXIMA_ERRORS,
    };
    std::set<int> observed;

    SmallInput data;
    const EventPropInput input = make_input(data);

    for (int nth = 1; nth <= 16; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        MaximaOutput out = eventprop_compute_maxima(input);
        EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
        observed.insert(out.error_no);
        eventprop_free_maxima_output(out);
    }

    ExpectCodesReached(observed, expected, std::size(expected), "compute_maxima");
}

// ---------------------------------------------------------------------------
// backward_spikes
// ---------------------------------------------------------------------------
TEST_F(AllocFaultInjection, BackwardSpikesFailsCleanly) {
    static const EventPropError expected[] = {
        EVENTPROP_ALLOC_BWD_GRAD,        EVENTPROP_ALLOC_BWD_SPIKE_ERRORS,
        EVENTPROP_ALLOC_BWD_MERGED,      EVENTPROP_ALLOC_BWD_ALL_ERRORS,
        EVENTPROP_ALLOC_BWD_LAMBDA_V,    EVENTPROP_ALLOC_BWD_LAMBDA_I,
        EVENTPROP_ALLOC_BWD_JUMP_TIME,   EVENTPROP_ALLOC_BWD_JUMP_MAG,
        EVENTPROP_ALLOC_BWD_JUMP_COUNT,  EVENTPROP_ALLOC_BWD_JUMP_CAP,
        EVENTPROP_ALLOC_BWD_PER_NEURON,  EVENTPROP_ALLOC_BWD_CUR_SUM,
        EVENTPROP_ALLOC_BWD_CURRENTS,
    };
    std::set<int> observed;

    SmallInput data;
    const EventPropInput input = make_input(data);

    eventprop_test_alloc_reset();
    eventprop_test_alloc_fail_nth(0);
    SpikeOutput spikes = eventprop_compute_spikes(input);
    ASSERT_EQ(spikes.error_no, EVENTPROP_OK) << "fixture must produce spikes";

    /* The post-spike derivatives are caller-owned now. */
    std::vector<double> errs((size_t)spikes.n_spikes, 1.0);

    for (int nth = 1; nth <= 128; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        BackwardOutput out = eventprop_backward_spikes(input, spikes, errs.data());
        EXPECT_TRUE(out.error_no == EVENTPROP_OK || IsAllocErrorCode(out.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(out.error_no);
        observed.insert(out.error_no);
        eventprop_free_backward_output(out);
    }
    eventprop_free_spike_output(spikes);

    ExpectCodesReached(observed, expected, std::size(expected), "backward_spikes");
}

// ---------------------------------------------------------------------------
// Batch entry points (batch_size = 1 keeps them off the OpenMP path)
// ---------------------------------------------------------------------------
TEST_F(AllocFaultInjection, BatchEntryPointsFailCleanly) {
    static const EventPropError expected[] = {
        EVENTPROP_ALLOC_BATCH_OUTPUTS,
        EVENTPROP_ALLOC_DROPOUT_OFFSETS, EVENTPROP_ALLOC_DROPOUT_SAMPLE,
    };
    std::set<int> observed;

    SmallInput data;
    const EventPropInput input = make_input(data);

    std::vector<double> dt = {0.05};
    std::vector<int> dsrc  = {0};
    std::vector<int> dn    = {1};

    for (int nth = 1; nth <= 64; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        SpikesBatch spikes = eventprop_compute_spikes_batch(&input, 1);
        EXPECT_TRUE(spikes.error_no == EVENTPROP_OK || IsAllocErrorCode(spikes.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(spikes.error_no);
        observed.insert(spikes.error_no);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        MaximaBatch maxima = eventprop_compute_maxima_batch(&input, 1);
        EXPECT_TRUE(maxima.error_no == EVENTPROP_OK || IsAllocErrorCode(maxima.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(maxima.error_no);
        observed.insert(maxima.error_no);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        DropoutBatch drop = eventprop_dropout_spikes_batch(
            dt.data(), dsrc.data(), dn.data(), 1, 0.5, 42);
        EXPECT_TRUE(drop.error_no == EVENTPROP_OK || IsAllocErrorCode(drop.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(drop.error_no);
        observed.insert(drop.error_no);

        eventprop_free_spikes_batch(spikes);
        eventprop_free_maxima_batch(maxima);
        eventprop_free_dropout_batch(drop);
    }

    ExpectCodesReached(observed, expected, std::size(expected), "batch entry points");
}

// ---------------------------------------------------------------------------
// Accumulate entry points — the only site reporting ALLOC_BATCH_GRAD
// ---------------------------------------------------------------------------
TEST_F(AllocFaultInjection, AccumulateEntryPointsFailCleanly) {
    static const EventPropError expected[] = {
        EVENTPROP_ALLOC_BATCH_GRAD,       EVENTPROP_ALLOC_BATCH_OUTPUTS,
        EVENTPROP_ALLOC_BWD_GRAD,         EVENTPROP_ALLOC_BWD_SPIKE_ERRORS,
        EVENTPROP_ALLOC_BWD_MERGED,       EVENTPROP_ALLOC_BWD_ALL_ERRORS,
        EVENTPROP_ALLOC_BWD_LAMBDA_V,     EVENTPROP_ALLOC_BWD_LAMBDA_I,
        EVENTPROP_ALLOC_BWD_JUMP_TIME,    EVENTPROP_ALLOC_BWD_JUMP_MAG,
        EVENTPROP_ALLOC_BWD_JUMP_COUNT,   EVENTPROP_ALLOC_BWD_JUMP_CAP,
        EVENTPROP_ALLOC_BWD_PER_NEURON,   EVENTPROP_ALLOC_BWD_CUR_SUM,
        EVENTPROP_ALLOC_BWD_CURRENTS,
    };
    std::set<int> observed;

    SmallInput data;
    const EventPropInput input = make_input(data);

    eventprop_test_alloc_reset();
    eventprop_test_alloc_fail_nth(0);
    SpikesBatch spikes = eventprop_compute_spikes_batch(&input, 1);
    ASSERT_EQ(spikes.error_no, EVENTPROP_OK) << "fixture must produce spikes";

    /* The post-spike derivatives are caller-owned: one array per sample. */
    std::vector<double> errs((size_t)spikes.spikes[0].n_spikes, 1.0);
    const double* ptrs[1] = { errs.data() };

    for (int nth = 1; nth <= 128; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        AccumulatedBackwardResult acc =
            eventprop_backward_spikes_batch_accumulate(&input, spikes, ptrs);
        EXPECT_TRUE(acc.error_no == EVENTPROP_OK || IsExpectedInjectionCode(acc.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(acc.error_no);
        observed.insert(acc.error_no);
        eventprop_free_accumulated_backward_result(acc);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        BackwardsBatch bw = eventprop_backward_spikes_batch(&input, spikes, ptrs);
        EXPECT_TRUE(bw.error_no == EVENTPROP_OK || IsExpectedInjectionCode(bw.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(bw.error_no);
        observed.insert(bw.error_no);
        /* Per-sample codes carry the allocation failure of each sample. */
        for (int i = 0; i < bw.batch_size && bw.outputs; ++i)
            observed.insert(bw.outputs[i].error_no);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        GradientOutput grad = eventprop_accumulate_gradients(bw, (int)data.w.size(), 1);
        EXPECT_TRUE(grad.error_no == EVENTPROP_OK || IsExpectedInjectionCode(grad.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(grad.error_no);
        observed.insert(grad.error_no);
        eventprop_free_gradient_output(grad);

        eventprop_free_backwards_batch(bw);
    }
    eventprop_free_spikes_batch(spikes);

    ExpectCodesReached(observed, expected, std::size(expected), "accumulate entry points");
}

// ---------------------------------------------------------------------------
// Diagnostic entry points
// ---------------------------------------------------------------------------
TEST_F(AllocFaultInjection, DiagEntryPointsFailCleanly) {
    static const EventPropError expected[] = {
        EVENTPROP_ALLOC_DIAG_V_TRACE, EVENTPROP_ALLOC_DIAG_TS,
        EVENTPROP_ALLOC_DIAG_LAMBDA,
    };
    std::set<int> observed;

    SmallInput data;
    const EventPropInput input = make_input(data);

    eventprop_test_alloc_reset();
    eventprop_test_alloc_fail_nth(0);
    SpikeOutput fwd = eventprop_compute_spikes(input);
    VoltageTraceInput vin{input, 0, fwd.times, fwd.sources, fwd.n_spikes, 0.5, 0.05};
    LambdaITraceInput lin{fwd.times, fwd.n_spikes, 0, kVTh, kTauMem, kTauSyn, 0.5, 0.05};

    for (int nth = 1; nth <= 32; ++nth) {
        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);

        VoltageTraceOutput vt = eventprop_compute_voltage_trace(vin);
        EXPECT_TRUE(vt.error_no == EVENTPROP_OK || IsAllocErrorCode(vt.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(vt.error_no);
        observed.insert(vt.error_no);
        eventprop_free_voltage_trace_output(vt);

        eventprop_test_alloc_reset();
        eventprop_test_alloc_fail_nth(nth);
        LambdaITraceOutput li = eventprop_compute_lambda_i_trace(lin);
        EXPECT_TRUE(li.error_no == EVENTPROP_OK || IsAllocErrorCode(li.error_no))
            << "nth=" << nth << " returned " << eventprop_error_string(li.error_no);
        observed.insert(li.error_no);
        eventprop_free_lambda_i_trace_output(li);
    }
    eventprop_free_spike_output(fwd);

    ExpectCodesReached(observed, expected, std::size(expected), "diagnostics");
}
