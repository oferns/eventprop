/**
 * @file eventprop_memory.c
 * @brief All public output-freeing functions for the EventProp C library.
 *
 * Every heap-backed result type has exactly one free function, collected here:
 * - Core single-sample outputs (eventprop.h)
 * - Batch results (eventprop_batch.h)
 * - Diagnostic traces (eventprop_diag.h)
 * - Loss readouts (eventprop_loss.h)
 *
 * Also hosts the allocation-fault-injection shim used to exercise error paths
 * in tests (see eventprop_test_alloc_fail_nth).
 *
 * All take their struct by value. Freeing a zeroed/empty struct is safe.
 */
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"
#include "eventprop_loss.h"
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Allocation fault injection (test-only)                              */
/* ------------------------------------------------------------------ */
/* Library code allocates through these wrappers so tests can force the */
/* n-th allocation to fail and exercise every EVENTPROP_ALLOC_* path.   */
/* When disabled (the default) the cost is a single predictable branch. */
static int alloc_fail_target = 0;   /* <= 0 disables injection */
static int alloc_count = 0;         /* allocations since last reset */

void eventprop_test_alloc_reset(void)
{
    alloc_count = 0;
}

void eventprop_test_alloc_fail_nth(const int nth)
{
    alloc_fail_target = nth;
}

void* eventprop_try_malloc(const size_t size)
{
    if (alloc_fail_target > 0 && ++alloc_count == alloc_fail_target)
        return NULL;
    return malloc(size);
}

void* eventprop_try_calloc(const size_t count, const size_t elem_size)
{
    if (alloc_fail_target > 0 && ++alloc_count == alloc_fail_target)
        return NULL;
    return calloc(count, elem_size);
}

/* realloc is counted too: the per-neuron growth paths are the ones that can */
/* overflow, so they must be injectable like every other allocation. */
void* eventprop_try_realloc(void* ptr, const size_t size)
{
    if (alloc_fail_target > 0 && ++alloc_count == alloc_fail_target)
        return NULL;
    return realloc(ptr, size);
}

/* ------------------------------------------------------------------ */
/* Core single-sample outputs                                          */
/* ------------------------------------------------------------------ */

void eventprop_free_spike_output(const SpikeOutput output)
{
    free(output.times);
    free(output.sources);
}

void eventprop_free_sums_output(const SumsOutput output)
{
    free(output.sum0);
    free(output.sum1);
}

void eventprop_free_maxima_output(const MaximaOutput output)
{
    free(output.times);
    free(output.values);
    free(output.errors);
}

void eventprop_free_backward_output(const BackwardOutput output)
{
    free(output.grad);
    free(output.spike_errors);
}

/* ------------------------------------------------------------------ */
/* Batch results                                                       */
/* ------------------------------------------------------------------ */

void eventprop_free_spikes_batch(const SpikesBatch batch)
{
    if (batch.spikes) {
        for (int b = 0; b < batch.batch_size; b++) {
            eventprop_free_spike_output(batch.spikes[b]);
        }
    }
    free(batch.spikes);
}

void eventprop_free_maxima_batch(const MaximaBatch batch)
{
    if (batch.outputs) {
        for (int b = 0; b < batch.batch_size; b++) {
            eventprop_free_maxima_output(batch.outputs[b]);
        }
    }
    free(batch.outputs);
}

void eventprop_free_backwards_batch(const BackwardsBatch batch)
{
    if (batch.outputs) {
        for (int b = 0; b < batch.batch_size; b++) {
            eventprop_free_backward_output(batch.outputs[b]);
        }
    }
    free(batch.outputs);
}

void eventprop_free_gradient_output(const GradientOutput output)
{
    free(output.data);
}

void eventprop_free_accumulated_backward_result(const AccumulatedBackwardResult result)
{
    if (result.summed_grads) {
        eventprop_free_gradient_output(*result.summed_grads);
        free(result.summed_grads);
    }

    if (result.outputs) {
        for (int b = 0; b < result.batch_size; b++) {
            eventprop_free_backward_output(result.outputs[b]);
        }
    }
    free(result.outputs);
}

void eventprop_free_dropout_output(const DropoutOutput output)
{
    free(output.times);
    free(output.sources);
}

void eventprop_free_dropout_batch(const DropoutBatch batch)
{
    if (batch.outputs) {
        for (int b = 0; b < batch.batch_size; b++) {
            eventprop_free_dropout_output(batch.outputs[b]);
        }
    }
    free(batch.outputs);
}

/* ------------------------------------------------------------------ */
/* Diagnostic traces                                                   */
/* ------------------------------------------------------------------ */

void eventprop_free_lambda_i_trace_output(const LambdaITraceOutput output)
{
    free(output.lambda_trace);
    free(output.ts);
}

void eventprop_free_voltage_trace_output(const VoltageTraceOutput output)
{
    free(output.v_trace);
    free(output.ts);
}

/* ------------------------------------------------------------------ */
/* Loss readouts                                                       */
/* ------------------------------------------------------------------ */

void eventprop_free_maxima_regression_output(const MaximaRegressionOutput output)
{
    free(output.errors);
    free(output.decoder_grad);
}
