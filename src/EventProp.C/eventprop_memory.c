/**
 * @file eventprop_memory.c
 * @brief All public output-freeing functions for the EventProp C library.
 *
 * Every heap-backed result type has exactly one free function, collected here:
 * - Core single-sample outputs (eventprop.h)
 * - Batch results (eventprop_batch.h)
 * - Diagnostic traces (eventprop_diag.h)
 *
 * All take their struct by value. Freeing a zeroed/empty struct is safe.
 */
#include "eventprop.h"
#include "eventprop_batch.h"
#include "eventprop_diag.h"
#include <stdlib.h>

/* ------------------------------------------------------------------ */
/* Core single-sample outputs                                          */
/* ------------------------------------------------------------------ */

void eventprop_free_spike_output(const SpikeOutput output)
{
    free(output.times);
    free(output.sources);
    free(output.currents);
    free(output.errors);
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
