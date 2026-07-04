#include "eventprop_batch.h"
#include <stdlib.h>
#include <stdint.h>

void eventprop_free_spikes_batch(SpikesBatch* batch)
{
    if (batch == NULL) return;
    for (int b = 0; b < batch->batch_size; b++) {
        eventprop_free_spike_output(&batch->spikes[b]);
    }
    free(batch->spikes);
    batch->spikes = NULL;
    batch->batch_size = 0;
}

void eventprop_free_spikes_with_inputs_batch(SpikesBatchWithInputs* batch)
{
    if (batch == NULL) return;
    for (int b = 0; b < batch->batch_size; b++) {
        eventprop_free_spike_output(&batch->spikes[b]);
    }
    free(batch->spikes);
    free(batch->inputs);
    batch->spikes = NULL;
    batch->inputs = NULL;
    batch->batch_size = 0;
}

void eventprop_free_maxima_batch(MaximaBatch* batch)
{
    if (batch == NULL) return;
    for (int b = 0; b < batch->batch_size; b++) {
        eventprop_free_maxima_output(&batch->outputs[b]);
    }
    free(batch->outputs);
    batch->outputs = NULL;
    batch->batch_size = 0;
}

void eventprop_free_maxima_with_inputs_batch(MaximaBatchWithInputs* batch)
{
    if (batch == NULL) return;
    for (int b = 0; b < batch->batch_size; b++) {
        eventprop_free_maxima_output(&batch->outputs[b]);
    }
    free(batch->outputs);
    free(batch->inputs);
    batch->outputs = NULL;
    batch->inputs = NULL;
    batch->batch_size = 0;
}

void eventprop_free_backwards_batch(BackwardsBatch* batch)
{
    if (batch == NULL) return;
    for (int b = 0; b < batch->batch_size; b++) {
        eventprop_free_backward_output(&batch->outputs[b]);
    }
    free(batch->outputs);
    batch->outputs = NULL;
    batch->batch_size = 0;
}

SpikesBatch eventprop_compute_spikes_batch(const EventPropInput* inputs, int batch_size)
{
    SpikesBatch result;
    result.batch_size = batch_size;
    result.has_errors = 0;

    /* Allocate result array */
    result.spikes = (SpikeOutput*)calloc(batch_size, sizeof(SpikeOutput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            result.spikes[b] = eventprop_compute_spikes(inputs[b]);
            if (result.spikes[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            result.spikes[b] = eventprop_compute_spikes(inputs[b]);
            if (result.spikes[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
        }
    }
    return result;
}

SpikesBatchWithInputs eventprop_compute_spikes_batch_flat(
    const double* w,
    int n_pre, int n_post,
    const double* times,
    const int* sources,
    const int* n_per_sample,
    int batch_size,
    double tau_mem, double tau_syn, double v_th)
{
    SpikesBatchWithInputs result;
    result.batch_size = batch_size;
    result.has_errors = 0;

    /* Allocate result arrays — malloc is sufficient; zero-init from calloc adds ~400us overhead per call on Windows CRT */
    result.spikes = (SpikeOutput*)malloc(batch_size * sizeof(SpikeOutput));
    result.inputs = (EventPropInput*)malloc(batch_size * sizeof(EventPropInput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
        /* Parallel path — OpenMP runtime init cost (~800us on Windows MSVC) is amortized across samples */
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            const double* t_ptr   = times;
            const int* s_ptr      = sources;
            for (int k = 0; k < b; k++) {
                t_ptr += n_per_sample[k];
                s_ptr += n_per_sample[k];
            }
            EventPropInput input = { .w=w, .n_pre=n_pre, .n_post=n_post,
                                     .times=t_ptr, .sources=s_ptr, .n_spikes=n_per_sample[b],
                                     .tau_mem=tau_mem, .tau_syn=tau_syn, .v_th=v_th };
            result.spikes[b] = eventprop_compute_spikes(input);
            if (result.spikes[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
            result.inputs[b] = input;
        }
    } else
#endif
    {
        /* Sequential path — avoids OpenMP runtime overhead for small batches */
        for (b = 0; b < batch_size; b++) {
            const double* t_ptr   = times;
            const int* s_ptr      = sources;
            for (int k = 0; k < b; k++) {
                t_ptr += n_per_sample[k];
                s_ptr += n_per_sample[k];
            }
            EventPropInput input = { .w=w, .n_pre=n_pre, .n_post=n_post,
                                     .times=t_ptr, .sources=s_ptr, .n_spikes=n_per_sample[b],
                                     .tau_mem=tau_mem, .tau_syn=tau_syn, .v_th=v_th };
            result.spikes[b] = eventprop_compute_spikes(input);
            if (result.spikes[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
            result.inputs[b] = input;
        }
    }
    return result;
}

BackwardsBatch eventprop_backward_spikes_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    int batch_size = spikes_batch.batch_size;
    BackwardsBatch batch;
    batch.batch_size = batch_size;
    batch.has_errors = 0;

    /* Parallel per-sample backward */
    batch.outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                batch.has_errors = 1;
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                batch.has_errors = 1;
        }
    }

    return batch;
}

MaximaBatch eventprop_compute_maxima_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    int batch_size = spikes_batch.batch_size;
    MaximaBatch batch;
    batch.batch_size = batch_size;
    batch.has_errors = 0;

    /* Parallel per-sample maxima */
    batch.outputs = (MaximaOutput*)calloc(batch_size, sizeof(MaximaOutput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_compute_maxima(inputs[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                batch.has_errors = 1;
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_compute_maxima(inputs[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                batch.has_errors = 1;
        }
    }

    return batch;
}

MaximaBatchWithInputs eventprop_compute_maxima_batch_flat(
    const double* w,
    int n_pre, int n_post,
    const double* times,
    const int* sources,
    const int* n_per_sample,
    int batch_size,
    double tau_mem, double tau_syn, double v_th)
{
    MaximaBatchWithInputs result;
    result.batch_size = batch_size;
    result.has_errors = 0;

    /* Allocate arrays */
    result.outputs = (MaximaOutput*)calloc(batch_size, sizeof(MaximaOutput));
    result.inputs  = (EventPropInput*)calloc(batch_size, sizeof(EventPropInput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            const double* t_ptr   = times;
            const int* s_ptr      = sources;
            for (int k = 0; k < b; k++) {
                t_ptr += n_per_sample[k];
                s_ptr += n_per_sample[k];
            }
            EventPropInput input = { .w=w, .n_pre=n_pre, .n_post=n_post,
                                     .times=t_ptr, .sources=s_ptr, .n_spikes=n_per_sample[b],
                                     .tau_mem=tau_mem, .tau_syn=tau_syn, .v_th=v_th };
            result.outputs[b] = eventprop_compute_maxima(input);
            if (result.outputs[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
            result.inputs[b]  = input;
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            const double* t_ptr   = times;
            const int* s_ptr      = sources;
            for (int k = 0; k < b; k++) {
                t_ptr += n_per_sample[k];
                s_ptr += n_per_sample[k];
            }
            EventPropInput input = { .w=w, .n_pre=n_pre, .n_post=n_post,
                                     .times=t_ptr, .sources=s_ptr, .n_spikes=n_per_sample[b],
                                     .tau_mem=tau_mem, .tau_syn=tau_syn, .v_th=v_th };
            result.outputs[b] = eventprop_compute_maxima(input);
            if (result.outputs[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
            result.inputs[b]  = input;
        }
    }
    return result;
}

BackwardsBatch eventprop_backward_maxima_batch(const MaximaBatchWithInputs maxima_batch)
{
    int batch_size = maxima_batch.batch_size;
    BackwardsBatch batch;
    batch.batch_size = batch_size;
    batch.has_errors = 0;

    /* Parallel per-sample backward */
    batch.outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_maxima(
                maxima_batch.inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                batch.has_errors = 1;
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_maxima(
                maxima_batch.inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                batch.has_errors = 1;
        }
    }

    return batch;
}

void eventprop_free_gradient_output(GradientOutput* output)
{
    if (output == NULL) return;
    free(output->data);
    output->data = NULL;
    output->rows = 0;
    output->cols = 0;
}

GradientOutput eventprop_accumulate_gradients(BackwardsBatch batch, int n_pre, int n_post)
{
    GradientOutput result;
    result.rows = n_pre;
    result.cols = n_post;
    result.data = (double*)calloc(n_pre * n_post, sizeof(double));

    for (int b = 0; b < batch.batch_size; b++) {
        if (batch.outputs[b].error_no != EVENTPROP_OK) continue;
        for (int i = 0; i < n_pre * n_post; i++)
            result.data[i] += batch.outputs[b].grad[i];
    }

    return result;
}

void eventprop_free_accumulated_backward_result(AccumulatedBackwardResult* result)
{
    if (result == NULL) return;
        
    eventprop_free_gradient_output(&result->summed_grads);

    for (int b = 0; b < result->batch_size; b++) {
        eventprop_free_backward_output(&result->outputs[b]);
    }
    free(result->outputs);
    result->outputs = NULL;
    result->batch_size = 0;
}

AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(
    const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    int batch_size = spikes_batch.batch_size;
    AccumulatedBackwardResult result;
    result.batch_size = batch_size;
    result.has_errors = 0;

    /* Parallel per-sample backward */
    BackwardOutput* outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (outputs[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (outputs[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
        }
    }

    BackwardsBatch batch = {
        .batch_size = batch_size,
        .has_errors = 0,   /* not used here */
        .outputs = outputs
    };
    
    result.summed_grads = eventprop_accumulate_gradients(batch, inputs[0].n_pre, inputs[0].n_post);
    result.outputs = outputs;
    return result;
}

AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
    const MaximaBatchWithInputs maxima_batch)
{
    int batch_size = maxima_batch.batch_size;
    AccumulatedBackwardResult result;
    result.batch_size = batch_size;
    result.has_errors = 0;

    /* Parallel per-sample backward */
    BackwardOutput* outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_maxima(
                maxima_batch.inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (outputs[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_maxima(
                maxima_batch.inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (outputs[b].error_no != EVENTPROP_OK)
                result.has_errors = 1;
        }
    }

    BackwardsBatch batch = {
        .batch_size = batch_size,
        .has_errors = 0,   /* not used here */
        .outputs = outputs
    };

    result.summed_grads = eventprop_accumulate_gradients(batch, maxima_batch.inputs[0].n_pre, maxima_batch.inputs[0].n_post);

    result.outputs = outputs;
    return result;
}

static inline double lcg_rand(uint64_t* state)
{
    *state = (*state * 6364136223846793005ULL + 1442695040888963407ULL);
    return ((double)(*state >> 11)) / 9007199254740992.0;  /* [0, 1), ~53 bits of precision */
}

/**
 * Free a single dropout output.
 */
void eventprop_free_dropout_output(DropoutOutput* output)
{
    if (output == NULL) return;
    free(output->times);
    free(output->sources);
    output->times = NULL;
    output->sources = NULL;
}

/**
 * Free a dropout batch result.
 */
void eventprop_free_dropout_batch(DropoutBatch* batch)
{
    if (batch == NULL) return;
    for (int b = 0; b < batch->batch_size; b++) {
        eventprop_free_dropout_output(&batch->outputs[b]);
    }
    free(batch->outputs);
    batch->outputs = NULL;
    batch->batch_size = 0;
}

/**
 * Flat-array dropout — randomly drop spikes from each sample.
 *
 * Accepts flat concatenated spike arrays and per-sample offset counts.
 * Designed for FFI callers (Cython, P/Invoke) that have flat numpy buffers.
 */
DropoutBatch eventprop_dropout_spikes_batch(
    const double* times,
    const int* sources,
    const int* n_per_sample,
    int batch_size,
    double p_drop,
    int seed
)
{
    DropoutBatch result;
    result.batch_size = batch_size;
    result.has_errors = 0;   /* dropout has no per-sample errors */
    result.outputs = (DropoutOutput*)calloc(batch_size, sizeof(DropoutOutput));

    /* Compute sample offsets */
    int* offsets = (int*)malloc((batch_size + 1) * sizeof(int));
    offsets[0] = 0;
    for (int i = 0; i < batch_size; i++) {
        offsets[i+1] = offsets[i] + n_per_sample[i];
    }

    /* Process each sample — sequential (malloc per-sample) */
    for (int b = 0; b < batch_size; b++) {
        int offset = offsets[b];
        int n_spikes = n_per_sample[b];

        /* Per-sample seed for reproducibility */
        uint64_t state = (uint64_t)(seed + b) + 1;

        if (n_spikes == 0 || p_drop <= 0.0) {
            /* No dropout — copy all spikes */
            result.outputs[b].n_spikes = n_spikes;
            if (n_spikes > 0) {
                result.outputs[b].times = (double*)malloc(n_spikes * sizeof(double));
                result.outputs[b].sources = (int*)malloc(n_spikes * sizeof(int));
                for (int i = 0; i < n_spikes; i++) {
                    result.outputs[b].times[i] = times[offset + i];
                    result.outputs[b].sources[i] = sources[offset + i];
                }
            }
            continue;
        }

        /* Single-pass: pre-allocate max size, then compact */
        result.outputs[b].times = (double*)malloc(n_spikes * sizeof(double));
        result.outputs[b].sources = (int*)malloc(n_spikes * sizeof(int));

        int idx = 0;
        for (int i = 0; i < n_spikes; i++) {
            if (lcg_rand(&state) > p_drop) {
                result.outputs[b].times[idx] = times[offset + i];
                result.outputs[b].sources[idx] = sources[offset + i];
                idx++;
            }
        }
        result.outputs[b].n_spikes = idx;
    }

    free(offsets);
    return result;
}
