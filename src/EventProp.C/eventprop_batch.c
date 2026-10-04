#include "eventprop_batch.h"
#include "eventprop.h"
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

static inline double lcg_rand(uint64_t* state)
{
    *state = (*state * 6364136223846793005ULL + 1442695040888963407ULL);
    return ((double)(*state >> 11)) / 9007199254740992.0;  /* [0, 1), ~53 bits of precision */
}

void eventprop_free_spikes_batch(const SpikesBatch batch)
{
    for (int b = 0; b < batch.batch_size; b++) {
        eventprop_free_spike_output(batch.spikes[b]);
    }
    free(batch.spikes);
}

void eventprop_free_maxima_batch(const MaximaBatch batch)
{
    for (int b = 0; b < batch.batch_size; b++) {
        eventprop_free_maxima_output(batch.outputs[b]);
    }
    free(batch.outputs);
}

void eventprop_free_backwards_batch(const BackwardsBatch batch)
{
    for (int b = 0; b < batch.batch_size; b++) {
        eventprop_free_backward_output(batch.outputs[b]);
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

    for (int b = 0; b < result.batch_size; b++) {
        eventprop_free_backward_output(result.outputs[b]);
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
    for (int b = 0; b < batch.batch_size; b++) {
        eventprop_free_dropout_output(batch.outputs[b]);
    }
    free(batch.outputs);
}

SpikesBatch eventprop_compute_spikes_batch(const EventPropInput* inputs, const int batch_size)
{
    SpikesBatch result = (SpikesBatch){ .spikes = 0 };
  
    if (batch_size <= 0) {
        atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        result.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return result;
    }

    if (inputs == NULL) {
        atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        result.error_no = EVENTPROP_BATCH_EMPTY; 
        return result;
    }

    result.batch_size = batch_size;

    result.spikes = (SpikeOutput*)calloc((size_t)batch_size, sizeof(SpikeOutput));
    
    if (!result.spikes) {
        atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        result.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
        return result;
    }

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            result.spikes[b] = eventprop_compute_spikes(inputs[b]);
            if (result.spikes[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            result.spikes[b] = eventprop_compute_spikes(inputs[b]);
            if (result.spikes[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        }
    }
    return result;
}

BackwardsBatch eventprop_backward_spikes_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    BackwardsBatch batch = (BackwardsBatch){ .error_no = EVENTPROP_OK };
 
    const int batch_size = spikes_batch.batch_size;

    if (batch_size <= 0) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return batch;
    }

    if (inputs == NULL) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_BATCH_EMPTY;
        return batch;
    }

    /* Parallel per-sample backward */
    batch.outputs = (BackwardOutput*)calloc((size_t)batch_size, sizeof(BackwardOutput));
    if (!batch.outputs) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
        return batch;
    }

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        }
    }

    batch.batch_size = batch_size;
    return batch;
}

MaximaBatch eventprop_compute_maxima_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    MaximaBatch batch = (MaximaBatch) { .error_no = EVENTPROP_OK };

    const int batch_size = spikes_batch.batch_size;
 
    if (batch_size <= 0) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return batch;
    }

    if (inputs == NULL) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_BATCH_EMPTY; 
        return batch;
    }

    /* Parallel per-sample maxima */
    batch.outputs = (MaximaOutput*)calloc((size_t)batch_size, sizeof(MaximaOutput));
    if (!batch.outputs) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
        return batch;
    }

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_compute_maxima(inputs[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_compute_maxima(inputs[b]);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        }
    }

    batch.batch_size = batch_size;
    return batch;
}

BackwardsBatch eventprop_backward_maxima_batch(const EventPropInput* inputs, const MaximaBatch maxima_batch)
{
    BackwardsBatch batch = (BackwardsBatch){ .error_no = EVENTPROP_OK };
 
    const int batch_size = maxima_batch.batch_size;

    if (batch_size <= 0) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return batch;
    }

    if (inputs == NULL) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_BATCH_EMPTY;
        return batch;
    }

    /* Parallel per-sample backward */
    batch.outputs = (BackwardOutput*)calloc((size_t)batch_size, sizeof(BackwardOutput));
    if (!batch.outputs) {
        atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        batch.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
        return batch;
    }

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_maxima(
                inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            batch.outputs[b] = eventprop_backward_maxima(
                inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (batch.outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&batch.has_errors, 1, memory_order_relaxed);
        }
    }

    batch.batch_size = batch_size;
    return batch;
}

GradientOutput eventprop_accumulate_gradients(const BackwardsBatch batch, const int n_pre, const int n_post)
{
    GradientOutput result = (GradientOutput) { .error_no = EVENTPROP_OK  };
 
    if (n_pre <= 0 || n_post <= 0) {
        result.error_no = EVENTPROP_INVALID_DIMENSION;
        return result;
    }

    result.rows = n_pre;
    result.cols = n_post;
 
    if (batch.batch_size <= 0) {
        result.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return result;
    }

 
    /* Check for integer overflow */
    if (n_pre > INT_MAX / n_post) {
        result.error_no = EVENTPROP_DIMENSION_TOO_LARGE;
        return result;
    }

    result.data = (double*)calloc((size_t)n_pre * (size_t)n_post, sizeof(double));
    if (!result.data) {
        result.error_no = EVENTPROP_ALLOC_BATCH_GRAD;
        return result;
    }

    int skipped = 0;
    for (int b = 0; b < batch.batch_size; b++) {
        if (batch.outputs[b].error_no != EVENTPROP_OK) { skipped++; continue; }
        for (int i = 0; i < n_pre * n_post; i++)
            result.data[i] += batch.outputs[b].grad[i];
    }

    /* Partial sum is still returned, but must not masquerade as a full one */
    if (skipped > 0)
        result.error_no = EVENTPROP_BATCH_PARTIAL_ACCUMULATION;

    return result;
}

AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(
    const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    AccumulatedBackwardResult result = (AccumulatedBackwardResult)  { .error_no = EVENTPROP_OK };

    const int batch_size = spikes_batch.batch_size;

     if (batch_size <= 0) {
        atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        result.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return result;
    }

    result.batch_size = batch_size;

    if (inputs == NULL) {
        atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        result.error_no = EVENTPROP_BATCH_EMPTY; 
        return result;
    }

    /* Parallel per-sample backward */
    BackwardOutput* outputs = calloc((size_t)batch_size, sizeof(BackwardOutput));
    if (!outputs) {
        result.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
        return result;
    }

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
            if (outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        }
    }

    BackwardsBatch batch = {
        .batch_size = batch_size,
        .has_errors = ATOMIC_VAR_INIT(0),   /* not used here */
        .outputs = outputs
    };

    GradientOutput* grad = (GradientOutput*)malloc(sizeof(GradientOutput));
    if (!grad) {
        free(outputs);
        result.error_no = EVENTPROP_ALLOC_BATCH_GRAD;
        return result;
    }
    *grad = eventprop_accumulate_gradients(batch, inputs[0].n_pre, inputs[0].n_post);

    result.batch_size = batch_size;
    result.summed_grads = grad;
    result.outputs = outputs;
    return result;
}

AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
    const EventPropInput* inputs, const MaximaBatch maxima_batch)
{
    const int batch_size = maxima_batch.batch_size;
    AccumulatedBackwardResult result  = (AccumulatedBackwardResult) { .error_no = EVENTPROP_OK };
  
    if (batch_size <= 0) {
        result.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return result;
    }
    if (inputs == NULL) {
        result.error_no = EVENTPROP_BATCH_EMPTY;
        return result;
    }

    /* Parallel per-sample backward */
    BackwardOutput* outputs = (BackwardOutput*)calloc((size_t)batch_size, sizeof(BackwardOutput));
    if (!outputs) {
        result.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
        return result;
    }

    int b;
#ifdef _OPENMP
    if (batch_size >= 2) {
#pragma omp parallel for schedule(static)
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_maxima(
                inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        }
    } else
#endif
    {
        for (b = 0; b < batch_size; b++) {
            outputs[b] = eventprop_backward_maxima(
                inputs[b],
                maxima_batch.outputs[b].times,
                maxima_batch.outputs[b].errors);
            if (outputs[b].error_no != EVENTPROP_OK)
                atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
        }
    }

    BackwardsBatch batch = {
        .batch_size = batch_size,
        .has_errors = ATOMIC_VAR_INIT(0),   /* not used here */
        .outputs = outputs
    };

    GradientOutput* grad = (GradientOutput*)malloc(sizeof(GradientOutput));
    if (!grad) {
        free(outputs);
        result.error_no = EVENTPROP_ALLOC_BATCH_GRAD;
        return result;
    }
    *grad = eventprop_accumulate_gradients(batch, inputs[0].n_pre, inputs[0].n_post);

    result.batch_size = batch_size;
    result.summed_grads = grad;
    result.outputs = outputs;
    return result;
}

DropoutBatch eventprop_dropout_spikes_batch(
    const double* times,
    const int* sources,
    const int* n_per_sample,
    const int batch_size,
    double p_drop,
    int seed
)
{
    DropoutBatch result;
    result.error_no = EVENTPROP_OK;
    result.batch_size = 0;
    atomic_store_explicit(&result.has_errors, 0, memory_order_relaxed);
    result.outputs = NULL;

    if (batch_size <= 0) {
        result.error_no = EVENTPROP_BATCHSIZE_ZERO_OR_NEG;
        return result;
    }
    if (times == NULL || sources == NULL || n_per_sample == NULL) {
        result.error_no = EVENTPROP_BATCH_EMPTY;
        return result;
    }

    result.batch_size = batch_size;
    result.outputs = (DropoutOutput*)calloc((size_t)batch_size, sizeof(DropoutOutput));
    if (!result.outputs) {
        result.batch_size = 0;
        result.error_no = EVENTPROP_ALLOC_BATCH_OUTPUTS;
        return result;
    }

    /* Compute sample offsets */
    int* offsets = (int*)malloc(((size_t)batch_size + 1) * sizeof(int));
    if (!offsets) {
        eventprop_free_dropout_batch(result);
        result.error_no = EVENTPROP_ALLOC_DROPOUT_OFFSETS;
        return result;
    }
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
                result.outputs[b].times = (double*)malloc((size_t)n_spikes * sizeof(double));
                result.outputs[b].sources = (int*)malloc((size_t)n_spikes * sizeof(int));
                if (!result.outputs[b].times || !result.outputs[b].sources) {
                    free(result.outputs[b].times);
                    free(result.outputs[b].sources);
                    result.outputs[b].times = NULL;
                    result.outputs[b].sources = NULL;
                    atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
                    eventprop_free_dropout_batch(result);
                    result.outputs = NULL; /* already freed above — keep caller's free() safe */
                    result.error_no = EVENTPROP_ALLOC_DROPOUT_SAMPLE;
                    result.batch_size = 0;
                    return result;
                }
                for (int i = 0; i < n_spikes; i++) {
                    result.outputs[b].times[i] = times[offset + i];
                    result.outputs[b].sources[i] = sources[offset + i];
                }
            }
            continue;
        }

        /* Single-pass: pre-allocate max size, then compact */
        result.outputs[b].times = (double*)malloc((size_t)n_spikes * sizeof(double));
        result.outputs[b].sources = (int*)malloc((size_t)n_spikes * sizeof(int));
        if (!result.outputs[b].times || !result.outputs[b].sources) {
            free(result.outputs[b].times);
            free(result.outputs[b].sources);
            result.outputs[b].times = NULL;
            result.outputs[b].sources = NULL;
            atomic_store_explicit(&result.has_errors, 1, memory_order_relaxed);
            eventprop_free_dropout_batch(result);
            result.outputs = NULL; /* already freed above — keep caller's free() safe */
            result.error_no = EVENTPROP_ALLOC_DROPOUT_SAMPLE;
            result.batch_size = 0;
            return result;
        }

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