#include "eventprop_batch.h"
#include <stdlib.h>

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

    /* Allocate result array */
    result.spikes = (SpikeOutput*)calloc(batch_size, sizeof(SpikeOutput));

    /* Process each sample in parallel (if available) */
    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_size >= 2)
#endif
    for (b = 0; b < batch_size; b++) {
        result.spikes[b] = eventprop_compute_spikes(inputs[b]);
    }
    return result;
}

BackwardsBatch eventprop_backward_spikes_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    int batch_size = spikes_batch.batch_size;
    BackwardsBatch batch;
    batch.batch_size = batch_size;

    /* Parallel per-sample backward */
    batch.outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_size >= 2)
#endif
    for (b = 0; b < batch_size; b++) {
        batch.outputs[b] = eventprop_backward_spikes(inputs[b], spikes_batch.spikes[b]);
    }

    return batch;
}

MaximaBatch eventprop_compute_maxima_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    int batch_size = spikes_batch.batch_size;
    MaximaBatch batch;
    batch.batch_size = batch_size;

    /* Parallel per-sample maxima */
    batch.outputs = (MaximaOutput*)calloc(batch_size, sizeof(MaximaOutput));

    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_size >= 2)
#endif
    for (b = 0; b < batch_size; b++) {
        batch.outputs[b] = eventprop_compute_maxima(inputs[b]);
    }

    return batch;
}

BackwardsBatch eventprop_backward_maxima_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch)
{
    int batch_size = spikes_batch.batch_size;
    BackwardsBatch batch;
    batch.batch_size = batch_size;

    /* Parallel per-sample backward */
    batch.outputs = (BackwardOutput*)calloc(batch_size, sizeof(BackwardOutput));

    int b;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(batch_size >= 2)
#endif
    for (b = 0; b < batch_size; b++) {
        batch.outputs[b] = eventprop_backward_maxima(inputs[b], spikes_batch.spikes[b]);
    }

    return batch;
}

double* eventprop_accumulate_gradients(BackwardsBatch batch, int grad_size)
{
    double* grad = (double*)calloc(grad_size, sizeof(double));
    for (int b = 0; b < batch.batch_size; b++) {
        for (int i = 0; i < grad_size; i++)
            grad[i] += batch.outputs[b].grad[i];
    }
    return grad;
}
