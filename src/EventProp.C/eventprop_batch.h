/**
 * @file eventprop_batch.h
 * @brief Batch operations for EventProp — parallel processing of multiple samples.
 *
 * All batch functions accept an array of per-sample inputs and return a batch result
 * struct. The caller is responsible for freeing returned structs via the corresponding
 * _free_ function.
 */

#ifndef EVENTPROP_BATCH_H
#define EVENTPROP_BATCH_H

#include "eventprop.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Batch forward output — per-sample spike results.
 */
typedef struct SpikesBatch {
    int batch_size;                    /* Number of samples in the batch */
    SpikeOutput* spikes;               /* Per-sample outputs (malloc'd, length = batch_size) */
} SpikesBatch;

void eventprop_free_spikes_batch(SpikesBatch* batch);

/**
 * Batch backward output — per-sample results.
 */
typedef struct BackwardsBatch {
    int batch_size;                    /* Number of samples in the batch */
    BackwardOutput* outputs;           /* Per-sample outputs (malloc'd, length = batch_size) */
} BackwardsBatch;

void eventprop_free_backwards_batch(BackwardsBatch* batch);

/**
 * Batch maxima output — per-sample membrane potential maxima.
 */
typedef struct MaximaBatch {
    int batch_size;                    /* Number of samples in the batch */
    MaximaOutput* outputs;             /* Per-sample outputs (malloc'd, length = batch_size) */
} MaximaBatch;

void eventprop_free_maxima_batch(MaximaBatch* batch);

/**
 * Dropout output — per-sample surviving spikes.
 */
typedef struct DropoutOutput {
    int n_spikes;              /* Number of surviving spikes */
    double* times;             /* Surviving spike times (malloc'd) */
    int* sources;              /* Surviving source neurons (malloc'd) */
} DropoutOutput;


/**
 * Batch forward pass — compute post-synaptic spikes for each sample.
 */
SpikesBatch eventprop_compute_spikes_batch(const EventPropInput* inputs, int batch_size);

/**
 * Batch backward pass (spike loss) — returns per-sample gradients.
 */
BackwardsBatch eventprop_backward_spikes_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch);

/**
 * Batch forward pass — compute membrane potential maxima for each sample.
 */
MaximaBatch eventprop_compute_maxima_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch);

/**
 * Batch backward pass (maxima loss) — returns per-sample gradients.
 */
BackwardsBatch eventprop_backward_maxima_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch);

/**
 * Accumulated gradient output with dimensions.
 */
typedef struct GradientOutput {
    double* data;     /* malloc'd, size = rows * cols (row-major) */
    int rows;         /* n_pre */
    int cols;         /* n_post */
} GradientOutput;

void eventprop_free_gradient_output(GradientOutput* output);

/**
 * Sum per-sample gradients into a single tensor.
 * Caller must free via eventprop_free_gradient_output().
 */
GradientOutput eventprop_accumulate_gradients(BackwardsBatch batch, int n_pre, int n_post);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_BATCH_H */
