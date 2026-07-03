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
    int has_errors;                    /* 1 if any sample returned an error, 0 otherwise — safe to set from parallel loop */
    SpikeOutput* spikes;               /* Per-sample outputs (malloc'd, length = batch_size) */
} SpikesBatch;

void eventprop_free_spikes_batch(SpikesBatch* batch);

typedef struct SpikesBatchWithInputs {
    EventPropInput* inputs;
    int batch_size;                    /* Number of samples in the batch */
    int has_errors;                    /* 1 if any sample returned an error, 0 otherwise — safe to set from parallel loop */
    SpikeOutput* spikes;               /* Per-sample outputs (malloc'd, length = batch_size) */
} SpikesBatchWithInputs;

void eventprop_free_spikes_with_inputs_batch(SpikesBatchWithInputs* batch);

/**
 * Batch backward output — per-sample results.
 */
typedef struct BackwardsBatch {
    int batch_size;                    /* Number of samples in the batch */
    int has_errors;                    /* 1 if any sample returned an error, 0 otherwise — safe to set from parallel loop */
    BackwardOutput* outputs;           /* Per-sample outputs (malloc'd, length = batch_size) */
} BackwardsBatch;

void eventprop_free_backwards_batch(BackwardsBatch* batch);

/**
 * Batch maxima output — per-sample membrane potential maxima.
 */
typedef struct MaximaBatch {
    int batch_size;                    /* Number of samples in the batch */
    int has_errors;                    /* 1 if any sample returned an error, 0 otherwise — safe to set from parallel loop */
    MaximaOutput* outputs;             /* Per-sample outputs (malloc'd, length = batch_size) */
} MaximaBatch;

void eventprop_free_maxima_batch(MaximaBatch* batch);

typedef struct MaximaBatchWithInputs {
    EventPropInput* inputs;            /* Owned copy of per-sample inputs (calloc'd) */
    int batch_size;                    /* Number of samples in the batch */
    int has_errors;                    /* 1 if any sample returned an error, 0 otherwise — safe to set from parallel loop */
    MaximaOutput* outputs;             /* Per-sample outputs (malloc'd, length = batch_size) */
} MaximaBatchWithInputs;

void eventprop_free_maxima_with_inputs_batch(MaximaBatchWithInputs* batch);

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
 * Dropout output — per-sample surviving spikes.
 */
typedef struct DropoutOutput {
    int n_spikes;              /* Number of surviving spikes */
    double* times;             /* Surviving spike times (malloc'd) */
    int* sources;              /* Surviving source neurons (malloc'd) */
} DropoutOutput;

void eventprop_free_dropout_output(DropoutOutput* output);

/**
 * Batch dropout result — per-sample outputs.
 */
typedef struct DropoutBatch {
    int batch_size;
    int has_errors;            /* 1 if any sample returned an error, 0 otherwise — safe to set from parallel loop */
    DropoutOutput* outputs;    /* Per-sample surviving spikes (malloc'd, length = batch_size) */
} DropoutBatch;

void eventprop_free_dropout_batch(DropoutBatch* batch);

/**
 * Combined backward + accumulate result.
 *
 * Contains:
 * - Summed gradient tensor (ready for weight update)
 * - Per-sample BackwardOutput with spike_errors preserved but .grad freed after summation
 *
 * This is the primary FFI entry point — one call, zero intermediate allocations in Python.
 */
typedef struct AccumulatedBackwardResult {
    GradientOutput summed_grads;
    int has_errors;            /* 1 if any sample returned an error, 0 otherwise — safe to set from parallel loop */
    BackwardOutput* outputs;   /* Per-sample outputs (.grad freed, .spike_errors kept) */
    int batch_size;
} AccumulatedBackwardResult;

void eventprop_free_accumulated_backward_result(AccumulatedBackwardResult* result);

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
    int seed);

/**
 * Batch forward pass — compute post-synaptic spikes for each sample.
 */
SpikesBatch eventprop_compute_spikes_batch(const EventPropInput* inputs, int batch_size);

/**
 * Flat-array variant of compute_spikes_batch.
 *
 * Accepts flat concatenated spike arrays and per-sample offset counts.
 * Internally constructs EventPropInput structs and dispatches in parallel.
 * Designed for FFI callers (Cython, P/Invoke) that have flat numpy buffers.
 */
SpikesBatchWithInputs eventprop_compute_spikes_batch_flat(
    const double* w,
    int n_pre, int n_post,
    const double* times,
    const int* sources,
    const int* n_per_sample,
    int batch_size,
    double tau_mem, double tau_syn, double v_th);

/**
 * Batch backward pass (spike loss) — returns per-sample gradients.
 */
BackwardsBatch eventprop_backward_spikes_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch);

/**
 * Batch forward pass — compute membrane potential maxima for each sample.
 */
MaximaBatch eventprop_compute_maxima_batch(const EventPropInput* inputs, const SpikesBatch spikes_batch);

/**
 * Flat-array variant of compute_maxima_batch.
 *
 * Accepts flat concatenated spike arrays and per-sample offset counts.
 * Internally constructs EventPropInput structs and dispatches in parallel.
 * Designed for FFI callers (Cython, P/Invoke) that have flat numpy buffers.
 */
MaximaBatchWithInputs eventprop_compute_maxima_batch_flat(
    const double* w,
    int n_pre, int n_post,
    const double* times,
    const int* sources,
    const int* n_per_sample,
    int batch_size,
    double tau_mem, double tau_syn, double v_th);

/**
 * Batch backward pass (maxima loss) — returns per-sample gradients.
 *
 * Uses `MaximaOutput.errors` written by caller before this call.
 */
BackwardsBatch eventprop_backward_maxima_batch(const MaximaBatchWithInputs maxima_batch);



/**
 * Sum per-sample gradients into a single tensor.
 * Caller must free via eventprop_free_gradient_output().
 *
 * NOTE: This does NOT free the BackwardsBatch. Call eventprop_free_backwards_batch() separately.
 */
GradientOutput eventprop_accumulate_gradients(BackwardsBatch batch, int n_pre, int n_post);

/**
 * Batch backward (spike loss) + accumulate gradients in one call.
 *
 * Parallel backward pass → sequential gradient sum → free per-sample .grad buffers.
 * Returns combined result with summed gradient and preserved spike errors.
 */
AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(
    const EventPropInput* inputs, const SpikesBatch spikes_batch);

/**
 * Batch backward (maxima loss) + accumulate gradients in one call.
 *
 * Parallel backward pass → sequential gradient sum → free per-sample .grad buffers.
 * Returns combined result with summed gradient and preserved spike errors.
 *
 * Uses `MaximaOutput.errors` written by caller before this call.
 */
AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
    const MaximaBatchWithInputs maxima_batch);


#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_BATCH_H */
