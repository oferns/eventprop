/**
 * @file eventprop_batch.h
 * @brief Batch operations for EventProp — parallel processing of multiple samples.
 *
 * All batch functions accept a BatchEventPropInput struct containing shared layer
 * parameters (weights, time constants) and per-sample spike counts. Input spike
 * data (times, sources) must be provided as flat concatenated arrays; the batch
 * functions compute offsets internally.
 */

#ifndef EVENTPROP_BATCH_H
#define EVENTPROP_BATCH_H

#include "eventprop.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Batch input — shared layer params + per-sample spike counts.
 *
 * times and sources in the referenced EventPropInput must be flat concatenated
 * arrays: sample 0's spikes first, then sample 1's, etc.
 * n_spikes_per_sample gives the spike count for each sample.
 */
typedef struct BatchEventPropInput {
    const EventPropInput* input;          /* Shared layer params + concatenated spike data */
    const int* n_spikes_per_sample;       /* Spike count per sample (length: batch_size) */
    int batch_size;                       /* Number of samples in batch */
} BatchEventPropInput;

/**
 * Pointer-based batch input — no flattening required.
 *
 * Each sample's spike data is passed as a separate pointer, mirroring how
 * pybind11 iterates std::vector<Spikes>. C processes each sample directly
 * without computing offsets into concatenated arrays.
 */
typedef struct BatchEventPropInputPtrs {
    const double** times;       /* Array of pointers: one per sample's spike times */
    const int** sources;        /* Array of pointers: one per sample's source neurons */
    const int* n_spikes;        /* Spike count per sample (length: batch_size) */
    const double* w;            /* Weight matrix (row-major, shared across samples) */
    int n_pre;                  /* Pre-synaptic neurons */
    int n_post;                 /* Post-synaptic neurons */
    int batch_size;             /* Number of samples */
    double tau_mem;             /* Membrane time constant */
    double tau_syn;             /* Synaptic time constant */
    double v_th;                /* Spike threshold voltage */
} BatchEventPropInputPtrs;

/**
 * Direct batch input — array of EventPropInput structs.
 *
 * Each sample's data is passed as a complete EventPropInput struct, eliminating
 * the need for separate pointer arrays (const double**, const int**). This is
 * the most efficient API for Cython/Python bindings.
 */
typedef struct BatchEventPropInputDirect {
    const EventPropInput* inputs; /* Array of EventPropInput structs (length: batch_size) */
    int batch_size;               /* Number of samples */
} BatchEventPropInputDirect;

/**
 * Dropout output — per-sample surviving spikes.
 */
typedef struct DropoutOutput {
    int n_spikes;              /* Number of surviving spikes */
    double* times;             /* Surviving spike times (malloc'd) */
    int* sources;              /* Surviving source neurons (malloc'd) */
} DropoutOutput;

/* ── Free functions ─────────────────────────────────────────────── */

void eventprop_free_batch_spike_output(SpikeOutput* output, int batch_size);
void eventprop_free_backward_batch_output(BackwardOutput* output, int batch_size);
void eventprop_free_batch_maxima_output(MaximaOutput* output, int batch_size);
void eventprop_free_dropout_output(DropoutOutput* output, int batch_size);

/* ── Batch forward pass (spikes) ─────────────────────────────────── */

/**
 * Batch forward pass.
 * Processes multiple samples in parallel using OpenMP (if available).
 *
 * @param input  BatchEventPropInput with per-sample spike counts
 *
 * @return Array of SpikeOutput structs (one per sample), freed via eventprop_free_batch_spike_output()
 */
SpikeOutput* eventprop_compute_spikes_batch(const BatchEventPropInput* input);

/**
 * Pointer-based batch forward pass (spikes).
 * No flattening required — each sample's data is a separate pointer.
 */
SpikeOutput* eventprop_compute_spikes_batch_ptrs(const BatchEventPropInputPtrs* input);

/**
 * Direct batch forward pass (spikes) — array of EventPropInput structs.
 * Most efficient API for Cython/Python bindings — no pointer array allocation needed.
 */
SpikeOutput* eventprop_compute_spikes_batch_direct(const BatchEventPropInputDirect* input);

/* ── Batch backward pass (spikes) ────────────────────────────────── */

/**
 * Batch backward pass (spikes).
 * Processes multiple samples in parallel using OpenMP.
 * Returns array of BackwardOutput structs (one per sample), freed via eventprop_free_backward_batch_output().
 */
BackwardOutput* eventprop_backward_spikes_batch(
    const BatchEventPropInput* input,
    SpikeOutput* fwd_batch,        /* Forward spike outputs from compute_spikes_batch */
    const double* post_errors
    );

/**
 * Batch backward pass (spikes) with gradient accumulation.
 * Processes multiple samples in parallel using OpenMP and accumulates gradients
 * into a single BackwardOutput. Equivalent to summing per-sample outputs.
 *
 * @return Single BackwardOutput, freed via eventprop_free_backward_output()
 */
BackwardOutput* eventprop_backward_spikes_batch_accumulate(
    const BatchEventPropInput* input,
    SpikeOutput* fwd_batch,
    const double* post_errors
    );

/**
 * Pointer-based batch backward pass (spikes) with gradient accumulation.
 */
BackwardOutput* eventprop_backward_spikes_batch_accumulate_ptrs(
    const BatchEventPropInputPtrs* input,
    SpikeOutput* fwd_batch,
    const double** post_errors,     /* Array of pointers: one per sample's errors */
    const int* n_post_errors        /* Error count per sample (length: batch_size) */
    );

/* ── Batch forward pass (maxima) ─────────────────────────────────── */

/**
 * Batch forward pass — compute membrane potential maxima.
 * Processes multiple samples in parallel using OpenMP (if available).
 *
 * @param input  BatchEventPropInput with per-sample spike counts
 *
 * @return Array of MaximaOutput structs (one per sample), freed via eventprop_free_batch_maxima_output()
 */
MaximaOutput* eventprop_compute_maxima_batch(const BatchEventPropInput* input);

/**
 * Pointer-based batch forward pass (maxima).
 */
MaximaOutput* eventprop_compute_maxima_batch_ptrs(const BatchEventPropInputPtrs* input);

/**
 * Direct batch forward pass (maxima) — array of EventPropInput structs.
 */
MaximaOutput* eventprop_compute_maxima_batch_direct(const BatchEventPropInputDirect* input);

/* ── Batch backward pass (maxima) ────────────────────────────────── */

/**
 * Batch backward pass (maxima).
 * Processes multiple samples in parallel using OpenMP.
 * Returns array of BackwardOutput structs (one per sample), freed via eventprop_free_backward_batch_output().
 */
BackwardOutput* eventprop_backward_maxima_batch(
    const BatchEventPropInput* input,
    MaximaOutput* fwd_batch,       /* Forward maxima outputs from compute_maxima_batch */
    const double* maxima_errors
    );

/**
 * Batch backward pass (maxima) with gradient accumulation.
 * Processes multiple samples in parallel using OpenMP and accumulates gradients
 * into a single BackwardOutput. Equivalent to summing per-sample outputs.
 *
 * @return Single BackwardOutput, freed via eventprop_free_backward_output()
 */
BackwardOutput* eventprop_backward_maxima_batch_accumulate(
    const BatchEventPropInput* input,
    MaximaOutput* fwd_batch,
    const double* maxima_errors
    );

/**
 * Pointer-based batch backward pass (maxima) with gradient accumulation.
 */
BackwardOutput* eventprop_backward_maxima_batch_accumulate_ptrs(
    const BatchEventPropInputPtrs* input,
    MaximaOutput* fwd_batch,
    const double** maxima_errors,   /* Array of pointers: one per sample's errors (length n_post each) */
    int n_post                      /* Number of post-synaptic neurons (= error count per sample) */
    );

/* ── Batch dropout ─────────────────────────────────────────────── */

/**
 * Randomly drop spikes from each sample in a batch.
 *
 * For each spike, a Bernoulli trial with probability (1 - p_drop) decides
 * whether to keep it. Uses a simple LCG PRNG seeded per-sample for reproducibility.
 *
 * @param input  BatchEventPropInput with per-sample spike counts
 * @param p_drop Dropout probability (0 = no dropout, 1.0 = drop all)
 * @param seed   Random seed (applied per-sample: sample i uses seed + i)
 *
 * @return Array of DropoutOutput structs (one per sample), freed via eventprop_free_dropout_output()
 */
DropoutOutput* eventprop_dropout_spikes_batch(
    const BatchEventPropInput* input,
    double p_drop,
    int seed
    );

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_BATCH_H */
