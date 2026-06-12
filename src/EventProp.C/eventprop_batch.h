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

/* ── Free functions ─────────────────────────────────────────────── */

void eventprop_free_batch_spike_output(SpikeOutput* output, int batch_size);
void eventprop_free_backward_batch_output(BackwardOutput* output, int batch_size);
void eventprop_free_batch_maxima_output(MaximaOutput* output, int batch_size);

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

/* ── Batch backward pass (spikes) ────────────────────────────────── */

/**
 * Batch backward pass (spikes).
 * Processes multiple samples in parallel using OpenMP.
 * Returns array of BackwardOutput structs (one per sample).
 */
BackwardOutput* eventprop_backward_spikes_batch(
    const BatchEventPropInput* input,
    SpikeOutput* fwd_batch,        /* Forward spike outputs from compute_spikes_batch */
    const double* post_errors
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

/* ── Batch backward pass (maxima) ────────────────────────────────── */

/**
 * Batch backward pass (maxima).
 * Processes multiple samples in parallel using OpenMP.
 * Returns array of BackwardOutput structs (one per sample).
 */
BackwardOutput* eventprop_backward_maxima_batch(
    const BatchEventPropInput* input,
    MaximaOutput* fwd_batch,       /* Forward maxima outputs from compute_maxima_batch */
    const double* maxima_errors
    );

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_BATCH_H */
