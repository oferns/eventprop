/**
 * @file eventprop.h
 * @brief EventProp: Event-based backpropagation for SNN training.
 *
 * Pure C implementation with optional SIMD (AVX2/NEON). 
 *
 * Parameters (SI units):
 *   tau_mem  — membrane time constant (seconds, typically 0.02)
 *   tau_syn  — synaptic time constant (seconds, typically 0.005)
 *   v_th     — spike threshold voltage (typically 1.0)
 */

#ifndef EVENTPROP_H
#define EVENTPROP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EventPropInput {
    const double* w;           /* Weight matrix (row-major) */
    int n_pre;                 /* Pre-synaptic neurons */
    int n_post;                /* Post-synaptic neurons */
    const double* times;       /* Input spike times */
    const int* sources;        /* Input source neurons  */
    int n_spikes;              /* Number of input spikes */
    double tau_mem;            /* Membrane time constant (seconds) */
    double tau_syn;            /* Synaptic time constant (seconds) */
    double v_th;               /* Spike threshold voltage */
} EventPropInput;

typedef struct BatchEventPropInput {
    const EventPropInput* input;
    const int* n_spikes_per_sample;
    int batch_size;
} BatchEventPropInput;

typedef struct SpikeOutput {
    int n_spikes;            /* Number of output spikes */
    double* times;           /* Output spike times (malloc'd) */
    int* sources;        /* Output source neurons (malloc'd) */
    double* currents;        /* Synaptic currents at spike times (malloc'd) */
} SpikeOutput;

typedef struct SumsOutput {
    int n_rows;              /* n_spikes + 1 */
    int n_post;              /* Number of post-synaptic neurons */
    double* sum0;            /* Cumulative sums (malloc'd, n_rows * n_post) */
    double* sum1;            /* Cumulative sums (malloc'd, n_rows * n_post) */
} SumsOutput;

typedef struct MaximaOutput {
    int n_post;              /* Number of post-synaptic neurons */
    double* times;           /* Maxima times (malloc'd, n_post) */
    double* values;          /* Maxima values (malloc'd, n_post) */
} MaximaOutput;

typedef struct BackwardSpikesInput {
    const EventPropInput* input;      /* Shared layer params + input spike data (borrowed) */
    const double* fwd_times;           /* Forward spike times (borrowed) */
    const int* fwd_sources;        /* Forward spike sources (borrowed) */
    const double* fwd_currents;         /* Forward spike currents (borrowed) */
    const double* post_errors;          /* Post-synaptic errors (borrowed) */
    int n_post_spikes;                  /* Number of post-synaptic spikes */
} BackwardSpikesInput;

typedef struct BackwardMaximaInput {
    const EventPropInput* input;      /* Shared layer params + input spike data (borrowed) */
    const double* maxima_times;        /* Maxima times (borrowed) */
    const double* maxima_values;         /* Maxima values (borrowed) */
    const double* maxima_errors;          /* Maxima errors (borrowed) */
    int n_maxima;                       /* Number of maxima */
} BackwardMaximaInput;

typedef struct BackwardOutput {
    int n_pre;               /* Number of pre-synaptic neurons */
    int n_post;              /* Number of post-synaptic neurons */
    double* grad;            /* Weight gradient (malloc'd, n_pre * n_post) */
    double* spike_errors;   /* Input spike errors (malloc'd, n_spikes) */
} BackwardOutput;

void eventprop_free_spike_output(SpikeOutput* output);
void eventprop_free_sums_output(SumsOutput* output);
void eventprop_free_maxima_output(MaximaOutput* output);
void eventprop_free_backward_output(BackwardOutput* output);
void eventprop_free_batch_spike_output(SpikeOutput* output, int batch_size);
void eventprop_free_backward_batch_output(BackwardOutput* output, int batch_size);

/**
 * Forward pass
 * Returns a SpikeOutput struct — caller must free via eventprop_free_spike_output().
 */
SpikeOutput eventprop_compute_spikes(const EventPropInput* input);

/**
 * Compute cumulative sums
 * Returns a SumsOutput struct — caller must free via eventprop_free_sums_output().
 */
SumsOutput eventprop_compute_sums(const EventPropInput* input);

/**
 * Compute membrane potential maxima
 * Returns a MaximaOutput struct — caller must free via eventprop_free_maxima_output().
 */
MaximaOutput eventprop_compute_maxima(const EventPropInput* input);

/**
 * Backward pass (spikes).
 * Returns a BackwardOutput struct — caller must free via eventprop_free_backward_output().
 */
BackwardOutput eventprop_backward_spikes(const BackwardSpikesInput* input);

/**
 * Backward pass (maxima).
 * Returns a BackwardOutput struct — caller must free via eventprop_free_backward_output().
 */
BackwardOutput eventprop_backward_maxima(const BackwardMaximaInput* input);

/**
 * Batch forward pass.
 * Processes multiple samples in parallel using OpenMP (if available).
 *
 * @param input                Shared layer params (weights, tau_mem, tau_syn, v_th)
 * @param n_spikes_per_sample  Spike count per sample (length: batch_size)
 * @param batch_size           Number of samples in batch
 *
 * @return Array of SpikeOutput structs (one per sample), freed via eventprop_free_batch_spike_output()
 */
SpikeOutput* eventprop_compute_spikes_batch(const BatchEventPropInput* input);

/**
 * Batch backward pass (spikes) with automatic memory allocation.
 * Processes multiple samples in parallel using OpenMP.
 * Returns array of BackwardOutput structs (one per sample).
 */
BackwardOutput* eventprop_backward_spikes_batch(
    const BatchEventPropInput* input,
    SpikeOutput* fwd_batch,        /* Forward spike outputs from compute_spikes_batch */
    const double* post_errors
    );

/**
 * Free batch maxima output array.
 */
void eventprop_free_batch_maxima_output(MaximaOutput* output, int batch_size);

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

#endif /* EVENTPROP_H */
