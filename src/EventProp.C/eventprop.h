/**
 * @file eventprop.h
 * @brief EventProp: Event-based backpropagation for SNN training.
 *
 * Pure C implementation with optional SIMD (AVX2/NEON). Zero-allocation hot paths
 * via pre-allocated scratch buffers. Flat row-major double[] arrays throughout.
 *
 * Parameters (SI units):
 *   tau_mem  — membrane time constant (seconds, typically 0.02)
 *   tau_syn  — synaptic time constant (seconds, typically 0.005)
 *   v_th     — spike threshold voltage (typically 1.0)
 */

#ifndef EVENTPROP_H
#define EVENTPROP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Spike input (borrowed data, no allocation) ─────────────────────── */

/** Input spike data — borrowed pointers, no memory management. */
typedef struct {
    const double* times;           /* Spike times (borrowed) */
    const int32_t* sources;        /* Source neurons (borrowed) */
    int n_spikes;                   /* Number of input spikes */
} SpikeInput;

/* ── Forward pass input structs (borrowed data, no allocation) ─────── */

/** Base forward pass input — common parameters for sums/maxima/spikes. */
typedef struct {
    const double* w;           /* Weight matrix (row-major, borrowed) */
    int n_pre;                 /* Pre-synaptic neurons */
    int n_post;                /* Post-synaptic neurons */
    const double* times;       /* Input spike times (borrowed) */
    const int32_t* sources;   /* Input source neurons (borrowed) */
    int n_spikes;              /* Number of input spikes */
    double tau_mem;            /* Membrane time constant (seconds) */
    double tau_syn;             /* Synaptic time constant (seconds) */
    double v_th;                /* Spike threshold voltage */
} ForwardInputBase;

/** Extended forward pass input for spike computation. */
typedef struct {
    ForwardInputBase base;   /* Base parameters (must be first member) */
} ForwardInputSpikes;

/* ── Spike output (owned memory) ─────────────────────────────────────── */

typedef struct {
    int n_spikes;            /* Number of output spikes */
    double* times;           /* Output spike times (malloc'd) */
    int32_t* sources;        /* Output source neurons (malloc'd) */
    double* currents;        /* Synaptic currents at spike times (malloc'd) */
} SpikeOutput;

/**
 * Forward pass with automatic memory allocation.
 * Returns a SpikeOutput struct — caller must free via eventprop_free_spike_output().
 */
SpikeOutput eventprop_compute_spikes(const ForwardInputSpikes* input);

/** Free memory owned by a SpikeOutput struct. */
void eventprop_free_spike_output(SpikeOutput* output);

/* ── Cumulative sums output (owned memory) ───────────────────────────── */

typedef struct {
    int n_rows;              /* n_spikes + 1 */
    int n_post;              /* Number of post-synaptic neurons */
    double* sum0;            /* Cumulative sums (malloc'd, n_rows * n_post) */
    double* sum1;            /* Cumulative sums (malloc'd, n_rows * n_post) */
} SumsOutput;

/**
 * Compute cumulative sums with automatic memory allocation.
 * Returns a SumsOutput struct — caller must free via eventprop_free_sums_output().
 */
SumsOutput eventprop_compute_sums(const ForwardInputBase* input);

/** Free memory owned by a SumsOutput struct. */
void eventprop_free_sums_output(SumsOutput* output);

/* ── Maxima output (owned memory) ─────────────────────────────────────── */

typedef struct {
    int n_post;              /* Number of post-synaptic neurons */
    double* times;           /* Maxima times (malloc'd, n_post) */
    double* values;          /* Maxima values (malloc'd, n_post) */
} MaximaOutput;

/**
 * Compute membrane potential maxima with automatic memory allocation.
 * Returns a MaximaOutput struct — caller must free via eventprop_free_maxima_output().
 */
MaximaOutput eventprop_compute_maxima(const ForwardInputBase* input);

/** Free memory owned by a MaximaOutput struct. */
void eventprop_free_maxima_output(MaximaOutput* output);

/* ── Backward pass input structs (borrowed data, no allocation) ─────── */

/** Base backward pass input — common parameters for spikes/maxima. */
typedef struct {
    const ForwardInputBase* base;      /* Common forward pass params + input spike data (borrowed) */
} BackwardInputBase;

/** Backward spikes input — extends base with forward spike data and errors. */
typedef struct {
    BackwardInputBase base;
    const double* fwd_times;           /* Forward spike times (borrowed) */
    const int32_t* fwd_sources;        /* Forward spike sources (borrowed) */
    const double* fwd_currents;         /* Forward spike currents (borrowed) */
    const double* post_errors;          /* Post-synaptic errors (borrowed) */
    int n_post_spikes;                  /* Number of post-synaptic spikes */
} BackwardSpikesInput;

/** Backward maxima input — extends base with membrane potential maxima data. */
typedef struct {
    BackwardInputBase base;
    const double* maxima_times;        /* Maxima times (borrowed) */
    const double* maxima_values;         /* Maxima values (borrowed) */
    const double* maxima_errors;          /* Maxima errors (borrowed) */
    int n_maxima;                       /* Number of maxima */
} BackwardMaximaInput;

/* ── Backward pass outputs (owned memory) ───────────────────────────── */

typedef struct {
    int n_pre;               /* Number of pre-synaptic neurons */
    int n_post;              /* Number of post-synaptic neurons */
    double* grad;            /* Weight gradient (malloc'd, n_pre * n_post) */
    double* spike_errors;   /* Input spike errors (malloc'd, n_spikes) */
} BackwardSpikesOutput;

/**
 * Backward pass (spikes) with automatic memory allocation.
 * Returns a BackwardSpikesOutput struct — caller must free via eventprop_free_backward_spikes_output().
 */
BackwardSpikesOutput eventprop_backward_spikes(const BackwardSpikesInput* input);


/** Free memory owned by a BackwardSpikesOutput struct. */
void eventprop_free_backward_spikes_output(BackwardSpikesOutput* output);

typedef struct {
    int n_pre;               /* Number of pre-synaptic neurons */
    int n_post;              /* Number of post-synaptic neurons */
    double* gradient;        /* Weight gradient (malloc'd, n_pre * n_post) */
    double* spike_errors;   /* Input spike errors (malloc'd, n_input_spikes) */
} BackwardMaximaOutput;

/**
 * Backward pass (maxima) with automatic memory allocation.
 * Returns a BackwardMaximaOutput struct — caller must free via eventprop_free_backward_maxima_output().
 */
BackwardMaximaOutput eventprop_backward_maxima(const BackwardMaximaInput* input);

/** Free memory owned by a BackwardMaximaOutput struct. */
void eventprop_free_backward_maxima_output(BackwardMaximaOutput* output);

/* ── Diagnostic functions are in eventprop_diag.h ───────────────────── */

/* ── Batch operations (parallel processing) ──────────────────────────── */

/**
 * Batch forward pass with automatic memory allocation.
 * Processes multiple samples in parallel using OpenMP.
 *
 * @param w           Weight matrix (flat, row-major)
 * @param n_pre        Number of pre-synaptic neurons
 * @param n_post       Number of post-synaptic neurons
 * @param times        Flat array of all input spike times (sorted per sample)
 * @param sources      Flat array of all source neurons
 * @param n_spikes_per_sample  Spike count per sample (length: batch_size)
 * @param batch_size   Number of samples in batch
 * @param v_th          Spike threshold voltage
 * @param tau_mem       Membrane time constant (seconds)
 * @param tau_syn       Synaptic time constant (seconds)
 *
 * @return Array of SpikeOutput structs (one per sample), freed via eventprop_free_batch_spike_output()
 */
SpikeOutput* eventprop_compute_spikes_batch(const ForwardInputSpikes* input,
                                                  const int* n_spikes_per_sample, int batch_size);

/** Free memory owned by batch SpikeOutput array. */
void eventprop_free_batch_spike_output(SpikeOutput* output, int batch_size);

/**
 * Batch backward pass (spikes) with automatic memory allocation.
 * Processes multiple samples in parallel using OpenMP.
 * Returns array of BackwardSpikesOutput structs (one per sample).
 */
BackwardSpikesOutput* eventprop_backward_spikes_batch(
    const BackwardInputBase* base,
    const int* n_spikes_per_sample,
    SpikeOutput* fwd_batch,        /* Forward spike outputs from compute_spikes_batch */
    const double* post_errors,
    int batch_size);


/** Free memory owned by batch BackwardSpikesOutput array. */
void eventprop_free_backward_spikes_batch_output(BackwardSpikesOutput* output, int batch_size);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_H */
