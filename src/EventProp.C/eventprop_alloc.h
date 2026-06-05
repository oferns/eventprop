/**
 * @file eventprop_alloc.h
 * @brief EventProp complementary API with automatic memory allocation.
 *
 * Functions allocate output buffers internally and return ownership to the caller.
 * Use corresponding free functions to release memory.
 */

#ifndef EVENTPROP_ALLOC_H
#define EVENTPROP_ALLOC_H

#include <stdint.h>
#include "eventprop.h"

#ifdef __cplusplus
extern "C" {
#endif

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
SpikeOutput eventprop_compute_spikes_alloc(const double* w, int n_pre, int n_post,
                                           const double* times, const int32_t* sources, int n_spikes,
                                           double v_th, double tau_mem, double tau_syn);

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
SumsOutput eventprop_compute_sums_alloc(const double* w, int n_pre, int n_post,
                                        const double* times, const int32_t* sources, int n_spikes,
                                        double tau_mem, double tau_syn);

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
MaximaOutput eventprop_compute_maxima_alloc(const double* w, int n_pre, int n_post,
                                           const double* times, const int32_t* sources, int n_spikes,
                                           double tau_mem, double tau_syn);

/** Free memory owned by a MaximaOutput struct. */
void eventprop_free_maxima_output(MaximaOutput* output);

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
BackwardSpikesOutput eventprop_backward_spikes_alloc(const double* times, const int32_t* sources,
                                                     int n_spikes,
                                                     const double* fwd_times, const int32_t* fwd_sources,
                                                     const double* fwd_currents,
                                                     const double* post_errors, int n_post_spikes,
                                                     const double* w, int n_pre, int n_post,
                                                     double v_th, double tau_mem, double tau_syn);

/** Free memory owned by a BackwardSpikesOutput struct. */
void eventprop_free_backward_spikes_output(BackwardSpikesOutput* output);

/* ── Backward pass batch operations ─────────────────────────────── */

/** Batch backward pass (spikes) with automatic memory allocation.
 * 
 * Processes multiple samples in parallel via OpenMP.
 * Returns array of BackwardSpikesOutput structs (one per sample).
 */
BackwardSpikesOutput* eventprop_backward_spikes_batch_alloc(
    const double* times, const int32_t* sources,
    const int* n_spikes_per_sample,
    const double* fwd_times, const int32_t* fwd_sources,
    const double* fwd_currents,
    const double* post_errors,
    const int* n_fwd_spikes_per_sample,
    int batch_size,
    const double* w, int n_pre, int n_post,
    double v_th, double tau_mem, double tau_syn);

/** Free memory owned by batch BackwardSpikesOutput array. */
void eventprop_free_backward_spikes_batch_output(BackwardSpikesOutput* output, int batch_size);

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
BackwardMaximaOutput eventprop_backward_maxima_alloc(const double* input_times, const int32_t* input_sources,
                                                     int n_input_spikes,
                                                     const double* maxima_times, const double* maxima_values,
                                                     const double* maxima_errors, int n_maxima,
                                                     const double* w, int n_pre, int n_post,
                                                     double tau_mem, double tau_syn);

/** Free memory owned by a BackwardMaximaOutput struct. */
void eventprop_free_backward_maxima_output(BackwardMaximaOutput* output);

/* ── Diagnostic functions (alloc versions) ─────────────────────────── */

/** Voltage trace output structure. */
typedef struct {
    int n_t;                     /* Number of time points (t_max/dt) */
    double* v_trace;             /* Voltage trace values (malloc'd, n_t) */
    double* ts;                   /* Time array (malloc'd, n_t) */
} VoltageTraceOutput;

/**
 * Compute membrane potential voltage trace with automatic memory allocation.
 */
VoltageTraceOutput eventprop_compute_voltage_trace_alloc(double t_max, double dt,
                                                         int target_nrn_idx,
                                                         const double* w, int n_pre, int n_post,
                                                         const double* times, const int32_t* sources, int n_spikes,
                                                         const double* fwd_times, const int32_t* fwd_sources, int n_post_spikes,
                                                         double v_th, double tau_mem, double tau_syn);

/** Free memory owned by a VoltageTraceOutput struct. */
void eventprop_free_voltage_trace_output(VoltageTraceOutput* output);

/** Lambda-I trace output structure. */
typedef struct {
    int n_t;                     /* Number of time points (t_max/dt) */
    double* lambda_trace;        /* Lambda-I trace values (malloc'd, n_t) */
    double* ts;                   /* Time array (malloc'd, n_t) */
} LambdaITraceOutput;

/**
 * Compute lambda-I trace with automatic memory allocation.
 */
LambdaITraceOutput eventprop_compute_lambda_i_trace_alloc(double t_max, double dt,
                                                          int target_nrn_idx,
                                                          const double* fwd_times, int n_post_spikes,
                                                          double v_th, double tau_mem, double tau_syn);

/** Free memory owned by a LambdaITraceOutput struct. */
void eventprop_free_lambda_i_trace_output(LambdaITraceOutput* output);

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
 * Returns: Array of SpikeOutput structs (malloc'd, length batch_size). Caller must free each via
 *          eventprop_free_spike_output(), then free the array itself.
 */
SpikeOutput* eventprop_compute_spikes_batch_alloc(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources,
    const int* n_spikes_per_sample, int batch_size,
    double v_th, double tau_mem, double tau_syn);

/**
 * Free batch spike output array.
 * Frees each SpikeOutput struct's data, then the array itself.
 */
void eventprop_free_batch_spike_output(SpikeOutput* output, int batch_size);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_ALLOC_H */
