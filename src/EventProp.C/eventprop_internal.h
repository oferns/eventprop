/**
 * @file eventprop_internal.h
 * @brief Internal constants and context types shared across EventProp modules.
 */
#ifndef EVENTPROP_INTERNAL_H
#define EVENTPROP_INTERNAL_H

#include "eventprop.h"
#include <stddef.h>

#define BACKWARD_DIFF_STACK_SIZE 512

#define TOMS748_MAX_ITER 1024
#define TOMS748_TOL 1e-14
#define TOMS748_NEAR_ZERO 1e-300   /* near-zero guard to avoid div-by-zero in curvature checks */

/* Voltage threshold crossing tolerance — has v genuinely exceeded v_th? */
#define SPIKE_THRESHOLD_EPS 1e-14

/* Evaluate v(t) - v_th for root-finding */
#define VOLTAGE(c, t) \
    (compute_v_factored(t, (c)->i, (c)->spike_count, (c)->post_spike_times, (c)->neuron_offset, \
                 (c)->sum0, (c)->sum1, (c)->target_nrn, (c)->n_post, \
                 (c)->k_prefactor, (c)->tau_mem_inv, (c)->tau_syn_inv, (c)->v_th, \
                 (c)->post_spike_exp_st) - (c)->v_th)

typedef struct RootContext {
    const double* sum0; const double* sum1;
    int i, spike_count, target_nrn, n_post;
    const double* post_spike_times; int neuron_offset;
    double k_prefactor, tau_mem_inv, tau_syn_inv, v_th;
    /* Precomputed: exp(post_spike_times[s] * tau_mem_inv) for each spike.
     * Factorization: exp(-(t-st)*tau-1) = exp(-t*tau-1) × exp(st*tau-1)
     * Eliminates N exp() calls per iteration → just 1 (for base terms). */
    const double* post_spike_exp_st;   /* NULL if not precomputed */
} RootContext;

typedef struct SpikeEntry {
     double time;
     int source;
} SpikeEntry;

/* Merge spike comparison for qsort (descending by time) */
typedef struct MergedSpike {
     double time; 
     int idx; 
     int is_input; 
} MergedSpike;

/** Context for eventprop_backward_spikes — groups all heap allocations. */
typedef struct BackwardSpikesCtx {
    MergedSpike* merged;
    double* all_errors;
    double* lambda_v;
    double* lambda_i;
    double* cur_sum;        // prefix row of synaptic sums, one n_post row
    double* currents;       // rebuilt synaptic current at each post spike
    double** jump_time;   // array of pointers
    double** jump_mag;
    int* jump_count;
    int* jump_cap;
    int n_post_alloc;     // how many per-neuron arrays were actually allocated
} BackwardSpikesCtx;

/**
 * Rebuild the synaptic current at each post-synaptic spike.
 *
 * Shared by the backward sweep and the lambda-I diagnostic on purpose: the jump amplitude
 * in the adjoint is 1/(current - v_th) * (v_th * lambda_v + error), so any consumer of the
 * adjoint needs the same current the training kernel computes.
 *
 * @param input         forward-pass parameters (weights, input spikes, taus, v_th)
 * @param post_times    post-synaptic spike times, ascending (borrowed)
 * @param post_sources  post-synaptic spike sources, same length (borrowed)
 * @param n_post_spikes length of post_times / post_sources / currents
 * @param currents      out, n_post_spikes doubles, caller-owned
 * @param row_scratch   one n_post-row of doubles, ZERO-initialised by the caller and
 *                      left in an unspecified state; it is the running synaptic sum
 *
 * Requires input->w, input->times, input->sources to be non-NULL and the post-spike times
 * ascending — the cursor only ever moves forward, so unsorted input silently under-counts.
 * See eventprop_currents.c for why the accumulation direction is not interchangeable.
 */
void eventprop_rebuild_post_currents(const EventPropInput* input,
                                     const double* post_times,
                                     const int* post_sources,
                                     int n_post_spikes,
                                     double* currents,
                                     double* row_scratch);

/** Per-thread scratch during spike detection.
 *  One row is reused across neurons and grown x3 through the injection shim
 *  when a neuron needs more entries than it currently holds, and each thread
 *  accumulates its own entries, so peak memory no longer scales with n_post. */
typedef struct SweepThread {
    double* scratch_times;
    SpikeEntry* entries;      /* grows x2 as this thread collects */
    int row_cap;
    int entries_cap;
    int n_entries;
} SweepThread;

// ── Internal context for compute_spikes memory management ─────────────────────
typedef struct ComputeSpikesCtx {
    SumsOutput sums;              // eventprop_free_sums_output
    SweepThread* threads;         // array of slots; each slot is heap-backed
    int n_threads;                // slots actually allocated
    /* Output fields (set by collect_spikes_result) */
    double* result_times;
    int*    result_sources;
    int     result_n_spikes;
} ComputeSpikesCtx;

/** Per-neuron mutable state during spike detection. */
typedef struct NeuronState {
    double* p_times;
    int cap;
    int count;
} NeuronState;

#endif /* EVENTPROP_INTERNAL_H */
