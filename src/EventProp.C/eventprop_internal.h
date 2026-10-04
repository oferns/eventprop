/**
 * @file eventprop_internal.h
 * @brief Internal constants and context types shared across EventProp modules.
 */
#ifndef EVENTPROP_INTERNAL_H
#define EVENTPROP_INTERNAL_H

#include "eventprop.h"
#include <stddef.h>

#define EXP_STACK_SIZE 1024
#define DIFF_STACK_SIZE 512
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
     double current; 
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
    double** jump_time;   // array of pointers
    double** jump_mag;
    int* jump_count;
    int* jump_cap;
    int n_post_alloc;     // how many per-neuron arrays were actually allocated
} BackwardSpikesCtx;

// ── Internal context for compute_spikes memory management ─────────────────────
typedef struct ComputeSpikesCtx {
    int n_post;  // needed for per-neuron clean-up loop
    SumsOutput sums;              // eventprop_free_sums_output
    double** post_times_arr;      // array + each [n]
    double** post_currents_arr;   // array + each [n]
    int* spike_caps;
    int* spike_counts;
    SpikeEntry* entries;          // sorted collection buffer
    /* Output fields (set by collect_spikes_result) */
    double* result_times;
    int*    result_sources;
    double* result_currents;
    double* result_errors;
    int     result_n_spikes;
} ComputeSpikesCtx;

/** Per-neuron mutable state during spike detection. */
typedef struct NeuronState {
    double* p_times;
    double* p_currents;
    int cap;
    int count;
} NeuronState;

#endif /* EVENTPROP_INTERNAL_H */
