/**
 * @file eventprop.h
 * @brief EventProp: Event-based backpropagation for SNN training.
 *
 * Pure C implementation with optional SIMD (AVX2/NEON) and threading OpenMP
 */

#ifndef EVENTPROP_H
#define EVENTPROP_H

#ifdef __cplusplus
extern "C" {
#endif

/** Error codes returned by EventProp functions.
 *  Set via the `error_no` field on output structs when validation is enabled */
typedef enum EventPropError {
    EVENTPROP_OK                      = 0,
    EVENTPROP_INVALID_DIMENSION       = 1,   /* n_pre <= 0 or n_post <= 0 */
    EVENTPROP_NEGATIVE_SPIKES         = 2,   /* n_spikes < 0 */
    EVENTPROP_NULL_POINTER            = 3,   /* w/times/sources is NULL when required */
    EVENTPROP_INVALID_SOURCE          = 4,   /* sources[i] out of [0, n_pre) range */
    EVENTPROP_INVALID_TAU             = 5,   /* tau_mem <= 0, tau_syn <= 0, or tau_mem == tau_syn */
    EVENTPROP_ALLOCATION_FAILED       = 6,   /* calloc/malloc returned NULL */
    EVENTPROP_INVALID_CURRENT         = 7,   /* spike.currents[i] == v_th (division by zero) */
    EVENTPROP_NEGATIVE_SPIKE_TIME     = 8,   /* times[i] < 0 */
    EVENTPROP_EXCESSIVE_SPIKE_TIME    = 9,   /* max(times) > MAX_SPIKE_TIME */
    EVENTPROP_TAU_TOO_SMALL           = 10,  /* tau_mem or tau_syn < MIN_TAU */
    EVENTPROP_TAU_TOO_LARGE           = 11,  /* tau_mem or tau_syn > MAX_TAU */
    EVENTPROP_INVALID_VTH             = 12,  /* v_th <= 0 */
    EVENTPROP_DIMENSION_TOO_LARGE     = 13,  /* n_pre or n_post > MAX_DIM */
    EVENTPROP_TIMES_TAU_MISMATCH      = 14,  /* max(times)/min(tau) >= EXP_OVERFLOW_THRESHOLD */
    EVENTPROP_TAU_TOO_CLOSE           = 15,  /* |tau_mem - tau_syn| / min(tau) < TAU_REL_DIFF_MIN */
} EventPropError;

/** ── Validation limits ──────────────────────────────────────────────── */

/// Maximum allowed spike time (seconds). ~11.6 days.
#define EVENTPROP_MAX_SPIKE_TIME        1e6
/// Minimum allowed tau (seconds). Below this, exp(-t/tau) underflows instantly.
#define EVENTPROP_MIN_TAU               1e-9
/// Maximum allowed tau (seconds). ~17 minutes; impractical above for SNN timescales.
#define EVENTPROP_MAX_TAU               1e3
/// Maximum allowed dimension per axis. Prevents int overflow in indexing and impossible allocations.
#define EVENTPROP_MAX_DIM               100000
/// exp(x) overflows to INF at x ≈ 709.78. We use a safety margin of 700.
#define EVENTPROP_EXP_OVERFLOW_THRESHOLD 700.0
/// Minimum relative difference between tau_mem and tau_syn (tau_mem == tau_syn already caught).
/// Below this, k_prefactor = tau_syn / (tau_mem - tau_syn) blows up.
#define EVENTPROP_TAU_REL_DIFF_MIN      1e-6

/** Non-owning view into a flat row-major double array.
 *  Mimics Eigen::Ref<RowMatrixXd const>: attached dimensions,
 *  .row(i) access, and element-wise indexing without allocation. */
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

typedef struct SpikeOutput {
    EventPropError error_no;  /* Error code (EVENTPROP_OK on success) */
    int n_spikes;            /* Number of output spikes */
    double* times;           /* Output spike times (malloc'd) */
    int* sources;        /* Output source neurons (malloc'd) */
    double* currents;        /* Synaptic currents at spike times (malloc'd) */
    double* errors;          /* Post-synaptic errors (malloc'd, same size as n_spikes) — written by caller before backward pass */
} SpikeOutput;

typedef struct SumsOutput {
    EventPropError error_no;  /* Error code (EVENTPROP_OK on success) */
    int n_rows;               /* Pre-synaptic neurons */
    int n_post;               /* Post-synaptic neurons */
    double* sum0;             /* Cumulative sums (malloc'd, n_rows * n_post) */
    double* sum1;             /* Cumulative sums (malloc'd, n_rows * n_post) */
} SumsOutput;

typedef struct MaximaOutput {
    EventPropError error_no;  /* Error code (EVENTPROP_OK on success) */
    int n_post;               /* Post-synaptic neurons */
    double* times;           /* Maxima times (malloc'd, n_post) */
    double* values;          /* Maxima values (malloc'd, n_post) */
    double* errors;          /* Maxima errors per post-neuron (malloc'd, n_post) — written by caller before backward pass */
} MaximaOutput;



typedef struct BackwardOutput {
    EventPropError error_no;  /* Error code (EVENTPROP_OK on success) */
    int n_pre;                /* Pre-synaptic neurons */
    int n_post;               /* Post-synaptic neurons */
    int n_spikes;             /* Number of input spikes */
    double* grad;            /* Weight gradient (malloc'd, n_pre * n_post) */
    double* spike_errors;   /* Input spike errors (malloc'd, n_spikes) */
} BackwardOutput;


void eventprop_free_spike_output(SpikeOutput output);
void eventprop_free_sums_output(SumsOutput output);
void eventprop_free_maxima_output(MaximaOutput output);
void eventprop_free_backward_output(BackwardOutput output);

/**
 * Forward pass
 * Returns a SpikeOutput struct — caller must free via eventprop_free_spike_output().
 */
SpikeOutput eventprop_compute_spikes(const EventPropInput input);

/**
 * Compute cumulative sums
 * Returns a SumsOutput struct — caller must free via eventprop_free_sums_output().
 */
SumsOutput eventprop_compute_sums(const EventPropInput input);

/**
 * Compute membrane potential maxima
 * Returns a MaximaOutput struct — caller must free via eventprop_free_maxima_output().
 */
MaximaOutput eventprop_compute_maxima(const EventPropInput input);

/**
 * Backward pass (spikes).
 * Returns a BackwardOutput struct — caller must free via eventprop_free_backward_output().
 */
BackwardOutput eventprop_backward_spikes(const EventPropInput input, const SpikeOutput spike);

/**
 * Backward pass (maxima loss).
 * `maxima_times` and `maxima_errors` are borrowed arrays of length n_post.
 * Returns a BackwardOutput struct — caller must free via eventprop_free_backward_output().
 */
BackwardOutput eventprop_backward_maxima(const EventPropInput input,
                                         const double* maxima_times,
                                         const double* maxima_errors);

/** Validate an EventPropInput and return the first error found (or OK). */
EventPropError eventprop_validate_input(const EventPropInput input);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_H */
