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

/* ── Scratch buffers (caller allocates memory, this sets pointers) ───────── */

typedef struct {
    double* sum0;        /* n_spikes_plus_1 * n_post  — cumulative exponential sums */
    double* sum1;        /* n_spikes_plus_1 * n_post  — cumulative exponential sums */
    double* fwd_times;       /* max_out = n_spikes * n_post — post-synaptic spike times */
    int32_t* fwd_sources;    /* max_out — post-synaptic spike source neurons */
    double* fwd_currents;    /* max_out — post-synaptic synaptic currents at spike times */
} EventPropScratch;

typedef struct {
    double* times;   /* n_post — membrane potential maxima per neuron */
    double* values;  /* n_post — maximum membrane potential value per neuron */
} EventPropMaxima;

/* ── Buffer initialization (sets pointers, does NOT allocate memory) ──────── */

void eventprop_scratch_init(EventPropScratch* buf, int n_pre, int n_post, int n_spikes);
void eventprop_maxima_init(EventPropMaxima* maxima, int n_post);

/* ── Core algorithms ─────────────────────────────────────────────────────── */

/**
 * Compute cumulative exponential-weighted sums for fast membrane potential evaluation.
 *
 * Pre-computes sum0 and sum1 arrays that allow O(1) membrane potential evaluation
 * at any time t via Eq. 12 in the EventProp paper.
 */
void eventprop_compute_sums(const double* w, int n_pre, int n_post,
                            const double* times, const int32_t* sources, int n_spikes,
                            double tau_mem, double tau_syn,
                            double* sum0, double* sum1);

/**
 * Forward pass: predict post-synaptic spike times via root-finding.
 *
 * For each post-synaptic neuron, finds the earliest time t > prev_spike_time
 * where membrane potential v(t) >= v_th. Uses Toms748 (Brent) root-finding.
 *
 * @return Number of post-synaptic spikes generated (<= max_out)
 */
int eventprop_compute_spikes(const double* w, int n_pre, int n_post,
                             const double* times, const int32_t* sources, int n_spikes,
                             double v_th, double tau_mem, double tau_syn,
                             double* out_times, int32_t* out_sources, double* out_currents);

/**
 * Backward pass: adjoint method computing weight gradients.
 *
 * Propagates λ_v (membrane) and λ_i (synaptic) adjoints in time-reversed order,
 * accumulating weight gradients via Eq. 18-20 in the EventProp paper.
 */
void eventprop_backward_spikes(const double* times, const int32_t* sources, int n_spikes,
                               const double* fwd_times, const int32_t* fwd_sources,
                               const double* fwd_currents, const double* post_errors, int n_post_spikes,
                               const double* w, int n_pre, int n_post,
                               double v_th, double tau_mem, double tau_syn,
                               double* grad);

/**
 * Find membrane potential maxima per neuron (for voltage-based loss).
 *
 * Evaluates the membrane potential trajectory for each post-synaptic neuron and
 * records the time and value of its global maximum.
 */
void eventprop_compute_maxima(const double* w, int n_pre, int n_post,
                              const double* times, const int32_t* sources, int n_spikes,
                              double tau_mem, double tau_syn, EventPropMaxima* maxima);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_H */
