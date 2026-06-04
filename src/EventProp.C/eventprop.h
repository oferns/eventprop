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

/* Deprecation macros */
#if defined(_MSC_VER)
    #define EVENTPROP_DEPRECATED __declspec(deprecated("Use alloc version instead"))
#elif defined(__GNUC__) || defined(__clang__)
    #define EVENTPROP_DEPRECATED __attribute__((deprecated("Use alloc version instead")))
#else
    #define EVENTPROP_DEPRECATED
#endif

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

/* ── Buffer initialization (sets pointers, does NOT allocate memory) ──────── */

void eventprop_scratch_init(EventPropScratch* buf, int n_pre, int n_post, int n_spikes);

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
                            double* sum0, double* sum1) EVENTPROP_DEPRECATED;

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
                             double* out_times, int32_t* out_sources, double* out_currents) EVENTPROP_DEPRECATED;

/**
 * Compute first_spike per neuron from sorted spike output.
 *
 * For each post-synaptic neuron, finds the time and index of its first spike
 * in the sorted output array. Neurons with no spikes get NaN/0.
 *
 * @param n_post           Number of post-synaptic neurons
 * @param out_times        Sorted spike times (from eventprop_compute_spikes)
 * @param out_sources      Sorted spike sources (from eventprop_compute_spikes)
 * @param n_total_spikes   Total number of spikes in the output
 * @param first_spike_times Output: time of neuron i's first spike (NaN if none)
 * @param first_spike_idxs  Output: index in out_times of neuron i's first spike (0 if none)
 */
void eventprop_compute_first_spike(int n_post, const double* out_times,
                                    const int32_t* out_sources, int n_total_spikes,
                                    double* first_spike_times, int32_t* first_spike_idxs) EVENTPROP_DEPRECATED;

/**
 * Backward pass: adjoint method computing weight gradients and input spike errors.
 *
 * Propagates λ_v (membrane) and λ_i (synaptic) adjoints in time-reversed order,
 * accumulating weight gradients via Eq. 18-20 in the EventProp paper.
 *
 * @param grad          Output: weight gradient (row-major, n_pre x n_post), zero-initialized by caller
 * @param spike_errors  Output: input spike errors (n_spikes), mutated in-place by caller
 */
void eventprop_backward_spikes(const double* times, const int32_t* sources, int n_spikes,
                               const double* fwd_times, const int32_t* fwd_sources,
                               const double* fwd_currents, const double* post_errors, int n_post_spikes,
                               const double* w, int n_pre, int n_post,
                               double v_th, double tau_mem, double tau_syn,
                               double* grad,
                               double* spike_errors) EVENTPROP_DEPRECATED;

/**
 * Find membrane potential maxima per neuron (for voltage-based loss).
 *
 * Evaluates the membrane potential trajectory for each post-synaptic neuron and
 * records the time and value of its global maximum.
 */
void eventprop_compute_maxima(const double* w, int n_pre, int n_post,
                              const double* times, const int32_t* sources, int n_spikes,
                              double tau_mem, double tau_syn,
                              double* maxima_times, double* maxima_values) EVENTPROP_DEPRECATED;

/**
 * Backward pass for maxima-based loss.
 *
 * Propagates error signals from membrane potential maxima backward through the
 * input spike timeline. Gradient and spike errors are written zero-copy.
 *
 * @param gradient      Output: weight gradient (row-major, n_pre x n_post), zero-initialized by caller
 * @param spike_errors  Output: input spike errors (n_input_spikes), mutated in-place by caller
 */
void eventprop_backward_maxima(const double* input_times, const int32_t* input_sources,
                               int n_input_spikes,
                               const double* maxima_times, const double* maxima_values,
                               const double* maxima_errors, int n_maxima,
                               const double* w, int n_pre, int n_post,
                               double tau_mem, double tau_syn,
                               double* gradient,
                               double* spike_errors) EVENTPROP_DEPRECATED;

/* ── Diagnostic functions ─────────────────────────────────────────── */

/**
 * Compute membrane potential voltage trace for a single post-synaptic neuron.
 *
 * Evaluates v(t) at uniformly spaced time points [0, dt, 2*dt, ..., t_max).
 * Includes contributions from both pre-synaptic and post-synaptic spikes.
 *
 * @param t_max          Maximum simulation time
 * @param dt             Time step
 * @param target_nrn_idx Post-synaptic neuron index
 * @param w              Weight matrix (row-major, n_pre x n_post)
 * @param times          Pre-synaptic spike times
 * @param sources        Pre-synaptic spike source neurons
 * @param n_spikes       Number of pre-synaptic spikes
 * @param fwd_times      Post-synaptic spike times (from eventprop_compute_spikes)
 * @param fwd_sources    Post-synaptic spike source neurons
 * @param n_post_spikes  Number of post-synaptic spikes
 * @param v_th           Spike threshold
 * @param tau_mem        Membrane time constant
 * @param tau_syn        Synaptic time constant
 * @param v_trace        Output: voltage trace (size = t_max/dt)
 * @param ts             Output: time array (size = t_max/dt)
 */
void eventprop_compute_voltage_trace(double t_max, double dt, int target_nrn_idx,
                                     const double* w, int n_pre, int n_post,
                                     const double* times, const int32_t* sources, int n_spikes,
                                     const double* fwd_times, const int32_t* fwd_sources, int n_post_spikes,
                                     double v_th, double tau_mem, double tau_syn,
                                     double* v_trace, double* ts);

/**
 * Compute λ_i (synaptic adjoint) at a specific time for a single neuron.
 *
 * @param t              Time at which to compute λ_i
 * @param target_nrn_idx Post-synaptic neuron index
 * @param fwd_times      Post-synaptic spike times
 * @param n_post_spikes  Number of post-synaptic spikes
 * @param v_th           Spike threshold
 * @param tau_mem        Membrane time constant
 * @param tau_syn        Synaptic time constant
 * @return λ_i value at time t for the target neuron
 */
double eventprop_compute_lambda_i(double t, int target_nrn_idx,
                                  const double* fwd_times, int n_post_spikes,
                                  double v_th, double tau_mem, double tau_syn);

/**
 * Compute λ_i trace over time for a single post-synaptic neuron.
 *
 * @param t_max          Maximum simulation time
 * @param dt             Time step
 * @param target_nrn_idx Post-synaptic neuron index
 * @param fwd_times      Post-synaptic spike times
 * @param n_post_spikes  Number of post-synaptic spikes
 * @param v_th           Spike threshold
 * @param tau_mem        Membrane time constant
 * @param tau_syn        Synaptic time constant
 * @param lambda_trace   Output: λ_i trace (size = t_max/dt)
 * @param ts             Output: time array (size = t_max/dt)
 */
void eventprop_compute_lambda_i_trace(double t_max, double dt, int target_nrn_idx,
                                      const double* fwd_times, int n_post_spikes,
                                      double v_th, double tau_mem, double tau_syn,
                                      double* lambda_trace, double* ts);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_H */
