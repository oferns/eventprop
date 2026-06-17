// C-style API for P/Invoke from C#
// Wraps the original eventprop_cpp.cpp functions with flat arrays and sizes.

#ifndef EVENTPROP_C_API_H
#define EVENTPROP_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// C-compatible matrix descriptor for zero-copy output.
// Caller allocates `rows * cols` doubles; library writes directly into it
// via Eigen::Map (no intermediate copy).
typedef struct CMatrix {
    double* data;
    int rows;
    int cols;
} CMatrix;

// --- compute_sums ---
// Writes sum0 and sum1 matrices (row-major, n_spikes+1 rows x n_post cols)
// Caller must allocate buffers of size (n_spikes+1)*n_post each.
void eventprop_compute_sums(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double tau_mem, double tau_syn,
    double* sum0, double* sum1);

// --- compute_spikes ---
// Writes output spike times, sources, and currents.
// Returns the number of output spikes (0 if none).
int eventprop_compute_spikes(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double v_th, double tau_mem, double tau_syn,
    double* out_times, int32_t* out_sources, double* out_currents);

// --- compute_maxima ---
// Writes maxima times and values (one per post-synaptic neuron).
void eventprop_compute_maxima(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double tau_mem, double tau_syn,
    double* maxima_times, double* maxima_values);

// --- backward (Spikes loss) ---
// Computes weight gradient and spike errors.
// Caller must allocate: gradient[n_pre * n_post], spike_errors[n_input_spikes]
void eventprop_backward_spikes(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* post_times, const int32_t* post_sources, const double* post_currents,
    const double* post_errors, int n_post_spikes,
    const double* w, int n_pre, int n_post,
    double v_th, double tau_mem, double tau_syn,
    double* gradient,      // row-major output: n_pre x n_post
    double* spike_errors); // output: n_input_spikes (mutated)

// --- backward (Maxima loss) ---
// Caller must allocate: gradient[n_pre * n_post], spike_errors[n_input_spikes]
void eventprop_backward_maxima(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* maxima_times, const double* maxima_values, const double* maxima_errors, int n_maxima,
    const double* w, int n_pre, int n_post,
    double tau_mem, double tau_syn,
    double* gradient,      // row-major output: n_pre x n_post
    double* spike_errors); // output: n_input_spikes (mutated)

#ifdef __cplusplus
}
#endif

#endif // EVENTPROP_C_API_H
