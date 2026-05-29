// eventprop_c_api.cpp — C-style wrapper that calls into EventProp.Python's Eigen core.
// Converts flat arrays ↔ Eigen/Spikes types for P/Invoke compatibility.

#include "eventprop_c_api.h"
#include "eventprop_types.h"  // Spikes, Maxima, RowMatrixXd
#include <Eigen/Dense>

// ============================================================
// Helpers: flat array ↔ Eigen/Spikes conversion
// All conversions are zero-copy via Eigen::Map.
// Input: Map caller's flat array as read-only Eigen view.
// Output: Library overloads write directly into caller's buffer
//         via Eigen::Map (no intermediate copy).
// ============================================================

static RowMatrixXd make_matrix(const double* data, int rows, int cols) {
    // Zero-copy: Map flat row-major array directly as Eigen matrix.
    return RowMatrixXd::Map(data, rows, cols);
}

// Helper to create a named Eigen::Map for zero-copy output.
// Must be kept as a local variable so the Ref can bind to it (not a temporary).
#define MAP_OUTPUT_MATRIX(dst, rows, cols) \
    RowMatrixXd::Map((dst), (rows), (cols))
#define MAP_OUTPUT_ARRAY(dst, n) \
    Eigen::Map<Eigen::ArrayXd>((dst), (n))

static Spikes make_spikes(const double* times, const int32_t* sources, int n) {
    return Spikes(
        Eigen::Map<const Eigen::ArrayXd>(times, n),
        Eigen::Map<const Eigen::ArrayXi>((int*)sources, n)
    );
}

static Spikes make_spikes_with_currents(const double* times, const int32_t* sources,
                                         const double* currents, int n) {
    return Spikes(
        Eigen::Map<const Eigen::ArrayXd>(times, n),
        Eigen::Map<const Eigen::ArrayXi>((int*)sources, n),
        Eigen::Map<const Eigen::ArrayXd>(currents, n),
        std::vector<double>(),  // first_spike_times (empty)
        std::vector<int>()      // first_spike_idxs (empty)
    );
}

// ============================================================
// compute_sums — delegates to EventProp.Python::compute_sums (zero-copy output)
// ============================================================

void eventprop_compute_sums(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double tau_mem, double tau_syn,
    double* sum0_out, double* sum1_out)
{
    auto w_mat = make_matrix(w, n_pre, n_post);
    Spikes spikes = make_spikes(times, sources, n_spikes);

    // Zero-copy: library writes directly into caller's buffers via Eigen::Map.
    auto sum0_map = MAP_OUTPUT_MATRIX(sum0_out, n_spikes + 1, n_post);
    auto sum1_map = MAP_OUTPUT_MATRIX(sum1_out, n_spikes + 1, n_post);
    ::compute_sums(w_mat, spikes, tau_mem, tau_syn, sum0_map, sum1_map);
}

// ============================================================
// compute_spikes — delegates to EventProp.Python::compute_spikes
// ============================================================

int eventprop_compute_spikes(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double v_th, double tau_mem, double tau_syn,
    double* out_times, int32_t* out_sources, double* out_currents)
{
    auto w_mat = make_matrix(w, n_pre, n_post);
    Spikes spikes = make_spikes(times, sources, n_spikes);

    Spikes result = ::compute_spikes(w_mat, spikes, v_th, tau_mem, tau_syn);

    int n_out = result.n_spikes;
    for (int i = 0; i < n_out; i++) {
        out_times[i] = result.times(i);
        out_sources[i] = (int32_t)result.sources(i);
        out_currents[i] = result.currents(i);
    }

    return n_out;
}

// ============================================================
// compute_maxima — delegates to EventProp.Python::compute_maxima
// ============================================================

void eventprop_compute_maxima(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double tau_mem, double tau_syn,
    double* maxima_times, double* maxima_values)
{
    auto w_mat = make_matrix(w, n_pre, n_post);
    Spikes spikes = make_spikes(times, sources, n_spikes);

    Maxima result = ::compute_maxima(w_mat, spikes, tau_mem, tau_syn);

    for (int i = 0; i < n_post; i++) {
        maxima_times[i] = result.times(i);
        maxima_values[i] = result.values(i);
    }
}

// ============================================================
// backward (Spikes loss) — delegates to EventProp.Python::backward (zero-copy output)
// ============================================================

void eventprop_backward_spikes(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* post_times, const int32_t* post_sources, const double* post_currents,
    const double* post_errors, int n_post_spikes,
    const double* w, int n_pre, int n_post,
    double v_th, double tau_mem, double tau_syn,
    double* gradient,      // row-major n_pre x n_post, caller zero-initialized
    double* spike_errors)  // output: n_input_spikes (mutated)
{
    auto w_mat = make_matrix(w, n_pre, n_post);

    Spikes input_spikes = make_spikes(input_times, input_sources, n_input_spikes);
    Spikes post_spikes;

    if (n_post_spikes > 0) {
        // Build post-spikes with currents and errors
        Spikes tmp = make_spikes_with_currents(post_times, post_sources, post_currents, n_post_spikes);
        tmp.errors = Eigen::Map<const Eigen::ArrayXd>(post_errors, n_post_spikes);
        post_spikes = tmp;
    } else {
        // Empty post-spikes — create minimal Spikes object
        post_spikes = Spikes();
    }

    // Zero-copy: library writes gradient and spike errors directly into caller's buffers.
    auto grad_map = MAP_OUTPUT_MATRIX(gradient, n_pre, n_post);
    auto errors_map = MAP_OUTPUT_ARRAY(spike_errors, n_input_spikes);
    ::backward(input_spikes, post_spikes, w_mat, v_th, tau_mem, tau_syn,
               grad_map, errors_map);
}

// ============================================================
// backward (Maxima loss) — delegates to EventProp.Python::backward (zero-copy output)
// ============================================================

void eventprop_backward_maxima(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* maxima_times, const double* maxima_values, const double* maxima_errors, int n_maxima,
    const double* w, int n_pre, int n_post,
    double tau_mem, double tau_syn,
    double* gradient,      // row-major n_pre x n_post, caller zero-initialized
    double* spike_errors)  // output: n_input_spikes (mutated)
{
    auto w_mat = make_matrix(w, n_pre, n_post);

    Spikes input_spikes = make_spikes(input_times, input_sources, n_input_spikes);

    Maxima maxima;
    if (n_maxima > 0) {
        Eigen::ArrayXd m_times = Eigen::Map<const Eigen::ArrayXd>(maxima_times, n_maxima);
        Eigen::ArrayXd m_values = Eigen::Map<const Eigen::ArrayXd>(maxima_values, n_maxima);
        Eigen::ArrayXd m_errors = Eigen::Map<const Eigen::ArrayXd>(maxima_errors, n_maxima);
        maxima = Maxima(m_times, m_values, m_errors);
    } else {
        maxima = Maxima();
    }

    // Zero-copy: library writes gradient and spike errors directly into caller's buffers.
    auto grad_map = MAP_OUTPUT_MATRIX(gradient, n_pre, n_post);
    auto errors_map = MAP_OUTPUT_ARRAY(spike_errors, n_input_spikes);
    ::backward(input_spikes, maxima, w_mat, tau_mem, tau_syn,
               grad_map, errors_map);
}
