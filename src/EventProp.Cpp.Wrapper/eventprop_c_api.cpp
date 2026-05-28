// eventprop_c_api.cpp — C-style wrapper that calls into EventProp.cpp (Eigen-based algorithm).
// Converts flat arrays ↔ Eigen/Spikes types for P/Invoke compatibility.

#include "eventprop_c_api.h"
#include "../EventProp.cpp/eventprop_types.h"  // Spikes, Maxima, RowMatrixXd
#include <Eigen/Dense>

// ============================================================
// Helpers: flat array ↔ Eigen/Spikes conversion
// ============================================================

static RowMatrixXd make_matrix(const double* data, int rows, int cols) {
    RowMatrixXd m(rows, cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            m(r, c) = data[r * cols + c];
    return m;
}

static void copy_matrix_to_flat(const RowMatrixXd& src, double* dst) {
    for (int r = 0; r < src.rows(); r++)
        for (int c = 0; c < src.cols(); c++)
            dst[r * src.cols() + c] = src(r, c);
}

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
// compute_sums — delegates to EventProp.cpp::compute_sums
// ============================================================

void eventprop_compute_sums(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double tau_mem, double tau_syn,
    double* sum0_out, double* sum1_out)
{
    auto w_mat = make_matrix(w, n_pre, n_post);
    Spikes spikes = make_spikes(times, sources, n_spikes);

    RowMatrixXd sum0, sum1;
    std::tie(sum0, sum1) = ::compute_sums(w_mat, spikes, tau_mem, tau_syn);

    copy_matrix_to_flat(sum0, sum0_out);
    copy_matrix_to_flat(sum1, sum1_out);
}

// ============================================================
// compute_spikes — delegates to EventProp.cpp::compute_spikes
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
// compute_maxima — delegates to EventProp.cpp::compute_maxima
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
// backward (Spikes loss) — delegates to EventProp.cpp::backward
// ============================================================

void eventprop_backward_spikes(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* post_times, const int32_t* post_sources, const double* post_currents,
    const double* post_errors, int n_post_spikes,
    const double* w, int n_pre, int n_post,
    double v_th, double tau_mem, double tau_syn,
    double* gradient)  // row-major n_pre x n_post, caller zero-initialized
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

    RowMatrixXd grad;
    Eigen::ArrayXd spike_errors;
    std::tie(grad, spike_errors) = ::backward(input_spikes, post_spikes, w_mat, v_th, tau_mem, tau_syn);

    copy_matrix_to_flat(grad, gradient);
}

// ============================================================
// backward (Maxima loss) — delegates to EventProp.cpp::backward
// ============================================================

void eventprop_backward_maxima(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* maxima_times, const double* maxima_values, const double* maxima_errors, int n_maxima,
    const double* w, int n_pre, int n_post,
    double tau_mem, double tau_syn,
    double* gradient)  // row-major n_pre x n_post, caller zero-initialized
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

    RowMatrixXd grad;
    Eigen::ArrayXd spike_errors;
    std::tie(grad, spike_errors) = ::backward(input_spikes, maxima, w_mat, tau_mem, tau_syn);

    copy_matrix_to_flat(grad, gradient);
}
