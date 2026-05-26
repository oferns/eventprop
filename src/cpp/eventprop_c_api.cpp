// eventprop_c_api.cpp — C-style wrapper around the original EventProp algorithm.
// Uses Eigen internally but exposes a flat-array API for P/Invoke.

#include "eventprop_c_api.h"
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>

#ifndef __APPLE__
#include <execution>
#endif

#define EP_INF std::numeric_limits<double>::infinity()
#define EP_EPS std::numeric_limits<double>::epsilon()

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ============================================================
// Helpers
// ============================================================
static RowMatrixXd make_matrix(const double* data, int rows, int cols) {
    // Copy to ensure correct row-major layout (Map can misbehave with flat arrays)
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

// ============================================================
// compute_sums
// ============================================================
void eventprop_compute_sums(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double tau_mem, double tau_syn,
    double* sum0_out, double* sum1_out)
{
    auto w_mat = make_matrix(w, n_pre, n_post);

    // Compute exp(t/tau) for each spike time
    Eigen::ArrayXd exp_mem(n_spikes), exp_syn(n_spikes);
    for (int i = 0; i < n_spikes; i++) {
        exp_mem[i] = std::exp(times[i] / tau_mem);
        exp_syn[i] = std::exp(times[i] / tau_syn);
    }

    RowMatrixXd s0(n_spikes + 1, n_post), s1(n_spikes + 1, n_post);
    s0.setZero();
    s1.setZero();

    for (int i = 0; i < n_spikes; i++) {
        s0.row(i + 1) = s0.row(i).array() + exp_syn[i] * w_mat.row(sources[i]).array();
        s1.row(i + 1) = s1.row(i).array() + exp_mem[i] * w_mat.row(sources[i]).array();
    }

    copy_matrix_to_flat(s0, sum0_out);
    copy_matrix_to_flat(s1, sum1_out);
}

// ============================================================
// compute_spikes — full algorithm
// ============================================================
int eventprop_compute_spikes(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double v_th, double tau_mem, double tau_syn,
    double* out_times, int32_t* out_sources, double* out_currents)
{
    auto w_mat = make_matrix(w, n_pre, n_post);

    double k_prefactor = tau_syn / (tau_mem - tau_syn);
    double tmax_prefactor = 1.0 / (1.0 / tau_mem - 1.0 / tau_syn);
    double tmax_summand = std::log(tau_syn / tau_mem);

    // Compute sums
    Eigen::ArrayXd exp_mem(n_spikes), exp_syn(n_spikes);
    for (int i = 0; i < n_spikes; i++) {
        exp_mem[i] = std::exp(times[i] / tau_mem);
        exp_syn[i] = std::exp(times[i] / tau_syn);
    }

    RowMatrixXd sum0(n_spikes + 1, n_post), sum1(n_spikes + 1, n_post);
    sum0.setZero();
    sum1.setZero();

    for (int i = 0; i < n_spikes; i++) {
        sum0.row(i + 1) = sum0.row(i).array() + exp_syn[i] * w_mat.row(sources[i]).array();
        sum1.row(i + 1) = sum1.row(i).array() + exp_mem[i] * w_mat.row(sources[i]).array();
    }

    // Per-neuron spike lists
    std::vector<std::vector<std::pair<double, double>>> post_spikes(n_post);

    // Voltage at time t for interval t_pre_idx
    auto v_func = [&](double t, int target_nrn, int t_pre_idx) {
        double mem = k_prefactor * (sum1(t_pre_idx, target_nrn) * std::exp(-t / tau_mem)
                          - sum0(t_pre_idx, target_nrn) * std::exp(-t / tau_syn));
        for (auto& spike : post_spikes[target_nrn]) {
            if (spike.first > t) break;
            mem -= v_th * std::exp(-(t - spike.first) / tau_mem);
        }
        return mem;
    };

    // Time of maximum voltage in interval [t_pre_idx-1, t_pre_idx]
    auto get_tmax = [&](int t_pre_idx, int target_nrn) {
        if (t_pre_idx == 0) return 0.0;
        double t_input = (t_pre_idx == n_spikes) ? EP_INF : times[t_pre_idx];
        double sum0_e = sum0(t_pre_idx, target_nrn);
        double sum1_e = sum1(t_pre_idx, target_nrn);
        for (auto& spike : post_spikes[target_nrn]) {
            if (t_input <= spike.first) break;
            sum1_e += -v_th * std::exp(spike.first / tau_mem);
        }
        double tmax = tmax_prefactor * (tmax_summand + std::log(sum1_e / sum0_e));
        double t_before = (t_pre_idx > 0) ? times[t_pre_idx - 1] : 0.0;
        for (auto& spike : post_spikes[target_nrn]) {
            if (t_input <= spike.first) break;
            if (spike.first > t_before) t_before = spike.first;
        }
        if (t_before <= tmax && tmax <= t_input) return tmax;
        return EP_INF;
    };

    // Brent root finder (inline)
    auto bracket_spike = [&](double a, double b, int target_nrn, int t_pre_idx) {
        auto f = [&](double t) { return v_func(t, target_nrn, t_pre_idx) - v_th; };
        double fa = f(a), fb = f(b);
        if (fa * fb > 0) return EP_INF;

        double c = a, fc = fa;
        for (int iter = 0; iter < 1024; iter++) {
            double m = 0.5 * (b - a);
            if (std::abs(m) <= 1e-14 || fb == 0.0) return b;

            double s = 0.0, d = 0.0;
            bool use_bisect = true;

            if (std::abs(fc) > std::abs(fb) && c != a && c != b) {
                double qba = (fb - fa) / (b - a);
                double qcb = (fc - fb) / (c - b);
                double qcbba = (qcb - qba) / (c - a);
                if (std::abs(qcbba) > 1e-300 && std::abs(qba + qcb) > 1e-300) {
                    s = b - fb / (qba + qcb * (b - a) * qcbba / (qba + qcb));
                    if ((s > a && s < b) && std::abs(s - b) < 0.5 * (b - a))
                        use_bisect = false;
                }
            }

            if (use_bisect) s = a + 0.5 * (b - a);

            double prev_b = b;
            b = s;
            fb = f(b);

            if (std::abs(fb) <= 1e-14 || std::abs(b - prev_b) <= 1e-14 * std::max(1.0, std::abs(prev_b)))
                return b;

            if (fa * fb < 0.0) { c = prev_b; fc = f(prev_b); }
            else if (fb * fc < 0.0) { a = prev_b; fa = f(prev_b); }
            else { a = prev_b; fa = f(prev_b); }
        }
        return b;
    };

    // Current at time t
    auto get_i = [&](double t, int target_nrn) {
        int idx = 0;
        for (; idx < n_spikes; idx++) {
            if (times[idx] > t) break;
        }
        return sum0(idx, target_nrn) * std::exp(-t / tau_syn);
    };

    // Main loop over neurons (parallel on non-Apple)
#ifndef __APPLE__
#pragma omp parallel for
#endif
    for (int target_nrn = 0; target_nrn < n_post; target_nrn++) {
        bool finished = false;
        int processed_up_to = 0;

        while (!finished) {
            for (int i = processed_up_to; i < n_spikes + 1; i++) {
                if (i == n_spikes) {
                    // Last interval — check after last input spike
                    double sum0_e = sum0(i, target_nrn);
                    double sum1_e = sum1(i, target_nrn);
                    for (auto& spike : post_spikes[target_nrn]) {
                        sum1_e += -v_th * std::exp(spike.first / tau_mem);
                    }
                    double tmax = tmax_prefactor * (tmax_summand + std::log(sum1_e / sum0_e));
                    if (std::isfinite(tmax)) {
                        double vmax = v_func(tmax, target_nrn, i);
                        if (vmax > v_th + EP_EPS) {
                            double t_lower = post_spikes[target_nrn].empty() ? 0.0 :
                                             post_spikes[target_nrn].back().first;
                            double t_post = bracket_spike(t_lower, tmax, target_nrn, i);
                            if (t_post > t_lower && t_post < tmax) {
                                post_spikes[target_nrn].push_back({t_post, get_i(t_post, target_nrn)});
                            }
                        }
                    }
                    finished = true;
                } else {
                    double t_pre = times[i];
                    double tmax = get_tmax(i, target_nrn);

                    if (tmax != EP_INF) {
                        double vmax = v_func(tmax, target_nrn, i);
                        if (vmax > v_th + EP_EPS) {
                            double t_before = 0.0;
                            if (i > 0) {
                                t_before = times[i - 1];
                                for (auto& spike : post_spikes[target_nrn]) {
                                    if (spike.first > t_pre) break;
                                    if (t_before < spike.first) t_before = spike.first;
                                }
                            }
                            double t_post = bracket_spike(t_before, tmax, target_nrn, i);
                            post_spikes[target_nrn].push_back({t_post, get_i(t_post, target_nrn)});
                            processed_up_to = i;
                            break;
                        }
                    }

                    // Check at input spike time itself
                    double v_at_input = v_func(t_pre, target_nrn, i);
                    if (v_at_input > v_th + EP_EPS) {
                        double t_before = 0.0;
                        if (i > 0) {
                            t_before = times[i - 1];
                            for (auto& spike : post_spikes[target_nrn]) {
                                if (spike.first > t_pre) break;
                                if (t_before < spike.first) t_before = spike.first;
                            }
                        }
                        double t_post = bracket_spike(t_before, t_pre, target_nrn, i);
                        post_spikes[target_nrn].push_back({t_post, get_i(t_post, target_nrn)});
                        processed_up_to = i;
                        break;
                    }

                    if (i == n_spikes) finished = true;
                }
            }
        }
    }

    // Collect and sort all spikes by time
    std::vector<std::pair<std::pair<double, double>, int32_t>> all_spikes;
    for (int nrn = 0; nrn < n_post; nrn++) {
        for (auto& spike : post_spikes[nrn]) {
            all_spikes.push_back({spike, (int32_t)nrn});
        }
    }

#ifndef __APPLE__
    std::sort(std::execution::par_unseq, all_spikes.begin(), all_spikes.end(),
        [](const auto& a, const auto& b) { return a.first.first < b.first.first; });
#else
    std::sort(all_spikes.begin(), all_spikes.end(),
        [](const auto& a, const auto& b) { return a.first.first < b.first.first; });
#endif

    int n_out = (int)all_spikes.size();
    for (int i = 0; i < n_out; i++) {
        out_times[i] = all_spikes[i].first.first;
        out_sources[i] = all_spikes[i].second;
        out_currents[i] = all_spikes[i].first.second;
    }

    return n_out;
}

// ============================================================
// compute_maxima
// ============================================================
void eventprop_compute_maxima(
    const double* w, int n_pre, int n_post,
    const double* times, const int32_t* sources, int n_spikes,
    double tau_mem, double tau_syn,
    double* maxima_times, double* maxima_values)
{
    auto w_mat = make_matrix(w, n_pre, n_post);

    double k_prefactor = tau_syn / (tau_mem - tau_syn);
    double tmax_prefactor = 1.0 / (1.0 / tau_mem - 1.0 / tau_syn);
    double tmax_summand = std::log(tau_syn / tau_mem);

    // Compute sums
    Eigen::ArrayXd exp_mem(n_spikes), exp_syn(n_spikes);
    for (int i = 0; i < n_spikes; i++) {
        exp_mem[i] = std::exp(times[i] / tau_mem);
        exp_syn[i] = std::exp(times[i] / tau_syn);
    }

    RowMatrixXd sum0(n_spikes + 1, n_post), sum1(n_spikes + 1, n_post);
    sum0.setZero();
    sum1.setZero();

    for (int i = 0; i < n_spikes; i++) {
        sum0.row(i + 1) = sum0.row(i).array() + exp_syn[i] * w_mat.row(sources[i]).array();
        sum1.row(i + 1) = sum1.row(i).array() + exp_mem[i] * w_mat.row(sources[i]).array();
    }

    auto v_func = [&](double t, int target_nrn, int t_pre_idx) {
        return k_prefactor * (sum1(t_pre_idx, target_nrn) * std::exp(-t / tau_mem)
                     - sum0(t_pre_idx, target_nrn) * std::exp(-t / tau_syn));
    };

    auto get_tmax = [&](int t_pre_idx, int target_nrn) {
        if (t_pre_idx == 0) return 0.0;
        double t_input = (t_pre_idx == n_spikes) ? EP_INF : times[t_pre_idx];
        double sum0_e = sum0(t_pre_idx, target_nrn);
        double sum1_e = sum1(t_pre_idx, target_nrn);
        double t_before = (t_pre_idx > 0) ? times[t_pre_idx - 1] : 0.0;
        double tmax = tmax_prefactor * (tmax_summand + std::log(sum1_e / sum0_e));
        if (t_before <= tmax && tmax <= t_input) return tmax;
        return EP_INF;
    };

    // Initialize output
    for (int i = 0; i < n_post; i++) {
        maxima_times[i] = std::numeric_limits<double>::quiet_NaN();
        maxima_values[i] = 0.0;
    }

#ifndef __APPLE__
#pragma omp parallel for
#endif
    for (int target_nrn = 0; target_nrn < n_post; target_nrn++) {
        for (int i = 0; i <= n_spikes; i++) {
            double t_input = (i == n_spikes) ? EP_INF : times[i];
            double t_before = (i > 0) ? times[i - 1] : 0.0;

            double tmax = get_tmax(i, target_nrn);
            if (tmax != EP_INF) {
                double vmax = v_func(tmax, target_nrn, i);
                if (vmax > maxima_values[target_nrn]) {
                    maxima_values[target_nrn] = vmax;
                    maxima_times[target_nrn] = tmax;
                }
            }

            double v_at_input = v_func(t_input, target_nrn, i);
            if (v_at_input > maxima_values[target_nrn]) {
                maxima_values[target_nrn] = v_at_input;
                maxima_times[target_nrn] = (t_input == EP_INF) ? t_before : t_input;
            }
        }
    }
}

// ============================================================
// backward (Spikes loss) — full adjoint method with merged spike timeline
// ============================================================
void eventprop_backward_spikes(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* post_times, const int32_t* post_sources, const double* post_currents,
    const double* post_errors, int n_post_spikes,
    const double* w, int n_pre, int n_post,
    double v_th, double tau_mem, double tau_syn,
    double* gradient)  // row-major n_pre x n_post, caller zero-initialized
{
    // No post-spike activity — nothing to compute
    if (n_post_spikes == 0)
        return;

    auto w_mat = make_matrix(w, n_pre, n_post);
    RowMatrixXd grad = RowMatrixXd::Map(gradient, n_pre, n_post);

    int total_spikes = n_input_spikes + n_post_spikes;
    int max_neurons = std::max(n_pre, n_post);

    // Merge input + post-spike indices with pre-computed times for sorting
    struct MergedSpike {
        int idx;
        bool is_input;
        double time;
    };

    std::vector<MergedSpike> merged(total_spikes);
    for (int i = 0; i < n_input_spikes; i++)
        merged[i] = {i, true, input_times[i]};
    for (int i = 0; i < n_post_spikes; i++)
        merged[n_input_spikes + i] = {i, false, post_times[i]};

    // Sort descending by time
    std::sort(merged.begin(), merged.end(),
        [](const MergedSpike& a, const MergedSpike& b) { return a.time > b.time; });

    // Combined error array: input spike errors [0..n_input_spikes-1] + post-spike errors [n_input_spikes..total]
    std::vector<double> all_errors(total_spikes, 0.0);
    for (int i = 0; i < n_post_spikes; i++)
        all_errors[n_input_spikes + i] = post_errors[i]; // caller provides post-spike errors

    // Initialize λ_v = 0, previous_t = largest_time
    std::vector<double> lambda_v(n_post, 0.0);
    double largest_time = merged[0].time;
    double previous_t = largest_time;

    // Backward prefactor
    double k_bwd_prefactor = tau_mem / (tau_mem - tau_syn);

    // Per-neuron jump counters and flat jump arrays
    std::vector<int> jump_end(max_neurons, 0);
    std::vector<double> lambda_i_jumps_mag(total_spikes, 0.0);
    std::vector<double> lambda_i_jumps_offset(total_spikes, 0.0);

    // Main backward sweep through merged timeline (single pass)
    for (int s = 0; s < total_spikes; s++)
    {
        int spike_idx = merged[s].idx;
        bool is_input = merged[s].is_input;
        double spike_time = merged[s].time;
        int spike_source = is_input ? input_sources[spike_idx] : post_sources[spike_idx];

        // Time elapsed since largest time (increases as we go backward)
        double t_bwd = largest_time - spike_time;

        // Propagate λ_v backward through membrane decay
        double dt = previous_t - t_bwd;
        for (int nrn = 0; nrn < n_post; nrn++)
            lambda_v[nrn] *= std::exp(-dt / tau_mem);

        if (is_input)
        {
            // Input spike: compute λ_i from accumulated jumps for this source neuron
            std::vector<double> lambda_i(max_neurons, 0.0);
            int start = 0, end = jump_end[spike_source];
            for (int j = start; j < end; j++)
            {
                double jump_time_offset = lambda_i_jumps_offset[j];
                double jump_mag = lambda_i_jumps_mag[j];
                lambda_i[spike_source] += jump_mag * k_bwd_prefactor *
                    (std::exp(-(t_bwd - jump_time_offset) / tau_mem)
                   - std::exp(-(t_bwd - jump_time_offset) / tau_syn));
            }

            // Add gradient contribution: grad[source, nrn] += -tau_syn * lambda_i[nrn]
            for (int nrn = 0; nrn < n_post; nrn++)
                grad(spike_source, nrn) -= tau_syn * lambda_i[nrn];

            // Compute outbound signal: sum_nrn(w[nrn, source] * (lambda_v[nrn] - lambda_i[nrn]))
            // w is stored row-major as n_pre x n_post: w[source * n_post + nrn]
            double outbound_signal = 0.0;
            for (int nrn = 0; nrn < n_post; nrn++)
            {
                double w_val = w[spike_source * n_post + nrn];
                outbound_signal += w_val * (lambda_v[nrn] - lambda_i[nrn]);
            }

            // Add outbound signal to input spike error
            all_errors[spike_idx] += outbound_signal;
        }
        else
        {
            // Post-synaptic spike: record λ_i jump and update λ_v
            double current = post_currents[spike_idx];
            double inv_i_current = 1.0 / (current - v_th);

            // Record λ_i jump for this neuron in the flat array
            int j_idx = jump_end[spike_source];
            double post_error = all_errors[n_input_spikes + spike_idx]; // post-spikes at offset
            lambda_i_jumps_mag[j_idx] = inv_i_current * (v_th * lambda_v[spike_source] + post_error);
            lambda_i_jumps_offset[j_idx] = t_bwd;
            jump_end[spike_source]++;

            // Update λ_v for this neuron
            lambda_v[spike_source] =
                (current / (current - v_th)) * lambda_v[spike_source]
              + inv_i_current * post_error;
        }

        previous_t = t_bwd;
    }

    copy_matrix_to_flat(grad, gradient);
}

// ============================================================
// backward (Maxima loss) — simplified adjoint method
// ============================================================
void eventprop_backward_maxima(
    const double* input_times, const int32_t* input_sources, int n_input_spikes,
    const double* maxima_times, const double* maxima_values, const double* maxima_errors, int n_maxima,
    const double* w, int n_pre, int n_post,
    double tau_mem, double tau_syn,
    double* gradient)  // row-major n_pre x n_post, caller zero-initialized
{
    auto w_mat = make_matrix(w, n_pre, n_post);
    RowMatrixXd grad = RowMatrixXd::Map(gradient, n_pre, n_post);

    double largest_time = input_times[n_input_spikes - 1];
    double k_bwd_prefactor = tau_mem / (tau_mem - tau_syn);

    for (int spike_idx = n_input_spikes - 1; spike_idx >= 0; spike_idx--) {
        double t_bwd = largest_time - input_times[spike_idx];

        for (int max_idx = 0; max_idx < n_maxima; max_idx++) {
            if (std::isnan(maxima_times[max_idx])) continue;

            double t_vmax_bwd = largest_time - maxima_times[max_idx];
            if (t_vmax_bwd > t_bwd) continue;

            double lambda_v = -std::exp(-(t_bwd - t_vmax_bwd) / tau_mem) * maxima_errors[max_idx] / tau_mem;
            double k_bwd_val = k_bwd_prefactor * (std::exp(-t_bwd / tau_mem) - std::exp(-t_bwd / tau_syn));
            double lambda_i = -k_bwd_val * maxima_errors[max_idx] / tau_mem;

            grad(input_sources[spike_idx], max_idx) -= tau_syn * lambda_i;
        }
    }

    copy_matrix_to_flat(grad, gradient);
}
