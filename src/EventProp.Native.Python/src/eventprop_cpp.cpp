#include "eventprop.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <random>

namespace py = pybind11;
using namespace pybind11::literals;

// ── compute_spikes_batch (zero-copy) ────────────────────────────────────────

py::list compute_spikes_batch(
    py::array_t<double, py::array::c_style> w_in,
    int n_pre, int n_post,
    py::list batch, double v_th, double tau_mem, double tau_syn)
{
    const double* w = w_in.data();

    py::list result;

    // Find max spikes needed for scratch allocation
    int max_spikes = 64;
    for (auto item : batch) {
        auto tup = item.cast<py::tuple>();
        auto arr = tup[0].cast<py::array_t<double>>();
        max_spikes = std::max(max_spikes, (int)(arr.size() * n_post));
    }

    double* fwd_times      = new double[max_spikes];
    int32_t* fwd_sources   = new int32_t[max_spikes];
    double* fwd_currents   = new double[max_spikes];

    for (auto item : batch) {
        auto tup = item.cast<py::tuple>();  // (times_array, sources_array)
        auto times_arr = tup[0].cast<py::array_t<double>>();  // keeps ref alive
        auto sources_arr = tup[1].cast<py::array_t<int32_t>>();
        int n_spikes = (int)times_arr.size();

        int n = eventprop_compute_spikes(
            w, n_pre, n_post,
            times_arr.data(), sources_arr.data(), n_spikes,
            v_th, tau_mem, tau_syn,
            fwd_times, fwd_sources, fwd_currents);

        if (n > 0) {
            result.append(py::make_tuple(
                py::array_t<double>(n, fwd_times),
                py::array_t<int32_t>(n, fwd_sources),
                py::array_t<double>(n, fwd_currents)
            ));
        } else {
            result.append(py::make_tuple(
                py::array_t<double>(0),
                py::array_t<int32_t>(0),
                py::array_t<double>(0)
            ));
        }
    }

    delete[] fwd_times;
    delete[] fwd_sources;
    delete[] fwd_currents;

    return result;
}

// ── backward_spikes_batch (zero-copy) ───────────────────────────────────────

void backward_spikes_batch(
    py::list input_batch, py::list post_batch,
    py::array_t<double, py::array::c_style | py::array::forcecast> w_in,
    int n_pre, int n_post,
    py::array_t<double, py::array::c_style | py::array::forcecast> gradient_in,
    double v_th, double tau_mem, double tau_syn)
{
    const double* w = w_in.data();
    auto grad_view = gradient_in.mutable_unchecked<2>();

    for (Py_ssize_t bi = 0; bi < (Py_ssize_t)post_batch.size(); bi++) {
        auto post_tup = post_batch[bi].cast<py::tuple>();
        auto ptimes_arr = post_tup[0].cast<py::array_t<double>>();
        auto psources_arr = post_tup[1].cast<py::array_t<int32_t>>();
        auto pcurrents_arr = post_tup[2].cast<py::array_t<double>>();
        int n_post_spikes = (int)ptimes_arr.size();

        if (n_post_spikes == 0) continue;

        auto input_tup = input_batch[bi].cast<py::tuple>();
        auto itimes_arr = input_tup[0].cast<py::array_t<double>>();
        auto isources_arr = input_tup[1].cast<py::array_t<int32_t>>();
        int n_input_spikes = (int)itimes_arr.size();

        // Get post_errors from Spikes.errors attribute
        py::array_t<double> post_errors;
        try {
            post_errors = post_batch[bi].attr("errors").cast<py::array_t<double>>();
        } catch (...) {
            post_errors = py::array_t<double>(n_post_spikes);
        }

        std::vector<double> grad_buf(n_pre * n_post, 0.0);
        eventprop_backward_spikes(
            itimes_arr.data(), isources_arr.data(), n_input_spikes,
            ptimes_arr.data(), psources_arr.data(), pcurrents_arr.data(),
            post_errors.mutable_data(), n_post_spikes,
            w, n_pre, n_post,
            v_th, tau_mem, tau_syn,
            grad_buf.data());

        for (int i = 0; i < n_pre * n_post; i++) {
            int row = i / n_post;
            int col = i % n_post;
            grad_view(row, col) += grad_buf[i];
        }
    }
}

// ── compute_maxima_batch (zero-copy) ────────────────────────────────────────

py::list compute_maxima_batch(
    py::array_t<double, py::array::c_style> w_in,
    int n_pre, int n_post,
    py::list batch, double tau_mem, double tau_syn)
{
    const double* w = w_in.data();

    py::list result;

    for (auto item : batch) {
        auto tup = item.cast<py::tuple>();
        auto times_arr = tup[0].cast<py::array_t<double>>();
        auto sources_arr = tup[1].cast<py::array_t<int32_t>>();
        int n_spikes = (int)times_arr.size();

        std::vector<double> maxima_times(n_post, NAN);
        std::vector<double> maxima_values(n_post, 0.0);

        eventprop_compute_maxima(
            w, n_pre, n_post,
            times_arr.data(), sources_arr.data(), n_spikes,
            tau_mem, tau_syn,
            maxima_times.data(), maxima_values.data());

        // Copy data into numpy arrays (pybind11 3.x std::vector→array has edge cases)
        py::array_t<double> arr_times(n_post);
        py::array_t<double> arr_values(n_post);
        std::copy(maxima_times.begin(), maxima_times.end(),
                  reinterpret_cast<double*>(arr_times.mutable_data()));
        std::copy(maxima_values.begin(), maxima_values.end(),
                  reinterpret_cast<double*>(arr_values.mutable_data()));
        result.append(py::make_tuple(arr_times, arr_values));
    }

    return result;
}

// ── backward_maxima_batch (zero-copy) ───────────────────────────────────────

void backward_maxima_batch(
    py::list input_batch, py::list maxima_list,
    py::array_t<double, py::array::c_style | py::array::forcecast> w_in,
    int n_pre, int n_post,
    py::array_t<double, py::array::c_style | py::array::forcecast> gradient_in,
    double tau_mem, double tau_syn)
{
    const double* w = w_in.data();
    auto grad_view = gradient_in.mutable_unchecked<2>();

    for (Py_ssize_t bi = 0; bi < (Py_ssize_t)maxima_list.size(); bi++) {
        auto input_tup = input_batch[bi].cast<py::tuple>();
        auto itimes_arr = input_tup[0].cast<py::array_t<double>>();
        auto isources_arr = input_tup[1].cast<py::array_t<int32_t>>();
        int n_input_spikes = (int)itimes_arr.size();

        if (n_input_spikes == 0) continue;

        // Get maxima from tuple or Maxima object
        std::vector<double> mtimes, mvalues;
        if (py::isinstance<py::tuple>(maxima_list[bi])) {
            auto tup = maxima_list[bi].cast<py::tuple>();
            mtimes = tup[0].cast<std::vector<double>>();
            mvalues = tup[1].cast<std::vector<double>>();
        } else {
            mtimes = maxima_list[bi].attr("times").cast<std::vector<double>>();
            mvalues = maxima_list[bi].attr("values").cast<std::vector<double>>();
        }

        // Count valid maxima
        int n_valid = 0;
        for (int i = 0; i < n_post; i++) {
            if (!std::isnan(mtimes[i])) n_valid++;
        }
        if (n_valid == 0) continue;

        // Get maxima_errors from Maxima.errors attribute
        py::array_t<double> merrors;
        try {
            merrors = maxima_list[bi].attr("errors").cast<py::array_t<double>>();
        } catch (...) {
            merrors = py::array_t<double>(n_post);
        }

        // Build filtered arrays for valid maxima only
        std::vector<double> mt(n_valid), mv(n_valid), me(n_valid);
        int idx = 0;
        for (int i = 0; i < n_post; i++) {
            if (!std::isnan(mtimes[i])) {
                mt[idx] = mtimes[i];
                mv[idx] = mvalues[i];
                me[idx] = merrors.mutable_data()[i];
                idx++;
            }
        }

        std::vector<double> grad_buf(n_pre * n_post, 0.0);
        eventprop_backward_maxima(
            itimes_arr.data(), isources_arr.data(), n_input_spikes,
            mt.data(), mv.data(), me.data(), n_valid,
            w, n_pre, n_post,
            tau_mem, tau_syn,
            grad_buf.data());

        for (int i = 0; i < n_pre * n_post; i++) {
            int row = i / n_post;
            int col = i % n_post;
            grad_view(row, col) += grad_buf[i];
        }
    }
}

// ── Diagnostic: compute_voltage_trace (pure C++) ────────────────────────────

std::pair<std::vector<double>, std::vector<double>> compute_voltage_trace(
    double t_max, double dt, int target_nrn_idx,
    py::array_t<double, py::array::c_style> w_in, int n_pre, int n_post,
    py::handle spikes, double v_th, double tau_mem, double tau_syn)
{
    // Handle both tuple (times, sources) and numpy array directly
    py::array_t<double> times_arr;
    py::array_t<int32_t> sources_arr;
    if (py::isinstance<py::tuple>(spikes)) {
        auto tup = spikes.cast<py::tuple>();
        times_arr = tup[0].cast<py::array_t<double>>();
        sources_arr = tup[1].cast<py::array_t<int32_t>>();
    } else {
        times_arr = spikes.cast<py::array_t<double>>();
        // No sources for direct array — create dummy
        int n = (int)times_arr.size();
        sources_arr = py::array_t<int32_t>(n);
    }
    int n_spikes = (int)times_arr.size();

    // Pre-compute sums using C API
    int n_spikes_plus_1 = n_spikes + 1;
    std::vector<double> sum0(n_spikes_plus_1 * n_post, 0.0);
    std::vector<double> sum1(n_spikes_plus_1 * n_post, 0.0);
    eventprop_compute_sums(w_in.data(), n_pre, n_post,
                           times_arr.data(), sources_arr.data(), n_spikes,
                           tau_mem, tau_syn,
                           sum0.data(), sum1.data());

    double k_prefactor = tau_syn / (tau_mem - tau_syn);
    int n_t = static_cast<int>(t_max / dt);

    std::vector<double> ts(n_t), v_trace(n_t);
    for (int ti = 0; ti < n_t; ti++) {
        double t = dt * ti;

        int t_pre_idx = 0;
        for (int i = n_spikes - 1; i >= 0; i--) {
            if (times_arr.data()[i] <= t) { t_pre_idx = i + 1; break; }
        }

        double mem = k_prefactor * (
            sum1[t_pre_idx * n_post + target_nrn_idx] * std::exp(-t / tau_mem) -
            sum0[t_pre_idx * n_post + target_nrn_idx] * std::exp(-t / tau_syn)
        );

        v_trace[ti] = mem;
        ts[ti] = t;
    }

    return {ts, v_trace};
}

// ── Diagnostic: compute_lambda_i (pure C++) ─────────────────────────────────

double compute_lambda_i(double t, int target_nrn_idx,
                        py::handle post_spikes, double v_th, double tau_mem, double tau_syn)
{
    py::array_t<double> times_arr;
    if (py::isinstance<py::tuple>(post_spikes)) {
        times_arr = post_spikes.cast<py::tuple>()[0].cast<py::array_t<double>>();
    } else {
        times_arr = post_spikes.cast<py::array_t<double>>();
    }
    int n_spikes = (int)times_arr.size();

    if (n_spikes == 0) return 0.0;

    double largest_time = times_arr.data()[n_spikes - 1];
    struct Jump { double mag; double time_bwd; };
    std::vector<Jump> jumps;

    for (int i = n_spikes - 1; i >= 0; i--) {
        if (times_arr.data()[i] < t) break;
        double t_bwd = largest_time - times_arr.data()[i];
        jumps.push_back({1.0 / (v_th + 0.001), t_bwd});
    }

    double lambda_i = 0.0;
    double t_bwd = largest_time - t;
    auto k_bwd_prefactor = tau_mem / (tau_mem - tau_syn);
    for (const auto& jump : jumps) {
        double kb = k_bwd_prefactor * (std::exp(-t_bwd / tau_mem) - std::exp(-t_bwd / tau_syn));
        lambda_i += jump.mag * kb;
    }

    return lambda_i;
}

// ── Diagnostic: compute_lambda_i_trace ──────────────────────────────────────

std::pair<std::vector<double>, std::vector<double>> compute_lambda_i_trace(
    double t_max, double dt, int target_nrn_idx,
    py::handle post_spikes, double v_th, double tau_mem, double tau_syn)
{
    int n_t = static_cast<int>(t_max / dt);
    std::vector<double> ts(n_t), lambda_trace(n_t);

    for (int ti = 0; ti < n_t; ti++) {
        lambda_trace[ti] = compute_lambda_i(dt * ti, target_nrn_idx, post_spikes, v_th, tau_mem, tau_syn);
        ts[ti] = dt * ti;
    }

    return {ts, lambda_trace};
}

// ── Dropout (pure C++) ─────────────────────────────────────────────────────

py::list dropout_spikes(py::list spikes, double p_drop, unsigned int random_seed) {
    py::list dropped;
    std::mt19937 gen{random_seed};
    std::binomial_distribution dist{1, p_drop};

    for (auto item : spikes) {
        auto tup = item.cast<py::tuple>();
        auto times_arr = tup[0].cast<py::array_t<double>>();
        auto sources_arr = tup[1].cast<py::array_t<int32_t>>();
        int n_spikes = (int)times_arr.size();

        std::vector<double> kept_times;
        std::vector<int32_t> kept_sources;
        for (int i = 0; i < n_spikes; i++) {
            if (dist(gen) == 0) {
                kept_times.push_back(times_arr.data()[i]);
                kept_sources.push_back(sources_arr.data()[i]);
            }
        }

        if (!kept_times.empty()) {
            dropped.append(py::make_tuple(
                py::array_t<double>(kept_times),
                py::array_t<int32_t>(kept_sources)
            ));
        } else {
            dropped.append(py::make_tuple(
                py::array_t<double>(0),
                py::array_t<int32_t>(0)
            ));
        }
    }

    return dropped;
}

// ── pybind11 module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_native, m) {
    m.doc() = "EventProp native batch functions — calls C library";

    m.def("_compute_spikes_batch_raw", &compute_spikes_batch,
          "w"_a.noconvert(), "n_pre"_a, "n_post"_a,
          "batch"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a)
      .def("_backward_spikes_batch_raw", &backward_spikes_batch,
          "input_batch"_a, "post_batch"_a.noconvert(),
          "w"_a.noconvert(), "n_pre"_a, "n_post"_a,
          "gradient"_a.noconvert(), "v_th"_a, "tau_mem"_a, "tau_syn"_a)
      .def("_compute_maxima_batch_raw", &compute_maxima_batch,
          "w"_a.noconvert(), "n_pre"_a, "n_post"_a,
          "batch"_a, "tau_mem"_a, "tau_syn"_a)
      .def("_backward_maxima_batch_raw", &backward_maxima_batch,
          "input_batch"_a, "maxima"_a.noconvert(),
          "w"_a.noconvert(), "n_pre"_a, "n_post"_a,
          "gradient"_a.noconvert(), "tau_mem"_a, "tau_syn"_a)
      .def("_compute_voltage_trace_raw", &compute_voltage_trace,
          "t_max"_a, "dt"_a, "target_nrn_idx"_a,
          "w"_a.noconvert(), "n_pre"_a, "n_post"_a,
          "spikes"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a)
      .def("_compute_lambda_i_raw", &compute_lambda_i,
          "t"_a, "target_nrn_idx"_a, "post_spikes"_a,
          "v_th"_a, "tau_mem"_a, "tau_syn"_a)
      .def("_compute_lambda_i_trace_raw", &compute_lambda_i_trace,
          "t_max"_a, "dt"_a, "target_nrn_idx"_a,
          "post_spikes"_a, "v_th"_a, "tau_mem"_a, "tau_syn"_a)
      .def("_dropout_spikes_raw", &dropout_spikes,
          "spikes"_a.noconvert(), "p_drop"_a, "random_seed"_a);
}
