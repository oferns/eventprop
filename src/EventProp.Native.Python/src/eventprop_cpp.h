#ifndef EVENTPROP_CPP_H
#define EVENTPROP_CPP_H

#include <vector>
#include <cstdint>
#include "eventprop.h"

// ── Python-exposed types ────────────────────────────────────────────────────
// Minimal wrappers around flat arrays so the existing Python code works unchanged.

struct Spikes {
    std::vector<double> times;
    std::vector<int32_t> sources;
    std::vector<double> errors;   // mutable — set_error() writes here
    std::vector<double> currents;
    std::vector<double> first_spike_times;
    std::vector<int32_t> first_spike_idxs;
    int n_spikes;

    Spikes() : errors(), currents(), first_spike_times(), first_spike_idxs(), n_spikes(0) {}
    Spikes(const std::vector<double>& t, const std::vector<int32_t>& s)
        : times(t), sources(s), errors(t.size(), 0.0), currents(t.size(), 0.0),
          first_spike_times(), first_spike_idxs(), n_spikes(static_cast<int>(t.size())) {}
    Spikes(const std::vector<double>& t, const std::vector<int32_t>& s,
           const std::vector<double>& c, const std::vector<double>& fst, const std::vector<int32_t>& fsi)
        : times(t), sources(s), errors(t.size(), 0.0), currents(c),
          first_spike_times(fst), first_spike_idxs(fsi), n_spikes(static_cast<int>(t.size())) {}

    void set_error(int spike_idx, double error) { errors[spike_idx] = error; }
    void set_time(int spike_idx, double time)   { times[spike_idx] = time; }
};

struct Maxima {
    std::vector<double> times;
    std::vector<double> values;
    std::vector<double> errors;

    Maxima() : times(), values(), errors() {}
    Maxima(const std::vector<double>& t, const std::vector<double>& v, const std::vector<double>& e)
        : times(t), values(v), errors(e) {}
    Maxima(const std::vector<double>& t, const std::vector<double>& v)
        : times(t), values(v), errors(t.size(), 0.0) {}

    void set_error(int nrn_idx, double error) { errors[nrn_idx] = error; }
    void set_value(int nrn_idx, double value) { values[nrn_idx] = value; }
};

// ── Thin wrappers around C API (batched) ────────────────────────────────────

std::vector<Spikes> compute_spikes_batch(
    const std::vector<double>& w, int n_pre, int n_post,
    const std::vector<Spikes>& batch, double v_th, double tau_mem, double tau_syn);

void backward_spikes_batch(
    std::vector<Spikes>& input_batch, const std::vector<Spikes>& post_batch,
    const std::vector<double>& w, int n_pre, int n_post,
    std::vector<double>& gradient, double v_th, double tau_mem, double tau_syn);

std::vector<Maxima> compute_maxima_batch(
    const std::vector<double>& w, int n_pre, int n_post,
    const std::vector<Spikes>& batch, double tau_mem, double tau_syn);

void backward_maxima_batch(
    std::vector<Spikes>& input_batch, const std::vector<Maxima>& maxima,
    const std::vector<double>& w, int n_pre, int n_post,
    std::vector<double>& gradient, double tau_mem, double tau_syn);

#endif // EVENTPROP_CPP_H
