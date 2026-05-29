/**
 * @file eventprop_eigen.h
 * @brief Stripped Eigen core header for EventProp parity testing.
 *        Extracted from EventProp.Python/src/eventprop_cpp.h with pybind11 removed.
 */

#ifndef EVENTPROP_EIGEN_H
#define EVENTPROP_EIGEN_H

#include <Eigen/Dense>
#include <limits>
#include <vector>

inline double inf = std::numeric_limits<double>::infinity();
inline double eps = std::numeric_limits<double>::epsilon();

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Spikes {
  Eigen::ArrayXd times;
  Eigen::ArrayXi sources;
  Eigen::ArrayXd errors;
  Eigen::ArrayXd currents;
  std::vector<double> first_spike_times;
  std::vector<int> first_spike_idxs;
  int n_spikes;

  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), first_spike_times(), first_spike_idxs(), n_spikes(times.size())  {}
  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources, std::vector<double> first_spike_times, std::vector<int> first_spike_idxs)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), first_spike_times(first_spike_times), first_spike_idxs(first_spike_idxs), n_spikes(times.size()){}
  Spikes(Eigen::ArrayXd times, Eigen::ArrayXi sources, Eigen::ArrayXd currents, std::vector<double> first_spike_times, std::vector<int> first_spike_idxs)
      : times(times), sources(sources),
        errors(Eigen::ArrayXd::Zero(times.size())), currents(currents), first_spike_times(first_spike_times), first_spike_idxs(first_spike_idxs), n_spikes(times.size()) {}
  Spikes() : errors(), currents(), first_spike_times(), first_spike_idxs(), n_spikes(0) {}
  void set_error(int spike_idx, double error) { errors(spike_idx) = error; }
  void set_time(int spike_idx, double time) { times(spike_idx) = time; }
};

struct Maxima {
  Eigen::ArrayXd times;
  Eigen::ArrayXd values;
  Eigen::ArrayXd errors;

  Maxima(Eigen::ArrayXd times, Eigen::ArrayXd values, Eigen::ArrayXd errors) : times(times), values(values), errors(errors) {}
  Maxima(Eigen::ArrayXd times, Eigen::ArrayXd values) : times(times), values(values), errors(Eigen::ArrayXd::Zero(times.size())) {}
  Maxima() : times(), values(), errors() {}
  void set_error(int nrn_idx, double error) {errors(nrn_idx) = error; }
  void set_value(int nrn_idx, double value) {values(nrn_idx) = value; }
};

/* ── Function declarations (from eventprop_cpp.h, pybind11 bindings removed) ── */

std::pair<RowMatrixXd, RowMatrixXd> compute_sums(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn);
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_voltage_trace(double t_max, double dt, int target_nrn_idx, Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double v_th, double tau_mem, double tau_syn);
double compute_lambda_i(double t, int target_nrn_idx, Spikes const& post_spikes, double v_th, double tau_mem, double tau_syn);
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_lambda_i_trace(double t_max, double dt, int target_nrn_idx, Spikes const& post_spikes, double v_th, double tau_mem, double tau_syn);
Spikes compute_spikes(Eigen::Ref<RowMatrixXd const> w, Spikes const &spikes,
               double v_th, double tau_mem, double tau_syn);
std::vector<Spikes> compute_spikes_batch(Eigen::Ref<RowMatrixXd const> w,
                     std::vector<Spikes> const &batch, double v_th,
                     double tau_mem, double tau_syn);

/* backward for spikes — note: first arg is non-const (mutates input_spikes.errors) */
std::pair<RowMatrixXd, Eigen::ArrayXd> backward(Spikes &input_spikes, Spikes const &post_spikes,
              Eigen::Ref<RowMatrixXd const> w,
              double v_th, double tau_mem, double tau_syn);

void backward_spikes_batch(std::vector<Spikes> &input_batch,
                    std::vector<Spikes> const &post_batch,
                    Eigen::Ref<RowMatrixXd const> w,
                    Eigen::Ref<RowMatrixXd> gradient, double v_th,
                    double tau_mem, double tau_syn);

Maxima compute_maxima(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn);

/* backward for maxima — first arg is non-const */
std::pair<RowMatrixXd, Eigen::ArrayXd> backward(Spikes &input_spikes, Maxima const& maxima, Eigen::Ref<RowMatrixXd const> w, double tau_mem, double tau_syn);

std::vector<Maxima> compute_maxima_batch(Eigen::Ref<RowMatrixXd const> w, std::vector<Spikes> const& batch, double tau_mem, double tau_syn);

void backward_maxima_batch(std::vector<Spikes> &input_batch, std::vector<Maxima> const& maxima, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double tau_mem, double tau_syn);

std::vector<Spikes> dropout_spikes(std::vector<Spikes> const& spikes, double p_drop, unsigned int random_seed);

#endif /* EVENTPROP_EIGEN_H */
