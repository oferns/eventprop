// Minimal C++ Eigen header for native benchmarks (no pybind11/Python dependency)
// Stripped from eventprop_cpp.h to allow native benchmarking

#ifndef EVENTPROP_CPP_BARE_H
#define EVENTPROP_CPP_BARE_H

#include <Eigen/Dense>
#include <vector>

// Eigen types used by the C++ implementation
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Spikes structure (matches eventprop_cpp.h)
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
        errors(Eigen::ArrayXd::Zero(times.size())), first_spike_times(), first_spike_idxs(), n_spikes(times.size()) {}
  Spikes() {}
  
  void set_error(int spike_idx, double error) { errors(spike_idx) = error; }
};

// Maxima structure (matches eventprop_cpp.h)
struct Maxima {
  Eigen::ArrayXd times;
  Eigen::ArrayXd values;
  Eigen::ArrayXd errors;

  Maxima(Eigen::ArrayXd times, Eigen::ArrayXd values) 
      : times(times), values(values), errors(Eigen::ArrayXd::Zero(times.size())) {}
  Maxima() {}
  
  void set_error(int nrn_idx, double error) { errors(nrn_idx) = error; }
};

// Function declarations (from eventprop_cpp.h, no pybind11)
std::pair<RowMatrixXd, RowMatrixXd> compute_sums(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn);
Spikes compute_spikes(Eigen::Ref<RowMatrixXd const> w, Spikes const &spikes, double v_th, double tau_mem, double tau_syn);
std::vector<Spikes> compute_spikes_batch(Eigen::Ref<RowMatrixXd const> w, std::vector<Spikes> const &batch, double v_th, double tau_mem, double tau_syn);
std::pair<RowMatrixXd, Eigen::ArrayXd> backward(Spikes &input_spikes, Spikes const &post_spikes, Eigen::Ref<RowMatrixXd const> w, double v_th, double tau_mem, double tau_syn);
void backward_spikes_batch(std::vector<Spikes> &input_batch, std::vector<Spikes> const &post_batch, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double v_th, double tau_mem, double tau_syn);
Maxima compute_maxima(Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double tau_mem, double tau_syn);
std::vector<Maxima> compute_maxima_batch(Eigen::Ref<RowMatrixXd const> w, std::vector<Spikes> const& batch, double tau_mem, double tau_syn);
std::pair<RowMatrixXd, Eigen::ArrayXd> backward(Spikes &input_spikes, Maxima const& maxima, Eigen::Ref<RowMatrixXd const> w, double tau_mem, double tau_syn);
void backward_maxima_batch(std::vector<Spikes> &input_batch, std::vector<Maxima> const& maxima, Eigen::Ref<RowMatrixXd const> w, Eigen::Ref<RowMatrixXd> gradient, double tau_mem, double tau_syn);
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_voltage_trace(double t_max, double dt, int target_nrn_idx, Eigen::Ref<RowMatrixXd const> w, Spikes const& spikes, double v_th, double tau_mem, double tau_syn);
double compute_lambda_i(double t, int target_nrn_idx, Spikes const& post_spikes, double v_th, double tau_mem, double tau_syn);
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_lambda_i_trace(double t_max, double dt, int target_nrn_idx, Spikes const& post_spikes, double v_th, double tau_mem, double tau_syn);

#endif // EVENTPROP_CPP_BARE_H