#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "eventprop_types.h"

// Generate identical data for both libraries
static void GeneratePoissonData(int n_pre, int n_post, int n_spikes,
                                 double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);

    double rate = 200.0;
    double dt = 1.0 / (rate * n_spikes);
    std::uniform_real_distribution<double> u_dist(0.0, 1.0);

    for (int idx = 0; idx < n_spikes; idx++) {
        double u = u_dist(rng);
        times[idx] = -std::log(1.0 - u) / rate;
        sources[idx] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);
        if (idx > 0 && times[idx] <= times[idx - 1])
            times[idx] = times[idx - 1] + dt;
    }
}

int main() {
    int n_pre = 50, n_post = 20, n_spikes_in = 100;
    
    std::vector<double> w_data(n_pre * n_post);
    std::vector<double> times(n_spikes_in);
    std::vector<int32_t> sources(n_spikes_in);
    
    GeneratePoissonData(n_pre, n_post, n_spikes_in, w_data.data(), times.data(), sources.data());
    
    Eigen::MatrixXd weights = Eigen::Map<Eigen::MatrixXd>(w_data.data(), n_pre, n_post);
    
    Spikes input_spikes;
    input_spikes.times = Eigen::Map<Eigen::ArrayXd>(times.data(), n_spikes_in);
    input_spikes.sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), n_spikes_in);
    
    double tau_mem = 0.02, tau_syn = 0.005, v_th = 1.0;
    
    auto result = compute_spikes(weights, input_spikes, v_th, tau_mem, tau_syn);
    
    std::cout << "Eigen result: " << result.times.size() << " spikes" << std::endl;
    std::cout << "\nFirst 20 spikes:" << std::endl;
    for (size_t i = 0; i < result.times.size() && i < 20; i++) {
        std::cout << "  [" << i << "] t=" << result.times[i] 
                  << " src=" << result.sources[i] 
                  << " I=" << result.currents[i] << std::endl;
    }
    
    // Check for NaN/Inf
    int nan_count = 0, inf_count = 0;
    for (size_t i = 0; i < result.times.size(); i++) {
        if (std::isnan(result.times[i])) nan_count++;
        if (std::isinf(result.times[i])) inf_count++;
    }
    std::cout << "\nNaN count: " << nan_count << ", Inf count: " << inf_count << std::endl;
    
    // Print weight matrix stats
    double w_min = w_data[0], w_max = w_data[0];
    for (size_t i = 1; i < w_data.size(); i++) {
        if (w_data[i] < w_min) w_min = w_data[i];
        if (w_data[i] > w_max) w_max = w_data[i];
    }
    std::cout << "Weights: min=" << w_min << ", max=" << w_max << std::endl;
    
    // Print time range
    double t_min = times[0], t_max = times[n_spikes_in-1];
    for (int i = 1; i < n_spikes_in; i++) {
        if (times[i] < t_min) t_min = times[i];
        if (times[i] > t_max) t_max = times[i];
    }
    std::cout << "Input spike times: [" << t_min << ", " << t_max << "]" << std::endl;
    
    return 0;
}
