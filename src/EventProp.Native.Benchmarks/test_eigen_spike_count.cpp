#include <iostream>
#include <vector>
#include "eventprop_types.h"

int main() {
    int n_pre = 50, n_post = 20, n_spikes_in = 100;
    
    std::vector<double> w_data(n_pre * n_post, 1.5);
    Eigen::MatrixXd w = Eigen::Map<Eigen::MatrixXd>(w_data.data(), n_pre, n_post);
    
    Spikes input_spikes;
    input_spikes.times.resize(n_spikes_in);
    input_spikes.sources.resize(n_spikes_in);
    
    for (int i = 0; i < n_spikes_in; i++) {
        input_spikes.times[i] = (i + 1) / 501.0;
        input_spikes.sources[i] = i % n_pre;
    }
    
    double tau_mem = 0.02, tau_syn = 0.005, v_th = 1.0;
    
    auto result = compute_spikes(w, input_spikes, v_th, tau_mem, tau_syn);
    
    std::cout << "Eigen result: " << result.times.size() << " spikes" << std::endl;
    std::cout << "First 20 spikes:" << std::endl;
    for (size_t i = 0; i < result.times.size() && i < 20; i++) {
        std::cout << "  t=" << result.times[i] << " src=" << result.sources[i] 
                  << " I=" << result.currents[i] << std::endl;
    }
    
    return 0;
}
