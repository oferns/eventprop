/**
 * @file diagnose_eigen_low_threshold.cpp
 * @brief Diagnose low threshold (v_th=0.01) bus error in Eigen implementation
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

/* Eigen core (from EventProp.Python, stripped of pybind11) */
#include "../EventProp.Cpp.Wrapper/eventprop_eigen_stripped.h"

int main() {
    printf("Diagnosing low threshold (v_th=0.01) bus error in Eigen\n");
    printf("========================================================\n\n");

    // Test parameters
    int n_pre = 50, n_post = 20, n_spikes = 100;
    double v_th_low = 0.01;     // Low threshold (causes bus error)
    double v_th_high = 100.0;   // High threshold (works)
    double tau_mem = 0.02, tau_syn = 0.005;

    // Generate test data
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)calloc(n_spikes, sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0), time_dist(0.01, 0.99);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    for (int i = 0; i < n_spikes; i++) {
        times[i] = time_dist(rng);
        sources[i] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);
    }

    printf("Test: %d×%d, %d spikes\n", n_pre, n_post, n_spikes);
    
    // Convert to Eigen format
    RowMatrixXd w_mat(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++) w_mat(i) = w[i];
    
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd const>(times, n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi const>(sources, n_spikes);
    Spikes input_eig(std::move(e_times), std::move(e_sources));

    // Test with high threshold first (should work)
    printf("\n--- High threshold v_th=%.1f ---\n", v_th_high);
    
    auto result_high = compute_spikes(w_mat, input_eig, v_th_high, tau_mem, tau_syn);
    printf("High threshold: %zu output spikes\n", result_high.times.size());

    // Test with low threshold (should fail)
    printf("\n--- Low threshold v_th=%.2f ---\n", v_th_low);
    
    auto result_low = compute_spikes(w_mat, input_eig, v_th_low, tau_mem, tau_syn);
    printf("Low threshold: %zu output spikes\n", result_low.times.size());

    // Test with medium threshold
    printf("\n--- Medium threshold v_th=0.1 ---\n");
    
    auto result_medium = compute_spikes(w_mat, input_eig, 0.1, tau_mem, tau_syn);
    printf("Medium threshold: %zu output spikes\n", result_medium.times.size());

    // Test with very low threshold
    printf("\n--- Very low threshold v_th=0.001 ---\n");
    
    auto result_very_low = compute_spikes(w_mat, input_eig, 0.001, tau_mem, tau_syn);
    printf("Very low threshold: %zu output spikes\n", result_very_low.times.size());

    free(w); free(times); free(sources);
    
    return 0;
}
