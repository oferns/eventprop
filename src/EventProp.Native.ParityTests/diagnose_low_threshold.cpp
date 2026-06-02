/**
 * @file diagnose_low_threshold.cpp
 * @brief Diagnose low threshold (v_th=0.01) bus error - run C and Eigen separately
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <dlfcn.h>

typedef int  (*cfn_compute_spikes)(const double*, int, int, const double*, const int32_t*, int, double, double, double, double*, int32_t*, double*);

int main() {
    printf("Diagnosing low threshold (v_th=0.01) bus error\n");
    printf("================================================\n\n");

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
    
    // Test with high threshold first (should work)
    printf("\n--- High threshold v_th=%.1f ---\n", v_th_high);
    
    void* lib = dlopen("/Users/ollie/repos/eventprop.net/src/EventProp.C/build/libeventprop_c_api.dylib", RTLD_NOW);
    if (!lib) { printf("Failed to load library: %s\n", dlerror()); return 1; }
    
    cfn_compute_spikes fn = (cfn_compute_spikes)dlsym(lib, "eventprop_compute_spikes");
    if (!fn) { printf("Failed to find symbol\n"); return 1; }
    
    int max_out = 10 * (n_spikes + 1) * n_post;
    double* out_times = (double*)malloc(max_out * sizeof(double));
    int32_t* out_sources = (int32_t*)malloc(max_out * sizeof(int32_t));
    double* out_currents = (double*)malloc(max_out * sizeof(double));
    
    int n_post_spikes = fn(w, n_pre, n_post, times, sources, n_spikes, v_th_high, tau_mem, tau_syn, 
                           out_times, out_sources, out_currents);
    printf("High threshold: %d output spikes\n", n_post_spikes);

    // Test with low threshold (should fail)
    printf("\n--- Low threshold v_th=%.2f ---\n", v_th_low);
    
    n_post_spikes = fn(w, n_pre, n_post, times, sources, n_spikes, v_th_low, tau_mem, tau_syn, 
                       out_times, out_sources, out_currents);
    printf("Low threshold: %d output spikes\n", n_post_spikes);
    
    // Check if buffer overflowed
    printf("Max output capacity: %d\n", max_out);
    
    // Try with larger buffer
    printf("\n--- Low threshold v_th=%.2f, 10x buffer ---\n", v_th_low);
    int max_out_large = 100 * (n_spikes + 1) * n_post;
    double* out_times_large = (double*)malloc(max_out_large * sizeof(double));
    int32_t* out_sources_large = (int32_t*)malloc(max_out_large * sizeof(int32_t));
    double* out_currents_large = (double*)malloc(max_out_large * sizeof(double));
    
    n_post_spikes = fn(w, n_pre, n_post, times, sources, n_spikes, v_th_low, tau_mem, tau_syn, 
                       out_times_large, out_sources_large, out_currents_large);
    printf("Low threshold (large buffer): %d output spikes\n", n_post_spikes);
    
    // Test with medium threshold
    printf("\n--- Medium threshold v_th=0.1 ---\n");
    n_post_spikes = fn(w, n_pre, n_post, times, sources, n_spikes, 0.1, tau_mem, tau_syn, 
                       out_times_large, out_sources_large, out_currents_large);
    printf("Medium threshold: %d output spikes\n", n_post_spikes);
    
    // Test with very low threshold
    printf("\n--- Very low threshold v_th=0.001 ---\n");
    n_post_spikes = fn(w, n_pre, n_post, times, sources, n_spikes, 0.001, tau_mem, tau_syn, 
                       out_times_large, out_sources_large, out_currents_large);
    printf("Very low threshold: %d output spikes\n", n_post_spikes);

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    free(out_times_large); free(out_sources_large); free(out_currents_large);
    dlclose(lib);
    
    return 0;
}
