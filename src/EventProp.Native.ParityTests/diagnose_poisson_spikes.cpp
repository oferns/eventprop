/**
 * @file diagnose_poisson_spikes.cpp
 * @brief Debug: why C and Eigen produce different spike counts for poisson data.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <dlfcn.h>

#include "eventprop_eigen.h"

typedef int  (*cfn_compute_spikes)(const double*, int, int, const double*, const int32_t*, int, double, double, double, double*, int32_t*, double*);

int main(int argc, char* argv[])
{
    const char* c_lib_path = (argc > 1) ? argv[1] : "./libeventprop_c_api.dylib";
    void* c_lib = dlopen(c_lib_path, RTLD_NOW);
    if (!c_lib) { fprintf(stderr, "Cannot load: %s\n", dlerror()); return 1; }
    cfn_compute_spikes fn = (cfn_compute_spikes)dlsym(c_lib, "eventprop_compute_spikes");

    /* Generate poisson data (same as parity_main.cpp) */
    int n_pre = 50, n_post = 20, n_spikes_in = 100;
    double rate = 200.0;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    std::vector<double> w(n_pre * n_post);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);

    std::vector<double> times(n_spikes_in);
    std::vector<int32_t> sources(n_spikes_in);

    for (int idx = 0; idx < n_spikes_in; idx++) {
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);
        double u = u_dist(rng);
        times[idx] = -std::log(1.0 - u) / rate;
        sources[idx] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);
        if (idx > 0 && times[idx] <= times[idx - 1])
            times[idx] = times[idx - 1] + 1.0 / (rate * n_spikes_in);
    }

    /* Sort spikes */
    for (int i = 1; i < n_spikes_in; i++) {
        if (times[i] < times[i-1]) {
            double t = times[i]; int32_t s = sources[i];
            int j = i;
            while (j > 0 && times[j-1] > t) {
                times[j] = times[j-1]; sources[j] = sources[j-1]; j--;
            }
            times[j] = t; sources[j] = s;
        }
    }

    printf("Input: %d spikes, n_pre=%d, n_post=%d\n", n_spikes_in, n_pre, n_post);
    printf("First 10 input spikes:\n");
    for (int i = 0; i < 10 && i < n_spikes_in; i++)
        printf("  [%d] t=%.6f src=%d\n", i, times[i], sources[i]);
    printf("Last 5 input spikes:\n");
    for (int i = n_spikes_in - 5; i < n_spikes_in; i++)
        printf("  [%d] t=%.6f src=%d\n", i, times[i], sources[i]);

    /* C library */
    int max_out = n_spikes_in * n_post;
    std::vector<double> out_times_c(max_out);
    std::vector<int32_t> out_sources_c(max_out);
    std::vector<double> out_currents_c(max_out);
    int n_post_spikes_c = fn(w.data(), n_pre, n_post, times.data(), sources.data(), n_spikes_in,
                             1.0, 0.02, 0.005, out_times_c.data(), out_sources_c.data(), out_currents_c.data());
    printf("\nC: %d post-spike spikes\n", n_post_spikes_c);

    /* Eigen */
    RowMatrixXd w_mat(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++) w_mat(i) = w[i];
    Spikes spikes_eig(Eigen::Map<Eigen::ArrayXd>(times.data(), n_spikes_in),
                      Eigen::Map<Eigen::ArrayXi>(sources.data(), n_spikes_in));
    auto result_eig = compute_spikes(w_mat, spikes_eig, 1.0, 0.02, 0.005);
    int n_post_spikes_eig = (int)result_eig.times.size();
    printf("Eigen: %d post-spike spikes\n", n_post_spikes_eig);

    /* Per-neuron spike count comparison */
    printf("\nPer-neuron spike counts:\n");
    for (int nrn = 0; nrn < n_post; nrn++) {
        int c_count = 0, e_count = 0;
        for (int i = 0; i < n_post_spikes_c; i++)
            if (out_sources_c[i] == nrn) c_count++;
        for (int i = 0; i < n_post_spikes_eig; i++)
            if (result_eig.sources(i) == nrn) e_count++;
        if (c_count != e_count) {
            printf("  Neuron %d: C=%d, Eigen=%d (MISMATCH)\n", nrn, c_count, e_count);
        }
    }

    /* Print first 20 spikes from each */
    printf("\nC first 15 spikes:\n");
    for (int i = 0; i < n_post_spikes_c && i < 15; i++)
        printf("  [%d] t=%.6f src=%d I=%.4f\n", i, out_times_c[i], out_sources_c[i], out_currents_c[i]);

    printf("\nEigen first 15 spikes:\n");
    for (int i = 0; i < n_post_spikes_eig && i < 15; i++)
        printf("  [%d] t=%.6f src=%d I=%.4f\n", i, result_eig.times(i), (int)result_eig.sources(i), result_eig.currents(i));

    dlclose(c_lib);
    return 0;
}
