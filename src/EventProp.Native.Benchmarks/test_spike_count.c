#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "eventprop.h"

int main(void) {
    int n_pre = 50, n_post = 20, n_spikes = 100;
    int w_size = n_pre * n_post;
    
    // Same weights as benchmark (1.5)
    double* w = malloc(w_size * sizeof(double));
    for (int i = 0; i < w_size; i++) w[i] = 1.5;
    
    // Poisson-like pattern (same as benchmark)
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    
    for (int i = 0; i < n_spikes; i++) {
        times[i] = (i + 1) / 501.0;
        sources[i] = i % n_pre;
    }
    
    int max_out = n_spikes * n_post;
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));
    
    int count = eventprop_compute_spikes(w, n_pre, n_post,
                                          times, sources, n_spikes,
                                          1.0, 0.02, 0.005,
                                          out_times, out_sources, out_currents);
    
    printf("C result: %d spikes\n", count);
    
    // Print first 20 spike times for inspection
    printf("First 20 spikes:\n");
    for (int i = 0; i < count && i < 20; i++) {
        printf("  t=%.6f src=%d I=%.4f\n", out_times[i], out_sources[i], out_currents[i]);
    }
    
    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    return 0;
}
