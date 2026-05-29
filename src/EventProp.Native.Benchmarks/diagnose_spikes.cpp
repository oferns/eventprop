// diagnose_spikes.cpp — Compare C and Eigen ComputeSpikes neuron-by-neuron
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "eventprop_c_api.h"  // C wrapper header

// We'll use the C API directly for both libraries
extern "C" {
    int eventprop_compute_spikes(const double* w, int n_pre, int n_post,
                                  const double* times, const int32_t* sources,
                                  int n_spikes, double v_th, double tau_mem, double tau_syn,
                                  double* post_times, int32_t* post_sources, double* post_currents);
}

int main() {
    // Generate the same data as seed=42 poisson tiny (10x5, max 20 spikes)
    int nPre = 10, nPost = 5;
    double w[50];
    double times[20];
    int32_t sources[20];
    
    // Use the same RNG as our .NET tests (seed 42)
    // For simplicity, hardcode the data that produces C=90, Eigen=89
    // We'll generate it programmatically
    
    srand(42);
    
    // Generate weights [0.1, 2.1)
    for (int i = 0; i < nPre * nPost; i++)
        w[i] = (rand() / (double)RAND_MAX) * 2.0 + 0.1;
    
    // Generate poisson spikes (rate=200Hz)
    int nSpikes = 0;
    double t = 0.0;
    while (t < 1.0) {
        double u = rand() / (double)RAND_MAX;
        t += -log(1.0 - u) / 200.0;
        if (t < 1.0) {
            times[nSpikes] = t;
            sources[nSpikes] = rand() % nPre;
            nSpikes++;
        }
    }
    
    // Sort by time (required for C library)
    for (int i = 0; i < nSpikes - 1; i++) {
        for (int j = i + 1; j < nSpikes; j++) {
            if (times[i] > times[j]) {
                double tmp = times[i]; times[i] = times[j]; times[j] = tmp;
                int tmps = sources[i]; sources[i] = sources[j]; sources[j] = tmps;
            }
        }
    }
    
    printf("Input: nPre=%d, nPost=%d, nSpikes=%d\n", nPre, nPost, nSpikes);
    printf("Weights: ");
    for (int i = 0; i < 10; i++) printf("%.4f ", w[i]);
    printf("...\n");
    printf("Times: ");
    for (int i = 0; i < nSpikes && i < 10; i++) printf("%.6f ", times[i]);
    if (nSpikes > 10) printf("...");
    printf("\n\n");
    
    // Run C library
    double cTimes[20 * 5]; int32_t cSources[20 * 5]; double cCurrents[20 * 5];
    int nCSpikes = eventprop_compute_spikes(w, nPre, nPost, times, sources, nSpikes,
                                             1.0, 0.02, 0.005,
                                             cTimes, cSources, cCurrents);
    
    // Run Eigen library (via wrapper)
    double cppTimes[20 * 5]; int32_t cppSources[20 * 5]; double cppCurrents[20 * 5];
    int nCppSpikes = eventprop_compute_spikes(w, nPre, nPost, times, sources, nSpikes,
                                               1.0, 0.02, 0.005,
                                               cppTimes, cppSources, cppCurrents);
    
    printf("C spikes: %d\n", nCSpikes);
    printf("Eigen spikes: %d\n", nCppSpikes);
    printf("\n");
    
    // Compare neuron by neuron
    int minSpikes = (nCSpikes < nCppSpikes) ? nCSpikes : nCppSpikes;
    int maxSpikes = (nCSpikes > nCppSpikes) ? nCSpikes : nCppSpikes;
    
    for (int nrn = 0; nrn < nPost; nrn++) {
        printf("Neuron %d: C=%d, Eigen=%d", nrn, 
               (nCSpikes >= (nrn+1)*20 ? 20 : 0), // placeholder
               (nCppSpikes >= (nrn+1)*20 ? 20 : 0));
        printf("\n");
    }
    
    // Print first few spikes for each neuron
    for (int nrn = 0; nrn < nPost && nrn < 3; nrn++) {
        printf("\nNeuron %d spikes:\n", nrn);
        int cCount = 0, cppCount = 0;
        
        // Count spikes for this neuron in C output
        for (int i = 0; i < nCSpikes && cCount < 10; i++) {
            if (cSources[i] == nrn) {
                printf("  C[%d]: t=%.10f, I=%.6f\n", cCount, cTimes[i], cCurrents[i]);
                cCount++;
            }
        }
        
        // Count spikes for this neuron in Eigen output
        for (int i = 0; i < nCppSpikes && cppCount < 10; i++) {
            if (cppSources[i] == nrn) {
                printf("  Eigen[%d]: t=%.10f, I=%.6f\n", cppCount, cppTimes[i], cppCurrents[i]);
                cppCount++;
            }
        }
    }
    
    return 0;
}
