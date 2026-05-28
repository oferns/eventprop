// shared_test_data.h — Identical test data generator for cross-language benchmarks.
// Uses a simple LCG PRNG so C++, Pure C, and .NET produce bit-identical datasets.
//
// Include this header in C/C++ benchmarks. The .NET equivalent is inline in
// CrossLanguageComparison.cs (same algorithm, same seed).

#ifndef SHARED_TEST_DATA_H
#define SHARED_TEST_DATA_H

#include <stdint.h>

// ── LCG PRNG (glibc parameters) — deterministic across platforms ───────────

static inline uint32_t lcg_next(uint32_t* state) {
    *state = (*state) * 1103515245u + 12345u;
    return (*state >> 16) & 0x7fff;
}

static inline double lcg_uniform(double lo, double hi, uint32_t* state) {
    // Returns value in [lo, hi), ~15-bit precision (sufficient for benchmarks)
    return lo + (hi - lo) * ((double)lcg_next(state)) / 32768.0;
}

static inline int lcg_int(int lo, int hi, uint32_t* state) {
    // Returns value in [lo, hi] inclusive
    return lo + (int)(lcg_next(state)) % (hi - lo + 1);
}

// ── Data generation — uniform pattern ──────────────────────────────────────

static inline void generate_uniform_data(int n_pre, int n_post, int n_spikes,
                                          double* w, double* times, int32_t* sources)
{
    uint32_t state = 12345u; // fixed seed

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = lcg_uniform(0.5, 2.0, &state);

    for (int i = 0; i < n_spikes; i++) {
        times[i] = (i + 1) / (double)(n_spikes + 1);
        sources[i] = (int32_t)lcg_int(0, n_pre - 1, &state);
    }
}

// ── Data generation — poisson pattern ──────────────────────────────────────

static inline void generate_poisson_data(int n_pre, int n_post, int n_spikes,
                                          double* w, double* times, int32_t* sources)
{
    uint32_t state = 12345u; // fixed seed

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = lcg_uniform(0.5, 2.0, &state);

    // Poisson: exponential inter-spike intervals (~200 Hz per neuron)
    int spike_idx = 0;
    for (int nrn = 0; nrn < n_pre && spike_idx < n_spikes; nrn++) {
        double t = 0.0;
        while (t < 1.0 && spike_idx < n_spikes) {
            // Approximate exponential: -ln(u) * mean_interval
            double u = lcg_uniform(0.0001, 1.0, &state);
            t -= 0.005 * (-log(u)); // mean interval = 5ms
            if (t > 1.0) t = 0.999;
            times[spike_idx] = t;
            sources[spike_idx] = (int32_t)nrn;
            spike_idx++;
        }
    }
}

// ── Problem sizes ──────────────────────────────────────────────────────────

#define SHARED_SIZE_TINY   10,   5,   20
#define SHARED_SIZE_SMALL  50,  20,  100
#define SHARED_SIZE_MEDIUM 200, 80,  500
#define SHARED_SIZE_LARGE  500, 200, 1000
#define SHARED_SIZE_XL    2000, 500, 5000

// ── Default parameters (match all benchmark suites) ────────────────────────

#define SHARED_TAU_MEM 0.02
#define SHARED_TAU_SYN 0.005
#define SHARED_V_TH    1.0

#endif // SHARED_TEST_DATA_H
