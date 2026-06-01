// C vs C++ EventProp benchmark — side-by-side comparison
// Two separate executables: bench_c (links C library) and bench_cpp (links C++ wrapper)
// Same code, different backend. Run both and compare.

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

// Include the C API header — both libraries export the same extern "C" symbols
#ifdef USE_CPP_WRAPPER
#include "eventprop_c_api.h"
#else
#include "eventprop.h"
#endif

// ============================================================
// Data generation (same for both)
// ============================================================

static void GenerateUniformData(int n_pre, int n_post, int n_spikes,
                                 double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    for (int i = 0; i < n_spikes; i++) {
        times[i] = (i + 1) / static_cast<double>(n_spikes + 1);
        sources[i] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);
    }
}

static void GeneratePoissonData(int n_pre, int n_post, int n_spikes,
                                 double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    /* Poisson: exponential inter-spike intervals, ~200 Hz per neuron.
       Matches .NET BenchPatterns.cs generation. */
    double rate = 200.0; // Hz
    double dt = 1.0 / (rate * n_spikes);
    std::uniform_real_distribution<double> u_dist(0.0, 1.0);

    for (int idx = 0; idx < n_spikes; idx++) {
        double u = u_dist(rng);
        times[idx] = -std::log(1.0 - u) / rate;
        sources[idx] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);

        /* Ensure strictly increasing times */
        if (idx > 0 && times[idx] <= times[idx - 1])
            times[idx] = times[idx - 1] + dt;
    }
}

static void GenerateBurstData(int n_pre, int n_post, int n_spikes,
                               double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    int spike_idx = 0;
    for (int nrn = 0; nrn < n_pre && spike_idx < n_spikes; nrn++) {
        double burst_start = 0.05 + nrn * 0.1;
        for (int b = 0; burst_start < 1.0 && spike_idx < n_spikes; b++) {
            for (int s = 0; s < 5 && spike_idx < n_spikes; s++) {
                times[spike_idx] = burst_start + s * 0.002;
                sources[spike_idx] = nrn;
                spike_idx++;
            }
            burst_start += 0.12;
        }
    }
}

// ============================================================
// Problem sizes (match .NET and Python benchmarks)
// ============================================================

struct ProblemSize {
    const char* name;
    int n_pre, n_post, n_spikes;
};

static const ProblemSize kSizes[] = {
    {"Tiny",   10,   5,   20},
    {"Small",  50,   20,  100},
    {"Medium", 200,  80,  500},
    {"Large",  500,  200, 1000},
};

static const char* kPatterns[] = {"uniform", "poisson", "burst"};
using DataGenerator = void(*)(int, int, int, double*, double*, int32_t*);
static DataGenerator kGen[] = {GenerateUniformData, GeneratePoissonData, GenerateBurstData};

// ============================================================
// Benchmark: ComputeSums
// ============================================================

static void BM_ComputeSums(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    int sum_rows = sz.n_spikes + 1;
    std::vector<double> sum0(sum_rows * sz.n_post);
    std::vector<double> sum1(sum_rows * sz.n_post);

    for (auto _ : state) {
        std::memset(sum0.data(), 0, sum0.size() * sizeof(double));
        std::memset(sum1.data(), 0, sum1.size() * sizeof(double));

#ifdef USE_CPP_WRAPPER
        eventprop_compute_sums(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes, 0.02, 0.005, sum0.data(), sum1.data());
#else
        eventprop_compute_sums(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes, 0.02, 0.005, sum0.data(), sum1.data());
#endif
        benchmark::DoNotOptimize(sum0.data());
        benchmark::DoNotOptimize(sum1.data());
    }

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define COMPUTESUMS_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_ComputeSums, SizeName##_uniform,                                     \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                  \
        "ComputeSums/" #SizeName "/uniform/default");                                          \
    BENCHMARK_CAPTURE(BM_ComputeSums, SizeName##_poisson,                                     \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                  \
        "ComputeSums/" #SizeName "/poisson/default");                                          \
    BENCHMARK_CAPTURE(BM_ComputeSums, SizeName##_burst,                                       \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                    \
        "ComputeSums/" #SizeName "/burst/default")

COMPUTESUMS_BENCH(Tiny,    10,   5,   20);
COMPUTESUMS_BENCH(Small,   50,   20,  100);
COMPUTESUMS_BENCH(Medium,  200,  80,  500);
COMPUTESUMS_BENCH(Large,   500,  200, 1000);

// ============================================================
// Benchmark: ComputeSpikes (forward pass)
// ============================================================

static void BM_ComputeSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    int max_out = sz.n_spikes * sz.n_post;
    std::vector<double> out_times(max_out);
    std::vector<int32_t> out_sources(max_out);
    std::vector<double> out_currents(max_out);

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        eventprop_compute_spikes(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            1.0, 0.02, 0.005, out_times.data(), out_sources.data(), out_currents.data());
#else
        eventprop_compute_spikes(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            1.0, 0.02, 0.005, out_times.data(), out_sources.data(), out_currents.data());
#endif
        benchmark::DoNotOptimize(out_times.data());
        benchmark::DoNotOptimize(out_sources.data());
        benchmark::DoNotOptimize(out_currents.data());
    }

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define COMPUTESPIKES_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_ComputeSpikes, SizeName##_uniform,                                     \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                    \
        "ComputeSpikes/" #SizeName "/uniform/default");                                          \
    BENCHMARK_CAPTURE(BM_ComputeSpikes, SizeName##_poisson,                                     \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                    \
        "ComputeSpikes/" #SizeName "/poisson/default");                                          \
    BENCHMARK_CAPTURE(BM_ComputeSpikes, SizeName##_burst,                                       \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                      \
        "ComputeSpikes/" #SizeName "/burst/default")

COMPUTESPIKES_BENCH(Tiny,    10,   5,   20);
COMPUTESPIKES_BENCH(Small,   50,   20,  100);
COMPUTESPIKES_BENCH(Medium,  200,  80,  500);
COMPUTESPIKES_BENCH(Large,   500,  200, 1000);

// ============================================================
// Benchmark: BackwardSpikes (backward pass)
// Requires post-spike data from a forward pass.
// Forward pass is run once outside the timing loop; backward is benchmarked.
// ============================================================

static void BM_BackwardSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    // Run forward pass ONCE to get post-spike data (not benchmarked)
    int max_out = sz.n_spikes * sz.n_post;
    std::vector<double> fwd_times(max_out);
    std::vector<int32_t> fwd_sources(max_out);
    std::vector<double> fwd_currents(max_out);

#ifdef USE_CPP_WRAPPER
    int n_post_spikes = eventprop_compute_spikes(w.data(), sz.n_pre, sz.n_post,
        times.data(), sources.data(), sz.n_spikes,
        1.0, 0.02, 0.005, fwd_times.data(), fwd_sources.data(), fwd_currents.data());
#else
    int n_post_spikes = eventprop_compute_spikes(w.data(), sz.n_pre, sz.n_post,
        times.data(), sources.data(), sz.n_spikes,
        1.0, 0.02, 0.005, fwd_times.data(), fwd_sources.data(), fwd_currents.data());
#endif

    // Resize to actual spike count (trim unused capacity)
    fwd_times.resize(n_post_spikes);
    fwd_sources.resize(n_post_spikes);
    fwd_currents.resize(n_post_spikes);

    // Post-spike errors (all 1.0, matching .NET benchmark convention)
    std::vector<double> post_errors(n_post_spikes, 1.0);

    // Gradient output buffer
    std::vector<double> grad(sz.n_pre * sz.n_post);
    // Spike errors output buffer
    std::vector<double> spike_errors(sz.n_spikes);

    for (auto _ : state) {
        std::memset(grad.data(), 0, grad.size() * sizeof(double));

#ifdef USE_CPP_WRAPPER
        eventprop_backward_spikes(times.data(), sources.data(), sz.n_spikes,
            fwd_times.data(), fwd_sources.data(), fwd_currents.data(),
            post_errors.data(), n_post_spikes,
            w.data(), sz.n_pre, sz.n_post,
            1.0, 0.02, 0.005, grad.data(), spike_errors.data());
#else
        eventprop_backward_spikes(times.data(), sources.data(), sz.n_spikes,
            fwd_times.data(), fwd_sources.data(), fwd_currents.data(),
            post_errors.data(), n_post_spikes,
            w.data(), sz.n_pre, sz.n_post,
            1.0, 0.02, 0.005, grad.data(), spike_errors.data());
#endif
        benchmark::DoNotOptimize(grad.data());
    }

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define BACKWARDSPIKES_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_BackwardSpikes, SizeName##_uniform,                                     \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                    \
        "BackwardSpikes/" #SizeName "/uniform/default");                                          \
    BENCHMARK_CAPTURE(BM_BackwardSpikes, SizeName##_poisson,                                     \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                    \
        "BackwardSpikes/" #SizeName "/poisson/default");                                          \
    BENCHMARK_CAPTURE(BM_BackwardSpikes, SizeName##_burst,                                       \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                      \
        "BackwardSpikes/" #SizeName "/burst/default")

BACKWARDSPIKES_BENCH(Tiny,    10,   5,   20);
BACKWARDSPIKES_BENCH(Small,   50,   20,  100);
BACKWARDSPIKES_BENCH(Medium,  200,  80,  500);
BACKWARDSPIKES_BENCH(Large,   500,  200, 1000);

BENCHMARK_MAIN();
