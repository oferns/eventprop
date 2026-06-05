// C vs C++ EventProp benchmark — side-by-side comparison
// Two separate executables: bench_c (links C library) and bench_cpp (calls Eigen directly)
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

#ifdef USE_CPP_WRAPPER
// C++ Eigen functions (bare header, no pybind11/Python dependency)
#include "eventprop_cpp_bare.h"
#else
// C library alloc functions
#include "eventprop_alloc.h"
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

    double rate = 200.0; // Hz
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
// Benchmark: ComputeSpikes (forward pass)
// ============================================================

static void BM_ComputeSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        // C++ Eigen: compute_spikes(w, spikes, v_th, tau_mem, tau_syn)
        // Create Spikes object from data
        Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes);
        Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes);
        Spikes spikes(e_times, e_sources);
        
        Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
        Spikes output = compute_spikes(w_map, spikes, 1.0, 0.02, 0.005);
        
        benchmark::DoNotOptimize(output.n_spikes);
#else
        // C library: eventprop_compute_spikes_alloc(w, n_pre, n_post, times, sources, v_th, tau_mem, tau_syn)
        SpikeOutput output = eventprop_compute_spikes_alloc(
            w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            1.0, 0.02, 0.005);
        
        benchmark::DoNotOptimize(output.n_spikes);
        eventprop_free_spike_output(&output);
#endif
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
// ============================================================

static void BM_BackwardSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

#ifdef USE_CPP_WRAPPER
    // C++ Eigen: Run forward pass to get post-spike data
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes);
    Spikes input_spikes(e_times, e_sources);
    
    Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
    Spikes post_spikes = compute_spikes(w_map, input_spikes, 1.0, 0.02, 0.005);
    
    // Set post-spike errors (all 1.0)
    for (int i = 0; i < post_spikes.n_spikes; i++)
        post_spikes.set_error(i, 1.0);

    // Gradient output buffer (row-major)
    RowMatrixXd gradient(sz.n_pre, sz.n_post);

    for (auto _ : state) {
        gradient.setZero();
        
        // C++ backward: returns (gradient, spike_errors)
        std::pair<RowMatrixXd, Eigen::ArrayXd> result = backward(input_spikes, post_spikes, w_map, 1.0, 0.02, 0.005);
        
        benchmark::DoNotOptimize(result.first(0, 0));
    }
#else
    // C library: Run forward pass to get post-spike data
    SpikeOutput fwd_result = eventprop_compute_spikes_alloc(
        w.data(), sz.n_pre, sz.n_post,
        times.data(), sources.data(), sz.n_spikes,
        1.0, 0.02, 0.005);

    int n_post_spikes = fwd_result.n_spikes;
    
    // Post-spike errors (all 1.0)
    std::vector<double> post_errors(n_post_spikes, 1.0);

    for (auto _ : state) {
        BackwardSpikesOutput result = eventprop_backward_spikes_alloc(
            times.data(), sources.data(), sz.n_spikes,
            fwd_result.times, fwd_result.sources, fwd_result.currents,
            post_errors.data(), n_post_spikes,
            w.data(), sz.n_pre, sz.n_post,
            1.0, 0.02, 0.005);

        benchmark::DoNotOptimize(result.grad[0]);
        eventprop_free_backward_spikes_output(&result);
    }

    eventprop_free_spike_output(&fwd_result);
#endif

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