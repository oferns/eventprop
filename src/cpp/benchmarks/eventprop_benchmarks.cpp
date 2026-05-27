// eventprop_benchmarks.cpp — Comprehensive C++ benchmarks for EventProp algorithm
// Mirrors the .NET BenchmarkDotNet suite with expanded coverage.

#include "eventprop_c_api.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <omp.h>
#else
#include <omp.h>
#endif

// ============================================================
// Parameters (match .NET benchmarks exactly)
// ============================================================
static constexpr double kTauMemDefault = 0.02;   // 20ms
static constexpr double kTauSynDefault = 0.005;  // 5ms
static constexpr double kVThDefault    = 1.0;

// ============================================================
// Problem sizes (5 tiers)
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
    {"XL",     2000, 500, 5000},
};

// ============================================================
// Helpers — data generation
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

    // Poisson: exponential inter-spike intervals, ~200 Hz per neuron
    std::exponential_distribution<double> inter_spike(1.0 / 200.0); // mean 5ms
    int spike_idx = 0;
    for (int nrn = 0; nrn < n_pre && spike_idx < n_spikes; nrn++) {
        double t = 0.0;
        while (t < 1.0 && spike_idx < n_spikes) {
            t += inter_spike(rng);
            times[spike_idx] = std::min(t, 0.999);
            sources[spike_idx] = nrn;
            spike_idx++;
        }
    }
}

static void GenerateBurstData(int n_pre, int n_post, int n_spikes,
                               double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    // Burst: clusters of 5 spikes at t, t+2ms, t+4ms, t+6ms, t+8ms
    // Gap between bursts: 10ms. Distribute across neurons.
    int spike_idx = 0;
    for (int nrn = 0; nrn < n_pre && spike_idx < n_spikes; nrn++) {
        double burst_start = 0.05 + nrn * 0.1; // stagger bursts
        for (int b = 0; burst_start < 1.0 && spike_idx < n_spikes; b++) {
            for (int s = 0; s < 5 && spike_idx < n_spikes; s++) {
                times[spike_idx] = burst_start + s * 0.002;
                sources[spike_idx] = nrn;
                spike_idx++;
            }
            burst_start += 0.12; // 10ms gap between bursts
        }
    }
}

using DataGenerator = void(*)(int, int, int, double*, double*, int32_t*);

static const struct {
    const char* name;
    DataGenerator gen;
} kPatterns[] = {
    {"uniform", GenerateUniformData},
    {"poisson", GeneratePoissonData},
    {"burst",   GenerateBurstData},
};

// ============================================================
// Forward pass helper (for backward + training loop)
// ============================================================
static int RunForward(int n_pre, int n_post, int n_spikes,
                      const double* w, const double* times, const int32_t* sources,
                      std::vector<double>& out_times,
                      std::vector<int32_t>& out_sources,
                      std::vector<double>& out_currents)
{
    int max_out = n_spikes * n_post;
    out_times.resize(max_out);
    out_sources.resize(max_out);
    out_currents.resize(max_out);

    int n = eventprop_compute_spikes(
        w, n_pre, n_post, times, sources, n_spikes,
        kVThDefault, kTauMemDefault, kTauSynDefault,
        out_times.data(), out_sources.data(), out_currents.data());

    out_times.resize(n);
    out_sources.resize(n);
    out_currents.resize(n);
    return n;
}

// ============================================================
// Benchmark: ComputeSums — all sizes × patterns
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
        eventprop_compute_sums(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            kTauMemDefault, kTauSynDefault, sum0.data(), sum1.data());
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
COMPUTESUMS_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: ComputeSpikes — all sizes × patterns
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
        std::memset(out_times.data(), 0, out_times.size() * sizeof(double));
        std::memset(out_sources.data(), 0, out_sources.size() * sizeof(int32_t));
        std::memset(out_currents.data(), 0, out_currents.size() * sizeof(double));
        eventprop_compute_spikes(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            kVThDefault, kTauMemDefault, kTauSynDefault,
            out_times.data(), out_sources.data(), out_currents.data());
    }
    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define COMPUTESPIKES_BENCH(SizeName, NP, NPO, NS)                                             \
    BENCHMARK_CAPTURE(BM_ComputeSpikes, SizeName##_uniform,                                    \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                   \
        "ComputeSpikes/" #SizeName "/uniform/default");                                         \
    BENCHMARK_CAPTURE(BM_ComputeSpikes, SizeName##_poisson,                                    \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                   \
        "ComputeSpikes/" #SizeName "/poisson/default");                                         \
    BENCHMARK_CAPTURE(BM_ComputeSpikes, SizeName##_burst,                                      \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                     \
        "ComputeSpikes/" #SizeName "/burst/default")

COMPUTESPIKES_BENCH(Tiny,    10,   5,   20);
COMPUTESPIKES_BENCH(Small,   50,   20,  100);
COMPUTESPIKES_BENCH(Medium,  200,  80,  500);
COMPUTESPIKES_BENCH(Large,   500,  200, 1000);
COMPUTESPIKES_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: BackwardSpikes — all sizes × patterns
// ============================================================
static void BM_BackwardSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    std::vector<double> post_times, post_currents;
    std::vector<int32_t> post_sources_vec;
    int n_post_spikes = RunForward(sz.n_pre, sz.n_post, sz.n_spikes,
                                   w.data(), times.data(), sources.data(),
                                   post_times, post_sources_vec, post_currents);

    std::vector<double> post_errors(n_post_spikes, 1.0);
    std::vector<double> grad(sz.n_pre * sz.n_post);

    for (auto _ : state) {
        std::memset(grad.data(), 0, grad.size() * sizeof(double));
        eventprop_backward_spikes(times.data(), sources.data(), sz.n_spikes,
            post_times.data(), post_sources_vec.data(),
            post_currents.data(), post_errors.data(), n_post_spikes,
            w.data(), sz.n_pre, sz.n_post,
            kVThDefault, kTauMemDefault, kTauSynDefault, grad.data());
    }
    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define BACKWARDSPIKES_BENCH(SizeName, NP, NPO, NS)                                            \
    BENCHMARK_CAPTURE(BM_BackwardSpikes, SizeName##_uniform,                                   \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                   \
        "BackwardSpikes/" #SizeName "/uniform/default");                                        \
    BENCHMARK_CAPTURE(BM_BackwardSpikes, SizeName##_poisson,                                   \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                   \
        "BackwardSpikes/" #SizeName "/poisson/default");                                        \
    BENCHMARK_CAPTURE(BM_BackwardSpikes, SizeName##_burst,                                     \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                     \
        "BackwardSpikes/" #SizeName "/burst/default")

BACKWARDSPIKES_BENCH(Tiny,    10,   5,   20);
BACKWARDSPIKES_BENCH(Small,   50,   20,  100);
BACKWARDSPIKES_BENCH(Medium,  200,  80,  500);
BACKWARDSPIKES_BENCH(Large,   500,  200, 1000);
BACKWARDSPIKES_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: ComputeMaxima — all sizes × patterns (NEW)
// ============================================================
static void BM_ComputeMaxima(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    std::vector<double> maxima_times(sz.n_post);
    std::vector<double> maxima_values(sz.n_post);

    for (auto _ : state) {
        eventprop_compute_maxima(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            kTauMemDefault, kTauSynDefault,
            maxima_times.data(), maxima_values.data());
    }
    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define COMPUTEMAXIMA_BENCH(SizeName, NP, NPO, NS)                                               \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_uniform,                                      \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                     \
        "ComputeMaxima/" #SizeName "/uniform/default");                                           \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_poisson,                                      \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                     \
        "ComputeMaxima/" #SizeName "/poisson/default");                                           \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_burst,                                        \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                       \
        "ComputeMaxima/" #SizeName "/burst/default")

COMPUTEMAXIMA_BENCH(Tiny,    10,   5,   20);
COMPUTEMAXIMA_BENCH(Small,   50,   20,  100);
COMPUTEMAXIMA_BENCH(Medium,  200,  80,  500);
COMPUTEMAXIMA_BENCH(Large,   500,  200, 1000);
COMPUTEMAXIMA_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: Training loop — forward + backward (all sizes)
// ============================================================
static void BM_TrainingLoop(benchmark::State& state, const ProblemSize& sz)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    GenerateUniformData(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    // Pre-compute forward pass
    std::vector<double> post_times, post_currents;
    std::vector<int32_t> post_sources_vec;
    int n_post_spikes = RunForward(sz.n_pre, sz.n_post, sz.n_spikes,
                                   w.data(), times.data(), sources.data(),
                                   post_times, post_sources_vec, post_currents);

    std::vector<double> post_errors(n_post_spikes, 1.0);
    std::vector<double> grad(sz.n_pre * sz.n_post);

    // Pre-allocate output buffers (max possible spikes = n_spikes * n_post)
    int max_out = sz.n_spikes * sz.n_post;
    std::vector<double> fwd_times(max_out);
    std::vector<int32_t> fwd_sources(max_out);
    std::vector<double> fwd_currents(max_out);

    for (auto _ : state) {
        // Forward pass
        int n_fwd = eventprop_compute_spikes(
            w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            kVThDefault, kTauMemDefault, kTauSynDefault,
            fwd_times.data(), fwd_sources.data(), fwd_currents.data());

        // Backward pass
        std::memset(grad.data(), 0, grad.size() * sizeof(double));
        eventprop_backward_spikes(times.data(), sources.data(), sz.n_spikes,
            fwd_times.data(), fwd_sources.data(),
            fwd_currents.data(), post_errors.data(), n_fwd,
            w.data(), sz.n_pre, sz.n_post,
            kVThDefault, kTauMemDefault, kTauSynDefault, grad.data());
    }
    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define TRAININGLOOP_BENCH(SizeName, NP, NPO, NS)                                                \
    BENCHMARK_CAPTURE(BM_TrainingLoop, SizeName,                                                 \
        {#SizeName, NP, NPO, NS})->UseRealTime()->Name(                                          \
        "TrainingLoop/" #SizeName "/uniform/default")

TRAININGLOOP_BENCH(Tiny,    10,   5,   20);
TRAININGLOOP_BENCH(Small,   50,   20,  100);
TRAININGLOOP_BENCH(Medium,  200,  80,  500);
TRAININGLOOP_BENCH(Large,   500,  200, 1000);
TRAININGLOOP_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Multi-threading comparison benchmarks
// Runs each algorithm/size with 1 thread vs. all cores.
// Requires OpenMP (brew install libomp on macOS).
// ============================================================

// Multi-threading comparison: 1 thread vs. all cores (EP_BENCH_THREADS from CMake)
#ifndef EP_BENCH_THREADS
#define EP_BENCH_THREADS 8
#endif

static void BM_ComputeSums_MultiThreaded(benchmark::State& state, const ProblemSize& sz)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    GenerateUniformData(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    int sum_rows = sz.n_spikes + 1;
    std::vector<double> sum0(sum_rows * sz.n_post);
    std::vector<double> sum1(sum_rows * sz.n_post);

    int n_threads = (state.thread_index() == 0) ? 1 : EP_BENCH_THREADS;

    for (auto _ : state) {
        omp_set_num_threads(n_threads);
        std::memset(sum0.data(), 0, sum0.size() * sizeof(double));
        std::memset(sum1.data(), 0, sum1.size() * sizeof(double));
        eventprop_compute_sums(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            kTauMemDefault, kTauSynDefault, sum0.data(), sum1.data());
    }
    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define MULTITHREADED_COMPUTESUMS(SizeName, NP, NPO, NS)                                        \
    BENCHMARK_CAPTURE(BM_ComputeSums_MultiThreaded, SizeName##_threads1,                         \
        {#SizeName, NP, NPO, NS})->UseRealTime()->Threads(1)->Name("ComputeSums/" #SizeName "/uniform/threads=1"); \
    BENCHMARK_CAPTURE(BM_ComputeSums_MultiThreaded, SizeName##_threadsN,                         \
        {#SizeName, NP, NPO, NS})->UseRealTime()->Threads(EP_BENCH_THREADS)->Name("ComputeSums/" #SizeName "/uniform/threads=N")

MULTITHREADED_COMPUTESUMS(Small,   50,   20,  100);
MULTITHREADED_COMPUTESUMS(Medium,  200,  80,  500);
MULTITHREADED_COMPUTESUMS(Large,   500,  200, 1000);

static void BM_ComputeSpikes_MultiThreaded(benchmark::State& state, const ProblemSize& sz)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    GenerateUniformData(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    int max_out = sz.n_spikes * sz.n_post;
    std::vector<double> out_times(max_out);
    std::vector<int32_t> out_sources(max_out);
    std::vector<double> out_currents(max_out);

    int n_threads = (state.thread_index() == 0) ? 1 : EP_BENCH_THREADS;

    for (auto _ : state) {
        omp_set_num_threads(n_threads);
        std::memset(out_times.data(), 0, out_times.size() * sizeof(double));
        std::memset(out_sources.data(), 0, out_sources.size() * sizeof(int32_t));
        std::memset(out_currents.data(), 0, out_currents.size() * sizeof(double));
        eventprop_compute_spikes(w.data(), sz.n_pre, sz.n_post,
            times.data(), sources.data(), sz.n_spikes,
            kVThDefault, kTauMemDefault, kTauSynDefault,
            out_times.data(), out_sources.data(), out_currents.data());
    }
    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define MULTITHREADED_COMPUTESPIKES(SizeName, NP, NPO, NS)                                        \
    BENCHMARK_CAPTURE(BM_ComputeSpikes_MultiThreaded, SizeName##_threads1,                        \
        {#SizeName, NP, NPO, NS})->UseRealTime()->Threads(1)->Name("ComputeSpikes/" #SizeName "/uniform/threads=1"); \
    BENCHMARK_CAPTURE(BM_ComputeSpikes_MultiThreaded, SizeName##_threadsN,                        \
        {#SizeName, NP, NPO, NS})->UseRealTime()->Threads(EP_BENCH_THREADS)->Name("ComputeSpikes/" #SizeName "/uniform/threads=N")

MULTITHREADED_COMPUTESPIKES(Small,   50,   20,  100);
MULTITHREADED_COMPUTESPIKES(Medium,  200,  80,  500);
MULTITHREADED_COMPUTESPIKES(Large,   500,  200, 1000);

static void BM_BackwardSpikes_MultiThreaded(benchmark::State& state, const ProblemSize& sz)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    GenerateUniformData(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    std::vector<double> post_times, post_currents;
    std::vector<int32_t> post_sources_vec;
    int n_post_spikes = RunForward(sz.n_pre, sz.n_post, sz.n_spikes,
                                   w.data(), times.data(), sources.data(),
                                   post_times, post_sources_vec, post_currents);

    std::vector<double> post_errors(n_post_spikes, 1.0);
    std::vector<double> grad(sz.n_pre * sz.n_post);

    int n_threads = (state.thread_index() == 0) ? 1 : EP_BENCH_THREADS;

    for (auto _ : state) {
        omp_set_num_threads(n_threads);
        std::memset(grad.data(), 0, grad.size() * sizeof(double));
        eventprop_backward_spikes(times.data(), sources.data(), sz.n_spikes,
            post_times.data(), post_sources_vec.data(),
            post_currents.data(), post_errors.data(), n_post_spikes,
            w.data(), sz.n_pre, sz.n_post,
            kVThDefault, kTauMemDefault, kTauSynDefault, grad.data());
    }
    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define MULTITHREADED_BACKWARDSPIKES(SizeName, NP, NPO, NS)                                       \
    BENCHMARK_CAPTURE(BM_BackwardSpikes_MultiThreaded, SizeName##_threads1,                       \
        {#SizeName, NP, NPO, NS})->UseRealTime()->Threads(1)->Name("BackwardSpikes/" #SizeName "/uniform/threads=1"); \
    BENCHMARK_CAPTURE(BM_BackwardSpikes_MultiThreaded, SizeName##_threadsN,                       \
        {#SizeName, NP, NPO, NS})->UseRealTime()->Threads(EP_BENCH_THREADS)->Name("BackwardSpikes/" #SizeName "/uniform/threads=N")

MULTITHREADED_BACKWARDSPIKES(Small,   50,   20,  100);
MULTITHREADED_BACKWARDSPIKES(Medium,  200,  80,  500);
MULTITHREADED_BACKWARDSPIKES(Large,   500,  200, 1000);

// Note: main() is provided by benchmark::benchmark_main library
