// eventprop_benchmarks.cpp — C++ benchmarks mirroring EventProp.Benchmarks (C#)
// Compares against the same problem sizes and parameters.

#include "eventprop_c_api.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ============================================================
// Parameters (match .NET benchmarks exactly)
// ============================================================
static constexpr double kTauMem = 0.02;
static constexpr double kTauSyn = 0.005;
static constexpr double kVTh    = 1.0;

// ============================================================
// Helpers — mirror .NET BenchBase Setup() exactly
// ============================================================
static void GenerateData(int n_pre, int n_post, int n_spikes,
                         double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42); // Same seed as .NET BenchBase
    std::uniform_real_distribution<double> w_dist(0.5, 2.0); // 0.5 + rng.NextDouble() * 1.5

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    for (int i = 0; i < n_spikes; i++) {
        times[i] = (i + 1) / static_cast<double>(n_spikes + 1);
        sources[i] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);
    }
}

// Run forward pass to get post-spike data for backward benchmarks
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
        w, n_pre, n_post,
        times, sources, n_spikes,
        kVTh, kTauMem, kTauSyn,
        out_times.data(), out_sources.data(), out_currents.data());

    out_times.resize(n);
    out_sources.resize(n);
    out_currents.resize(n);
    return n;
}

// ============================================================
// Benchmark: ComputeSums — Small (50, 20, 100)
// ============================================================
static void BM_ComputeSums_Small(benchmark::State& state)
{
    const int n_pre = 50, n_post = 20, n_spikes = 100;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    int sum_rows = n_spikes + 1;
    std::vector<double> sum0(sum_rows * n_post);
    std::vector<double> sum1(sum_rows * n_post);

    for (auto _ : state) {
        std::memset(sum0.data(), 0, sum0.size() * sizeof(double));
        std::memset(sum1.data(), 0, sum1.size() * sizeof(double));
        eventprop_compute_sums(
            w.data(), n_pre, n_post,
            times.data(), sources.data(), n_spikes,
            kTauMem, kTauSyn,
            sum0.data(), sum1.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ComputeSums_Small)->UseRealTime();

// ============================================================
// Benchmark: ComputeSums — Medium (200, 80, 500)
// ============================================================
static void BM_ComputeSums_Medium(benchmark::State& state)
{
    const int n_pre = 200, n_post = 80, n_spikes = 500;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    int sum_rows = n_spikes + 1;
    std::vector<double> sum0(sum_rows * n_post);
    std::vector<double> sum1(sum_rows * n_post);

    for (auto _ : state) {
        std::memset(sum0.data(), 0, sum0.size() * sizeof(double));
        std::memset(sum1.data(), 0, sum1.size() * sizeof(double));
        eventprop_compute_sums(
            w.data(), n_pre, n_post,
            times.data(), sources.data(), n_spikes,
            kTauMem, kTauSyn,
            sum0.data(), sum1.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ComputeSums_Medium)->UseRealTime();

// ============================================================
// Benchmark: ComputeSums — Large (500, 200, 1000)
// ============================================================
static void BM_ComputeSums_Large(benchmark::State& state)
{
    const int n_pre = 500, n_post = 200, n_spikes = 1000;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    int sum_rows = n_spikes + 1;
    std::vector<double> sum0(sum_rows * n_post);
    std::vector<double> sum1(sum_rows * n_post);

    for (auto _ : state) {
        std::memset(sum0.data(), 0, sum0.size() * sizeof(double));
        std::memset(sum1.data(), 0, sum1.size() * sizeof(double));
        eventprop_compute_sums(
            w.data(), n_pre, n_post,
            times.data(), sources.data(), n_spikes,
            kTauMem, kTauSyn,
            sum0.data(), sum1.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ComputeSums_Large)->UseRealTime();

// ============================================================
// Benchmark: ComputeSpikes — Small (50, 20, 100)
// ============================================================
static void BM_ComputeSpikes_Small(benchmark::State& state)
{
    const int n_pre = 50, n_post = 20, n_spikes = 100;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    int max_out = n_spikes * n_post;
    std::vector<double> out_times(max_out);
    std::vector<int32_t> out_sources(max_out);
    std::vector<double> out_currents(max_out);

    for (auto _ : state) {
        std::memset(out_times.data(), 0, out_times.size() * sizeof(double));
        std::memset(out_sources.data(), 0, out_sources.size() * sizeof(int32_t));
        std::memset(out_currents.data(), 0, out_currents.size() * sizeof(double));

        eventprop_compute_spikes(
            w.data(), n_pre, n_post,
            times.data(), sources.data(), n_spikes,
            kVTh, kTauMem, kTauSyn,
            out_times.data(), out_sources.data(), out_currents.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ComputeSpikes_Small)->UseRealTime();

// ============================================================
// Benchmark: ComputeSpikes — Medium (200, 80, 500)
// ============================================================
static void BM_ComputeSpikes_Medium(benchmark::State& state)
{
    const int n_pre = 200, n_post = 80, n_spikes = 500;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    int max_out = n_spikes * n_post;
    std::vector<double> out_times(max_out);
    std::vector<int32_t> out_sources(max_out);
    std::vector<double> out_currents(max_out);

    for (auto _ : state) {
        std::memset(out_times.data(), 0, out_times.size() * sizeof(double));
        std::memset(out_sources.data(), 0, out_sources.size() * sizeof(int32_t));
        std::memset(out_currents.data(), 0, out_currents.size() * sizeof(double));

        eventprop_compute_spikes(
            w.data(), n_pre, n_post,
            times.data(), sources.data(), n_spikes,
            kVTh, kTauMem, kTauSyn,
            out_times.data(), out_sources.data(), out_currents.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ComputeSpikes_Medium)->UseRealTime();

// ============================================================
// Benchmark: ComputeSpikes — Large (500, 200, 1000)
// ============================================================
static void BM_ComputeSpikes_Large(benchmark::State& state)
{
    const int n_pre = 500, n_post = 200, n_spikes = 1000;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    int max_out = n_spikes * n_post;
    std::vector<double> out_times(max_out);
    std::vector<int32_t> out_sources(max_out);
    std::vector<double> out_currents(max_out);

    for (auto _ : state) {
        std::memset(out_times.data(), 0, out_times.size() * sizeof(double));
        std::memset(out_sources.data(), 0, out_sources.size() * sizeof(int32_t));
        std::memset(out_currents.data(), 0, out_currents.size() * sizeof(double));

        eventprop_compute_spikes(
            w.data(), n_pre, n_post,
            times.data(), sources.data(), n_spikes,
            kVTh, kTauMem, kTauSyn,
            out_times.data(), out_sources.data(), out_currents.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ComputeSpikes_Large)->UseRealTime();

// ============================================================
// Benchmark: BackwardSpikes — Small (50, 20, 100)
// ============================================================
static void BM_BackwardSpikes_Small(benchmark::State& state)
{
    const int n_pre = 50, n_post = 20, n_spikes = 100;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    // Pre-compute forward pass once
    std::vector<double> post_times, post_currents;
    std::vector<int32_t> post_sources_vec;
    int n_post_spikes = RunForward(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data(),
                                   post_times, post_sources_vec, post_currents);

    // Post errors (all 1.0, matching .NET BenchBase)
    std::vector<double> post_errors(n_post_spikes, 1.0);

    // Gradient buffer (caller zero-initializes)
    std::vector<double> grad(n_pre * n_post);

    for (auto _ : state) {
        std::memset(grad.data(), 0, grad.size() * sizeof(double));
        eventprop_backward_spikes(
            times.data(), sources.data(), n_spikes,
            post_times.data(), post_sources_vec.data(),
            post_currents.data(), post_errors.data(), n_post_spikes,
            w.data(), n_pre, n_post,
            kVTh, kTauMem, kTauSyn,
            grad.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_BackwardSpikes_Small)->UseRealTime();

// ============================================================
// Benchmark: BackwardSpikes — Medium (200, 80, 500)
// ============================================================
static void BM_BackwardSpikes_Medium(benchmark::State& state)
{
    const int n_pre = 200, n_post = 80, n_spikes = 500;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    // Pre-compute forward pass once
    std::vector<double> post_times, post_currents;
    std::vector<int32_t> post_sources_vec;
    int n_post_spikes = RunForward(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data(),
                                   post_times, post_sources_vec, post_currents);

    // Post errors (all 1.0, matching .NET BenchBase)
    std::vector<double> post_errors(n_post_spikes, 1.0);

    // Gradient buffer (caller zero-initializes)
    std::vector<double> grad(n_pre * n_post);

    for (auto _ : state) {
        std::memset(grad.data(), 0, grad.size() * sizeof(double));
        eventprop_backward_spikes(
            times.data(), sources.data(), n_spikes,
            post_times.data(), post_sources_vec.data(),
            post_currents.data(), post_errors.data(), n_post_spikes,
            w.data(), n_pre, n_post,
            kVTh, kTauMem, kTauSyn,
            grad.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_BackwardSpikes_Medium)->UseRealTime();

// ============================================================
// Benchmark: BackwardSpikes — Large (500, 200, 1000)
// ============================================================
static void BM_BackwardSpikes_Large(benchmark::State& state)
{
    const int n_pre = 500, n_post = 200, n_spikes = 1000;
    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);
    GenerateData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    // Pre-compute forward pass once
    std::vector<double> post_times, post_currents;
    std::vector<int32_t> post_sources_vec;
    int n_post_spikes = RunForward(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data(),
                                   post_times, post_sources_vec, post_currents);

    // Post errors (all 1.0, matching .NET BenchBase)
    std::vector<double> post_errors(n_post_spikes, 1.0);

    // Gradient buffer (caller zero-initializes)
    std::vector<double> grad(n_pre * n_post);

    for (auto _ : state) {
        std::memset(grad.data(), 0, grad.size() * sizeof(double));
        eventprop_backward_spikes(
            times.data(), sources.data(), n_spikes,
            post_times.data(), post_sources_vec.data(),
            post_currents.data(), post_errors.data(), n_post_spikes,
            w.data(), n_pre, n_post,
            kVTh, kTauMem, kTauSyn,
            grad.data());
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_BackwardSpikes_Large)->UseRealTime();

// Note: main() is provided by benchmark::benchmark_main library
