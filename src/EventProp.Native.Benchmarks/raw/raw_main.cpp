// Raw C++ EventProp benchmark — calls EventProp.Python Eigen core directly
// No wrapper layer. Pure Eigen + Spikes/Maxima types.

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <algorithm>

// Include the C++ algorithm headers (EventProp.Python)
#include "eventprop_types.h"

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXi = Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// ============================================================
// Data generation (same RNG seed 42, same patterns)
// ============================================================

static void GenerateUniformData(int n_pre, int n_post, int n_spikes,
                                 double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);

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
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);

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
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);

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
// Problem sizes (match all other benchmarks)
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
// Helper: build Spikes from flat arrays (sorted)
// ============================================================

static Spikes MakeSpikes(const double* times, const int32_t* sources, int n)
{
    // Sort by time (required for root-finding)
    std::vector<std::pair<double, int32_t>> sorted(n);
    for (int i = 0; i < n; i++) sorted[i] = {times[i], sources[i]};
    std::sort(sorted.begin(), sorted.end());

    Eigen::ArrayXd t(n);
    Eigen::ArrayXi s(n);
    for (int i = 0; i < n; i++) { t(i) = sorted[i].first; s(i) = static_cast<int>(sorted[i].second); }
    return Spikes(t, s);
}

// ============================================================
// Benchmark: ComputeSums (raw C++)
// ============================================================

static void BM_ComputeSums(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    // Build Eigen matrix view over flat weight array
    RowMatrixXd w_mat(sz.n_pre, sz.n_post);
    for (int i = 0; i < sz.n_pre * sz.n_post; i++) w_mat(i) = w[i];

    Spikes spikes = MakeSpikes(times.data(), sources.data(), sz.n_spikes);
    double tau_mem = 0.02, tau_syn = 0.005;

    for (auto _ : state) {
        auto [sum0, sum1] = compute_sums(w_mat, spikes, tau_mem, tau_syn);
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
COMPUTESUMS_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: ComputeSpikes (raw C++)
// ============================================================

static void BM_ComputeSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    RowMatrixXd w_mat(sz.n_pre, sz.n_post);
    for (int i = 0; i < sz.n_pre * sz.n_post; i++) w_mat(i) = w[i];

    Spikes spikes = MakeSpikes(times.data(), sources.data(), sz.n_spikes);
    double v_th = 1.0, tau_mem = 0.02, tau_syn = 0.005;

    for (auto _ : state) {
        Spikes result = compute_spikes(w_mat, spikes, v_th, tau_mem, tau_syn);
        benchmark::DoNotOptimize(result.times.data());
        benchmark::DoNotOptimize(result.sources.data());
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
COMPUTESPIKES_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: ComputeMaxima (raw C++)
// ============================================================

static void BM_ComputeMaxima(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    RowMatrixXd w_mat(sz.n_pre, sz.n_post);
    for (int i = 0; i < sz.n_pre * sz.n_post; i++) w_mat(i) = w[i];

    Spikes spikes = MakeSpikes(times.data(), sources.data(), sz.n_spikes);
    double tau_mem = 0.02, tau_syn = 0.005;

    for (auto _ : state) {
        Maxima result = compute_maxima(w_mat, spikes, tau_mem, tau_syn);
        benchmark::DoNotOptimize(result.times.data());
        benchmark::DoNotOptimize(result.values.data());
    }

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define COMPUTEMAXIMA_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_uniform,                                     \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                    \
        "ComputeMaxima/" #SizeName "/uniform/default");                                          \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_poisson,                                     \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                    \
        "ComputeMaxima/" #SizeName "/poisson/default");                                          \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_burst,                                       \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                      \
        "ComputeMaxima/" #SizeName "/burst/default")

COMPUTEMAXIMA_BENCH(Tiny,    10,   5,   20);
COMPUTEMAXIMA_BENCH(Small,   50,   20,  100);
COMPUTEMAXIMA_BENCH(Medium,  200,  80,  500);
COMPUTEMAXIMA_BENCH(Large,   500,  200, 1000);
COMPUTEMAXIMA_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: BackwardMaxima (raw C++)
// Requires maxima data from ComputeMaxima forward pass.
// ============================================================

static void BM_BackwardMaxima(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

    RowMatrixXd w_mat(sz.n_pre, sz.n_post);
    for (int i = 0; i < sz.n_pre * sz.n_post; i++) w_mat(i) = w[i];

    Spikes input_spikes = MakeSpikes(times.data(), sources.data(), sz.n_spikes);
    Maxima maxima = compute_maxima(w_mat, input_spikes, 0.02, 0.005);

    // Set maxima errors (all -1.0 for gradient descent)
    for (int i = 0; i < sz.n_post; i++)
        maxima.set_error(i, -1.0);

    for (auto _ : state) {
        auto [gradient, spike_errors] = backward(input_spikes, maxima, w_mat, 0.02, 0.005);
        benchmark::DoNotOptimize(gradient.data());
    }

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define BACKWARDMAXIMA_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_BackwardMaxima, SizeName##_uniform,                                     \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                    \
        "BackwardMaxima/" #SizeName "/uniform/default");                                          \
    BENCHMARK_CAPTURE(BM_BackwardMaxima, SizeName##_poisson,                                     \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                    \
        "BackwardMaxima/" #SizeName "/poisson/default");                                          \
    BENCHMARK_CAPTURE(BM_BackwardMaxima, SizeName##_burst,                                       \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                      \
        "BackwardMaxima/" #SizeName "/burst/default")

BACKWARDMAXIMA_BENCH(Tiny,    10,   5,   20);
BACKWARDMAXIMA_BENCH(Small,   50,   20,  100);
BACKWARDMAXIMA_BENCH(Medium,  200,  80,  500);
BACKWARDMAXIMA_BENCH(Large,   500,  200, 1000);
BACKWARDMAXIMA_BENCH(XL,      2000, 500, 5000);
// ============================================================
// Threshold Voltage (v_th) Variation Benchmarks
// Tests how different threshold settings affect performance
// Lower v_th = more spikes = more root-finding work
// Higher v_th = fewer spikes = less root-finding work
// ============================================================

// ComputeSpikes with v_th variation (Small size, uniform)
static void BM_ComputeSpikesVth(benchmark::State& state)
{
    double v_th = static_cast<double>(state.range(0)) / 10.0; // 5→0.5, 10→1.0, 20→2.0
    int n_pre = 50, n_post = 20, n_spikes = 100;

    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);

    GenerateUniformData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    for (auto _ : state) {
        Eigen::Map<RowMatrixXd const> w_map(w.data(), n_pre, n_post);
        Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), n_spikes);
        Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), n_spikes);
        Spikes output = compute_spikes(w_map, Spikes(e_times, e_sources), v_th, 0.02, 0.005);
        benchmark::DoNotOptimize(output.n_spikes);
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}

BENCHMARK(BM_ComputeSpikesVth)->ArgNames({"v_th"})
    ->RangeMultiplier(1)
    ->DenseRange(5, 20, 5);

// BackwardSpikes with v_th variation (Small size, uniform)
static void BM_BackwardSpikesVth(benchmark::State& state)
{
    double v_th = static_cast<double>(state.range(0)) / 10.0;
    int n_pre = 50, n_post = 20, n_spikes = 100;

    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int32_t> sources(n_spikes);

    GenerateUniformData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    Eigen::Map<RowMatrixXd const> w_map(w.data(), n_pre, n_post);
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), n_spikes);
    Spikes input = Spikes(e_times, e_sources);
    
    // Run forward pass with default v_th=1.0 for all tests
    Spikes post = compute_spikes(w_map, input, 1.0, 0.02, 0.005);
    
    // Set errors for all post-synaptic spikes
    for (int i = 0; i < post.n_spikes; i++)
        post.set_error(i, -1.0);

    std::pair<RowMatrixXd, Eigen::ArrayXd> gradient;

    for (auto _ : state) {
        gradient = backward(input, post, w_map, v_th, 0.02, 0.005);
        benchmark::DoNotOptimize(gradient.first(0, 0));
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}

BENCHMARK(BM_BackwardSpikesVth)->ArgNames({"v_th"})
    ->RangeMultiplier(1)
    ->DenseRange(5, 20, 5);
BENCHMARK_MAIN();
