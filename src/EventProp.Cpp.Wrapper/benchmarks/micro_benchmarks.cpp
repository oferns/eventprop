// micro_benchmarks.cpp — Micro-benchmarks for individual hot paths in EventProp.
// These isolate specific algorithms to identify bottlenecks.

#include "eventprop_c_api.h"
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ============================================================
// Parameters
// ============================================================
static constexpr double kTauMem = 0.02;
static constexpr double kTauSyn = 0.005;
static constexpr double kVTh    = 1.0;

// ============================================================
// Helpers — data generation
// ============================================================
static void GenerateData(int n_pre, int n_post, int n_spikes,
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

// ============================================================
// Micro: ExpPrecompute — std::exp(t/τ) for all spike times
// ============================================================
static void BM_ExpPrecompute(benchmark::State& state)
{
    const int n_spikes = 100;
    std::vector<double> times(n_spikes);
    for (int i = 0; i < n_spikes; i++)
        times[i] = (i + 1) / static_cast<double>(n_spikes + 1);

    std::vector<double> exp_mem(n_spikes), exp_syn(n_spikes);

    for (auto _ : state) {
        for (int i = 0; i < n_spikes; i++) {
            exp_mem[i] = std::exp(times[i] / kTauMem);
            exp_syn[i] = std::exp(times[i] / kTauSyn);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ExpPrecompute)->UseRealTime()->Name("Micro/ExpPrecompute/n=100");

static void BM_ExpPrecompute_Medium(benchmark::State& state)
{
    const int n_spikes = 500;
    std::vector<double> times(n_spikes);
    for (int i = 0; i < n_spikes; i++)
        times[i] = (i + 1) / static_cast<double>(n_spikes + 1);

    std::vector<double> exp_mem(n_spikes), exp_syn(n_spikes);

    for (auto _ : state) {
        for (int i = 0; i < n_spikes; i++) {
            exp_mem[i] = std::exp(times[i] / kTauMem);
            exp_syn[i] = std::exp(times[i] / kTauSyn);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_spikes);
}
BENCHMARK(BM_ExpPrecompute_Medium)->UseRealTime()->Name("Micro/ExpPrecompute/n=500");

// ============================================================
// Micro: BrentRootFinder — inline Brent root finder (from eventprop_c_api.cpp)
// ============================================================
static constexpr double EP_INF = std::numeric_limits<double>::infinity();

// Simulated voltage function for root finding
static double MakeVFunc(const std::vector<double>& sum0, const std::vector<double>& sum1,
                        int target_nrn, int t_pre_idx)
{
    // Simplified: just return a value that crosses threshold
    double k_prefactor = kTauSyn / (kTauMem - kTauSyn);
    return k_prefactor * (sum1[t_pre_idx] * std::exp(-0.5 / kTauMem)
                         - sum0[t_pre_idx] * std::exp(-0.5 / kTauSyn));
}

static double BrentRootFinder(double a, double b, std::function<double(double)> f)
{
    double fa = f(a), fb = f(b);
    if (fa * fb > 0) return EP_INF;

    double c = a, fc = fa;
    for (int iter = 0; iter < 1024; iter++) {
        double m = 0.5 * (b - a);
        if (std::abs(m) <= 1e-14 || fb == 0.0) return b;

        double s = 0.0, d = 0.0;
        bool use_bisect = true;

        if (std::abs(fc) > std::abs(fb) && c != a && c != b) {
            double qba = (fb - fa) / (b - a);
            double qcb = (fc - fb) / (c - b);
            double qcbba = (qcb - qba) / (c - a);
            if (std::abs(qcbba) > 1e-300 && std::abs(qba + qcb) > 1e-300) {
                s = b - fb / (qba + qcb * (b - a) * qcbba / (qba + qcb));
                if ((s > a && s < b) && std::abs(s - b) < 0.5 * (b - a))
                    use_bisect = false;
            }
        }

        if (use_bisect) s = a + 0.5 * (b - a);

        double prev_b = b;
        b = s;
        fb = f(b);

        if (std::abs(fb) <= 1e-14 || std::abs(b - prev_b) <= 1e-14 * std::max(1.0, std::abs(prev_b)))
            return b;

        if (fa * fb < 0.0) { c = prev_b; fc = f(prev_b); }
        else if (fb * fc < 0.0) { a = prev_b; fa = f(prev_b); }
        else { a = prev_b; fa = f(prev_b); }
    }
    return b;
}

static void BM_BrentRootFinder(benchmark::State& state)
{
    const int n_nrns = 20;
    // Create a simple function that crosses zero (simulating voltage crossing threshold)
    auto f = [](double t) { return std::sin(10.0 * t) - 0.5; };

    for (auto _ : state) {
        double result = BrentRootFinder(0.0, 1.0, f);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_BrentRootFinder)->UseRealTime()->Name("Micro/BrentRootFinder/nrns=20");

static void BM_BrentRootFinder_Large(benchmark::State& state)
{
    const int n_nrns = 200;
    auto f = [](double t) { return std::sin(10.0 * t) - 0.5; };

    for (auto _ : state) {
        double result = BrentRootFinder(0.0, 1.0, f);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_BrentRootFinder_Large)->UseRealTime()->Name("Micro/BrentRootFinder/nrns=200");

// ============================================================
// Micro: GradientAccumulation — backward pass inner loop
// ============================================================
static void BM_GradientAccumulation(benchmark::State& state)
{
    const int n_pre = 50, n_post = 20;

    // Simulated adjoint values
    std::vector<double> lambda_v(n_post, 0.5);
    std::vector<double> lambda_i(n_post, 0.3);
    std::vector<double> w(n_pre * n_post);

    // Fill with random weights
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    std::vector<double> grad(n_pre * n_post, 0.0);
    double tau_syn = kTauSyn;

    for (auto _ : state) {
        // Inner loop of backward: grad[source, nrn] -= tau_syn * lambda_i[nrn]
        // Plus outbound signal computation
        for (int src = 0; src < n_pre; src++) {
            double outbound_signal = 0.0;
            for (int nrn = 0; nrn < n_post; nrn++) {
                grad[src * n_post + nrn] -= tau_syn * lambda_i[nrn];
                outbound_signal += w[src * n_post + nrn] * (lambda_v[nrn] - lambda_i[nrn]);
            }
            benchmark::DoNotOptimize(outbound_signal);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pre * n_post);
}
BENCHMARK(BM_GradientAccumulation)->UseRealTime()->Name("Micro/GradientAccumulation/pre=50/post=20");

static void BM_GradientAccumulation_Medium(benchmark::State& state)
{
    const int n_pre = 200, n_post = 80;

    std::vector<double> lambda_v(n_post, 0.5);
    std::vector<double> lambda_i(n_post, 0.3);
    std::vector<double> w(n_pre * n_post);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    std::vector<double> grad(n_pre * n_post, 0.0);
    double tau_syn = kTauSyn;

    for (auto _ : state) {
        for (int src = 0; src < n_pre; src++) {
            double outbound_signal = 0.0;
            for (int nrn = 0; nrn < n_post; nrn++) {
                grad[src * n_post + nrn] -= tau_syn * lambda_i[nrn];
                outbound_signal += w[src * n_post + nrn] * (lambda_v[nrn] - lambda_i[nrn]);
            }
            benchmark::DoNotOptimize(outbound_signal);
        }
    }
    state.SetItemsProcessed(state.iterations() * n_pre * n_post);
}
BENCHMARK(BM_GradientAccumulation_Medium)->UseRealTime()->Name("Micro/GradientAccumulation/pre=200/post=80");

// ============================================================
// Micro: SpikeMergeSort — merge input + post spikes, sort descending by time
// ============================================================
struct MergedSpike {
    int idx;
    bool is_input;
    double time;
};

static void BM_SpikeMergeSort(benchmark::State& state)
{
    const int n_input = 100;
    const int n_post = 30;

    std::vector<double> input_times(n_input);
    for (int i = 0; i < n_input; i++)
        input_times[i] = (i + 1) / static_cast<double>(n_input + 1);

    std::vector<double> post_times(n_post);
    for (int i = 0; i < n_post; i++)
        post_times[i] = (i + 1) / static_cast<double>(n_post + 2); // offset to interleave

    int total = n_input + n_post;
    std::vector<MergedSpike> merged(total);

    for (auto _ : state) {
        // Populate merged array
        for (int i = 0; i < n_input; i++)
            merged[i] = {i, true, input_times[i]};
        for (int i = 0; i < n_post; i++)
            merged[n_input + i] = {i, false, post_times[i]};

        // Sort descending by time
        std::sort(merged.begin(), merged.end(),
            [](const MergedSpike& a, const MergedSpike& b) { return a.time > b.time; });

        benchmark::DoNotOptimize(merged.size());
    }
    state.SetItemsProcessed(state.iterations() * total);
}
BENCHMARK(BM_SpikeMergeSort)->UseRealTime()->Name("Micro/SpikeMergeSort/input=100+post=30");

static void BM_SpikeMergeSort_Large(benchmark::State& state)
{
    const int n_input = 500;
    const int n_post = 150;

    std::vector<double> input_times(n_input);
    for (int i = 0; i < n_input; i++)
        input_times[i] = (i + 1) / static_cast<double>(n_input + 1);

    std::vector<double> post_times(n_post);
    for (int i = 0; i < n_post; i++)
        post_times[i] = (i + 1) / static_cast<double>(n_post + 2);

    int total = n_input + n_post;
    std::vector<MergedSpike> merged(total);

    for (auto _ : state) {
        for (int i = 0; i < n_input; i++)
            merged[i] = {i, true, input_times[i]};
        for (int i = 0; i < n_post; i++)
            merged[n_input + i] = {i, false, post_times[i]};

        std::sort(merged.begin(), merged.end(),
            [](const MergedSpike& a, const MergedSpike& b) { return a.time > b.time; });

        benchmark::DoNotOptimize(merged.size());
    }
    state.SetItemsProcessed(state.iterations() * total);
}
BENCHMARK(BM_SpikeMergeSort_Large)->UseRealTime()->Name("Micro/SpikeMergeSort/input=500+post=150");

// Note: main() is provided by benchmark::benchmark_main library
