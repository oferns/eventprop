// C vs C++ EventProp benchmark — side-by-side comparison
// Links against both libeventprop_c_api.dylib (C) and libeventprop_cpp.dylib (C++ Eigen)

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <string>

// ============================================================
// Function pointer types for both libraries
// ============================================================

typedef void (*compute_sums_fn)(const double*, int, int, const double*, const int32_t*, int, double, double, double*, double*);
typedef int  (*compute_spikes_fn)(const double*, int, int, const double*, const int32_t*, int, double, double, double, double*, int32_t*, double*);
typedef void (*backward_spikes_fn)(const double*, const int32_t*, int, const double*, const int32_t*, const double*, const double*, int,
                                    const double*, int, int, double, double, double, double*);

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

    std::exponential_distribution<double> inter_spike(1.0 / 200.0);
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
// Problem sizes
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
    std::vector<double> sum0_c(sum_rows * sz.n_post);
    std::vector<double> sum1_c(sum_rows * sz.n_post);
    std::vector<double> sum0_cpp(sum_rows * sz.n_post);
    std::vector<double> sum1_cpp(sum_rows * sz.n_post);

    // Load libraries
    void* h_c = dlopen("./libeventprop_c_api.dylib", RTLD_NOW);
    void* h_cpp = dlopen("./libeventprop_cpp.dylib", RTLD_NOW);

    if (!h_c || !h_cpp) {
        std::cerr << "Failed to load libraries: " << dlerror() << "\n";
        state.SkipWithError("Library load failed");
        return;
    }

    auto fn_c = (compute_sums_fn)dlsym(h_c, "eventprop_compute_sums");
    auto fn_cpp = (compute_sums_fn)dlsym(h_cpp, "eventprop_compute_sums");

    if (!fn_c || !fn_cpp) {
        std::cerr << "Failed to find symbols\n";
        state.SkipWithError("Symbol lookup failed");
        dlclose(h_c); dlclose(h_cpp);
        return;
    }

    for (auto _ : state) {
        std::memset(sum0_c.data(), 0, sum0_c.size() * sizeof(double));
        std::memset(sum1_c.data(), 0, sum1_c.size() * sizeof(double));
        fn_c(w.data(), sz.n_pre, sz.n_post, times.data(), sources.data(), sz.n_spikes, 0.02, 0.005, sum0_c.data(), sum1_c.data());
    }

    for (auto _ : state) {
        std::memset(sum0_cpp.data(), 0, sum0_cpp.size() * sizeof(double));
        std::memset(sum1_cpp.data(), 0, sum1_cpp.size() * sizeof(double));
        fn_cpp(w.data(), sz.n_pre, sz.n_post, times.data(), sources.data(), sz.n_spikes, 0.02, 0.005, sum0_cpp.data(), sum1_cpp.data());
    }

    dlclose(h_c); dlclose(h_cpp);
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
    std::vector<double> out_times_c(max_out);
    std::vector<int32_t> out_sources_c(max_out);
    std::vector<double> out_currents_c(max_out);
    std::vector<double> out_times_cpp(max_out);
    std::vector<int32_t> out_sources_cpp(max_out);
    std::vector<double> out_currents_cpp(max_out);

    void* h_c = dlopen("./libeventprop_c_api.dylib", RTLD_NOW);
    void* h_cpp = dlopen("./libeventprop_cpp.dylib", RTLD_NOW);

    if (!h_c || !h_cpp) {
        std::cerr << "Failed to load libraries: " << dlerror() << "\n";
        state.SkipWithError("Library load failed");
        return;
    }

    auto fn_c = (compute_spikes_fn)dlsym(h_c, "eventprop_compute_spikes");
    auto fn_cpp = (compute_spikes_fn)dlsym(h_cpp, "eventprop_compute_spikes");

    if (!fn_c || !fn_cpp) {
        std::cerr << "Failed to find symbols\n";
        state.SkipWithError("Symbol lookup failed");
        dlclose(h_c); dlclose(h_cpp);
        return;
    }

    for (auto _ : state) {
        fn_c(w.data(), sz.n_pre, sz.n_post, times.data(), sources.data(), sz.n_spikes,
             1.0, 0.02, 0.005, out_times_c.data(), out_sources_c.data(), out_currents_c.data());
    }

    for (auto _ : state) {
        fn_cpp(w.data(), sz.n_pre, sz.n_post, times.data(), sources.data(), sz.n_spikes,
               1.0, 0.02, 0.005, out_times_cpp.data(), out_sources_cpp.data(), out_currents_cpp.data());
    }

    dlclose(h_c); dlclose(h_cpp);
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

BENCHMARK_MAIN();
