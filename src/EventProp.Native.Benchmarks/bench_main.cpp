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
// C++ Eigen functions (stripped header, no pybind11/Python dependency)
#include "eventprop_eigen_stripped.h"
#else
// C library functions (consolidated header)
#include "eventprop.h"
#include "eventprop_batch.h"
#endif

// ============================================================
// Data generation (same for both)
// ============================================================

static void GenerateUniformData(int n_pre, int n_post, int n_spikes,
                                 double* w, double* times, int* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    for (int i = 0; i < n_spikes; i++) {
        times[i] = (i + 1) / static_cast<double>(n_spikes + 1);
        sources[i] = std::uniform_int_distribution<int>(0, n_pre - 1)(rng);
    }
}

static void GeneratePoissonData(int n_pre, int n_post, int n_spikes,
                                 double* w, double* times, int* sources)
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
        sources[idx] = std::uniform_int_distribution<int>(0, n_pre - 1)(rng);

        if (idx > 0 && times[idx] <= times[idx - 1])
            times[idx] = times[idx - 1] + dt;
    }
}

static void GenerateBurstData(int n_pre, int n_post, int n_spikes,
                               double* w, double* times, int* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    int spike_idx = 0;
    for (int nrn = 0; nrn < n_pre && spike_idx < n_spikes; nrn++) {
        /* Increased spacing between bursts to allow voltage decay */
        double burst_start = 0.05 + nrn * 0.15; /* Was 0.1 */
        for (int b = 0; burst_start < 1.0 && spike_idx < n_spikes; b++) {
            for (int s = 0; s < 5 && spike_idx < n_spikes; s++) {
                times[spike_idx] = burst_start + s * 0.002;
                sources[spike_idx] = nrn;
                spike_idx++;
            }
            /* Increased inter-burst gap to ensure voltage decay */
            burst_start += 0.2; /* Was 0.12 - need >5*tau_mem = 0.1s */
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
using DataGenerator = void(*)(int, int, int, double*, double*, int*);
static DataGenerator kGen[] = {GenerateUniformData, GeneratePoissonData, GenerateBurstData};

// ============================================================
// Benchmark: ComputeSpikes (forward pass)
// ============================================================

static void BM_ComputeSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

#ifdef USE_CPP_WRAPPER
    // Pre-construct Eigen objects outside the loop (avoids per-iteration Map + copy overhead)
    Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes);
    Spikes input_spikes(e_times, e_sources);
#endif

    // Warmup iteration to prime caches and OpenMP thread pools
{
#ifdef USE_CPP_WRAPPER
        Spikes warmup = compute_spikes(w_map, input_spikes, 1.0, 0.02, 0.005);
        benchmark::DoNotOptimize(warmup.times);
#else
        EventPropInput fi = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                    .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                    .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
        SpikeOutput warmup = eventprop_compute_spikes(fi);
        benchmark::DoNotOptimize(warmup.times);
        eventprop_free_spike_output(&warmup);
#endif
    }

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        Spikes output = compute_spikes(w_map, input_spikes, 1.0, 0.02, 0.005);
        benchmark::DoNotOptimize(output.times);
        benchmark::DoNotOptimize(output.currents);
#else
        EventPropInput fi = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                    .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                    .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
        SpikeOutput output = eventprop_compute_spikes(fi);
        benchmark::DoNotOptimize(output.times);
        benchmark::DoNotOptimize(output.currents);
        eventprop_free_spike_output(&output);
#endif
    }

#ifdef USE_CPP_WRAPPER
    Spikes last = compute_spikes(w_map, input_spikes, 1.0, 0.02, 0.005);
#else
    EventPropInput fi = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    SpikeOutput last = eventprop_compute_spikes(fi);
#endif
#ifdef USE_CPP_WRAPPER
    state.counters["output_spikes"] = benchmark::Counter(last.n_spikes);
#else
    state.counters["output_spikes"] = benchmark::Counter(last.n_spikes);
#endif

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
// Medium — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeSpikes, Medium_uniform,
    {"Medium", 200, 80, 500}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeSpikes/Medium/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeSpikes, Medium_poisson,
    {"Medium", 200, 80, 500}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeSpikes/Medium/poisson/default");
// Large — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeSpikes, Large_uniform,
    {"Large", 500, 200, 1000}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeSpikes/Large/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeSpikes, Large_poisson,
    {"Large", 500, 200, 1000}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeSpikes/Large/poisson/default");
// XL — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeSpikes, XL_uniform,
    {"XL", 2000, 500, 5000}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeSpikes/XL/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeSpikes, XL_poisson,
    {"XL", 2000, 500, 5000}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeSpikes/XL/poisson/default");

// ============================================================
// Benchmark: BackwardSpikes (backward pass)
// Requires post-spike data from a forward pass.
// ============================================================

static void BM_BackwardSpikes(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int> sources(sz.n_spikes);
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
        // backward() returns a new RowMatrixXd — no need to pre-allocate/zero
        std::pair<RowMatrixXd, Eigen::ArrayXd> result = backward(input_spikes, post_spikes, w_map, 1.0, 0.02, 0.005);
        benchmark::DoNotOptimize(result.first.data());
    }

    state.counters["output_spikes"] = benchmark::Counter(post_spikes.n_spikes);
#else
    // C library: Run forward pass to get post-spike data
    EventPropInput fi = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    SpikeOutput fwd_result = eventprop_compute_spikes(fi);

    int n_post_spikes = fwd_result.n_spikes;

    // Write errors into fwd_result.errors (allocated by forward pass)
    for (int i = 0; i < n_post_spikes; i++) fwd_result.errors[i] = 1.0;

    EventPropInput base = { .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                              .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                              .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0 };

    for (auto _ : state) {
        BackwardOutput result = eventprop_backward_spikes(base, fwd_result);

        benchmark::DoNotOptimize(result.grad);
        eventprop_free_backward_output(&result);
    }

    state.counters["output_spikes"] = benchmark::Counter(n_post_spikes);
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
// Medium — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_BackwardSpikes, Medium_uniform,
    {"Medium", 200, 80, 500}, GenerateUniformData)->UseRealTime()->Name(
    "BackwardSpikes/Medium/uniform/default");
BENCHMARK_CAPTURE(BM_BackwardSpikes, Medium_poisson,
    {"Medium", 200, 80, 500}, GeneratePoissonData)->UseRealTime()->Name(
    "BackwardSpikes/Medium/poisson/default");
// Large — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_BackwardSpikes, Large_uniform,
    {"Large", 500, 200, 1000}, GenerateUniformData)->UseRealTime()->Name(
    "BackwardSpikes/Large/uniform/default");
BENCHMARK_CAPTURE(BM_BackwardSpikes, Large_poisson,
    {"Large", 500, 200, 1000}, GeneratePoissonData)->UseRealTime()->Name(
    "BackwardSpikes/Large/poisson/default");
// XL — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_BackwardSpikes, XL_uniform,
    {"XL", 2000, 500, 5000}, GenerateUniformData)->UseRealTime()->Name(
    "BackwardSpikes/XL/uniform/default");
BENCHMARK_CAPTURE(BM_BackwardSpikes, XL_poisson,
    {"XL", 2000, 500, 5000}, GeneratePoissonData)->UseRealTime()->Name(
    "BackwardSpikes/XL/poisson/default");
// ============================================================
// Benchmark: FullTrainingStep (composite — Sums → Spikes → Backward)
// Measures the full forward + backward training pipeline end-to-end.
// Catches real-world regressions and measures cache effects between chained ops.
// ============================================================

static void BM_FullTrainingStep(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int> sources(sz.n_spikes);
    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

#ifdef USE_CPP_WRAPPER
    Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes);
    Spikes input_spikes(e_times, e_sources);

    // Warmup: full pipeline to prime caches and OpenMP thread pools
    {
        auto sums = compute_sums(w_map, input_spikes, 0.02, 0.005);
        benchmark::DoNotOptimize(sums.first.data());
        Spikes post = compute_spikes(w_map, input_spikes, 1.0, 0.02, 0.005);
        benchmark::DoNotOptimize(post.times);
    }

    for (auto _ : state) {
        // Forward: sums + spikes
        auto sums = compute_sums(w_map, input_spikes, 0.02, 0.005);
        Spikes post = compute_spikes(w_map, input_spikes, 1.0, 0.02, 0.005);

        // Set errors for backward pass
        for (int i = 0; i < post.n_spikes; i++)
            post.set_error(i, -1.0);

        // Backward: gradient computation
        auto result = backward(input_spikes, post, w_map, 1.0, 0.02, 0.005);
        benchmark::DoNotOptimize(result.first.data());
        benchmark::DoNotOptimize(sums.first.data());
    }
#else
    // C library: warmup forward pass to get post-spike data
    EventPropInput fi_warmup = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    SpikeOutput fwd_warmup = eventprop_compute_spikes(fi_warmup);
    benchmark::DoNotOptimize(fwd_warmup.times);
    eventprop_free_spike_output(&fwd_warmup);

    EventPropInput base_input = {
        .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
        .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
        .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
    };

    for (auto _ : state) {
        // Forward: sums + spikes
        SumsOutput sums = eventprop_compute_sums(base_input);

        SpikeOutput fwd = eventprop_compute_spikes(base_input);

        // Write errors into fwd.errors (allocated by forward pass)
        for (int i = 0; i < fwd.n_spikes; i++) fwd.errors[i] = -1.0;

        // Backward: gradient computation
        BackwardOutput result = eventprop_backward_spikes(base_input, fwd);

        benchmark::DoNotOptimize(result.grad);
        benchmark::DoNotOptimize(sums.sum0);

        eventprop_free_backward_output(&result);
        eventprop_free_spike_output(&fwd);
        eventprop_free_sums_output(&sums);
    }
#endif

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define FULLTRAINING_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_FullTrainingStep, SizeName##_uniform,                                 \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(                    \
        "FullTrainingStep/" #SizeName "/uniform/default");                                      \
    BENCHMARK_CAPTURE(BM_FullTrainingStep, SizeName##_poisson,                                 \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(                    \
        "FullTrainingStep/" #SizeName "/poisson/default");                                      \
    BENCHMARK_CAPTURE(BM_FullTrainingStep, SizeName##_burst,                                   \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(                      \
        "FullTrainingStep/" #SizeName "/burst/default")

FULLTRAINING_BENCH(Tiny,    10,   5,   20);
FULLTRAINING_BENCH(Small,   50,   20,  100);
// Medium — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_FullTrainingStep, Medium_uniform,
    {"Medium", 200, 80, 500}, GenerateUniformData)->UseRealTime()->Name(
    "FullTrainingStep/Medium/uniform/default");
BENCHMARK_CAPTURE(BM_FullTrainingStep, Medium_poisson,
    {"Medium", 200, 80, 500}, GeneratePoissonData)->UseRealTime()->Name(
    "FullTrainingStep/Medium/poisson/default");
// Large — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_FullTrainingStep, Large_uniform,
    {"Large", 500, 200, 1000}, GenerateUniformData)->UseRealTime()->Name(
    "FullTrainingStep/Large/uniform/default");
BENCHMARK_CAPTURE(BM_FullTrainingStep, Large_poisson,
    {"Large", 500, 200, 1000}, GeneratePoissonData)->UseRealTime()->Name(
    "FullTrainingStep/Large/poisson/default");
// XL — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_FullTrainingStep, XL_uniform,
    {"XL", 2000, 500, 5000}, GenerateUniformData)->UseRealTime()->Name(
    "FullTrainingStep/XL/uniform/default");
BENCHMARK_CAPTURE(BM_FullTrainingStep, XL_poisson,
    {"XL", 2000, 500, 5000}, GeneratePoissonData)->UseRealTime()->Name(
    "FullTrainingStep/XL/poisson/default");

// ============================================================
// Benchmark: BatchTrainingStep (multi-sample forward + backward)
// Tests batch processing at realistic ML batch sizes (4-32 samples).
// Matches actual Python/ML usage patterns.
// ============================================================

static void BM_BatchTrainingStep(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    int batch_size = state.range(0); // batch size from benchmark args
    
    std::vector<double> w(sz.n_pre * sz.n_post);
    gen(sz.n_pre, sz.n_post, 0, w.data(), nullptr, nullptr); // weights only
    
#ifdef USE_CPP_WRAPPER
    Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
    
    // Generate per-sample data
    std::vector<Spikes> input_batch(batch_size);
    for (int b = 0; b < batch_size; b++) {
        std::vector<double> times(sz.n_spikes);
        std::vector<int> sources(sz.n_spikes);
        // Vary RNG seed per sample
        std::mt19937 rng(42 + b);
        std::uniform_real_distribution<double> w_dist(0.5, 2.0);
        for (int i = 0; i < sz.n_spikes; i++) {
            times[i] = (i + 1) / static_cast<double>(sz.n_spikes + 1);
            sources[i] = std::uniform_int_distribution<int>(0, sz.n_pre - 1)(rng);
        }
        input_batch[b] = Spikes(Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes),
                               Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes));
    }

    // Collect results inside the loop and free after — avoids allocation
    // overhead per iteration for fair comparison with C backend.
    std::vector<std::vector<Spikes>> fwd_results;
    fwd_results.reserve(state.iterations());

    for (auto _ : state) {
        // Forward: batch compute spikes
        std::vector<Spikes> post_batch = compute_spikes_batch(w_map, input_batch, 1.0, 0.02, 0.005);
        
        // Set errors for backward pass
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < post_batch[b].n_spikes; i++)
                post_batch[b].set_error(i, -1.0);
        }

        // Backward: batch gradient computation (accumulates into shared buffer)
        RowMatrixXd gradient(sz.n_pre, sz.n_post);
        backward_spikes_batch(input_batch, post_batch, w_map, gradient, 1.0, 0.02, 0.005);
        benchmark::DoNotOptimize(gradient.data());

        fwd_results.push_back(std::move(post_batch));
    }
#else
    // C library: construct EventPropInput[] from flat arrays
    std::vector<EventPropInput> inputs(batch_size);
    for (int b = 0; b < batch_size; b++) {
        double* times = new double[sz.n_spikes];
        int* sources = new int[sz.n_spikes];
        std::mt19937 rng(42 + b);
        for (int i = 0; i < sz.n_spikes; i++) {
            times[i] = (i + 1) / static_cast<double>(sz.n_spikes + 1);
            sources[i] = std::uniform_int_distribution<int>(0, sz.n_pre - 1)(rng);
        }
        inputs[b] = {w.data(), sz.n_pre, sz.n_post, times, sources, sz.n_spikes, 0.02, 0.005, 1.0};
    }

    // Collect results inside the loop and free after — avoids malloc/free
    // overhead per iteration while keeping the C library's allocation philosophy.
    std::vector<SpikesBatch> fwd_results;
    std::vector<BackwardsBatch> bwd_results;
    fwd_results.reserve(state.iterations());
    bwd_results.reserve(state.iterations());

    for (auto _ : state) {
        SpikesBatch fwd_batch = eventprop_compute_spikes_batch(inputs.data(), batch_size);

        // Write errors into fwd_batch.spikes[b].errors (allocated by forward pass)
        for (int b = 0; b < batch_size; b++)
            for (int i = 0; i < fwd_batch.spikes[b].n_spikes; i++)
                fwd_batch.spikes[b].errors[i] = -1.0;

        BackwardsBatch result_batch = eventprop_backward_spikes_batch(inputs.data(), fwd_batch);

        benchmark::DoNotOptimize(result_batch.outputs[0].grad);

        fwd_results.push_back(std::move(fwd_batch));
        bwd_results.push_back(std::move(result_batch));
    }

    // Free all results after the benchmark loop
    for (auto& r : bwd_results) eventprop_free_backwards_batch(&r);
    for (auto& r : fwd_results) eventprop_free_spikes_batch(&r);
    for (int b = 0; b < batch_size; b++) {
        delete[] inputs[b].times;
        delete[] inputs[b].sources;
    }
#endif

    state.SetItemsProcessed(state.iterations() * batch_size * sz.n_spikes);
}

#define BATCHTRAINING_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_BatchTrainingStep, SizeName##_uniform,                                 \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->ArgNames({"batch_size"}) \
        ->DenseRange(4, 32, 4)->Name("BatchTrainingStep/" #SizeName "/uniform/default")

BATCHTRAINING_BENCH(Large,  500,  200, 1000);
BATCHTRAINING_BENCH(Small,   50,   20,  100);
BATCHTRAINING_BENCH(Medium,  200,  80,  500);

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
    std::vector<int> sources(n_spikes);

    GenerateUniformData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        Eigen::Map<RowMatrixXd const> w_map(w.data(), n_pre, n_post);
        Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), n_spikes);
        Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), n_spikes);
        Spikes output = compute_spikes(w_map, Spikes(e_times, e_sources), v_th, 0.02, 0.005);
        benchmark::DoNotOptimize(output.times);
        benchmark::DoNotOptimize(output.currents);
#else
        EventPropInput fi = {.w=w.data(), .n_pre=n_pre, .n_post=n_post,
                    .times=times.data(), .sources=sources.data(), .n_spikes=n_spikes,
                    .tau_mem=0.02, .tau_syn=0.005, .v_th=v_th};
        SpikeOutput output = eventprop_compute_spikes(fi);
        benchmark::DoNotOptimize(output.times);
        benchmark::DoNotOptimize(output.currents);
        eventprop_free_spike_output(&output);
#endif
    }

    state.SetItemsProcessed(state.iterations() * n_spikes);
}

BENCHMARK(BM_ComputeSpikesVth)->ArgNames({"v_th"})
    
    ->DenseRange(5, 20, 5);

// BackwardSpikes with v_th variation (Small size, uniform)
static void BM_BackwardSpikesVth(benchmark::State& state)
{
    double v_th = static_cast<double>(state.range(0)) / 10.0;
    int n_pre = 50, n_post = 20, n_spikes = 100;

    std::vector<double> w(n_pre * n_post);
    std::vector<double> times(n_spikes);
    std::vector<int> sources(n_spikes);

    GenerateUniformData(n_pre, n_post, n_spikes, w.data(), times.data(), sources.data());

#ifdef USE_CPP_WRAPPER
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
#else
    EventPropInput fi = {.w=w.data(), .n_pre=n_pre, .n_post=n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=n_spikes,
                .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    SpikeOutput fwd = eventprop_compute_spikes(fi);

    // Write errors into fwd.errors (allocated by forward pass)
    for (int i = 0; i < fwd.n_spikes; i++) fwd.errors[i] = -1.0;
#endif

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        gradient = backward(input, post, w_map, v_th, 0.02, 0.005);
        benchmark::DoNotOptimize(gradient.first.data());
#else
        EventPropInput base_input = {.w=w.data(), .n_pre=n_pre, .n_post=n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=n_spikes,
            .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
        BackwardOutput result = eventprop_backward_spikes(base_input, fwd);
        benchmark::DoNotOptimize(result.grad);
        eventprop_free_backward_output(&result);
#endif
    }

#ifdef USE_CPP_WRAPPER
    // Nothing to free for Eigen
#else
    eventprop_free_spike_output(&fwd);
#endif

    state.SetItemsProcessed(state.iterations() * n_spikes);
}

BENCHMARK(BM_BackwardSpikesVth)->ArgNames({"v_th"})
    
    ->DenseRange(5, 20, 5);

// ============================================================
// Benchmark: ComputeSums (cumulative sums)
// ============================================================

static void BM_ComputeSums(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int> sources(sz.n_spikes);

    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

#ifdef USE_CPP_WRAPPER
    Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes);
    Spikes input_spikes(e_times, e_sources);
#endif

    // Warmup iteration to prime caches and OpenMP thread pools
{
#ifdef USE_CPP_WRAPPER
        auto warmup = compute_sums(w_map, input_spikes, 0.02, 0.005);
        benchmark::DoNotOptimize(warmup.first.data());
#else
        EventPropInput base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
        };
        SumsOutput warmup = eventprop_compute_sums(base_input);
        benchmark::DoNotOptimize(warmup.sum0);
        eventprop_free_sums_output(&warmup);
#endif
    }

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        std::pair<RowMatrixXd, RowMatrixXd> sums = compute_sums(w_map, input_spikes, 0.02, 0.005);
        benchmark::DoNotOptimize(sums.first.data());
        benchmark::DoNotOptimize(sums.second.data());
#else
        EventPropInput base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
        };
        SumsOutput output = eventprop_compute_sums(base_input);
        benchmark::DoNotOptimize(output.sum0);
        benchmark::DoNotOptimize(output.sum1);
        eventprop_free_sums_output(&output);
#endif
    }

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define COMPUTESUMS_BENCH(SizeName, NP, NPO, NS)                                          \
    BENCHMARK_CAPTURE(BM_ComputeSums, SizeName##_uniform,                                  \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(   \
        "ComputeSums/" #SizeName "/uniform/default");                                      \
    BENCHMARK_CAPTURE(BM_ComputeSums, SizeName##_poisson,                                  \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(   \
        "ComputeSums/" #SizeName "/poisson/default");                                      \
    BENCHMARK_CAPTURE(BM_ComputeSums, SizeName##_burst,                                    \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(   \
        "ComputeSums/" #SizeName "/burst/default")

COMPUTESUMS_BENCH(Tiny,    10,   5,   20);
COMPUTESUMS_BENCH(Small,   50,   20,  100);
// Medium — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeSums, Medium_uniform,
    {"Medium", 200, 80, 500}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeSums/Medium/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeSums, Medium_poisson,
    {"Medium", 200, 80, 500}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeSums/Medium/poisson/default");
// Large — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeSums, Large_uniform,
    {"Large", 500, 200, 1000}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeSums/Large/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeSums, Large_poisson,
    {"Large", 500, 200, 1000}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeSums/Large/poisson/default");
// XL — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeSums, XL_uniform,
    {"XL", 2000, 500, 5000}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeSums/XL/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeSums, XL_poisson,
    {"XL", 2000, 500, 5000}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeSums/XL/poisson/default");

// ============================================================
// Benchmark: ComputeMaxima (membrane potential maxima)
// ============================================================

static void BM_ComputeMaxima(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int> sources(sz.n_spikes);

    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

#ifdef USE_CPP_WRAPPER
    Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes);
    Spikes input_spikes(e_times, e_sources);
#endif

    // Warmup iteration to prime caches and OpenMP thread pools
{
#ifdef USE_CPP_WRAPPER
        auto warmup = compute_maxima(w_map, input_spikes, 0.02, 0.005);
        benchmark::DoNotOptimize(warmup.times.data());
#else
        EventPropInput base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
        };
        MaximaOutput warmup = eventprop_compute_maxima(base_input);
        benchmark::DoNotOptimize(warmup.times);
        eventprop_free_maxima_output(&warmup);
#endif
    }

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        Maxima maxima = compute_maxima(w_map, input_spikes, 0.02, 0.005);
        benchmark::DoNotOptimize(maxima.times.data());
        benchmark::DoNotOptimize(maxima.values.data());
#else
        EventPropInput base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
        };
        MaximaOutput output = eventprop_compute_maxima(base_input);
        benchmark::DoNotOptimize(output.times);
        benchmark::DoNotOptimize(output.values);
        eventprop_free_maxima_output(&output);
#endif
    }

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define COMPUTEMAXIMA_BENCH(SizeName, NP, NPO, NS)                                        \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_uniform,                                \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(   \
        "ComputeMaxima/" #SizeName "/uniform/default");                                    \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_poisson,                                \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(   \
        "ComputeMaxima/" #SizeName "/poisson/default");                                    \
    BENCHMARK_CAPTURE(BM_ComputeMaxima, SizeName##_burst,                                  \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(   \
        "ComputeMaxima/" #SizeName "/burst/default")

COMPUTEMAXIMA_BENCH(Tiny,    10,   5,   20);
COMPUTEMAXIMA_BENCH(Small,   50,   20,  100);
// Medium — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeMaxima, Medium_uniform,
    {"Medium", 200, 80, 500}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeMaxima/Medium/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeMaxima, Medium_poisson,
    {"Medium", 200, 80, 500}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeMaxima/Medium/poisson/default");
// Large — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeMaxima, Large_uniform,
    {"Large", 500, 200, 1000}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeMaxima/Large/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeMaxima, Large_poisson,
    {"Large", 500, 200, 1000}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeMaxima/Large/poisson/default");
// XL — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_ComputeMaxima, XL_uniform,
    {"XL", 2000, 500, 5000}, GenerateUniformData)->UseRealTime()->Name(
    "ComputeMaxima/XL/uniform/default");
BENCHMARK_CAPTURE(BM_ComputeMaxima, XL_poisson,
    {"XL", 2000, 500, 5000}, GeneratePoissonData)->UseRealTime()->Name(
    "ComputeMaxima/XL/poisson/default");

// ============================================================
// Benchmark: BackwardMaxima (backward pass via maxima)
// ============================================================

static void BM_BackwardMaxima(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int> sources(sz.n_spikes);

    gen(sz.n_pre, sz.n_post, sz.n_spikes, w.data(), times.data(), sources.data());

#ifdef USE_CPP_WRAPPER
    Eigen::Map<RowMatrixXd const> w_map(w.data(), sz.n_pre, sz.n_post);
    Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes);
    Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes);
    Maxima maxima = compute_maxima(w_map, Spikes(e_times, e_sources), 0.02, 0.005);
    
    // Set errors for all maxima (gradient descent)
    for (int i = 0; i < sz.n_post; i++)
        maxima.set_error(i, -1.0);

    Spikes input = Spikes(e_times, e_sources);
    std::pair<RowMatrixXd, Eigen::ArrayXd> gradient;
#else
    EventPropInput base_input = {
        .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
        .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
        .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
    };
    MaximaOutput maxima = eventprop_compute_maxima(base_input);

    std::vector<double> maxima_errors(sz.n_post, -1.0);
#endif

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        gradient = backward(input, maxima, w_map, 0.02, 0.005);
        benchmark::DoNotOptimize(gradient.first.data());
#else
        BackwardOutput result = eventprop_backward_maxima(base_input, maxima.times, maxima_errors.data());
        benchmark::DoNotOptimize(result.grad[0]);
        eventprop_free_backward_output(&result);
#endif
    }

#ifdef USE_CPP_WRAPPER
    // Nothing to free for Eigen
#else
    eventprop_free_maxima_output(&maxima);
#endif

    state.SetItemsProcessed(state.iterations() * sz.n_spikes);
}

#define BACKWARDMAXIMA_BENCH(SizeName, NP, NPO, NS)                                        \
    BENCHMARK_CAPTURE(BM_BackwardMaxima, SizeName##_uniform,                                \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(   \
        "BackwardMaxima/" #SizeName "/uniform/default");                                    \
    BENCHMARK_CAPTURE(BM_BackwardMaxima, SizeName##_poisson,                                \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(   \
        "BackwardMaxima/" #SizeName "/poisson/default");                                    \
    BENCHMARK_CAPTURE(BM_BackwardMaxima, SizeName##_burst,                                  \
        {#SizeName, NP, NPO, NS}, GenerateBurstData)->UseRealTime()->Name(   \
        "BackwardMaxima/" #SizeName "/burst/default")

BACKWARDMAXIMA_BENCH(Tiny,    10,   5,   20);
BACKWARDMAXIMA_BENCH(Small,   50,   20,  100);
// Medium — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_BackwardMaxima, Medium_uniform,
    {"Medium", 200, 80, 500}, GenerateUniformData)->UseRealTime()->Name(
    "BackwardMaxima/Medium/uniform/default");
BENCHMARK_CAPTURE(BM_BackwardMaxima, Medium_poisson,
    {"Medium", 200, 80, 500}, GeneratePoissonData)->UseRealTime()->Name(
    "BackwardMaxima/Medium/poisson/default");
// Large — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_BackwardMaxima, Large_uniform,
    {"Large", 500, 200, 1000}, GenerateUniformData)->UseRealTime()->Name(
    "BackwardMaxima/Large/uniform/default");
BENCHMARK_CAPTURE(BM_BackwardMaxima, Large_poisson,
    {"Large", 500, 200, 1000}, GeneratePoissonData)->UseRealTime()->Name(
    "BackwardMaxima/Large/poisson/default");
// XL — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_BackwardMaxima, XL_uniform,
    {"XL", 2000, 500, 5000}, GenerateUniformData)->UseRealTime()->Name(
    "BackwardMaxima/XL/uniform/default");
BENCHMARK_CAPTURE(BM_BackwardMaxima, XL_poisson,
    {"XL", 2000, 500, 5000}, GeneratePoissonData)->UseRealTime()->Name(
    "BackwardMaxima/XL/poisson/default");

// ============================================================
// Benchmark: Dropout (batch)
// ============================================================

static void BM_Dropout(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    int batch_size = 8;
    int total_spikes = sz.n_spikes * batch_size;

    std::vector<double> times(total_spikes);
    std::vector<int> sources(total_spikes);

    // Generate per-sample data
    for (int b = 0; b < batch_size; b++) {
        int offset = b * sz.n_spikes;
        std::mt19937 rng(42 + b);
        std::uniform_real_distribution<double> t_dist(0.0, 1.0);
        std::uniform_int_distribution<int> s_dist(0, sz.n_pre - 1);
        for (int i = 0; i < sz.n_spikes; i++) {
            times[offset + i] = t_dist(rng);
            sources[offset + i] = s_dist(rng);
        }
    }

#ifdef USE_CPP_WRAPPER
    // Pre-construct Spikes vector for C++
    std::vector<Spikes> cpp_spikes;
    cpp_spikes.reserve(batch_size);
    for (int b = 0; b < batch_size; b++) {
        int offset = b * sz.n_spikes;
        Eigen::ArrayXd e_times = Eigen::Map<Eigen::ArrayXd>(times.data() + offset, sz.n_spikes);
        Eigen::ArrayXi e_sources = Eigen::Map<Eigen::ArrayXi>(sources.data() + offset, sz.n_spikes);
        cpp_spikes.emplace_back(e_times, e_sources);
    }
#endif

#ifndef USE_CPP_WRAPPER
    // Setup C flat-array input
    std::vector<int> n_per_sample(batch_size, sz.n_spikes);
#endif

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        auto dropped = dropout_spikes(cpp_spikes, 0.3, 42);
        benchmark::DoNotOptimize(dropped);
#else
        DropoutBatch dropped = eventprop_dropout_spikes_batch(
            times.data(), sources.data(), n_per_sample.data(), batch_size, 0.3, 42);
        int total = 0;
        for (int b = 0; b < batch_size; b++) total += dropped.outputs[b].n_spikes;
        benchmark::DoNotOptimize(total);
        eventprop_free_dropout_batch(&dropped);
#endif
    }

    state.SetItemsProcessed(state.iterations() * total_spikes);
}

#define DROPOUT_BENCH(SizeName, NP, NPO, NS)                                               \
    BENCHMARK_CAPTURE(BM_Dropout, SizeName##_uniform,                                       \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->Name(   \
        "Dropout/" #SizeName "/uniform");                                                  \
    BENCHMARK_CAPTURE(BM_Dropout, SizeName##_poisson,                                       \
        {#SizeName, NP, NPO, NS}, GeneratePoissonData)->UseRealTime()->Name(   \
        "Dropout/" #SizeName "/poisson")

DROPOUT_BENCH(Tiny,    10,   5,   20);
DROPOUT_BENCH(Small,   50,   20,  100);
// Medium — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_Dropout, Medium_uniform,
    {"Medium", 200, 80, 500}, GenerateUniformData)->UseRealTime()->Name(
    "Dropout/Medium/uniform");
BENCHMARK_CAPTURE(BM_Dropout, Medium_poisson,
    {"Medium", 200, 80, 500}, GeneratePoissonData)->UseRealTime()->Name(
    "Dropout/Medium/poisson");
// Large — burst excluded (Toms748 throws on MSVC)
BENCHMARK_CAPTURE(BM_Dropout, Large_uniform,
    {"Large", 500, 200, 1000}, GenerateUniformData)->UseRealTime()->Name(
    "Dropout/Large/uniform");
BENCHMARK_CAPTURE(BM_Dropout, Large_poisson,
    {"Large", 500, 200, 1000}, GeneratePoissonData)->UseRealTime()->Name(
    "Dropout/Large/poisson");

BENCHMARK_MAIN();
