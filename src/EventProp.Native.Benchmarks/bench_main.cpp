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
// C library functions (consolidated header)
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
        ForwardInputSpikes spikes_input = {
            .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                    .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                    .tau_mem=0.02, .tau_syn=0.005},
            .v_th = 1.0
        };
        SpikeOutput warmup = eventprop_compute_spikes(&spikes_input);
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
        ForwardInputSpikes spikes_input = {
            .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                    .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                    .tau_mem=0.02, .tau_syn=0.005},
            .v_th = 1.0
        };
        SpikeOutput output = eventprop_compute_spikes(&spikes_input);
        benchmark::DoNotOptimize(output.times);
        benchmark::DoNotOptimize(output.currents);
        eventprop_free_spike_output(&output);
#endif
    }

#ifdef USE_CPP_WRAPPER
    Spikes last = compute_spikes(w_map, input_spikes, 1.0, 0.02, 0.005);
#else
    ForwardInputSpikes spikes_input = {
        .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                .tau_mem=0.02, .tau_syn=0.005},
        .v_th = 1.0
    };
    SpikeOutput last = eventprop_compute_spikes(&spikes_input);
#endif
    state.counters["output_spikes"] = benchmark::Counter(last.n_spikes);

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
        // backward() returns a new RowMatrixXd — no need to pre-allocate/zero
        std::pair<RowMatrixXd, Eigen::ArrayXd> result = backward(input_spikes, post_spikes, w_map, 1.0, 0.02, 0.005);
        benchmark::DoNotOptimize(result.first.data());
    }

    state.counters["output_spikes"] = benchmark::Counter(post_spikes.n_spikes);
#else
    // C library: Run forward pass to get post-spike data
    ForwardInputSpikes spikes_input = {
        .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                .tau_mem=0.02, .tau_syn=0.005},
        .v_th = 1.0
    };
    SpikeOutput fwd_result = eventprop_compute_spikes(&spikes_input);

    int n_post_spikes = fwd_result.n_spikes;
    
    // Post-spike errors (all 1.0)
    std::vector<double> post_errors(n_post_spikes, 1.0);

    for (auto _ : state) {
        SpikeInput input = { .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes };
        ForwardInputBase base = { .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                                  .tau_mem=0.02, .tau_syn=0.005 };
        BackwardSpikesInput backward_input = {
            .base = {.input=&input, .base=&base},
            .fwd_times=fwd_result.times, .fwd_sources=fwd_result.sources,
            .fwd_currents=fwd_result.currents, .post_errors=post_errors.data(),
            .n_post_spikes=n_post_spikes
        };
        BackwardSpikesOutput result = eventprop_backward_spikes(&backward_input);

        benchmark::DoNotOptimize(result.grad);
        eventprop_free_backward_spikes_output(&result);
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
BACKWARDSPIKES_BENCH(Medium,  200,  80,  500);
BACKWARDSPIKES_BENCH(Large,   500,  200, 1000);
BACKWARDSPIKES_BENCH(XL,      2000, 500, 5000);
// ============================================================
// Benchmark: FullTrainingStep (composite — Sums → Spikes → Backward)
// Measures the full forward + backward training pipeline end-to-end.
// Catches real-world regressions and measures cache effects between chained ops.
// ============================================================

static void BM_FullTrainingStep(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);
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
    ForwardInputSpikes spikes_input = {
        .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                .tau_mem=0.02, .tau_syn=0.005},
        .v_th = 1.0
    };
    SpikeOutput fwd_warmup = eventprop_compute_spikes(&spikes_input);
    benchmark::DoNotOptimize(fwd_warmup.times);
    eventprop_free_spike_output(&fwd_warmup);

    for (auto _ : state) {
        // Forward: sums + spikes
        ForwardInputBase base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005
        };
        SumsOutput sums = eventprop_compute_sums(&base_input);

        ForwardInputSpikes spikes_input = {
            .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                    .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
                    .tau_mem=0.02, .tau_syn=0.005},
            .v_th = 1.0
        };
        SpikeOutput fwd = eventprop_compute_spikes(&spikes_input);

        // Set errors for backward pass (all -1.0)
        std::vector<double> post_errors(fwd.n_spikes, -1.0);

        // Backward: gradient computation
        SpikeInput input = { .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes };
        ForwardInputBase base = { .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                                  .tau_mem=0.02, .tau_syn=0.005 };
        BackwardSpikesInput backward_input = {
            .base = {.input=&input, .base=&base},
            .fwd_times=fwd.times, .fwd_sources=fwd.sources,
            .fwd_currents=fwd.currents, .post_errors=post_errors.data(),
            .n_post_spikes=fwd.n_spikes
        };
        BackwardSpikesOutput result = eventprop_backward_spikes(&backward_input);

        benchmark::DoNotOptimize(result.grad);
        benchmark::DoNotOptimize(sums.sum0);

        eventprop_free_backward_spikes_output(&result);
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
FULLTRAINING_BENCH(Medium,  200,  80,  500);
FULLTRAINING_BENCH(Large,   500,  200, 1000);
FULLTRAINING_BENCH(XL,      2000, 500, 5000);

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
        std::vector<int32_t> sources(sz.n_spikes);
        // Vary RNG seed per sample
        std::mt19937 rng(42 + b);
        std::uniform_real_distribution<double> w_dist(0.5, 2.0);
        for (int i = 0; i < sz.n_spikes; i++) {
            times[i] = (i + 1) / static_cast<double>(sz.n_spikes + 1);
            sources[i] = std::uniform_int_distribution<int32_t>(0, sz.n_pre - 1)(rng);
        }
        input_batch[b] = Spikes(Eigen::Map<Eigen::ArrayXd>(times.data(), sz.n_spikes),
                               Eigen::Map<Eigen::ArrayXi>(sources.data(), sz.n_spikes));
    }

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
    }
#else
    // C library: flat arrays + per-sample spike counts
    std::vector<double> times(batch_size * sz.n_spikes);
    std::vector<int32_t> sources(batch_size * sz.n_spikes);
    std::vector<int> n_spikes_per_sample(batch_size, sz.n_spikes);
    
    for (int b = 0; b < batch_size; b++) {
        std::mt19937 rng(42 + b);
        for (int i = 0; i < sz.n_spikes; i++) {
            times[b * sz.n_spikes + i] = (i + 1) / static_cast<double>(sz.n_spikes + 1);
            sources[b * sz.n_spikes + i] = std::uniform_int_distribution<int32_t>(0, sz.n_pre - 1)(rng);
        }
    }

    // Warmup: run forward pass to determine max output sizes
    {
        ForwardInputSpikes spikes_input = {
            .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                    .times=times.data(), .sources=sources.data(),
                    .tau_mem=0.02, .tau_syn=0.005},
            .v_th = 1.0
        };
        SpikeOutput* warmup = eventprop_compute_spikes_batch(&spikes_input,
            n_spikes_per_sample.data(), batch_size);
        int total_fwd = 0;
        for (int b = 0; b < batch_size; b++)
            total_fwd += warmup[b].n_spikes;
        // Pre-allocate buffers large enough for all samples
        std::vector<double> fwd_times(total_fwd);
        std::vector<int32_t> fwd_sources(total_fwd);
        std::vector<double> fwd_currents(total_fwd);
        std::vector<double> post_errors(total_fwd, -1.0);
        eventprop_free_batch_spike_output(warmup, batch_size);
    }

    // Pre-allocate buffers outside loop (reuse each iteration)
    std::vector<double> fwd_times(batch_size * sz.n_spikes);
    std::vector<int32_t> fwd_sources(batch_size * sz.n_spikes);
    std::vector<double> fwd_currents(batch_size * sz.n_spikes);
    std::vector<double> post_errors(batch_size * sz.n_spikes, -1.0);
    std::vector<int> n_fwd_spikes_per_sample(batch_size);

    for (auto _ : state) {
        // Forward: batch compute spikes
        ForwardInputSpikes spikes_input = {
            .base = {.w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                    .times=times.data(), .sources=sources.data(),
                    .tau_mem=0.02, .tau_syn=0.005},
            .v_th = 1.0
        };
        SpikeOutput* fwd_batch = eventprop_compute_spikes_batch(&spikes_input,
            n_spikes_per_sample.data(), batch_size);
        
        // Set errors for backward pass
        int total_fwd = 0;
        for (int b = 0; b < batch_size; b++)
            total_fwd += fwd_batch[b].n_spikes;
        std::vector<double> post_errors(total_fwd, -1.0);

        // Backward: batch gradient computation (pass fwd_batch directly, no copying)
        SpikeInput input = { .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes };
        ForwardInputBase base = { .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                                  .tau_mem=0.02, .tau_syn=0.005 };
        BackwardInputBase backward_base = { .input=&input, .base=&base };
        BackwardSpikesOutput* result_batch = eventprop_backward_spikes_batch(
            &backward_base, n_spikes_per_sample.data(),
            fwd_batch,  /* Pass forward outputs directly */
            post_errors.data(), batch_size);
        
        benchmark::DoNotOptimize(result_batch[0].grad);
        
        eventprop_free_backward_spikes_batch_output(result_batch, batch_size);
        eventprop_free_batch_spike_output(fwd_batch, batch_size);
    }
#endif

    state.SetItemsProcessed(state.iterations() * batch_size * sz.n_spikes);
}

#define BATCHTRAINING_BENCH(SizeName, NP, NPO, NS)                                              \
    BENCHMARK_CAPTURE(BM_BatchTrainingStep, SizeName##_uniform,                                 \
        {#SizeName, NP, NPO, NS}, GenerateUniformData)->UseRealTime()->ArgNames({"batch_size"}) \
        ->RangeMultiplier(1)->DenseRange(4, 32, 4)->Name("BatchTrainingStep/" #SizeName "/uniform/default")

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
    std::vector<int32_t> sources(n_spikes);

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
        ForwardInputSpikes spikes_input = {
            .base = {.w=w.data(), .n_pre=n_pre, .n_post=n_post,
                    .times=times.data(), .sources=sources.data(), .n_spikes=n_spikes,
                    .tau_mem=0.02, .tau_syn=0.005},
            .v_th = v_th
        };
        SpikeOutput output = eventprop_compute_spikes(&spikes_input);
        benchmark::DoNotOptimize(output.times);
        benchmark::DoNotOptimize(output.currents);
        eventprop_free_spike_output(&output);
#endif
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
    ForwardInputSpikes spikes_input = {
        .base = {.w=w.data(), .n_pre=n_pre, .n_post=n_post,
                .times=times.data(), .sources=sources.data(), .n_spikes=n_spikes,
                .tau_mem=0.02, .tau_syn=0.005},
        .v_th = 1.0
    };
    SpikeOutput fwd = eventprop_compute_spikes(&spikes_input);

    std::vector<double> post_errors(fwd.n_spikes, -1.0);
#endif

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        gradient = backward(input, post, w_map, v_th, 0.02, 0.005);
        benchmark::DoNotOptimize(gradient.first.data());
#else
        SpikeInput input = { .times=times.data(), .sources=sources.data(), .n_spikes=n_spikes };
        ForwardInputBase base = { .w=w.data(), .n_pre=n_pre, .n_post=n_post,
                                  .tau_mem=0.02, .tau_syn=0.005 };
        BackwardSpikesInput backward_input = {
            .base = {.input=&input, .base=&base},
            .fwd_times=fwd.times, .fwd_sources=fwd.sources,
            .fwd_currents=fwd.currents, .post_errors=post_errors.data(),
            .n_post_spikes=fwd.n_spikes
        };
        BackwardSpikesOutput result = eventprop_backward_spikes(&backward_input);
        benchmark::DoNotOptimize(result.grad);
        eventprop_free_backward_spikes_output(&result);
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
    ->RangeMultiplier(1)
    ->DenseRange(5, 20, 5);

// ============================================================
// Benchmark: ComputeSums (cumulative sums)
// ============================================================

static void BM_ComputeSums(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);

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
        ForwardInputBase base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005
        };
        SumsOutput warmup = eventprop_compute_sums(&base_input);
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
        ForwardInputBase base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005
        };
        SumsOutput output = eventprop_compute_sums(&base_input);
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
COMPUTESUMS_BENCH(Medium,  200,  80,  500);
COMPUTESUMS_BENCH(Large,   500,  200, 1000);
COMPUTESUMS_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: ComputeMaxima (membrane potential maxima)
// ============================================================

static void BM_ComputeMaxima(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);

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
        ForwardInputBase base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005
        };
        MaximaOutput warmup = eventprop_compute_maxima(&base_input);
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
        ForwardInputBase base_input = {
            .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
            .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
            .tau_mem=0.02, .tau_syn=0.005
        };
        MaximaOutput output = eventprop_compute_maxima(&base_input);
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
COMPUTEMAXIMA_BENCH(Medium,  200,  80,  500);
COMPUTEMAXIMA_BENCH(Large,   500,  200, 1000);
COMPUTEMAXIMA_BENCH(XL,      2000, 500, 5000);

// ============================================================
// Benchmark: BackwardMaxima (backward pass via maxima)
// ============================================================

static void BM_BackwardMaxima(benchmark::State& state, const ProblemSize& sz, DataGenerator gen)
{
    std::vector<double> w(sz.n_pre * sz.n_post);
    std::vector<double> times(sz.n_spikes);
    std::vector<int32_t> sources(sz.n_spikes);

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
    ForwardInputBase base_input = {
        .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
        .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes,
        .tau_mem=0.02, .tau_syn=0.005
    };
    MaximaOutput maxima = eventprop_compute_maxima(&base_input);

    std::vector<double> maxima_errors(sz.n_post, -1.0);
#endif

    for (auto _ : state) {
#ifdef USE_CPP_WRAPPER
        gradient = backward(input, maxima, w_map, 0.02, 0.005);
        benchmark::DoNotOptimize(gradient.first.data());
#else
        SpikeInput input = { .times=times.data(), .sources=sources.data(), .n_spikes=sz.n_spikes };
        ForwardInputBase base = { .w=w.data(), .n_pre=sz.n_pre, .n_post=sz.n_post,
                                  .tau_mem=0.02, .tau_syn=0.005 };
        BackwardMaximaInput backward_input = {
            .base = {.input=&input, .base=&base},
            .maxima_times=maxima.times, .maxima_values=maxima.values,
            .maxima_errors=maxima_errors.data(), .n_maxima=sz.n_post
        };
        BackwardMaximaOutput result = eventprop_backward_maxima(&backward_input);
        benchmark::DoNotOptimize(result.gradient[0]);
        eventprop_free_backward_maxima_output(&result);
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
BACKWARDMAXIMA_BENCH(Medium,  200,  80,  500);
BACKWARDMAXIMA_BENCH(Large,   500,  200, 1000);
BACKWARDMAXIMA_BENCH(XL,      2000, 500, 5000);
BENCHMARK_MAIN();
