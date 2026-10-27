#include <benchmark/benchmark.h>
#include <cstring>
#include <Eigen/Dense>
#include <boost/math/tools/toms748_solve.hpp>
#include <cmath>
#include <vector>
#include <random>

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct VoltageCtx {
    const double* sum0; const double* sum1;
    int i; int spike_count; int n_post; int target_nrn;
    const double* post_spike_times;
    double k_prefactor; double tau_mem_inv; double tau_syn_inv; double v_th;
};

static inline double compute_v(const VoltageCtx& ctx, double t) {
    int idx = ctx.target_nrn;
    double mem = ctx.k_prefactor * (
        ctx.sum1[ctx.i * ctx.n_post + idx] * std::exp(-t * ctx.tau_mem_inv) -
        ctx.sum0[ctx.i * ctx.n_post + idx] * std::exp(-t * ctx.tau_syn_inv));
    for (int s = 0; s < ctx.spike_count; s++) {
        double st = ctx.post_spike_times[s];
        if (st <= t) mem -= ctx.v_th * std::exp(-(t - st) * ctx.tau_mem_inv);
    }
    return mem;
}

int g_fn_calls = 0;
static inline double v_delta(const VoltageCtx& ctx, double t) {
    g_fn_calls++;
    return compute_v(ctx, t) - ctx.v_th;
}

#define TOL 1e-14
#define MAX_ITER 1024

static double toms748_c(const VoltageCtx& ctx, double a, double b) {
    g_fn_calls = 0;
    double fa = v_delta(ctx, a), fb = v_delta(ctx, b);
    if (!std::isfinite(fa) || !std::isfinite(fb)) return a;
    if (fa * fb > 0) return NAN;
    if (a > b) { double tmp=a; a=b; b=tmp; tmp=fa; fa=fb; fb=tmp; }
    double c = a, fc = fa, d = 0.0;
    int iter = 0;
    while (++iter <= MAX_ITER) {
        double m = 0.5 * (b - a);
        if (std::fabs(m) <= TOL || fb == 0.0) return b;
        double s = 0.0; int useBisection = 1;
        if (std::fabs(fc) > std::fabs(fb) && c != a && c != b) {
            double qba = (fb - fa) / (b - a);
            double qcb = (fc - fb) / (c - b);
            double qcbba = (qcb - qba) / (c - a);
            if (std::fabs(qcbba) > 1e-300) {
                double denom = qba + qcb;
                if (std::fabs(denom) > 1e-300) {
                    s = b - fb / (qba + qcb * (b - a) * qcbba / denom);
                    double mVal = 0.5 * (b - a);
                    if ((s > a && s < b) && std::fabs(s - b) < mVal && std::fabs(d - mVal) >= TOL * 100)
                        useBisection = 0;
                }
            }
        }
        if (useBisection) s = a + m;
        double prevB = b; b = s; fb = v_delta(ctx, b);
        if (!std::isfinite(fb)) return prevB;
        if (std::fabs(fb) <= TOL || std::fabs(b - prevB) <= TOL * std::fmax(1.0, std::fabs(prevB)))
            return b;
        if (fa * fb < 0.0) { c = prevB; fc = v_delta(ctx, prevB); }
        else if (fb * fc < 0.0) { a = prevB; fa = v_delta(ctx, prevB); }
        else { a = prevB; fa = v_delta(ctx, prevB); }
        if (!std::isfinite(fa) || !std::isfinite(fc)) return prevB;
        d = b - prevB;
    }
    return b;
}

// Track max iterations for Boost too
int g_boost_fn_calls = 0;

double toms748_boost(const VoltageCtx& ctx, double a, double b) {
    boost::math::tools::eps_tolerance<double> tol(std::numeric_limits<double>::digits);
    boost::uintmax_t max_iter = boost::math::policies::get_max_root_iterations<boost::math::policies::policy<>>();
    auto fn = [&](double t) { return v_delta(ctx, t); };
    std::pair<double, double> result = boost::math::tools::toms748_solve(fn, a, b, tol, max_iter);
    return (result.first + result.second) / 2;
}

struct TestCase { VoltageCtx ctx; double bracket_a; double bracket_b; };

// A bracket stated WITHOUT pointers: (i, nrn) index into the cumulative-sum rows owned by
// SumRows. Keeping raw pointers out of the type gen() returns is the whole point.
struct Bracket { int i; int nrn; double a; double b; };

// Owns the storage every VoltageCtx here borrows from.
//
// This type exists because of a use-after-free that made this binary segfault on its first
// benchmark (measured: SIGSEGV, exit 139, at HEAD — not a build-flag regression). gen()
// used to return std::vector<TestCase> whose VoltageCtx carried raw sum0 / sum1 /
// post_spike_times pointers into std::vector storage LOCAL TO gen(). Those three vectors
// were freed the moment gen() returned, so every benchmark iteration dereferenced dead
// 16 KB blocks and the surviving "cases" were whatever the allocator had reused them for.
// A post-synaptic neuron may fire any number of times, so nothing about the workload made
// that crash deterministic — it just happened to be immediate here.
//
// The rule this restores: a borrowed pointer must not outlive the function that borrowed
// it. gen() now returns only ownership plus index-based problems; the ctx is built by
// make_cases() from a fixture the CALLER holds, in the caller's own scope, and never
// escapes it. (Deliberately not "return the fixture and hope the vector move preserves
// the buffers" — that works on every real implementation but is not what the standard
// promises, and a silent reliance on it is exactly how this file got here.)
struct SumRows {
    std::vector<double> sum0, sum1, post_times;
    std::vector<Bracket> brackets;
    // The VoltageCtx fields that are constant across every bracket of one fixture.
    int n_post = 0;
    double k_prefactor = 0.0, tau_mem_inv = 0.0, tau_syn_inv = 0.0, v_th = 0.0;
};

// One VoltageCtx per bracket, borrowed from a fixture that is alive in the caller's frame.
// sum0/sum1/post_times are never resized after make_cases() is called, so .data() is
// stable for the lifetime of the returned cases.
static std::vector<TestCase> make_cases(const SumRows& rows) {
    std::vector<TestCase> cases;
    cases.reserve(rows.brackets.size());
    for (const Bracket& br : rows.brackets) {
        VoltageCtx ctx{ rows.sum0.data(), rows.sum1.data(), br.i, 0, rows.n_post, br.nrn,
                        rows.post_times.data(), rows.k_prefactor, rows.tau_mem_inv,
                        rows.tau_syn_inv, rows.v_th };
        cases.push_back({ ctx, br.a, br.b });
    }
    return cases;
}

static SumRows gen(int n_pre, int n_post, int n_spikes) {
    SumRows out;
    double tau_mem = 0.02, tau_syn = 0.005, v_th = 1.0;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> t_dist(0.0, 1.0);
    // Weights come from the SAME mt19937 as everything else in this generator, at the
    // cross-suite range U[0.5, 2.0) — see bench_main.cpp's w_dist and the data contract in
    // AGENTS.md. The previous line was
    //     RowMatrixXd w = RowMatrixXd::Random(n_pre, n_post).array() * 0.5;
    // and Random() is not the local RNG: Eigen::random_impl<double> is
    //     x + (y - x) * std::rand() / RAND_MAX      (Eigen/src/Core/MathFunctions.h:812)
    // so the weights came off the global std::rand() stream. Two measured consequences:
    //   • not reproducible — a row's data depended on how many draws earlier benchmarks in
    //     the same process had consumed, so the same row found 2 brackets in a full run and
    //     0 when run through --benchmark_filter;
    //   • wrong range — Random() is U[-1,1], scaled to U[-0.5,0.5], i.e. negative weights
    //     far outside the contract, which is why the 100-spike (Small) fixtures crossed
    //     v_th never and timed an empty loop.
    // Row-major fill, matching bench_main.cpp's flat w[i] loop over a row-major matrix.
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    RowMatrixXd w(n_pre, n_post);
    for (int r = 0; r < n_pre; r++)
        for (int c = 0; c < n_post; c++)
            w(r, c) = w_dist(rng);
    std::vector<double> times(n_spikes);
    std::vector<int> sources(n_spikes);
    for (int i = 0; i < n_spikes; i++) {
        times[i] = t_dist(rng);
        sources[i] = static_cast<int>(rng() % n_pre);
    }
    std::sort(times.begin(), times.end());
    int rows = n_spikes + 1;
    std::vector<double> sum0(rows * n_post, 0.0), sum1(rows * n_post, 0.0);
    double tau_mem_inv = 1.0 / tau_mem, tau_syn_inv = 1.0 / tau_syn;
    for (int i = 0; i < n_spikes; i++) {
        int src = sources[i];
        double em = std::exp(times[i] * tau_mem_inv), es = std::exp(times[i] * tau_syn_inv);
        for (int j = 0; j < n_post; j++) {
            sum0[(i+1)*n_post+j] = sum0[i*n_post+j] + w(src,j) * es;
            sum1[(i+1)*n_post+j] = sum1[i*n_post+j] + w(src,j) * em;
        }
    }
    std::vector<double> post_times(1024, 0.0);
    double kp = tau_mem / std::abs(tau_mem - tau_syn);
    for (int nrn = 0; nrn < n_post && (int)out.brackets.size() < 100; nrn++) {
        for (int i = 1; i <= n_spikes && (int)out.brackets.size() < 100; i++) {
            // A throwaway ctx, built and consumed entirely inside this loop — nothing here
            // hands a pointer to storage the loop does not own.
            VoltageCtx ctx{sum0.data(), sum1.data(), i, 0, n_post, nrn, post_times.data(), kp, tau_mem_inv, tau_syn_inv, v_th};
            double va = compute_v(ctx, times[i-1]);
            double vb = compute_v(ctx, (i < n_spikes) ? times[i] : 1.0);
            if ((va - v_th) * (vb - v_th) < 0)
                out.brackets.push_back({i, nrn, times[i-1], (i < n_spikes) ? times[i] : 1.0});
        }
    }
    // Hand over the storage together with the problems that index into it, and with the
    // per-fixture constants make_cases() needs. The move here transfers ownership of whole
    // buffers; it does not publish any pointer out of gen().
    out.sum0 = std::move(sum0);
    out.sum1 = std::move(sum1);
    out.post_times = std::move(post_times);
    out.n_post = n_post;
    out.k_prefactor = kp;
    out.tau_mem_inv = tau_mem_inv;
    out.tau_syn_inv = tau_syn_inv;
    out.v_th = v_th;
    return out;
}

static void BM_RootFind_C(benchmark::State& state) {
    auto [n_pre, n_post, n_spikes] = [&state]() -> std::tuple<int,int,int> {
        if (state.name().find("Small") != std::string::npos) return {50, 20, 100};
        if (state.name().find("Medium") != std::string::npos) return {200, 80, 500};
        return {500, 200, 1000};
    }();
    const SumRows rows = gen(n_pre, n_post, n_spikes);
    const std::vector<TestCase> cases = make_cases(rows);   // borrows from `rows`: same scope, both outlive the loop
    // An empty bracket list is a legitimate outcome of the scan (nothing crossed v_th in
    // this regime) but it times an empty loop, and google benchmark then reports 0.000 ns
    // with an absurd iteration count — which reads as "the C root finder is infinitely
    // fast". Publish the work each row actually did, exactly as bench_main.cpp publishes
    // c_error / c_fwd_error for the same reason. Exclude rows with cases=0 from any ratio.
    state.counters["cases"] = static_cast<double>(cases.size());
    for (auto _ : state) {
        for (const TestCase& tc : cases) { double r = toms748_c(tc.ctx, tc.bracket_a, tc.bracket_b); benchmark::DoNotOptimize(r); }
    }
}

static void BM_RootFind_Boost(benchmark::State& state) {
    auto [n_pre, n_post, n_spikes] = [&state]() -> std::tuple<int,int,int> {
        if (state.name().find("Small") != std::string::npos) return {50, 20, 100};
        if (state.name().find("Medium") != std::string::npos) return {200, 80, 500};
        return {500, 200, 1000};
    }();
    const SumRows rows = gen(n_pre, n_post, n_spikes);
    const std::vector<TestCase> cases = make_cases(rows);   // borrows from `rows`: same scope, both outlive the loop
    state.counters["cases"] = static_cast<double>(cases.size());   // see BM_RootFind_C
    for (auto _ : state) {
        for (const TestCase& tc : cases) { double r = toms748_boost(tc.ctx, tc.bracket_a, tc.bracket_b); benchmark::DoNotOptimize(r); }
    }
}

BENCHMARK(BM_RootFind_C)->UseRealTime()->Name("RootFind/C/Small");
BENCHMARK(BM_RootFind_C)->UseRealTime()->Name("RootFind/C/Medium");
BENCHMARK(BM_RootFind_C)->UseRealTime()->Name("RootFind/C/Large");
BENCHMARK(BM_RootFind_Boost)->UseRealTime()->Name("RootFind/Boost/Small");
BENCHMARK(BM_RootFind_Boost)->UseRealTime()->Name("RootFind/Boost/Medium");
BENCHMARK(BM_RootFind_Boost)->UseRealTime()->Name("RootFind/Boost/Large");

BENCHMARK_MAIN();
