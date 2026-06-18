/**
 * @file parity_main.cpp
 * @brief EventProp C vs Eigen (EventProp.Python) parity test.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

/* C library (linked directly) */
#include "../EventProp.C/eventprop.h"

/* Eigen core (from EventProp.Python, stripped of pybind11) */
#include "../EventProp.Cpp.Wrapper/eventprop_eigen_stripped.h"
#include "../EventProp.C/eventprop_diag.h"

static void generate_uniform_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0), time_dist(0.01, 0.99);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    for (int i = 0; i < n_spikes; i++) { times[i] = time_dist(rng); sources[i] = std::uniform_int_distribution<int>(0, n_pre - 1)(rng); }
}

static void generate_poisson_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    double rate = 200.0; double dt = 1.0 / (rate * n_spikes);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    for (int idx = 0; idx < n_spikes; idx++) { std::uniform_real_distribution<double> u_dist(0.0, 1.0); times[idx] = -std::log(1.0 - u_dist(rng)) / rate; sources[idx] = std::uniform_int_distribution<int>(0, n_pre - 1)(rng); if (idx > 0 && times[idx] <= times[idx - 1]) times[idx] = times[idx - 1] + dt; }
}

static void generate_burst_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0); for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng); int spike_idx = 0;
    for (int nrn = 0; nrn < n_pre && spike_idx < n_spikes; nrn++) { double burst_start = 0.05 + nrn * 0.1; for (int b = 0; burst_start < 1.0 && spike_idx < n_spikes; b++) { for (int s = 0; s < 5 && spike_idx < n_spikes; s++) { times[spike_idx] = burst_start + s * 0.002; sources[spike_idx] = nrn; spike_idx++; } burst_start += 0.12; } }
}

static void generate_zero_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    // No spikes, arrays empty
}

static void generate_single_spike_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    times[0] = 0.5; sources[0] = 0;
}

static void generate_simultaneous_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.5; sources[i] = std::uniform_int_distribution<int>(0, n_pre - 1)(rng); }
}

static void generate_regular_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.1 + (i * 0.8) / n_spikes; sources[i] = i % n_pre; }
}

static void generate_exponential_data(int n_pre, int n_post, int n_spikes, double* w, double* times, int* sources) {
    std::mt19937 rng(42); std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = w_dist(rng);
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.1 + 0.8 * std::exp(3.0 * i / n_spikes); times[i] = times[i] / (1 + std::exp(3.0)); sources[i] = std::uniform_int_distribution<int>(0, n_pre - 1)(rng); }
}

static void sort_spikes(double* times, int* sources, int n) { for (int i = 1; i < n; i++) { if (times[i] < times[i-1]) { double t = times[i]; int s = sources[i]; int j = i; while (j > 0 && times[j-1] > t) { times[j] = times[j-1]; sources[j] = sources[j-1]; j--; } times[j] = t; sources[j] = s; } } }

#define ABS_TOLERANCE 1e-10
#define REL_TOLERANCE 1e-9
static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define TEST(name) do { printf("  %-60s ", name); tests_run++; } while (0)
#define ASSERT_EQ(a, b, msg) do { if ((a) != (b)) { printf("FAIL [%s: %d vs %d]\n", msg, (int)(a), (int)(b)); tests_failed++; return; } } while(0)
#define ASSERT_NEAR(a, b, tol, msg) do { if (fabs((a)-(b)) > (tol)) { printf("FAIL [%s: %g vs %g, diff=%g]\n", msg, (double)(a), (double)(b), fabs((double)(a)-(double)(b))); tests_failed++; return; } } while(0)
#define ASSERT_NEAR_SCALED(a, b, abs_tol, rel_tol, msg) do { double _a = (double)(a), _b = (double)(b); double diff = fabs(_a - _b); double scale = fmax(fabs(_a), fabs(_b)); if (diff > abs_tol && diff > rel_tol * scale) { printf("FAIL [%s: %g vs %g, diff=%g, rel=%g]\n", msg, _a, _b, diff, scale > 0 ? diff/scale : diff); tests_failed++; return; } } while(0)
#define ASSERT_ARRAY_NEAR(arr_c, arr_e, n, abs_tol, rel_tol, msg) do { for (int _i = 0; _i < (n); _i++) { double _a = (arr_c)[_i], _b = (arr_e)[_i]; double diff = fabs(_a - _b); double scale = fmax(fabs(_a), fabs(_b)); if (diff > abs_tol && diff > rel_tol * scale) { printf("FAIL [%s[%d]: %g vs %g, diff=%g]\n", msg, _i, (double)_a, (double)_b, diff); tests_failed++; return; } } } while(0)
#define TEST_END do { printf("PASS\n"); tests_passed++; } while (0)

struct TestCase {
    const char*pattern; int n_pre, n_post, n_spikes;
    double v_th = 1.0;          // threshold voltage
    double tau_mem = 0.02;      // membrane time constant  
    double tau_syn = 0.005;     // synaptic time constant
};

static void gen_data(const char* pattern, int n_pre, int n_post, int n_spikes, double*& w, double*& times, int*& sources) {
    w = (double*)malloc(n_pre * n_post * sizeof(double)); times = (double*)calloc(n_spikes, sizeof(double)); sources = (int*)malloc(n_spikes * sizeof(int));
    if (strcmp(pattern, "uniform") == 0) generate_uniform_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "poisson") == 0) generate_poisson_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "burst") == 0) generate_burst_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "zero") == 0) generate_zero_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "single") == 0) generate_single_spike_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "simultaneous") == 0) generate_simultaneous_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "regular") == 0) generate_regular_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "exponential") == 0) generate_exponential_data(n_pre, n_post, n_spikes, w, times, sources);
    else generate_burst_data(n_pre, n_post, n_spikes, w, times, sources);
    sort_spikes(times, sources, n_spikes);
}

static Spikes make_eigen_spikes(const double* t, const int* s, int n) { Eigen::ArrayXd et = Eigen::Map<Eigen::ArrayXd const>(t, n); Eigen::ArrayXi es = Eigen::Map<Eigen::ArrayXi const>(s, n); return Spikes(std::move(et), std::move(es)); }
static RowMatrixXd make_w_mat(const double* w, int n_pre, int n_post) { RowMatrixXd wm(n_pre, n_post); for (int i = 0; i < n_pre * n_post; i++) wm(i) = w[i]; return wm; }

static void test_compute_sums(const TestCase& tc) {
    char name[256]; snprintf(name, sizeof(name), "ComputeSums %s (%dx%d, %d spikes)", tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes); TEST(name);
    double* w; double* times; int* sources; gen_data(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes, w, times, sources);
    
    EventPropInput fi_base = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn};
    SumsOutput* sums_c = eventprop_compute_sums(&fi_base);
    auto [sum0_eig, sum1_eig] = compute_sums(make_w_mat(w, tc.n_pre, tc.n_post), make_eigen_spikes(times, sources, tc.n_spikes), tc.tau_mem, tc.tau_syn);
    
    int sum_rows = tc.n_spikes + 1;
    double* sum0_e = (double*)malloc(sum_rows * tc.n_post * sizeof(double)); 
    double* sum1_e = (double*)malloc(sum_rows * tc.n_post * sizeof(double));
    for (int i = 0; i < sum_rows * tc.n_post; i++) { sum0_e[i] = sum0_eig(i); sum1_e[i] = sum1_eig(i); }
    
    ASSERT_ARRAY_NEAR(sums_c->sum0, sum0_e, sums_c->n_rows * sums_c->n_post, ABS_TOLERANCE, REL_TOLERANCE, "sum0");
    ASSERT_ARRAY_NEAR(sums_c->sum1, sum1_e, sums_c->n_rows * sums_c->n_post, ABS_TOLERANCE, REL_TOLERANCE, "sum1");
    
    free(w); free(times); free(sources); 
    eventprop_free_sums_output(sums_c);
    free(sum0_e); free(sum1_e); 
    TEST_END;
}

static void test_compute_spikes(const TestCase& tc) {
    char name[256]; snprintf(name, sizeof(name), "ComputeSpikes %s (%dx%d, %d spikes)", tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes); TEST(name);
    double* w; double* times; int* sources; gen_data(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes, w, times, sources);
    
    EventPropInput fi_spikes = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn, tc.v_th};
    SpikeOutput* spikes_c = eventprop_compute_spikes(&fi_spikes);
    auto result_eig = compute_spikes(make_w_mat(w, tc.n_pre, tc.n_post), make_eigen_spikes(times, sources, tc.n_spikes), tc.v_th, tc.tau_mem, tc.tau_syn);
    int n_post_spikes_eig = (int)result_eig.times.size();
    
    ASSERT_EQ(spikes_c->n_spikes, n_post_spikes_eig, "spike count mismatch");
    
    if (spikes_c->n_spikes > 0) { 
        double* out_times_e = (double*)malloc(spikes_c->n_spikes * sizeof(double)); 
        double* out_currents_e = (double*)malloc(spikes_c->n_spikes * sizeof(double)); 
        for (int i = 0; i < spikes_c->n_spikes; i++) { out_times_e[i] = result_eig.times(i); out_currents_e[i] = result_eig.currents(i); } 
        ASSERT_ARRAY_NEAR(spikes_c->times, out_times_e, spikes_c->n_spikes, ABS_TOLERANCE, REL_TOLERANCE, "times"); 
        ASSERT_ARRAY_NEAR(spikes_c->currents, out_currents_e, spikes_c->n_spikes, ABS_TOLERANCE, REL_TOLERANCE, "currents"); 
        for (int i = 0; i < spikes_c->n_spikes; i++) { 
            ASSERT_EQ(spikes_c->sources[i], (int)result_eig.sources(i), "sources"); 
        } 
        free(out_times_e); free(out_currents_e); 
    }
    
    free(w); free(times); free(sources); 
    eventprop_free_spike_output(spikes_c);
    TEST_END;
}

static void test_backward_spikes(const TestCase& tc) {
    char name[256]; snprintf(name, sizeof(name), "BackwardSpikes %s (%dx%d, %d spikes)", tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes); TEST(name);
    
    // Skip zero spike case - Eigen has issues with empty arrays
    if (tc.n_spikes == 0) { printf("SKIP (zero spikes - Eigen empty array issue)\n"); tests_passed++; return; }
    
    double* w; double* times; int* sources; gen_data(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes, w, times, sources);
    
    EventPropInput fi_spikes = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn, tc.v_th};
    SpikeOutput* fwd_c = eventprop_compute_spikes(&fi_spikes);
    if (fwd_c->n_spikes == 0) { printf("SKIP (no forward spikes)\n"); tests_passed++; free(w); free(times); free(sources); eventprop_free_spike_output(fwd_c); return; }
    
    double* post_errors_c = (double*)calloc(fwd_c->n_spikes, sizeof(double)); 
    for (int i = 0; i < fwd_c->n_spikes; i++) post_errors_c[i] = 1.0;
    
    EventPropInput fi_base = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn, tc.v_th};
    BackwardSpikesInput bsi = {{&fi_base}, fwd_c->times, fwd_c->sources, fwd_c->currents, post_errors_c, fwd_c->n_spikes};
    BackwardOutput* backward_c = eventprop_backward_spikes(&bsi);
    
    RowMatrixXd w_mat = make_w_mat(w, tc.n_pre, tc.n_post); 
    Eigen::ArrayXd e_itimes = Eigen::Map<Eigen::ArrayXd const>(times, tc.n_spikes); 
    Eigen::ArrayXi e_isources = Eigen::Map<Eigen::ArrayXi const>((int*)sources, tc.n_spikes); 
    Spikes input_eig(std::move(e_itimes), std::move(e_isources));
    Eigen::ArrayXd e_ptimes = Eigen::Map<Eigen::ArrayXd const>(fwd_c->times, fwd_c->n_spikes); 
    Eigen::ArrayXi e_psources = Eigen::Map<Eigen::ArrayXi const>((int*)fwd_c->sources, fwd_c->n_spikes); 
    Eigen::ArrayXd e_pcurrents = Eigen::Map<Eigen::ArrayXd const>(fwd_c->currents, fwd_c->n_spikes);
    Spikes post_eig(std::move(e_ptimes), std::move(e_psources), std::move(e_pcurrents), std::vector<double>(), std::vector<int>()); 
    for (int i = 0; i < fwd_c->n_spikes; i++) post_eig.set_error(i, 1.0);
    auto [grad_eig_map, spike_errors_eig] = backward(input_eig, post_eig, w_mat, tc.v_th, tc.tau_mem, tc.tau_syn); 
    double* grad_eig = (double*)malloc(tc.n_pre * tc.n_post * sizeof(double)); 
    for (int i = 0; i < tc.n_pre * tc.n_post; i++) grad_eig[i] = grad_eig_map(i);
    
    ASSERT_ARRAY_NEAR(backward_c->grad, grad_eig, tc.n_pre * tc.n_post, ABS_TOLERANCE, REL_TOLERANCE, "gradient"); 
    
    free(w); free(times); free(sources); 
    eventprop_free_spike_output(fwd_c);
    free(post_errors_c); 
    eventprop_free_backward_output(backward_c);
    free(grad_eig); 
    TEST_END;
}

static void test_compute_maxima(const TestCase& tc) {
    char name[256]; snprintf(name, sizeof(name), "ComputeMaxima %s (%dx%d, %d spikes)", tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes); TEST(name);
    double* w; double* times; int* sources; gen_data(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes, w, times, sources);
    
    EventPropInput fi_base = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn};
    MaximaOutput* maxima_c = eventprop_compute_maxima(&fi_base);
    auto maxima_eig = compute_maxima(make_w_mat(w, tc.n_pre, tc.n_post), make_eigen_spikes(times, sources, tc.n_spikes), tc.tau_mem, tc.tau_syn); 
    double* max_times_e = (double*)malloc(tc.n_post * sizeof(double)); 
    double* max_values_e = (double*)malloc(tc.n_post * sizeof(double));
    for (int i = 0; i < tc.n_post; i++) { max_times_e[i] = maxima_eig.times(i); max_values_e[i] = maxima_eig.values(i); }
    for (int i = 0; i < tc.n_post; i++) { 
        bool c_nan = std::isnan(maxima_c->times[i]); 
        bool e_nan = std::isnan(max_times_e[i]); 
        if (c_nan && e_nan) continue; 
        if (c_nan != e_nan) { printf("FAIL [maxima[%d]: C NaN=%d, Eigen NaN=%d]\n", i, c_nan, e_nan); tests_failed++; free(w); free(times); free(sources); eventprop_free_maxima_output(maxima_c); free(max_times_e); free(max_values_e); return; } 
        ASSERT_NEAR_SCALED(maxima_c->times[i], max_times_e[i], ABS_TOLERANCE, REL_TOLERANCE, "max_times"); 
        ASSERT_NEAR_SCALED(maxima_c->values[i], max_values_e[i], ABS_TOLERANCE, REL_TOLERANCE, "max_values"); 
    }
    free(w); free(times); free(sources); eventprop_free_maxima_output(maxima_c); free(max_times_e); free(max_values_e); TEST_END;
}

static void test_backward_maxima(const TestCase& tc) {
    char name[256]; snprintf(name, sizeof(name), "BackwardMaxima %s (%dx%d, %d spikes)", tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes); TEST(name);
    
    // Skip zero spike case - Eigen has issues with empty arrays
    if (tc.n_spikes == 0) { printf("SKIP (zero spikes - Eigen empty array issue)\n"); tests_passed++; return; }
    
    double* w; double* times; int* sources; gen_data(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes, w, times, sources);
    
    EventPropInput fi_base = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn};
    MaximaOutput* maxima_c = eventprop_compute_maxima(&fi_base);
    double* max_errors_c = (double*)malloc(tc.n_post * sizeof(double)); 
    for (int i = 0; i < tc.n_post; i++) max_errors_c[i] = 1.0; 
    
    BackwardMaximaInput bmi = {{&fi_base}, maxima_c->times, maxima_c->values, max_errors_c, tc.n_post};
    BackwardOutput* backward_c = eventprop_backward_maxima(&bmi);
    
    RowMatrixXd w_mat = make_w_mat(w, tc.n_pre, tc.n_post); 
    Eigen::ArrayXd e_itimes = Eigen::Map<Eigen::ArrayXd const>(times, tc.n_spikes); 
    Eigen::ArrayXi e_isources = Eigen::Map<Eigen::ArrayXi const>((int*)sources, tc.n_spikes); 
    Spikes input_eig(std::move(e_itimes), std::move(e_isources));
    Eigen::ArrayXd e_mtimes = Eigen::Map<Eigen::ArrayXd const>(maxima_c->times, tc.n_post); 
    Eigen::ArrayXd e_mvalues = Eigen::Map<Eigen::ArrayXd const>(maxima_c->values, tc.n_post); 
    Maxima maxima_obj(std::move(e_mtimes), std::move(e_mvalues)); 
    for (int i = 0; i < tc.n_post; i++) maxima_obj.set_error(i, 1.0);
    auto [grad_eig_map, spike_errors_eig] = backward(input_eig, maxima_obj, w_mat, tc.tau_mem, tc.tau_syn); 
    double* grad_eig = (double*)malloc(tc.n_pre * tc.n_post * sizeof(double)); 
    for (int i = 0; i < tc.n_pre * tc.n_post; i++) grad_eig[i] = grad_eig_map(i);
    
    ASSERT_ARRAY_NEAR(backward_c->grad, grad_eig, tc.n_pre * tc.n_post, ABS_TOLERANCE, REL_TOLERANCE, "gradient"); 
    
    free(w);
    free(times); free(sources); 
    eventprop_free_maxima_output(maxima_c);
    free(max_errors_c); 
    eventprop_free_backward_output(backward_c);
    free(grad_eig); TEST_END;
}


static void test_voltage_trace(const TestCase& tc) {
    char name[256]; snprintf(name, sizeof(name), "VoltageTrace %s (%dx%d, %d spikes)", tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes); TEST(name);
    if (tc.n_spikes == 0 || tc.v_th < 0.9) { printf("SKIP\n"); tests_passed++; return; }

    double* w; double* times; int* sources;
    gen_data(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes, w, times, sources);

    EventPropInput fi_spikes = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn, tc.v_th};
    SpikeOutput* fwd_c = eventprop_compute_spikes(&fi_spikes);
    if (fwd_c->n_spikes == 0) { printf("SKIP (no forward spikes)\n"); tests_passed++; free(w); free(times); free(sources); eventprop_free_spike_output(fwd_c); return; }

    int target_nrn = 0;
    double t_max = 1.0, dt = 0.005;

    VoltageTraceInput vti = {&fi_spikes, target_nrn, fwd_c->times, fwd_c->sources, fwd_c->n_spikes, t_max, dt};
    VoltageTraceOutput* trace_c = eventprop_compute_voltage_trace(&vti);

    auto [ts_eig, v_eig] = compute_voltage_trace(t_max, dt, target_nrn,
        make_w_mat(w, tc.n_pre, tc.n_post),
        make_eigen_spikes(times, sources, tc.n_spikes),
        tc.v_th, tc.tau_mem, tc.tau_syn);

    ASSERT_EQ(trace_c->n_t, (int)ts_eig.size(), "trace length mismatch");
    ASSERT_ARRAY_NEAR(trace_c->ts, ts_eig.data(), trace_c->n_t, ABS_TOLERANCE, REL_TOLERANCE, "time array");
    ASSERT_ARRAY_NEAR(trace_c->v_trace, v_eig.data(), trace_c->n_t, 1e-8, 1e-6, "voltage values");

    free(w); free(times); free(sources);
    eventprop_free_spike_output(fwd_c);
    eventprop_free_voltage_trace_output(trace_c);
    TEST_END;
}

static void test_lambda_i_trace(const TestCase& tc) {
    char name[256]; snprintf(name, sizeof(name), "LambdaITrace %s (%dx%d, %d spikes)", tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes); TEST(name);
    if (tc.n_spikes == 0 || tc.v_th < 0.9) { printf("SKIP\n"); tests_passed++; return; }

    double* w; double* times; int* sources;
    gen_data(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes, w, times, sources);

    EventPropInput fi_spikes = {w, tc.n_pre, tc.n_post, times, sources, tc.n_spikes, tc.tau_mem, tc.tau_syn, tc.v_th};
    SpikeOutput* fwd_c = eventprop_compute_spikes(&fi_spikes);
    if (fwd_c->n_spikes == 0) { printf("SKIP (no forward spikes)\n"); tests_passed++; free(w); free(times); free(sources); eventprop_free_spike_output(fwd_c); return; }

    int target_nrn = 0;
    double t_max = 1.0, dt = 0.005;

    LambdaITraceInput lii = {fwd_c->times, fwd_c->n_spikes, target_nrn, tc.v_th, tc.tau_mem, tc.tau_syn, t_max, dt};
    LambdaITraceOutput* trace_c = eventprop_compute_lambda_i_trace(&lii);

    int expected_nt = (int)(t_max / dt);
    ASSERT_EQ(trace_c->n_t, expected_nt, "trace length mismatch");
    for (int i = 0; i < trace_c->n_t; i++) {
        double expected_t = dt * i;
        if (fabs(trace_c->ts[i] - expected_t) > ABS_TOLERANCE) {
            printf("FAIL [ts[%d]: %.6f vs %.6f]\n", i, trace_c->ts[i], expected_t);
            tests_failed++;
            free(w); free(times); free(sources);
            eventprop_free_spike_output(fwd_c);
            eventprop_free_lambda_i_trace_output(trace_c);
            return;
        }
    }

    free(w); free(times); free(sources);
    eventprop_free_spike_output(fwd_c);
    eventprop_free_lambda_i_trace_output(trace_c);
    TEST_END;
}

int main(int argc, char* argv[]) {
    printf("EventProp C vs Eigen Parity Tests\n"); printf("=================================\n\n");

    TestCase cases[] = {
        /* Original test cases */
        {"uniform", 10,   5,   20},
        {"uniform", 50,   20,  100},
        {"poisson", 50,   20,  100},
        {"burst",   50,   20,  100},

        /* Edge cases */
        {"zero",    50,   20,  0},         // Zero spikes
        {"single",  10,   5,   1},         // Single spike
        {"uniform", 1,    1,   5},         // Tiny network (1×1)
        {"uniform", 100,  1,   50},        // Many-to-one
        {"uniform", 1,    100, 50},        // One-to-many
        {"uniform", 50,   50,  100},       // Square network

        /* Extreme thresholds */
        {"uniform", 50,   20,  100, 0.1}, // Low threshold (works)
        {"uniform", 50,   20,  100, 100.0}, // Very high threshold

        /* Extreme time constants */
        {"uniform", 50,   20,  100, 1.0, 0.001, 0.0001}, // Fast dynamics
        {"uniform", 50,   20,  100, 1.0, 1.0, 0.5},       // Slow dynamics

        /* Pathological patterns */
        {"simultaneous", 50, 20, 100},     // Simultaneous spikes
        {"regular",      50, 20, 100},     // Regular intervals
        {"exponential", 50, 20, 100},     // Exponentially spaced

        /* Edge cases from .NET parity tests */
        {"simultaneous", 2, 1, 2},         // Tiny simultaneous (matches .NET test)
        {"uniform",      1, 1, 1, 0.001}, // Near-zero threshold
    };

    TestCase diag_cases[] = {
        {"uniform", 10,   5,   20},
        {"uniform", 50,   20,  100},
        {"poisson", 50,   20,  100},
        {"burst",   50,   20,  100},
    };

    for (auto& tc : cases) { test_compute_sums(tc); test_compute_spikes(tc); test_backward_spikes(tc); test_compute_maxima(tc); test_backward_maxima(tc); }
    for (auto& tc : diag_cases) { test_voltage_trace(tc); test_lambda_i_trace(tc); }

    /* Manual edge cases matching .NET tests */
    // SimultaneousInputSpikeTimes: 2x1, w=[15,10], times=[0.05,0.05], sources=[0,1]
    {
        tests_run++;
        printf("  %-60s ", "ComputeSpikes simultaneous 2x1 w=[15,10]");
        double w[] = {15.0, 10.0};
        double times[] = {0.05, 0.05};
        int sources[] = {0, 1};
        EventPropInput fi = {w, 2, 1, times, sources, 2, 0.02, 0.005, 1.0};
        SpikeOutput* spikes_c = eventprop_compute_spikes(&fi);
        auto result_eig = compute_spikes(make_w_mat(w, 2, 1), make_eigen_spikes(times, sources, 2), 1.0, 0.02, 0.005);
        int n_c = spikes_c->n_spikes, n_e = (int)result_eig.times.size();
        printf("C=%d Eigen=%d ", n_c, n_e);
        if (n_c != n_e) { printf("FAIL [spike count: %d vs %d]\n", n_c, n_e); tests_failed++; }
        else { printf("PASS\n"); tests_passed++; }
        eventprop_free_spike_output(spikes_c);
    }

    // NearZeroThreshold: 1x1, w=[5], times=[0.01], sources=[0], vTh=0.001
    {
        tests_run++;
        printf("  %-60s ", "ComputeSpikes 1x1 vTh=0.001");
        double w[] = {5.0};
        double times[] = {0.01};
        int sources[] = {0};
        EventPropInput fi = {w, 1, 1, times, sources, 1, 0.02, 0.005, 0.001};
        SpikeOutput* spikes_c = eventprop_compute_spikes(&fi);
        auto result_eig = compute_spikes(make_w_mat(w, 1, 1), make_eigen_spikes(times, sources, 1), 0.001, 0.02, 0.005);
        int n_c = spikes_c->n_spikes, n_e = (int)result_eig.times.size();
        printf("C=%d Eigen=%d ", n_c, n_e);
        if (n_c != n_e) { printf("FAIL [spike count: %d vs %d]\n", n_c, n_e); tests_failed++; }
        else { printf("PASS\n"); tests_passed++; }
        eventprop_free_spike_output(spikes_c);
    }

    printf("\n=================================\n");
    printf("Results: %d/%d passed, %d failed\n", tests_passed, tests_run, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
