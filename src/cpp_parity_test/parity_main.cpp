/**
 * @file parity_main.cpp
 * @brief EventProp C vs Eigen (EventProp.Python) parity test.
 *
 * Compares the pure C implementation (libeventprop_c_api.dylib) against the Eigen-based
 * core from EventProp.Python. Uses identical input data for both and asserts
 * numerical equality within tolerance 1e-10.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <dlfcn.h>

/* Eigen core (from EventProp.Python, stripped of pybind11) */
#include "eventprop_eigen.h"

/* ── C library function pointer types (loaded via dlopen) ───────────── */
typedef void (*cfn_compute_sums)(const double*, int, int, const double*, const int32_t*, int, double, double, double*, double*);
typedef int  (*cfn_compute_spikes)(const double*, int, int, const double*, const int32_t*, int, double, double, double, double*, int32_t*, double*);
typedef void (*cfn_backward_spikes)(const double*, const int32_t*, int, const double*, const int32_t*, const double*, const double*, int,
                                    const double*, int, int, double, double, double, double*);
typedef void (*cfn_compute_maxima)(const double*, int, int, const double*, const int32_t*, int, double, double, double*, double*);
typedef void (*cfn_backward_maxima)(const double*, const int32_t*, int, const double*, const double*, const double*, int,
                                    const double*, int, int, double, double, double*);

/* ── C library runtime loader (dlopen/dlsym) ─────────────────────────── */

static void* g_c_lib = nullptr;
static cfn_compute_sums fn_compute_sums = nullptr;
static cfn_compute_spikes fn_compute_spikes = nullptr;
static cfn_backward_spikes fn_backward_spikes = nullptr;
static cfn_compute_maxima fn_compute_maxima = nullptr;
static cfn_backward_maxima fn_backward_maxima = nullptr;

static bool load_c_library(const char* path) {
    g_c_lib = dlopen(path, RTLD_NOW);
    if (!g_c_lib) {
        fprintf(stderr, "ERROR: Cannot load C library from '%s': %s\n", path, dlerror());
        return false;
    }

    fn_compute_sums = (cfn_compute_sums)dlsym(g_c_lib, "eventprop_compute_sums");
    fn_compute_spikes = (cfn_compute_spikes)dlsym(g_c_lib, "eventprop_compute_spikes");
    fn_backward_spikes = (cfn_backward_spikes)dlsym(g_c_lib, "eventprop_backward_spikes");
    fn_compute_maxima = (cfn_compute_maxima)dlsym(g_c_lib, "eventprop_compute_maxima");
    fn_backward_maxima = (cfn_backward_maxima)dlsym(g_c_lib, "eventprop_backward_maxima");

    if (!fn_compute_sums || !fn_compute_spikes || !fn_backward_spikes ||
        !fn_compute_maxima || !fn_backward_maxima) {
        fprintf(stderr, "ERROR: Failed to resolve C library symbols\n");
        dlclose(g_c_lib); g_c_lib = nullptr;
        return false;
    }

    printf("C library loaded from: %s\n", path);
    return true;
}

/* ── Test data generation (identical for both libraries) ─────────────── */

static void generate_uniform_data(int n_pre, int n_post, int n_spikes,
                                   double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    std::uniform_real_distribution<double> time_dist(0.01, 0.99);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    for (int i = 0; i < n_spikes; i++) {
        times[i] = time_dist(rng);
        sources[i] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);
    }
}

static void generate_poisson_data(int n_pre, int n_post, int n_spikes,
                                   double* w, double* times, int32_t* sources)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> w_dist(0.5, 2.0);
    double rate = 200.0;
    double dt = 1.0 / (rate * n_spikes);

    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = w_dist(rng);

    for (int idx = 0; idx < n_spikes; idx++) {
        std::uniform_real_distribution<double> u_dist(0.0, 1.0);
        double u = u_dist(rng);
        times[idx] = -std::log(1.0 - u) / rate;
        sources[idx] = std::uniform_int_distribution<int32_t>(0, n_pre - 1)(rng);
        if (idx > 0 && times[idx] <= times[idx - 1])
            times[idx] = times[idx - 1] + dt;
    }
}

static void generate_burst_data(int n_pre, int n_post, int n_spikes,
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

/* ── Helpers ─────────────────────────────────────────────────────────── */

static void sort_spikes(double* times, int32_t* sources, int n) {
    for (int i = 1; i < n; i++) {
        if (times[i] < times[i-1]) {
            double t = times[i]; int32_t s = sources[i];
            int j = i;
            while (j > 0 && times[j-1] > t) {
                times[j] = times[j-1]; sources[j] = sources[j-1]; j--;
            }
            times[j] = t; sources[j] = s;
        }
    }
}

#define ABS_TOLERANCE 1e-10
#define REL_TOLERANCE 1e-9
static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define TEST(name) \
    do { printf("  %-60s ", name); tests_run++; } while (0)

#define ASSERT_EQ(a, b, msg) \
    do { if ((a) != (b)) { printf("FAIL [%s: %d vs %d]\n", msg, (int)(a), (int)(b)); tests_failed++; return; } } while(0)

#define ASSERT_NEAR(a, b, tol, msg) \
    do { if (fabs((a)-(b)) > (tol)) { printf("FAIL [%s: %g vs %g, diff=%g]\n", msg, (double)(a), (double)(b), fabs((double)(a)-(double)(b))); tests_failed++; return; } } while(0)

#define ASSERT_NEAR_SCALED(a, b, abs_tol, rel_tol, msg) \
    do { double _a = (double)(a), _b = (double)(b); \
         double diff = fabs(_a - _b); \
         double scale = fmax(fabs(_a), fabs(_b)); \
         if (diff > abs_tol && diff > rel_tol * scale) { printf("FAIL [%s: %g vs %g, diff=%g, rel=%g]\n", msg, _a, _b, diff, scale > 0 ? diff/scale : diff); tests_failed++; return; } } while(0)

#define ASSERT_ARRAY_NEAR(arr_c, arr_e, n, abs_tol, rel_tol, msg) \
    do { for (int _i = 0; _i < (n); _i++) { double _a = (arr_c)[_i], _b = (arr_e)[_i]; \
         double diff = fabs(_a - _b); double scale = fmax(fabs(_a), fabs(_b)); \
         if (diff > abs_tol && diff > rel_tol * scale) { printf("FAIL [%s[%d]: %g vs %g, diff=%g]\n", msg, _i, (double)_a, (double)_b, diff); tests_failed++; return; } } } while(0)

#define TEST_END \
    do { printf("PASS\n"); tests_passed++; } while (0)

/* ── Test: ComputeSums ──────────────────────────────────────────────── */

static void test_compute_sums(const char* pattern, int n_pre, int n_post, int n_spikes)
{
    char name[256];
    snprintf(name, sizeof(name), "ComputeSums %s (%dx%d, %d spikes)", pattern, n_pre, n_post, n_spikes);
    TEST(name);

    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));

    if (strcmp(pattern, "uniform") == 0) generate_uniform_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "poisson") == 0) generate_poisson_data(n_pre, n_post, n_spikes, w, times, sources);
    else generate_burst_data(n_pre, n_post, n_spikes, w, times, sources);

    sort_spikes(times, sources, n_spikes);

    int sum_rows = n_spikes + 1;
    double* sum0_c = (double*)calloc(sum_rows * n_post, sizeof(double));
    double* sum1_c = (double*)calloc(sum_rows * n_post, sizeof(double));
    double* sum0_e = (double*)calloc(sum_rows * n_post, sizeof(double));
    double* sum1_e = (double*)calloc(sum_rows * n_post, sizeof(double));

    /* C library */
    fn_compute_sums(w, n_pre, n_post, times, sources, n_spikes,
                    0.02, 0.005, sum0_c, sum1_c);

    /* Eigen */
    RowMatrixXd w_mat(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++) w_mat(i) = w[i];
    Spikes spikes_eig(Eigen::Map<Eigen::ArrayXd>(times, n_spikes),
                      Eigen::Map<Eigen::ArrayXi>((int32_t*)sources, n_spikes));
    auto [sum0_eig, sum1_eig] = compute_sums(w_mat, spikes_eig, 0.02, 0.005);
    for (int i = 0; i < sum_rows * n_post; i++) {
        sum0_e[i] = sum0_eig(i);
        sum1_e[i] = sum1_eig(i);
    }

    ASSERT_ARRAY_NEAR(sum0_c, sum0_e, sum_rows * n_post, ABS_TOLERANCE, REL_TOLERANCE, "sum0");
    ASSERT_ARRAY_NEAR(sum1_c, sum1_e, sum_rows * n_post, ABS_TOLERANCE, REL_TOLERANCE, "sum1");

    free(w); free(times); free(sources);
    free(sum0_c); free(sum1_c); free(sum0_e); free(sum1_e);
    TEST_END;
}

/* ── Test: ComputeSpikes ───────────────────────────────────────────── */

static void test_compute_spikes(const char* pattern, int n_pre, int n_post, int n_spikes)
{
    char name[256];
    snprintf(name, sizeof(name), "ComputeSpikes %s (%dx%d, %d spikes)", pattern, n_pre, n_post, n_spikes);
    TEST(name);

    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));

    if (strcmp(pattern, "uniform") == 0) generate_uniform_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "poisson") == 0) generate_poisson_data(n_pre, n_post, n_spikes, w, times, sources);
    else generate_burst_data(n_pre, n_post, n_spikes, w, times, sources);

    sort_spikes(times, sources, n_spikes);

    int max_out = n_spikes * n_post;
    double* out_times_c = (double*)malloc(max_out * sizeof(double));
    int32_t* out_sources_c = (int32_t*)malloc(max_out * sizeof(int32_t));
    double* out_currents_c = (double*)malloc(max_out * sizeof(double));

    /* C library */
    int n_post_spikes_c = fn_compute_spikes(w, n_pre, n_post,
                                            times, sources, n_spikes,
                                            1.0, 0.02, 0.005,
                                            out_times_c, out_sources_c, out_currents_c);

    /* Eigen */
    RowMatrixXd w_mat(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++) w_mat(i) = w[i];
    Spikes spikes_eig(Eigen::Map<Eigen::ArrayXd>(times, n_spikes),
                      Eigen::Map<Eigen::ArrayXi>((int32_t*)sources, n_spikes));
    auto result_eig = compute_spikes(w_mat, spikes_eig, 1.0, 0.02, 0.005);

    int n_post_spikes_eig = (int)result_eig.times.size();

    /* Compare spike counts */
    ASSERT_EQ(n_post_spikes_c, n_post_spikes_eig, "spike count mismatch");

    if (n_post_spikes_c > 0) {
        /* Compare times */
        double* out_times_e = (double*)malloc(n_post_spikes_c * sizeof(double));
        double* out_currents_e = (double*)malloc(n_post_spikes_c * sizeof(double));
        for (int i = 0; i < n_post_spikes_c; i++) {
            out_times_e[i] = result_eig.times(i);
            out_currents_e[i] = result_eig.currents(i);
        }

        ASSERT_ARRAY_NEAR(out_times_c, out_times_e, n_post_spikes_c, ABS_TOLERANCE, REL_TOLERANCE, "times");
        ASSERT_ARRAY_NEAR(out_currents_c, out_currents_e, n_post_spikes_c, ABS_TOLERANCE, REL_TOLERANCE, "currents");

        /* Compare sources */
        for (int i = 0; i < n_post_spikes_c; i++) {
            ASSERT_EQ(out_sources_c[i], (int32_t)result_eig.sources(i), "sources");
        }

        free(out_times_e); free(out_currents_e);
    }

    free(w); free(times); free(sources);
    free(out_times_c); free(out_sources_c); free(out_currents_c);
    TEST_END;
}

/* ── Test: BackwardSpikes ──────────────────────────────────────────── */

static void test_backward_spikes(const char* pattern, int n_pre, int n_post, int n_spikes)
{
    char name[256];
    snprintf(name, sizeof(name), "BackwardSpikes %s (%dx%d, %d spikes)", pattern, n_pre, n_post, n_spikes);
    TEST(name);

    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));

    if (strcmp(pattern, "uniform") == 0) generate_uniform_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "poisson") == 0) generate_poisson_data(n_pre, n_post, n_spikes, w, times, sources);
    else generate_burst_data(n_pre, n_post, n_spikes, w, times, sources);

    sort_spikes(times, sources, n_spikes);

    int max_out = n_spikes * n_post;
    double* fwd_times_c = (double*)malloc(max_out * sizeof(double));
    int32_t* fwd_sources_c = (int32_t*)malloc(max_out * sizeof(int32_t));
    double* fwd_currents_c = (double*)malloc(max_out * sizeof(double));

    /* Forward pass: C */
    int n_fwd_c = fn_compute_spikes(w, n_pre, n_post,
                                     times, sources, n_spikes,
                                     1.0, 0.02, 0.005,
                                     fwd_times_c, fwd_sources_c, fwd_currents_c);

    if (n_fwd_c == 0) {
        printf("SKIP (no forward spikes)\n");
        tests_passed++;
        free(w); free(times); free(sources);
        free(fwd_times_c); free(fwd_sources_c); free(fwd_currents_c);
        return;
    }

    /* Set errors = 1.0 */
    double* post_errors_c = (double*)calloc(n_fwd_c, sizeof(double));
    for (int i = 0; i < n_fwd_c; i++) post_errors_c[i] = 1.0;

    /* C backward */
    double* grad_c = (double*)calloc(n_pre * n_post, sizeof(double));
    fn_backward_spikes(times, sources, n_spikes,
                        fwd_times_c, fwd_sources_c, fwd_currents_c,
                        post_errors_c, n_fwd_c,
                        w, n_pre, n_post, 1.0, 0.02, 0.005, grad_c);

    /* Eigen backward */
    RowMatrixXd w_mat(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++) w_mat(i) = w[i];

    Spikes input_eig(Eigen::Map<Eigen::ArrayXd>(times, n_spikes),
                     Eigen::Map<Eigen::ArrayXi>((int32_t*)sources, n_spikes));
    Spikes post_eig(Eigen::Map<Eigen::ArrayXd>(fwd_times_c, n_fwd_c),
                    Eigen::Map<Eigen::ArrayXi>((int32_t*)fwd_sources_c, n_fwd_c),
                    Eigen::Map<Eigen::ArrayXd>(fwd_currents_c, n_fwd_c),
                    std::vector<double>(), std::vector<int>());
    for (int i = 0; i < n_fwd_c; i++) post_eig.set_error(i, 1.0);

    auto [grad_eig_map, spike_errors_eig] = backward(input_eig, post_eig, w_mat, 1.0, 0.02, 0.005);

    double* grad_eig = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) grad_eig[i] = grad_eig_map(i);

    ASSERT_ARRAY_NEAR(grad_c, grad_eig, n_pre * n_post, ABS_TOLERANCE, REL_TOLERANCE, "gradient");

    free(w); free(times); free(sources);
    free(fwd_times_c); free(fwd_sources_c); free(fwd_currents_c);
    free(post_errors_c); free(grad_c); free(grad_eig);
    TEST_END;
}

/* ── Test: ComputeMaxima ───────────────────────────────────────────── */

static void test_compute_maxima(const char* pattern, int n_pre, int n_post, int n_spikes)
{
    char name[256];
    snprintf(name, sizeof(name), "ComputeMaxima %s (%dx%d, %d spikes)", pattern, n_pre, n_post, n_spikes);
    TEST(name);

    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));

    if (strcmp(pattern, "uniform") == 0) generate_uniform_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "poisson") == 0) generate_poisson_data(n_pre, n_post, n_spikes, w, times, sources);
    else generate_burst_data(n_pre, n_post, n_spikes, w, times, sources);

    sort_spikes(times, sources, n_spikes);

    /* C maxima */
    double* max_times_c = (double*)malloc(n_post * sizeof(double));
    double* max_values_c = (double*)malloc(n_post * sizeof(double));
    fn_compute_maxima(w, n_pre, n_post, times, sources, n_spikes,
                      0.02, 0.005, max_times_c, max_values_c);

    /* Eigen maxima */
    RowMatrixXd w_mat(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++) w_mat(i) = w[i];
    Spikes spikes_eig(Eigen::Map<Eigen::ArrayXd>(times, n_spikes),
                      Eigen::Map<Eigen::ArrayXi>((int32_t*)sources, n_spikes));
    auto maxima_eig = compute_maxima(w_mat, spikes_eig, 0.02, 0.005);

    double* max_times_e = (double*)malloc(n_post * sizeof(double));
    double* max_values_e = (double*)malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) {
        max_times_e[i] = maxima_eig.times(i);
        max_values_e[i] = maxima_eig.values(i);
    }

    /* Compare: NaN pairs are OK, otherwise values must match */
    for (int i = 0; i < n_post; i++) {
        bool c_nan = isnan(max_times_c[i]);
        bool e_nan = std::isnan(max_times_e[i]);
        if (c_nan && e_nan) continue;  /* both NaN — OK */
        if (c_nan != e_nan) {
            printf("FAIL [maxima[%d]: C NaN=%d, Eigen NaN=%d]\n", i, c_nan, e_nan);
            tests_failed++;
            free(w); free(times); free(sources);
            free(max_times_c); free(max_values_c); free(max_times_e); free(max_values_e);
            return;
        }
        ASSERT_NEAR_SCALED(max_times_c[i], max_times_e[i], ABS_TOLERANCE, REL_TOLERANCE, "max_times");
        ASSERT_NEAR_SCALED(max_values_c[i], max_values_e[i], ABS_TOLERANCE, REL_TOLERANCE, "max_values");
    }

    free(w); free(times); free(sources);
    free(max_times_c); free(max_values_c); free(max_times_e); free(max_values_e);
    TEST_END;
}

/* ── Test: BackwardMaxima ──────────────────────────────────────────── */

static void test_backward_maxima(const char* pattern, int n_pre, int n_post, int n_spikes)
{
    char name[256];
    snprintf(name, sizeof(name), "BackwardMaxima %s (%dx%d, %d spikes)", pattern, n_pre, n_post, n_spikes);
    TEST(name);

    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));

    if (strcmp(pattern, "uniform") == 0) generate_uniform_data(n_pre, n_post, n_spikes, w, times, sources);
    else if (strcmp(pattern, "poisson") == 0) generate_poisson_data(n_pre, n_post, n_spikes, w, times, sources);
    else generate_burst_data(n_pre, n_post, n_spikes, w, times, sources);

    sort_spikes(times, sources, n_spikes);

    /* C maxima */
    double* max_times_c = (double*)malloc(n_post * sizeof(double));
    double* max_values_c = (double*)malloc(n_post * sizeof(double));
    fn_compute_maxima(w, n_pre, n_post, times, sources, n_spikes,
                      0.02, 0.005, max_times_c, max_values_c);

    /* C backward maxima */
    double* max_errors_c = (double*)malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) max_errors_c[i] = 1.0;
    double* grad_c = (double*)calloc(n_pre * n_post, sizeof(double));
    fn_backward_maxima(times, sources, n_spikes,
                        max_times_c, max_values_c, max_errors_c, n_post,
                        w, n_pre, n_post, 0.02, 0.005, grad_c);

    /* Eigen maxima */
    RowMatrixXd w_mat(n_pre, n_post);
    for (int i = 0; i < n_pre * n_post; i++) w_mat(i) = w[i];
    Spikes spikes_eig(Eigen::Map<Eigen::ArrayXd>(times, n_spikes),
                      Eigen::Map<Eigen::ArrayXi>((int32_t*)sources, n_spikes));
    auto maxima_eig = compute_maxima(w_mat, spikes_eig, 0.02, 0.005);

    /* Eigen backward maxima */
    Spikes input_eig(Eigen::Map<Eigen::ArrayXd>(times, n_spikes),
                     Eigen::Map<Eigen::ArrayXi>((int32_t*)sources, n_spikes));
    Maxima maxima_obj(Eigen::Map<Eigen::ArrayXd>(max_times_c, n_post),
                      Eigen::Map<Eigen::ArrayXd>(max_values_c, n_post));
    for (int i = 0; i < n_post; i++) maxima_obj.set_error(i, 1.0);

    auto [grad_eig_map, spike_errors_eig] = backward(input_eig, maxima_obj, w_mat, 0.02, 0.005);

    double* grad_eig = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) grad_eig[i] = grad_eig_map(i);

    ASSERT_ARRAY_NEAR(grad_c, grad_eig, n_pre * n_post, ABS_TOLERANCE, REL_TOLERANCE, "gradient");

    free(w); free(times); free(sources);
    free(max_times_c); free(max_values_c); free(max_errors_c);
    free(grad_c); free(grad_eig);
    TEST_END;
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char* argv[])
{
    printf("EventProp C vs Eigen Parity Tests\n");
    printf("=================================\n\n");

    /* Determine C library path */
    const char* c_lib_path = nullptr;
    if (argc > 1) {
        c_lib_path = argv[1];
    } else {
        /* Default: look next to the test binary */
        c_lib_path = "./libeventprop_c_api.dylib";
    }

    if (!load_c_library(c_lib_path)) {
        return 1;
    }

    /* Problem sizes */
    struct TestCase { const char* pattern; int n_pre, n_post, n_spikes; };
    TestCase cases[] = {
        {"uniform", 10,   5,   20},
        {"uniform", 50,   20,  100},
        {"poisson", 50,   20,  100},
        {"burst",   50,   20,  100},
    };

    for (auto& tc : cases) {
        test_compute_sums(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes);
        test_compute_spikes(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes);
        test_backward_spikes(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes);
        test_compute_maxima(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes);
        test_backward_maxima(tc.pattern, tc.n_pre, tc.n_post, tc.n_spikes);
    }

    if (g_c_lib) dlclose(g_c_lib);

    printf("\n=================================\n");
    printf("Results: %d/%d passed, %d failed\n", tests_passed, tests_run, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
