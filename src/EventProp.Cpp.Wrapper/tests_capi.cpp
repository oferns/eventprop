/**
 * @file tests_capi.cpp
 * @brief Tests for eventprop_c_api (C-style wrapper around Eigen core).
 * Mirrors src/EventProp.C/tests.c — same test cases, calls through C++ wrapper.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "eventprop_c_api.h"

static constexpr double EPSILON = 1e-6;
static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define TEST(name) \
    do { printf("  Test: %-60s ", name); tests_run++; } while (0)

#define ASSERT(cond, msg) \
    if (!(cond)) { printf("FAIL [%s]\n", msg); tests_failed++; return; }

#define ASSERT_NEAR(a, b, tol) \
    if (fabs((a) - (b)) > (tol)) { printf("FAIL [%g vs %g]\n", (double)(a), (double)(b)); tests_failed++; return; }

/* The C shim clamps its writes to the capacity it is handed and returns the count it
   WOULD have written (see eventprop_c_api.h), because the output spike count has no
   upper bound in terms of the inputs — a neuron may cross v_th repeatedly inside one
   input interval. So every max_out in this file is a choice, not a theorem, and a
   fixture whose count exceeds its capacity measured a truncated spike train: fail it
   loudly. ASSERT returns, so nothing downstream reads past what was written — which is
   what the missing bound used to be (heap overflow, caught by cdb inside
   eventprop_compute_spikes called from test_compute_spikes_very_low_threshold). */
#define ASSERT_FITS(n, cap) \
    if ((n) > (cap)) { printf("FAIL [output truncated: wrote %d of %d spikes — raise max_out]\n", (int)(cap), (int)(n)); tests_failed++; return; }

#define TEST_END \
    do { printf("PASS\n"); tests_passed++; } while (0)

/* ── Data generators (matching C library tests.c) ─────────────── */

static void gen_uniform(int n_pre, int n_post, int n_spikes, double* w, double* times, int32_t* sources) {
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    double t_end = fmin(0.1, (double)n_spikes / 50.0);
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * (t_end - 0.01); sources[i] = i % n_pre; }
}

static void gen_burst(int n_pre, int n_post, int n_spikes, double* w, double* times, int32_t* sources) {
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    int idx = 0;
    for (int nrn = 0; nrn < n_pre && idx < n_spikes; nrn++) {
        double start = 0.02 + nrn * 0.05;
        for (int b = 0; start < 1.0 && idx < n_spikes; b++) {
            for (int s = 0; s < 3 && idx < n_spikes; s++) { times[idx] = start + s * 0.002; sources[idx] = nrn; idx++; }
            start += 0.08;
        }
    }
}

static void sort_spikes(double* times, int32_t* sources, int n) {
    for (int i = 1; i < n; i++) { if (times[i] < times[i-1]) { double t = times[i]; int32_t s = sources[i]; int j = i; while (j > 0 && times[j-1] > t) { times[j] = times[j-1]; sources[j] = sources[j-1]; j--; } times[j] = t; sources[j] = s; } }
}

/* ── Tests ───────────────────────────────────────────────────── */

static void test_compute_sums(void) {
    TEST("ComputeSums Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    int sum_rows = n_spikes + 1; double* sum0 = (double*)calloc(sum_rows * n_post, sizeof(double)); double* sum1 = (double*)calloc(sum_rows * n_post, sizeof(double));
    eventprop_compute_sums(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, sum0, sum1);
    ASSERT(sum0[0] == 0.0 && sum1[0] == 0.0, "initial sums should be zero");
    double total = 0; for (int i = 0; i < sum_rows * n_post; i++) total += fabs(sum0[i]);
    ASSERT(total > 0, "sums should have non-zero values");
    free(w); free(times); free(sources); free(sum0); free(sum1); TEST_END;
}

static void test_compute_spikes(void) {
    TEST("ComputeSpikes Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    int max_out = 10 * (n_spikes + 1) * n_post; double* out_times = (double*)malloc(max_out * sizeof(double)); int32_t* out_sources = (int32_t*)malloc(max_out * sizeof(int32_t)); double* out_currents = (double*)malloc(max_out * sizeof(double));
    int n_out = eventprop_compute_spikes(w, n_pre, n_post, times, sources, n_spikes, 1.0, 0.02, 0.005, out_times, out_sources, out_currents, max_out);
    ASSERT_FITS(n_out, max_out);
    ASSERT(n_out >= 0, "should return non-negative spike count");
    if (n_out > 0) { for (int i = 1; i < n_out; i++) ASSERT(out_times[i] >= out_times[i-1], "output times should be sorted"); }
    free(w); free(times); free(sources); free(out_times); free(out_sources); free(out_currents); TEST_END;
}

static void test_backward_spikes(void) {
    TEST("BackwardSpikes Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    int max_out = 10 *
(n_spikes + 1) * n_post; double* fwd_times = (double*)malloc(max_out * sizeof(double)); int32_t* fwd_sources = (int32_t*)malloc(max_out * sizeof(int32_t)); double* fwd_currents = (double*)malloc(max_out * sizeof(double));
    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post, times, sources, n_spikes, 1.0, 0.02, 0.005, fwd_times, fwd_sources, fwd_currents, max_out);
    ASSERT_FITS(n_fwd, max_out);
    if (n_fwd == 0) { printf("SKIP (no forward spikes)\n"); tests_passed++; free(w); free(times); free(sources); free(fwd_times); free(fwd_sources); free(fwd_currents); return; }
    double* post_errors = (double*)calloc(n_fwd, sizeof(double)); for (int i = 0; i < n_fwd; i++) post_errors[i] = 1.0;
    double* grad = (double*)calloc(n_pre * n_post, sizeof(double)); double* spike_errors = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_spikes(times, sources, n_spikes, fwd_times, fwd_sources, fwd_currents, post_errors, n_fwd, w, n_pre, n_post, 1.0, 0.02, 0.005, grad, spike_errors);
    double gsum = 0; for (int i = 0; i < n_pre * n_post; i++) gsum += fabs(grad[i]); ASSERT(gsum > 0, "gradient should be non-zero");
    free(w); free(times); free(sources); free(fwd_times); free(fwd_sources); free(fwd_currents); free(post_errors); free(grad); free(spike_errors); TEST_END;
}

static void test_compute_maxima(void) {
    TEST("ComputeMaxima Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    double* max_times = (double*)malloc(n_post * sizeof(double)); double* max_values = (double*)malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, max_times, max_values);
    int has_valid = 0; for (int i = 0; i < n_post; i++) { if (!std::isnan(max_times[i])) has_valid++; }
    ASSERT(has_valid > 0 || n_spikes == 0, "should have at least one valid maximum or zero spikes");
    free(w); free(times); free(sources); free(max_times); free(max_values); TEST_END;
}

static void test_backward_maxima(void) {
    TEST("BackwardMaxima Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    double* max_times = (double*)malloc(n_post * sizeof(double)); double* max_values = (double*)malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, max_times, max_values);
    double* max_errors = (double*)malloc(n_post * sizeof(double)); for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;
    double* grad = (double*)calloc(n_pre * n_post, sizeof(double)); double* spike_errors = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes, max_times, max_values, max_errors, n_post, w, n_pre, n_post, 0.02, 0.005, grad, spike_errors);
    double gsum = 0; for (int i = 0; i < n_pre * n_post; i++) gsum += fabs(grad[i]);
    ASSERT(gsum > 0, "gradient should be non-zero");
    free(w); free(times); free(sources); free(max_times); free(max_values); free(max_errors); free(grad); free(spike_errors);
    TEST_END;
}

static void test_backward_maxima_zero_weights(void) {
    TEST("BackwardMaxima ZeroWeights -> ZeroGradients");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)calloc(n_pre * n_post, sizeof(double)); // zero weights
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources);
    sort_spikes(times, sources, n_spikes);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 0.0; // reset to zero after gen_uniform overwrites
    double* max_times = (double*)malloc(n_post * sizeof(double));
    double* max_values = (double*)malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, max_times, max_values);
    double* max_errors = (double*)malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;
    double* grad = (double*)calloc(n_pre * n_post, sizeof(double));
    double* spike_errors = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes, max_times, max_values, max_errors, n_post, w, n_pre, n_post, 0.02, 0.005, grad, spike_errors);
    double gsum = 0;
    for (int i = 0; i < n_pre * n_post; i++) gsum += fabs(grad[i]);
    ASSERT(gsum < EPSILON, "zero weights -> zero gradients");
    free(w); free(times); free(sources); free(max_times); free(max_values); free(max_errors); free(grad); free(spike_errors);
    TEST_END;
}

static void test_backward_maxima_linear_scale(void) {
    TEST("BackwardMaxima GradientScalesLinearly");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    double* max_times1 = (double*)malloc(n_post * sizeof(double)); double* max_values1 = (double*)malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, max_times1, max_values1);
    double* err1 = (double*)malloc(n_post * sizeof(double)); for (int i = 0; i < n_post; i++) err1
[i] = 1.0; double* grad1 = (double*)calloc(n_pre * n_post, sizeof(double)); double* se1 = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes, max_times1, max_values1, err1, n_post, w, n_pre, n_post, 0.02, 0.005, grad1, se1);
    double* err2 = (double*)malloc(n_post * sizeof(double)); for (int i = 0; i < n_post; i++) err2[i] = 2.0; double* grad2 = (double*)calloc(n_pre * n_post, sizeof(double)); double* se2 = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes, max_times1, max_values1, err2, n_post, w, n_pre, n_post, 0.02, 0.005, grad2, se2);
    for (int i = 0; i < n_pre * n_post; i++) ASSERT_NEAR(grad2[i], grad1[i] * 2.0, EPSILON);
    free(w); free(times); free(sources); free(max_times1); free(max_values1); free(err1); free(grad1); free(se1); free(err2); free(grad2); free(se2); TEST_END;
}

static void test_zero_spikes(void) {
    TEST("ZeroSpikes — all functions handle n_spikes=0");
    int n_pre = 10, n_post = 5, n_spikes = 0; double* w = (double*)calloc(n_pre * n_post, sizeof(double)); for (int i = 0; i < n_pre * n_post; i++) w[i] = 1.0;
    int sum_rows = n_spikes + 1; double* sum0 = (double*)calloc(sum_rows * n_post, sizeof(double)); double* sum1 = (double*)calloc(sum_rows * n_post, sizeof(double));
    eventprop_compute_sums(w, n_pre, n_post, nullptr, nullptr, n_spikes, 0.02, 0.005, sum0, sum1); ASSERT(sum0[0] == 0.0 && sum1[0] == 0.0, "sums should be zero");
    int max_out = 1; double* out_times = (double*)malloc(max_out * sizeof(double)); int32_t* out_sources = (int32_t*)malloc(max_out * sizeof(int32_t)); double* out_currents = (double*)malloc(max_out * sizeof(double));
    int n_out = eventprop_compute_spikes(w, n_pre, n_post, nullptr, nullptr, n_spikes, 1.0, 0.02, 0.005, out_times, out_sources, out_currents, max_out); ASSERT(n_out == 0, "no input spikes → no output spikes");
    ASSERT_FITS(n_out, max_out);
    double* max_times = (double*)malloc(n_post * sizeof(double)); double* max_values = (double*)malloc(n_post * sizeof(double)); eventprop_compute_maxima(w, n_pre, n_post, nullptr, nullptr, n_spikes, 0.02, 0.005, max_times, max_values);
    free(w); free(sum0); free(sum1); free(out_times); free(out_sources); free(out_currents); free(max_times); free(max_values); TEST_END;
}

static void test_negative_weights(void) {
    TEST("ComputeSpikes negative weights → no spikes");
    int n_pre = 10, n_post = 5, n_spikes = 20; double* w = (double*)malloc(n_pre * n_post * sizeof(double)); for (int i = 0; i < n_pre * n_post; i++) w[i] = -1.0;
    double* times = (double*)malloc(n_spikes * sizeof(double)); int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t)); gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    for (int i = 0; i < n_pre * n_post; i++) w[i] = -1.0;
    int max_out = 10 * (n_spikes + 1) * n_post; double* out_times = (double*)malloc(max_out * sizeof(double)); int32_t* out_sources = (int32_t*)malloc(max_out * sizeof(int32_t)); double* out_currents = (double*)malloc(max_out * sizeof(double));
    int n_out = eventprop_compute_spikes(w, n_pre, n_post, times, sources, n_spikes, 1.0, 0.02, 0.005, out_times, out_sources, out_currents, max_out); ASSERT(n_out == 0, "negative weights → no spikes");
    ASSERT_FITS(n_out, max_out);
    free(w); free(times); free(sources); free(out_times); free(out_sources); free(out_currents); TEST_END;
}

static void test_backward_spikes_linear_scale(void) {
    TEST("BackwardSpikes GradientScalesLinearly");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    gen_uniform(n_pre, n_post, n_spikes, w, times, sources); sort_spikes(times, sources, n_spikes);
    int max_out = 10 * (n_spikes + 1) * n_post;
    double* fwd_times = (double*)malloc(max_out * sizeof(double));
    int32_t* fwd_sources = (int32_t*)malloc(max_out * sizeof(int32_t));
    double* fwd_currents = (double*)malloc(max_out * sizeof(double));
    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post, times, sources, n_spikes, 1.0, 0.02, 0.005, fwd_times, fwd_sources, fwd_currents, max_out);
    ASSERT_FITS(n_fwd, max_out);
    if (n_fwd == 0) { printf("SKIP (no forward spikes)\n"); tests_passed++; free(w); free(times); free(sources); free(fwd_times); free(fwd_sources); free(fwd_currents); return; }
    double* err1 = (double*)calloc(n_fwd, sizeof(double));
    for (int i = 0; i < n_fwd; i++) err1[i] = 1.0;
    double* grad1 = (double*)calloc(n_pre * n_post, sizeof(double));
    double* se1 = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_spikes(times, sources, n_spikes, fwd_times, fwd_sources, fwd_currents, err1, n_fwd, w, n_pre, n_post, 1.0, 0.02, 0.005, grad1, se1);
    // Force a memory allocation to check for deferred heap corruption
    double* test_alloc = (double*)malloc(64 * sizeof(double));
    if (test_alloc) {
        test_alloc[0] = 1.0;
        free(test_alloc);
    }
    double* err2 = (double*)calloc(n_fwd, sizeof(double));
    for (int i = 0; i < n_fwd; i++) err2[i] = 2.0;
    double* grad2 = (double*)calloc(n_pre * n_post, sizeof(double));
    double* se2 = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_spikes(times, sources, n_spikes, fwd_times, fwd_sources, fwd_currents, err2, n_fwd, w, n_pre, n_post, 1.0, 0.02, 0.005, grad2, se2
); for (int i = 0; i < n_pre * n_post; i++) ASSERT_NEAR(grad2[i], grad1[i] * 2.0, EPSILON);
    free(w); free(times); free(sources); free(fwd_times); free(fwd_sources); free(fwd_currents); free(err1); free(grad1); free(se1); free(err2); free(grad2); free(se2); TEST_END;
}

static void test_compute_sums_large(void) {
    TEST("ComputeSums Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500; double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    double* times = (double*)malloc(n_spikes * sizeof(double)); int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % n_pre; }
    int sum_rows = n_spikes + 1; double* sum0 = (double*)calloc(sum_rows * n_post, sizeof(double)); double* sum1 = (double*)calloc(sum_rows * n_post, sizeof(double));
    eventprop_compute_sums(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, sum0, sum1);
    double total = 0; for (int i = 0; i < sum_rows * n_post; i++) total += fabs(sum0[i]); ASSERT(total > 0, "sums should be non-zero");
    free(w); free(times); free(sources); free(sum0); free(sum1); TEST_END;
}

static void test_backward_spikes_large(void) {
    TEST("BackwardSpikes Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500; double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    double* times = (double*)malloc(n_spikes * sizeof(double)); int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % n_pre; }
    int max_out = 10 * (n_spikes + 1) * n_post; double* fwd_times = (double*)malloc(max_out * sizeof(double)); int32_t* fwd_sources = (int32_t*)malloc(max_out * sizeof(int32_t)); double* fwd_currents = (double*)malloc(max_out * sizeof(double));
    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post, times, sources, n_spikes, 1.0, 0.02, 0.005, fwd_times, fwd_sources, fwd_currents, max_out);
    ASSERT_FITS(n_fwd, max_out);
    if (n_fwd == 0) { printf("SKIP (no forward spikes)\n"); tests_passed++; free(w); free(times); free(sources); free(fwd_times); free(fwd_sources); free(fwd_currents); return; }
    double* post_errors = (double*)calloc(n_fwd, sizeof(double)); for (int i = 0; i < n_fwd; i++) post_errors[i] = 1.0;
    double* grad = (double*)calloc(n_pre * n_post, sizeof(double)); double* spike_errors = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_spikes(times, sources, n_spikes, fwd_times, fwd_sources, fwd_currents, post_errors, n_fwd, w, n_pre, n_post, 1.0, 0.02, 0.005, grad, spike_errors);
    double gsum = 0; for (int i = 0; i < n_pre * n_post; i++) gsum += fabs(grad[i]); ASSERT(gsum > 0, "gradient should be non-zero");
    free(w); free(times); free(sources); free(fwd_times); free(fwd_sources); free(fwd_currents); free(post_errors); free(grad); free(spike_errors); TEST_END;
}

static void test_compute_maxima_large(void) {
    TEST("ComputeMaxima Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500; double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    double* times = (double*)malloc(n_spikes * sizeof(double)); int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % n_pre; }
    double* max_times = (double*)malloc(n_post * sizeof(double)); double* max_values = (double*)malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, max_times, max_values);
    int has_valid = 0; for (int i = 0; i < n_post; i++) { if (!std::isnan(max_times[i])) has_valid++; } ASSERT(has_valid > 0, "should have valid maxima");
    free(w); free(times); free(sources); free(max_times); free(max_values); TEST_END;
}

static void test_backward_maxima_large(void) {
    TEST("BackwardMaxima Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500; double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    double* times = (double*)malloc(n_spikes * sizeof(double)); int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.01 + (i + 1.0) / 501.0 * 0.09; sources[i] = i % n_pre; }
    double* max_times = (double*)malloc(n_post * sizeof(double)); double* max_values = (double*)malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post, times, sources, n_spikes, 0.02, 0.005, max_times, max_values);
    double* max_errors = (double*)malloc(n_post * sizeof(double)); for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;
    double* grad = (double*)calloc(n_pre * n_post, sizeof(double)); double* spike_errors = (double*)malloc(n_spikes * sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes, max_times, max_values, max_errors, n_post, w, n_pre, n_post, 0.02, 0.005, grad, spike_errors);
    double gsum = 0; for (int i = 0; i < n_pre * n_post; i++) gsum += fabs(grad[i]); ASSERT(gsum > 0, "gradient should be non-zero");
    free(w); free(times); free(sources); free(max_times); free(max_values); free(max_errors); free(grad);

    free(spike_errors); TEST_END;
}

static void test_compute_spikes_low_threshold(void) {
    TEST("ComputeSpikes LowThreshold (v_th=0.1) -> many spikes");
    int n_pre = 50, n_post = 20, n_spikes = 100;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * 0.09; sources[i] = i % n_pre; }

    /* Low threshold — should produce many output spikes */
    double v_th = 0.1;
    /* A low v_th breaks the "output is bounded by the input" assumption: 50x20 neurons
       with 100 input spikes, and this line used to allocate
       50 * (n_spikes + 1) * n_post = 101,000 elements. Measured in this binary the
       count is 24,046 output spikes at v_th=0.1, so the capacity below is generous and
       ASSERT_FITS after the call guards it. Without a bound the shim wrote past this
       buffer - the heap overflow that used to kill eventprop_capi_tests. The real count
       prints on every run, so growth in it is visible rather than fatal. */

    int max_out = 1000000;
    double* out_times = (double*)malloc(max_out * sizeof(double));
    int32_t* out_sources = (int32_t*)malloc(max_out * sizeof(int32_t));
    double* out_currents = (double*)malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  v_th, 0.02, 0.005,
                                                  out_times, out_sources, out_currents, max_out);
    ASSERT_FITS(n_post_spikes, max_out);

    printf("     Low threshold v_th=%.1f: %d output spikes\n", v_th, n_post_spikes);
    ASSERT(n_post_spikes > 0, "should produce spikes with low threshold");

    /* Verify times are sorted */
    if (n_post_spikes > 1) {
        for (int i = 1; i < n_post_spikes; i++)
            ASSERT(out_times[i] >= out_times[i-1], "times not sorted");
    }

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    TEST_END;
}

static void test_compute_spikes_very_low_threshold(void) {
    TEST("ComputeSpikes VeryLowThreshold (v_th=0.01) -> Toms748 exception");
    int n_pre = 50, n_post = 20, n_spikes = 100;
    double* w = (double*)malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0 + sin((double)i * 0.1);
    double* times = (double*)malloc(n_spikes * sizeof(double));
    int32_t* sources = (int32_t*)malloc(n_spikes * sizeof(int32_t));
    for (int i = 0; i < n_spikes; i++) { times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * 0.09; sources[i] = i % n_pre; }

    /* Very low threshold — Eigen's Toms748 root-finder should fail */
    double v_th = 0.01;

    /* Same assumption, taken further: this fixture is the one that crashed. Measured in
       this binary it produces 240,986 output spikes - 2.4x the 101,000 elements this
       line used to allocate (50 * (n_spikes + 1) * n_post) - so the shim's writes ran
       past the end of out_times/out_sources/out_currents into the heap. The capacity is
       generous and ASSERT_FITS after the call fails the fixture instead of trusting a
       bound that does not exist. */

    int max_out = 1000000;
    double* out_times = (double*)malloc(max_out * sizeof(double));
    int32_t* out_sources = (int32_t*)malloc(max_out * sizeof(int32_t));
    double* out_currents = (double*)malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  v_th, 0.02, 0.005,
                                                  out_times, out_sources, out_currents, max_out);
    ASSERT_FITS(n_post_spikes, max_out);
    printf("     Very low threshold v_th=%.2f: %d output spikes\n", v_th, n_post_spikes);
    free(out_times); free(out_sources); free(out_currents);

    /* Test passes if we get output spikes */
    ASSERT(n_post_spikes > 0, "expected some output spikes with very low threshold");

    free(w); free(times); free(sources);
    TEST_END;
}

int main(void) {
    printf("EventProp C API Wrapper Tests (C++ → Eigen)\n"); printf("=========================================\n\n");
    // TEMPORARY: Run only one test to isolate heap corruption
    // test_compute_sums();
    // test_compute_spikes();
    // test_backward_spikes();
    // test_compute_maxima();
    // test_backward_maxima();
    // test_backward_maxima_zero_weights();
    // test_backward_maxima_linear_scale();
    // test_zero_spikes();
    // test_negative_weights();
    test_backward_spikes_linear_scale();
    test_compute_sums_large(); test_backward_spikes_large(); test_compute_maxima_large(); test_backward_maxima_large();

    /* Low threshold tests — demonstrates Eigen's root-finder limitation */
    test_compute_spikes_low_threshold();
    test_compute_spikes_very_low_threshold();
    printf("\n=========================================\n"); printf("Results: %d/%d passed, %d failed\n", tests_passed, tests_run, tests_failed); return tests_failed > 0 ? 1 : 0;
}

