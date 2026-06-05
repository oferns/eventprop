/**
 * @file tests.c
 * @brief EventProp C implementation test suite (alloc functions only).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "eventprop.h"

#define EPSILON 1e-6
static int tests_run = 0, tests_passed = 0, tests_failed = 0;

#define TEST(name) \
    do { printf("  Test: %-60s ", name); tests_run++; } while (0)

#define ASSERT(cond, msg) \
    if (!(cond)) { printf("FAIL [%s]\n", msg); tests_failed++; return; }

#define ASSERT_NEAR(a, b, tol) \
    if (fabs((a) - (b)) > (tol)) { printf("FAIL [%g vs %g]\n", (double)(a), (double)(b)); tests_failed++; return; }

#define TEST_END \
    do { printf("PASS\n"); tests_passed++; } while (0)

/* ── Deterministic test data generation ───────────────────────────── */

static void gen_data(int n_pre, int n_post, int n_spikes,
                     double* w, double* times, int32_t* sources)
{
    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = 5.0 + sin((double)i * 0.1);
    double t_end = fmin(0.1, (double)n_spikes / 50.0);
    for (int i = 0; i < n_spikes; i++)
        times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * (t_end - 0.01);
    for (int i = 0; i < n_spikes; i++)
        sources[i] = i % n_pre;
}

/* ── Alloc function tests ─────────────────────────────────────── */

static void test_compute_sums(void)
{
    TEST("ComputeSums Alloc Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    SumsOutput output = eventprop_compute_sums(w, n_pre, n_post,
                                                      times, sources, n_spikes,
                                                      0.02, 0.005);

    ASSERT(output.n_rows == n_spikes + 1, "wrong row count");
    ASSERT(output.n_post == n_post, "wrong post count");

    /* Row 0 should be zeros */
    for (int j = 0; j < n_post; j++) {
        ASSERT_NEAR(output.sum0[j], 0.0, EPSILON);
        ASSERT_NEAR(output.sum1[j], 0.0, EPSILON);
    }

    eventprop_free_sums_output(&output);
    TEST_END;
}

static void test_compute_sums_alloc_zero_spikes(void)
{
    TEST("ComputeSums Alloc zero spikes");
    SumsOutput output = eventprop_compute_sums(NULL, 10, 5,
                                                      NULL, NULL, 0,
                                                      0.02, 0.005);
    eventprop_free_sums_output(&output);
    TEST_END;
}

static void test_compute_sums_alloc_large(void)
{
    TEST("ComputeSums Alloc Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    SumsOutput output = eventprop_compute_sums(w, n_pre, n_post,
                                                      times, sources, n_spikes,
                                                      0.02, 0.005);

    ASSERT(output.n_rows == n_spikes + 1, "wrong row count");
    eventprop_free_sums_output(&output);

    free(w); free(times); free(sources);
    TEST_END;
}

static void test_compute_spikes(void)
{
    TEST("ComputeSpikes Alloc Tiny (10x5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    SpikeOutput output = eventprop_compute_spikes(w, n_pre, n_post,
                                                         times, sources, n_spikes,
                                                         1.0, 0.02, 0.005);

    ASSERT(output.n_spikes >= 0, "negative spike count");
    if (output.n_spikes > 1) {
        for (int i = 1; i < output.n_spikes; i++)
            ASSERT(output.times[i] >= output.times[i-1], "times not sorted");
    }

    eventprop_free_spike_output(&output);
    TEST_END;
}

static void test_backward_spikes(void)
{
    TEST("BackwardSpikes Alloc Tiny (10x5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Generate dummy forward pass data */
    double* fwd_times = malloc(n_spikes * sizeof(double));
    int32_t* fwd_sources = malloc(n_spikes * sizeof(int32_t));
    double* fwd_currents = malloc(n_spikes * sizeof(double));
    double* post_errors = malloc(n_spikes * sizeof(double));
    for (int i = 0; i < n_spikes; i++) {
        fwd_times[i] = times[i];
        fwd_sources[i] = i % n_post;
        fwd_currents[i] = 0.1;
        post_errors[i] = -1.0;
    }

    BackwardSpikesOutput output = eventprop_backward_spikes(
        times, sources, n_spikes,
        fwd_times, fwd_sources, fwd_currents, post_errors, n_spikes,
        w, n_pre, n_post, 1.0, 0.02, 0.005);

    ASSERT(output.n_pre == n_pre, "wrong pre count");
    ASSERT(output.n_post == n_post, "wrong post count");

    eventprop_free_backward_spikes_output(&output);
    free(fwd_times); free(fwd_sources); free(fwd_currents); free(post_errors);
    TEST_END;
}

static void test_compute_spikes_alloc_burst_medium(void)
{
    TEST("ComputeSpikes Alloc burst Medium (diagnostic)");
    int n_pre = 100, n_post = 40, n_spikes = 250;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    SpikeOutput output = eventprop_compute_spikes(w, n_pre, n_post,
                                                         times, sources, n_spikes,
                                                         1.0, 0.02, 0.005);

    printf(" (generated %d post spikes)\n", output.n_spikes);
    eventprop_free_spike_output(&output);

    free(w); free(times); free(sources);
    TEST_END;
}

static void test_compute_maxima(void)
{
    TEST("ComputeMaxima Alloc Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    MaximaOutput output = eventprop_compute_maxima(w, n_pre, n_post,
                                                          times, sources, n_spikes,
                                                          0.02, 0.005);

    ASSERT(output.n_post == n_post, "wrong post count");
    eventprop_free_maxima_output(&output);
    TEST_END;
}

static void test_backward_maxima(void)
{
    TEST("BackwardMaxima Alloc Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Generate dummy maxima data */
    double* maxima_times = malloc(n_post * sizeof(double));
    double* maxima_values = malloc(n_post * sizeof(double));
    double* maxima_errors = malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) {
        maxima_times[i] = times[n_spikes/2];
        maxima_values[i] = 0.5;
        maxima_errors[i] = -1.0;
    }

    BackwardMaximaOutput output = eventprop_backward_maxima(
        times, sources, n_spikes,
        maxima_times, maxima_values, maxima_errors, n_post,
        w, n_pre, n_post, 0.02, 0.005);

    ASSERT(output.n_pre == n_pre, "wrong pre count");
    eventprop_free_backward_maxima_output(&output);

    free(maxima_times); free(maxima_values); free(maxima_errors);
    TEST_END;
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(void)
{
    printf("EventProp C Tests (alloc functions only)\n");
    printf("==========================================\n\n");

    /* ComputeSums alloc tests */
    test_compute_sums();
    test_compute_sums_alloc_zero_spikes();
    test_compute_sums_alloc_large();

    /* ComputeSpikes alloc tests */
    test_compute_spikes();
    test_compute_spikes_alloc_burst_medium();

    /* BackwardSpikes alloc tests */
    test_backward_spikes();

    /* ComputeMaxima alloc tests */
    test_compute_maxima();

    /* BackwardMaxima alloc tests */
    test_backward_maxima();

    printf("\n==========================================\n");
    printf("Tests: %d passed, %d failed (of %d total)\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}
