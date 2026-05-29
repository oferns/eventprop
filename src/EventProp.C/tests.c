/**
 * @file tests.c
 * @brief EventProp C implementation test suite (TDD — tests first).
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
    /* Spread spikes over [0.01, 0.1] so maxima are near input times.
     * This keeps exp(-dt/tau_syn) from underflowing in BackwardMaxima. */
    double t_end = fmin(0.1, (double)n_spikes / 50.0);
    for (int i = 0; i < n_spikes; i++)
        times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * (t_end - 0.01);
    for (int i = 0; i < n_spikes; i++)
        sources[i] = i % n_pre;
}

/* ── Tests ───────────────────────────────────────────────────────── */

static void test_scratch_init(void)
{
    TEST("eventprop_scratch_init");
    EventPropScratch buf;
    eventprop_scratch_init(&buf, 10, 5, 20);
    ASSERT(buf.sum0 == NULL, "sum0 should be NULL");
    TEST_END;
}

static void test_compute_sums(void)
{
    TEST("ComputeSums Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    int n_sums = (n_spikes + 1) * n_post;
    double* sum0 = calloc(n_sums, sizeof(double));
    double* sum1 = calloc(n_sums, sizeof(double));

    eventprop_compute_sums(w, n_pre, n_post, times, sources, n_spikes,
                           0.02, 0.005, sum0, sum1);

    /* Row 0 should be zeros */
    for (int j = 0; j < n_post; j++) {
        ASSERT_NEAR(sum0[j], 0.0, EPSILON);
        ASSERT_NEAR(sum1[j], 0.0, EPSILON);
    }

    /* Row 1 should have contributions from first spike */
    int32_t src0 = sources[0];
    double exp_syn_0 = exp(times[0] / 0.005);
    for (int j = 0; j < n_post; j++) {
        double expected = w[src0 * n_post + j] * exp_syn_0;
        ASSERT_NEAR(sum0[n_post + j], expected, EPSILON * 10);
    }

    free(sum0); free(sum1);
    TEST_END;
}

static void test_compute_spikes(void)
{
    TEST("ComputeSpikes Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    int max_out = n_spikes * n_post;
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  1.0, 0.02, 0.005,
                                                  out_times, out_sources, out_currents);

    ASSERT(n_post_spikes >= 0, "negative spike count");
    if (n_post_spikes > 1) {
        for (int i = 1; i < n_post_spikes; i++)
            ASSERT(out_times[i] >= out_times[i-1], "times not sorted");
    }

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    TEST_END;
}

static void test_compute_spikes_medium(void)
{
    TEST("ComputeSpikes Medium (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    int max_out = (n_spikes + 1) * n_post; /* +1 per neuron for safety */
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  1.0, 0.02, 0.005,
                                                  out_times, out_sources, out_currents);

    if (n_post_spikes == 0 || n_post_spikes >= max_out)
        fprintf(stderr, "DEBUG: n_post_spikes=%d (expected >0 and <%d)\n", n_post_spikes, max_out);
    ASSERT(n_post_spikes > 0 && n_post_spikes < max_out, "spike count out of range");

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    TEST_END;
}

static void test_backward_spikes(void)
{
    TEST("BackwardSpikes Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    int max_out = n_spikes * n_post;
    double* fwd_times = malloc(max_out * sizeof(double));
    int32_t* fwd_sources = malloc(max_out * sizeof(int32_t));
    double* fwd_currents = malloc(max_out * sizeof(double));

    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post,
                                          times, sources, n_spikes,
                                          1.0, 0.02, 0.005,
                                          fwd_times, fwd_sources, fwd_currents);

    double* post_errors = calloc(n_fwd, sizeof(double));
    for (int i = 0; i < n_fwd; i++) post_errors[i] = 1.0;

    double* grad = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors = calloc(n_spikes, sizeof(double));
    eventprop_backward_spikes(times, sources, n_spikes,
                               fwd_times, fwd_sources, fwd_currents,
                               post_errors, n_fwd,
                               w, n_pre, n_post,
                               1.0, 0.02, 0.005, grad, spike_errors);

    /* No NaN/Inf in gradients */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT(!isnan(grad[i]), "grad NaN");
        ASSERT(!isinf(grad[i]), "grad Inf");
    }

    free(fwd_times); free(fwd_sources); free(fwd_currents);
    free(post_errors); free(grad); free(spike_errors);
    TEST_END;
}

static void test_compute_maxima(void)
{
    TEST("ComputeMaxima Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    double* max_times = malloc(n_post * sizeof(double));
    double* max_values = malloc(n_post * sizeof(double));

    eventprop_compute_maxima(w, n_pre, n_post,
                              times, sources, n_spikes,
                              0.02, 0.005, max_times, max_values);

    /* No NaN in times/values pairs */
    for (int i = 0; i < n_post; i++) {
        ASSERT(!isnan(max_times[i]) || isnan(max_values[i]), "times/values mismatch");
    }

    free(max_times); free(max_values);
    TEST_END;
}

static void test_backward_maxima(void)
{
    TEST("BackwardMaxima Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward: compute maxima */
    double* max_times = malloc(n_post * sizeof(double));
    double* max_values = malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post,
                              times, sources, n_spikes,
                              0.02, 0.005, max_times, max_values);

    /* Set up errors (all 1.0) */
    double* max_errors = malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;

    /* Backward pass */
    double* grad = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors = calloc(n_spikes, sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes,
                               max_times, max_values, max_errors, n_post,
                               w, n_pre, n_post,
                               0.02, 0.005, grad, spike_errors);

    /* No NaN/Inf in gradients */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT(!isnan(grad[i]), "grad NaN");
        ASSERT(!isinf(grad[i]), "grad Inf");
    }

    /* At least some gradient should be non-zero */
    double grad_sum = 0.0;
    for (int i = 0; i < n_pre * n_post; i++) grad_sum += fabs(grad[i]);
    ASSERT(grad_sum > 1e-12, "all gradients zero");

    free(max_times); free(max_values); free(max_errors); free(grad); free(spike_errors);
    TEST_END;
}

static void test_backward_maxima_zero_weights(void)
{
    TEST("BackwardMaxima ZeroWeights → ZeroGradients");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    memset(w, 0, sizeof(w)); /* all-zero weights */
    for (int i = 0; i < n_spikes; i++) {
        times[i] = (i + 1.0) / (n_spikes + 1.0);
        sources[i] = i % n_pre;
    }

    double* max_times = malloc(n_post * sizeof(double));
    double* max_values = malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post,
                              times, sources, n_spikes,
                              0.02, 0.005, max_times, max_values);

    /* With zero weights all maxima should be NaN or -Inf */
    double* max_errors = malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;

    double* grad = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors = calloc(n_spikes, sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes,
                               max_times, max_values, max_errors, n_post,
                               w, n_pre, n_post,
                               0.02, 0.005, grad, spike_errors);

    /* All gradients should be zero (NaN maxima are skipped) */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT_NEAR(grad[i], 0.0, EPSILON);
    }

    free(max_times); free(max_values); free(max_errors); free(grad); free(spike_errors);
    TEST_END;
}

static void test_backward_maxima_linear_scale(void)
{
    TEST("BackwardMaxima GradientScalesLinearly");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward: compute maxima */
    double* max_times = malloc(n_post * sizeof(double));
    double* max_values = malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post,
                              times, sources, n_spikes,
                              0.02, 0.005, max_times, max_values);

    /* Backward with errors=1.0 */
    double* max_errors = malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;
    double* grad1 = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors1 = calloc(n_spikes, sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes,
                               max_times, max_values, max_errors, n_post,
                               w, n_pre, n_post, 0.02, 0.005, grad1, spike_errors1);

    /* Backward with errors=2.0 */
    for (int i = 0; i < n_post; i++) max_errors[i] = 2.0;
    double* grad2 = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors2 = calloc(n_spikes, sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes,
                               max_times, max_values, max_errors, n_post,
                               w, n_pre, n_post, 0.02, 0.005, grad2, spike_errors2);

    /* grad2 should be exactly 2× grad1 */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT_NEAR(grad2[i], 2.0 * grad1[i], EPSILON);
    }

    free(max_times); free(max_values); free(max_errors);
    free(grad1); free(grad2); free(spike_errors1); free(spike_errors2);
    TEST_END;
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void)
{
    printf("EventProp C Implementation Tests\n=================================\n\n");

    test_scratch_init();
    test_compute_sums();
    test_compute_spikes();
    test_compute_spikes_medium();
    test_backward_spikes();
    test_compute_maxima();
    test_backward_maxima();
    test_backward_maxima_zero_weights();
    test_backward_maxima_linear_scale();

    printf("\n=================================\n");
    printf("Results: %d/%d passed, %d failed\n", tests_passed, tests_run, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
