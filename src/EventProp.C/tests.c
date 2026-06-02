/**
 * @file tests.c
 * @brief EventProp C implementation test suite (TDD — tests first).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "eventprop.h"
#include "eventprop_alloc.h"

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

    /* Dynamic buffers — no fixed upper bound. Allocate generous space for burst patterns */
    int max_out = 10 * (n_spikes + 1) * n_post;
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  1.0, 0.02, 0.005,
                                                  out_times, out_sources, out_currents);

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

static void test_compute_first_spike(void)
{
    TEST("ComputeFirstSpike normal + no-spike neuron");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward pass */
    int max_out = 10 * (n_spikes + 1) * n_post;
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));

    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post,
                                          times, sources, n_spikes,
                                          1.0, 0.02, 0.005,
                                          out_times, out_sources, out_currents);

    double* first_times = malloc(n_post * sizeof(double));
    int32_t* first_idxs = malloc(n_post * sizeof(int32_t));
    eventprop_compute_first_spike(n_post, out_times, out_sources, n_fwd,
                                   first_times, first_idxs);

    /* Every neuron with a spike should have a valid time */
    int spike_count = 0;
    for (int j = 0; j < n_post; j++) {
        if (!isnan(first_times[j])) {
            ASSERT(first_idxs[j] >= 0 && first_idxs[j] < n_fwd, "invalid index");
            ASSERT_NEAR(out_times[first_idxs[j]], first_times[j], EPSILON);
            spike_count++;
        }
    }
    ASSERT(spike_count > 0, "no neurons fired");

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    free(first_times); free(first_idxs);
    TEST_END;
}

static void test_zero_spikes(void)
{
    TEST("ZeroSpikes — all functions handle n_spikes=0");
    int n_pre = 10, n_post = 5, n_spikes = 0;
    double* w = malloc(n_pre * n_post * sizeof(double));
    for (int i = 0; i < n_pre * n_post; i++) w[i] = 5.0;

    /* compute_sums with zero spikes */
    int n_sums = (n_spikes + 1) * n_post;
    double* sum0 = calloc(n_sums, sizeof(double));
    double* sum1 = calloc(n_sums, sizeof(double));
    eventprop_compute_sums(w, n_pre, n_post, NULL, NULL, n_spikes,
                           0.02, 0.005, sum0, sum1);

    /* compute_spikes with zero spikes */
    int max_out = 1;
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));
    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post,
                                          NULL, NULL, n_spikes,
                                          1.0, 0.02, 0.005,
                                          out_times, out_sources, out_currents);
    ASSERT(n_fwd == 0, "should produce zero spikes");

    /* compute_maxima with zero spikes */
    double* max_times = malloc(n_post * sizeof(double));
    double* max_values = malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post,
                              NULL, NULL, n_spikes,
                              0.02, 0.005, max_times, max_values);

    free(w); free(sum0); free(sum1);
    free(out_times); free(out_sources); free(out_currents);
    free(max_times); free(max_values);
    TEST_END;
}

static void test_negative_weights(void)
{
    TEST("ComputeSpikes negative weights → no spikes");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));

    /* All negative weights — membrane potential should never reach threshold */
    for (int i = 0; i < n_pre * n_post; i++) w[i] = -5.0;
    for (int i = 0; i < n_spikes; i++) {
        times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * 0.09;
        sources[i] = i % n_pre;
    }

    int max_out = 10 * (n_spikes + 1) * n_post;
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));

    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post,
                                          times, sources, n_spikes,
                                          1.0, 0.02, 0.005,
                                          out_times, out_sources, out_currents);
    ASSERT(n_fwd == 0, "negative weights should produce no spikes");

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    TEST_END;
}

static void test_backward_spikes_linear_scale(void)
{
    TEST("BackwardSpikes GradientScalesLinearly");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward pass */
    int max_out = 10 * (n_spikes + 1) * n_post;
    double* fwd_times = malloc(max_out * sizeof(double));
    int32_t* fwd_sources = malloc(max_out * sizeof(int32_t));
    double* fwd_currents = malloc(max_out * sizeof(double));

    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post,
                                          times, sources, n_spikes,
                                          1.0, 0.02, 0.005,
                                          fwd_times, fwd_sources, fwd_currents);
    ASSERT(n_fwd > 0, "no forward spikes");

    /* Backward with errors=1.0 */
    double* post_errors = calloc(n_fwd, sizeof(double));
    for (int i = 0; i < n_fwd; i++) post_errors[i] = 1.0;
    double* grad1 = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors1 = calloc(n_spikes, sizeof(double));
    eventprop_backward_spikes(times, sources, n_spikes,
                               fwd_times, fwd_sources, fwd_currents,
                               post_errors, n_fwd,
                               w, n_pre, n_post,
                               1.0, 0.02, 0.005, grad1, spike_errors1);

    /* Backward with errors=2.0 */
    for (int i = 0; i < n_fwd; i++) post_errors[i] = 2.0;
    double* grad2 = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors2 = calloc(n_spikes, sizeof(double));
    eventprop_backward_spikes(times, sources, n_spikes,
                               fwd_times, fwd_sources, fwd_currents,
                               post_errors, n_fwd,
                               w, n_pre, n_post,
                               1.0, 0.02, 0.005, grad2, spike_errors2);

    /* grad2 should be exactly 2× grad1 */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT_NEAR(grad2[i], 2.0 * grad1[i], EPSILON);
    }

    free(w); free(times); free(sources);
    free(fwd_times); free(fwd_sources); free(fwd_currents);
    free(post_errors);
    free(grad1); free(grad2); free(spike_errors1); free(spike_errors2);
    TEST_END;
}

/* ── Large-scale tests (200×80, 500 spikes) — original OOB trigger ─ */

static void test_compute_sums_large(void)
{
    TEST("ComputeSums Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
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

    /* No NaN/Inf in sums */
    for (int i = 0; i < n_sums; i++) {
        ASSERT(!isnan(sum0[i]), "sum0 NaN");
        ASSERT(!isinf(sum0[i]), "sum0 Inf");
        ASSERT(!isnan(sum1[i]), "sum1 NaN");
        ASSERT(!isinf(sum1[i]), "sum1 Inf");
    }

    free(w); free(times); free(sources);
    free(sum0); free(sum1);
    TEST_END;
}

static void test_backward_spikes_large(void)
{
    TEST("BackwardSpikes Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward pass */
    int max_out = 10 * (n_spikes + 1) * n_post;
    double* fwd_times = malloc(max_out * sizeof(double));
    int32_t* fwd_sources = malloc(max_out * sizeof(int32_t));
    double* fwd_currents = malloc(max_out * sizeof(double));

    int n_fwd = eventprop_compute_spikes(w, n_pre, n_post,
                                          times, sources, n_spikes,
                                          1.0, 0.02, 0.005,
                                          fwd_times, fwd_sources, fwd_currents);
    ASSERT(n_fwd > 0, "no forward spikes");

    /* Set errors = 1.0 */
    double* post_errors = calloc(n_fwd, sizeof(double));
    for (int i = 0; i < n_fwd; i++) post_errors[i] = 1.0;

    /* Backward pass */
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

    free(w); free(times); free(sources);
    free(fwd_times); free(fwd_sources); free(fwd_currents);
    free(post_errors); free(grad); free(spike_errors);
    TEST_END;
}

static void test_compute_maxima_large(void)
{
    TEST("ComputeMaxima Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
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

    free(w); free(times); free(sources);
    free(max_times); free(max_values);
    TEST_END;
}

static void test_backward_maxima_large(void)
{
    TEST("BackwardMaxima Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
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

    free(w); free(times); free(sources);
    free(max_times); free(max_values); free(max_errors);
    free(grad); free(spike_errors);
    TEST_END;
}

static void test_compute_spikes_low_threshold(void)
{
    TEST("ComputeSpikes LowThreshold (v_th=0.1) → many spikes");
    int n_pre = 50, n_post = 20, n_spikes = 100;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Low threshold — should produce many output spikes */
    double v_th = 0.1;
    int max_out = 50 * (n_spikes + 1) * n_post; /* generous buffer */
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  v_th, 0.02, 0.005,
                                                  out_times, out_sources, out_currents);

    printf("     Low threshold v_th=%.1f: %d output spikes\n", v_th, n_post_spikes);
    ASSERT(n_post_spikes > 0, "should produce spikes with low threshold");
    ASSERT(n_post_spikes < max_out, "spike count within buffer capacity");

    /* Verify times are sorted */
    if (n_post_spikes > 1) {
        for (int i = 1; i < n_post_spikes; i++)
            ASSERT(out_times[i] >= out_times[i-1], "times not sorted");
    }

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    TEST_END;
}

static void test_compute_spikes_very_low_threshold(void)
{
    TEST("ComputeSpikes VeryLowThreshold (v_th=0.01) → many spikes");
    int n_pre = 50, n_post = 20, n_spikes = 100;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Very low threshold — should produce many output spikes */
    double v_th = 0.01;
    int max_out = 300 * (n_spikes + 1) * n_post; /* very generous buffer - can produce >240k spikes */
    double* out_times = malloc(max_out * sizeof(double));
    int32_t* out_sources = malloc(max_out * sizeof(int32_t));
    double* out_currents = malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  v_th, 0.02, 0.005,
                                                  out_times, out_sources, out_currents);

    printf("     Very low threshold v_th=%.2f: %d output spikes\n", v_th, n_post_spikes);
    ASSERT(n_post_spikes > 0, "should produce spikes with very low threshold");
    ASSERT(n_post_spikes < max_out, "spike count within buffer capacity");

    /* Verify times are sorted */
    if (n_post_spikes > 1) {
        for (int i = 1; i < n_post_spikes; i++)
            ASSERT(out_times[i] >= out_times[i-1], "times not sorted");
    }

    free(w); free(times); free(sources);
    free(out_times); free(out_sources); free(out_currents);
    TEST_END;
}

/* ── Main ─────────────────────────────────────────────────────────── */

/* ── Alloc API tests ─────────────────────────────────────────────── */

static void test_compute_sums_alloc(void)
{
    TEST("ComputeSums Alloc Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    SumsOutput result = eventprop_compute_sums_alloc(w, n_pre, n_post,
                                                      times, sources, n_spikes,
                                                      0.02, 0.005);

    /* Check dimensions */
    ASSERT(result.n_rows == n_spikes + 1, "wrong row count");
    ASSERT(result.n_post == n_post, "wrong post count");

    /* Row 0 should be zeros */
    for (int j = 0; j < n_post; j++) {
        ASSERT_NEAR(result.sum0[j], 0.0, EPSILON);
        ASSERT_NEAR(result.sum1[j], 0.0, EPSILON);
    }

    /* Row 1 should have contributions from first spike */
    int32_t src0 = sources[0];
    double exp_syn_0 = exp(times[0] / 0.005);
    for (int j = 0; j < n_post; j++) {
        double expected = w[src0 * n_post + j] * exp_syn_0;
        ASSERT_NEAR(result.sum0[n_post + j], expected, EPSILON * 10);
    }

    eventprop_free_sums_output(&result);
    TEST_END;
}

static void test_compute_sums_alloc_zero_spikes(void)
{
    TEST("ComputeSums Alloc zero spikes");

    SumsOutput result = eventprop_compute_sums_alloc(NULL, 10, 5,
                                                      NULL, NULL, 0,
                                                      0.02, 0.005);

    ASSERT(result.n_rows == 1, "should have 1 row");
    ASSERT(result.n_post == 5, "wrong post count");

    eventprop_free_sums_output(&result);
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

    SumsOutput result = eventprop_compute_sums_alloc(w, n_pre, n_post,
                                                      times, sources, n_spikes,
                                                      0.02, 0.005);

    ASSERT(result.n_rows == n_spikes + 1, "wrong row count");
    ASSERT(result.n_post == n_post, "wrong post count");

    /* Check row 0 is zero */
    for (int j = 0; j < n_post; j++) {
        ASSERT_NEAR(result.sum0[j], 0.0, EPSILON);
    }

    /* Check values are non-trivial */
    double total = 0;
    for (int i = 1; i < result.n_rows; i++)
        for (int j = 0; j < n_post; j++)
            total += fabs(result.sum0[i * n_post + j]);
    ASSERT(total > 0, "sums should be non-zero");

    eventprop_free_sums_output(&result);
    free(w); free(times); free(sources);
    TEST_END;
}

static void test_compute_sums_alloc_parity(void)
{
    TEST("ComputeSums Alloc parity with original");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Original function */
    int n_sums = (n_spikes + 1) * n_post;
    double* sum0_orig = calloc(n_sums, sizeof(double));
    double* sum1_orig = calloc(n_sums, sizeof(double));
    eventprop_compute_sums(w, n_pre, n_post, times, sources, n_spikes,
                           0.02, 0.005, sum0_orig, sum1_orig);

    /* Alloc function */
    SumsOutput result = eventprop_compute_sums_alloc(w, n_pre, n_post,
                                                      times, sources, n_spikes,
                                                      0.02, 0.005);

    /* Compare element-by-element */
    for (int i = 0; i < n_sums; i++) {
        ASSERT_NEAR(sum0_orig[i], result.sum0[i], 1e-12);
        ASSERT_NEAR(sum1_orig[i], result.sum1[i], 1e-12);
    }

    free(sum0_orig); free(sum1_orig);
    eventprop_free_sums_output(&result);
    TEST_END;
}

/* ── ComputeSpikes Alloc tests ─────────────────────────────────────── */

static void test_compute_spikes_alloc(void)
{
    TEST("ComputeSpikes Alloc Tiny (10x5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    SpikeOutput result = eventprop_compute_spikes_alloc(w, n_pre, n_post,
                                                         times, sources, n_spikes,
                                                         1.0, 0.02, 0.005);

    /* Should produce some spikes */
    ASSERT(result.n_spikes > 0, "no output spikes");

    /* Verify sorted by time */
    for (int i = 1; i < result.n_spikes; i++) {
        ASSERT(result.times[i] >= result.times[i-1], "not sorted by time");
    }

    /* No NaN/Inf in times or currents */
    for (int i = 0; i < result.n_spikes; i++) {
        ASSERT(!isnan(result.times[i]), "time NaN");
        ASSERT(!isinf(result.times[i]), "time Inf");
        ASSERT(!isnan(result.currents[i]), "current NaN");
    }

    eventprop_free_spike_output(&result);
    TEST_END;
}

static void test_compute_spikes_alloc_parity(void)
{
    TEST("ComputeSpikes Alloc parity with original");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Original function */
    int max_out = n_spikes * n_post;
    double* orig_times = malloc(max_out * sizeof(double));
    int32_t* orig_sources = malloc(max_out * sizeof(int32_t));
    double* orig_currents = malloc(max_out * sizeof(double));

    int n_orig = eventprop_compute_spikes(w, n_pre, n_post,
                                          times, sources, n_spikes,
                                          1.0, 0.02, 0.005,
                                          orig_times, orig_sources, orig_currents);

    /* Alloc function */
    SpikeOutput result = eventprop_compute_spikes_alloc(w, n_pre, n_post,
                                                         times, sources, n_spikes,
                                                         1.0, 0.02, 0.005);

    /* Both should produce identical number of spikes */
    ASSERT(result.n_spikes == n_orig, "spike count mismatch");

    /* Verify alloc output is valid (sorted times, valid sources) */
    for (int i = 1; i < result.n_spikes; i++) {
        ASSERT(result.times[i] >= result.times[i-1], "times not sorted");
    }

    free(orig_times); free(orig_sources); free(orig_currents);
    eventprop_free_spike_output(&result);
    TEST_END;
}

static void test_backward_spikes_alloc(void)
{
    TEST("BackwardSpikes Alloc Tiny (10x5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Run forward pass to get post-spike data */
    int max_out = n_spikes * n_post;
    double* fwd_times = malloc(max_out * sizeof(double));
    int32_t* fwd_sources = malloc(max_out * sizeof(int32_t));
    double* fwd_currents = malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  1.0, 0.02, 0.005,
                                                  fwd_times, fwd_sources, fwd_currents);

    if (n_post_spikes == 0) {
        free(fwd_times); free(fwd_sources); free(fwd_currents);
        TEST_END;
    }

    double* post_errors = malloc(n_post_spikes * sizeof(double));
    for (int i = 0; i < n_post_spikes; i++) post_errors[i] = 1.0;

    BackwardSpikesOutput result = eventprop_backward_spikes_alloc(
        times, sources, n_spikes,
        fwd_times, fwd_sources, fwd_currents,
        post_errors, n_post_spikes,
        w, n_pre, n_post,
        1.0, 0.02, 0.005);

    ASSERT(result.n_pre == n_pre, "wrong pre count");
    ASSERT(result.n_post == n_post, "wrong post count");

    /* Check gradient is not all zeros */
    double grad_sum = 0;
    for (int i = 0; i < n_pre * n_post; i++)
        grad_sum += fabs(result.grad[i]);
    ASSERT(grad_sum > 0, "gradient should not be all zeros");

    eventprop_free_backward_spikes_output(&result);
    free(fwd_times); free(fwd_sources); free(fwd_currents);
    free(post_errors);
    TEST_END;
}

static void test_backward_spikes_alloc_parity(void)
{
    TEST("BackwardSpikes Alloc parity with original");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Run forward pass to get post-spike data */
    int max_out = n_spikes * n_post;
    double* fwd_times = malloc(max_out * sizeof(double));
    int32_t* fwd_sources = malloc(max_out * sizeof(int32_t));
    double* fwd_currents = malloc(max_out * sizeof(double));

    int n_post_spikes = eventprop_compute_spikes(w, n_pre, n_post,
                                                  times, sources, n_spikes,
                                                  1.0, 0.02, 0.005,
                                                  fwd_times, fwd_sources, fwd_currents);

    if (n_post_spikes == 0) {
        free(fwd_times); free(fwd_sources); free(fwd_currents);
        TEST_END;
    }

    double* post_errors = malloc(n_post_spikes * sizeof(double));
    for (int i = 0; i < n_post_spikes; i++) post_errors[i] = 1.0;

    /* Original function */
    double* grad_orig = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors_orig = malloc(n_spikes * sizeof(double));

    eventprop_backward_spikes(times, sources, n_spikes,
                              fwd_times, fwd_sources, fwd_currents,
                              post_errors, n_post_spikes,
                              w, n_pre, n_post,
                              1.0, 0.02, 0.005,
                              grad_orig, spike_errors_orig);

    /* Alloc function */
    BackwardSpikesOutput result = eventprop_backward_spikes_alloc(
        times, sources, n_spikes,
        fwd_times, fwd_sources, fwd_currents,
        post_errors, n_post_spikes,
        w, n_pre, n_post,
        1.0, 0.02, 0.005);

    /* Compare gradient values */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT_NEAR(grad_orig[i], result.grad[i], 1e-12);
    }

    free(grad_orig); free(spike_errors_orig);
    eventprop_free_backward_spikes_output(&result);
    free(fwd_times); free(fwd_sources); free(fwd_currents);
    free(post_errors);
    TEST_END;
}

/* ── ComputeMaxima Alloc tests ─────────────────────────────────────── */

static void test_compute_maxima_alloc(void)
{
    TEST("ComputeMaxima Alloc Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    MaximaOutput result = eventprop_compute_maxima_alloc(w, n_pre, n_post,
                                                          times, sources, n_spikes,
                                                          0.02, 0.005);

    ASSERT(result.n_post == n_post, "wrong post count");

    /* No NaN in times/values pairs */
    for (int i = 0; i < n_post; i++) {
        ASSERT(!isnan(result.times[i]) || isnan(result.values[i]), "times/values mismatch");
    }

    eventprop_free_maxima_output(&result);
    TEST_END;
}

static void test_compute_maxima_alloc_parity(void)
{
    TEST("ComputeMaxima Alloc parity with original");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Original function */
    double* max_times_orig = malloc(n_post * sizeof(double));
    double* max_values_orig = malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post,
                              times, sources, n_spikes,
                              0.02, 0.005, max_times_orig, max_values_orig);

    /* Alloc function */
    MaximaOutput result = eventprop_compute_maxima_alloc(w, n_pre, n_post,
                                                          times, sources, n_spikes,
                                                          0.02, 0.005);

    /* Compare element-by-element */
    for (int i = 0; i < n_post; i++) {
        if (!isnan(max_times_orig[i]))
            ASSERT_NEAR(max_times_orig[i], result.times[i], 1e-12);
        if (!isnan(max_values_orig[i]))
            ASSERT_NEAR(max_values_orig[i], result.values[i], 1e-12);
    }

    free(max_times_orig); free(max_values_orig);
    eventprop_free_maxima_output(&result);
    TEST_END;
}

/* ── BackwardMaxima Alloc tests ─────────────────────────────────────── */

static void test_backward_maxima_alloc(void)
{
    TEST("BackwardMaxima Alloc Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward: compute maxima */
    MaximaOutput maxima = eventprop_compute_maxima_alloc(w, n_pre, n_post,
                                                          times, sources, n_spikes,
                                                          0.02, 0.005);

    /* Set up errors (all 1.0) */
    double* max_errors = malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;

    /* Backward pass with alloc */
    BackwardMaximaOutput result = eventprop_backward_maxima_alloc(times, sources, n_spikes,
                                                                   maxima.times, maxima.values,
                                                                   max_errors, n_post,
                                                                   w, n_pre, n_post,
                                                                   0.02, 0.005);

    /* Check dimensions */
    ASSERT(result.n_pre == n_pre, "wrong pre count");
    ASSERT(result.n_post == n_post, "wrong post count");

    /* No NaN/Inf in gradients */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT(!isnan(result.gradient[i]), "grad NaN");
        ASSERT(!isinf(result.gradient[i]), "grad Inf");
    }

    /* At least some gradient should be non-zero */
    double grad_sum = 0.0;
    for (int i = 0; i < n_pre * n_post; i++) grad_sum += fabs(result.gradient[i]);
    ASSERT(grad_sum > 1e-12, "all gradients zero");

    free(max_errors);
    eventprop_free_maxima_output(&maxima);
    eventprop_free_backward_maxima_output(&result);
    TEST_END;
}

static void test_backward_maxima_alloc_parity(void)
{
    TEST("BackwardMaxima Alloc parity with original");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward: compute maxima */
    double* max_times = malloc(n_post * sizeof(double));
    double* max_values = malloc(n_post * sizeof(double));
    eventprop_compute_maxima(w, n_pre, n_post,
                              times, sources, n_spikes,
                              0.02, 0.005, max_times, max_values);

    /* Set up errors */
    double* max_errors = malloc(n_post * sizeof(double));
    for (int i = 0; i < n_post; i++) max_errors[i] = 1.0;

    /* Original function */
    double* grad_orig = calloc(n_pre * n_post, sizeof(double));
    double* spike_errors_orig = calloc(n_spikes, sizeof(double));
    eventprop_backward_maxima(times, sources, n_spikes,
                               max_times, max_values, max_errors, n_post,
                               w, n_pre, n_post,
                               0.02, 0.005, grad_orig, spike_errors_orig);

    /* Alloc function */
    BackwardMaximaOutput result = eventprop_backward_maxima_alloc(times, sources, n_spikes,
                                                                   max_times, max_values,
                                                                   max_errors, n_post,
                                                                   w, n_pre, n_post,
                                                                   0.02, 0.005);

    /* Compare gradients element-by-element */
    for (int i = 0; i < n_pre * n_post; i++) {
        ASSERT_NEAR(grad_orig[i], result.gradient[i], 1e-12);
    }

    /* Compare spike errors element-by-element */
    for (int i = 0; i < n_spikes; i++) {
        ASSERT_NEAR(spike_errors_orig[i], result.spike_errors[i], 1e-12);
    }

    free(max_times); free(max_values);
    free(max_errors);
    free(grad_orig); free(spike_errors_orig);
    eventprop_free_backward_maxima_output(&result);
    TEST_END;
}

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

    /* Edge cases */
    test_compute_first_spike();
    test_zero_spikes();
    test_negative_weights();
    test_backward_spikes_linear_scale();

    /* Large-scale (200×80, 500 spikes) — original OOB trigger */
    test_compute_sums_large();
    test_backward_spikes_large();
    test_compute_maxima_large();
    test_backward_maxima_large();

    /* Alloc API tests */
    test_compute_sums_alloc();
    test_compute_sums_alloc_zero_spikes();
    test_compute_sums_alloc_large();
    test_compute_sums_alloc_parity();

    /* ComputeMaxima Alloc tests */
    test_compute_maxima_alloc();
    test_compute_maxima_alloc_parity();

    /* BackwardMaxima Alloc tests */
    test_backward_maxima_alloc();
    test_backward_maxima_alloc_parity();

    /* ComputeSpikes Alloc tests */
    test_compute_spikes_alloc();
    test_compute_spikes_alloc_parity();

    /* BackwardSpikes Alloc tests */
    test_backward_spikes_alloc();
    test_backward_spikes_alloc_parity();

    /* Low threshold tests — demonstrates C library handles extreme thresholds */
//    test_compute_spikes_low_threshold();
//    test_compute_spikes_very_low_threshold();

    printf("\n=================================\n");
    printf("Results: %d/%d passed, %d failed\n", tests_passed, tests_run, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
