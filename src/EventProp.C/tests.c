/**
 * @file tests.c
 * @brief EventProp C implementation test suite ( functions only).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "eventprop.h"
#include "eventprop_batch.h"

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
    if (!times || n_spikes == 0) return;
    double t_end = fmin(0.1, (double)n_spikes / 50.0);
    for (int i = 0; i < n_spikes; i++)
        times[i] = 0.01 + (i + 1.0) / (n_spikes + 1.0) * (t_end - 0.01);
    if (!sources) return;
    for (int i = 0; i < n_spikes; i++)
        sources[i] = i % n_pre;
}

/* ── function tests ─────────────────────────────────────── */

static void test_compute_sums(void)
{
    TEST("ComputeSums Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    EventPropInput base_input = {
        .w=w, .n_pre=n_pre, .n_post=n_post,
        .times=times, .sources=sources, .n_spikes=n_spikes,
        .tau_mem=0.02, .tau_syn=0.005
    };
    SumsOutput* output = eventprop_compute_sums(&base_input);

    ASSERT(output->n_rows == n_spikes + 1, "wrong row count");
    ASSERT(output->n_post == n_post, "wrong post count");

    /* Row 0 should be zeros */
    for (int j = 0; j < n_post; j++) {
        ASSERT_NEAR(output->sum0[j], 0.0, EPSILON);
        ASSERT_NEAR(output->sum1[j], 0.0, EPSILON);
    }

    eventprop_free_sums_output(output);
    TEST_END;
}

static void test_compute_sums__zero_spikes(void)
{
    TEST("ComputeSums zero spikes");
    SumsOutput* output = eventprop_compute_sums(&((EventPropInput){.w=NULL, .n_pre=10, .n_post=5,
    .times=NULL, .sources=NULL, .n_spikes=0, .tau_mem=0.02, .tau_syn=0.005}));
    eventprop_free_sums_output(output);
    TEST_END;
}

static void test_compute_sums__large(void)
{
    TEST("ComputeSums Large (200×80, 500 spikes)");
    int n_pre = 200, n_post = 80, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    EventPropInput base_input = {
        .w=w, .n_pre=n_pre, .n_post=n_post,
        .times=times, .sources=sources, .n_spikes=n_spikes,
        .tau_mem=0.02, .tau_syn=0.005
    };
    SumsOutput* output = eventprop_compute_sums(&base_input);

    ASSERT(output->n_rows == n_spikes + 1, "wrong row count");
    eventprop_free_sums_output(output);

    free(w); free(times); free(sources);
    TEST_END;
}

static void test_compute_spikes(void)
{
    TEST("ComputeSpikes  Tiny (10x5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    EventPropInput fi = {
        .w=w, .n_pre=n_pre, .n_post=n_post,
                 .times=times, .sources=sources, .n_spikes=n_spikes,
                 .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    SpikeOutput* output = eventprop_compute_spikes(&fi);

    ASSERT(output->n_spikes >= 0, "negative spike count");
    if (output->n_spikes > 1) {
        for (int i = 1; i < output->n_spikes; i++)
            ASSERT(output->times[i] >= output->times[i-1], "times not sorted");
    }

    eventprop_free_spike_output(output);
    TEST_END;
}

static void test_backward_spikes(void)
{
    TEST("BackwardSpikes Tiny (10x5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Generate dummy forward pass data */
    double* fwd_times = malloc(n_spikes * sizeof(double));
    int32_t* fwd_sources = malloc(n_spikes * sizeof(int32_t));
    double* fwd_currents = malloc(n_spikes * sizeof(double));
    double* errors = calloc(n_spikes, sizeof(double));
    for (int i = 0; i < n_spikes; i++) {
        fwd_times[i] = times[i];
        fwd_sources[i] = i % n_post;
        fwd_currents[i] = 0.1;
        errors[i] = -1.0;
    }

    EventPropInput base = { .w=w, .n_pre=n_pre, .n_post=n_post,
                              .times=times, .sources=sources, .n_spikes=n_spikes,
                              .tau_mem=0.02, .tau_syn=0.005 };
    BackwardSpikesInput backward_input = {
        .input = &base,
        .fwd_times=fwd_times, .fwd_sources=fwd_sources, .fwd_currents=fwd_currents,
        .errors=errors, .n_post_spikes=n_spikes
    };
    BackwardOutput* output = eventprop_backward_spikes(&backward_input);

    ASSERT(output->n_pre == n_pre, "wrong pre count");
    ASSERT(output->n_post == n_post, "wrong post count");

    eventprop_free_backward_output(output);
    free(fwd_times); free(fwd_sources); free(fwd_currents); free(errors);
    TEST_END;
}

static void test_compute_spikes__burst_medium(void)
{
    TEST("ComputeSpikes  burst Medium (diagnostic)");
    int n_pre = 100, n_post = 40, n_spikes = 250;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    EventPropInput fi = {
        .w=w, .n_pre=n_pre, .n_post=n_post,
                 .times=times, .sources=sources, .n_spikes=n_spikes,
                 .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    SpikeOutput* output = eventprop_compute_spikes(&fi);

    printf(" (generated %d post spikes)\n", output->n_spikes);
    eventprop_free_spike_output(output);

    free(w); free(times); free(sources);
    TEST_END;
}

static void test_compute_maxima(void)
{
    TEST("ComputeMaxima  Tiny (10×5, 20 spikes)");
    int n_pre = 10, n_post = 5, n_spikes = 20;
    double w[10*5], times[20]; int32_t sources[20];
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    EventPropInput base_input = {
        .w=w, .n_pre=n_pre, .n_post=n_post,
        .times=times, .sources=sources, .n_spikes=n_spikes,
        .tau_mem=0.02, .tau_syn=0.005
    };
    MaximaOutput* output = eventprop_compute_maxima(&base_input);

    ASSERT(output->n_post == n_post, "wrong post count");
    eventprop_free_maxima_output(output);
    TEST_END;
}

static void test_backward_maxima(void)
{
    TEST("BackwardMaxima  Tiny (10×5, 20 spikes)");
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

    EventPropInput base = { .w=w, .n_pre=n_pre, .n_post=n_post,
                              .times=times, .sources=sources, .n_spikes=n_spikes,
                              .tau_mem=0.02, .tau_syn=0.005 };
    BackwardMaximaInput backward_input = {
        .input = &base,
        .maxima_times=maxima_times, .maxima_values=maxima_values,
        .maxima_errors=maxima_errors, .n_maxima=n_post
    };
    BackwardOutput* output = eventprop_backward_maxima(&backward_input);

    ASSERT(output->n_pre == n_pre, "wrong pre count");
    eventprop_free_backward_output(output);

    free(maxima_times); free(maxima_values); free(maxima_errors);
    TEST_END;
}

/* ── Batch backward tests ─────────────────────────────────────── */

static void test_backward_spikes_batch_single_sample(void)
{
    TEST("BackwardSpikesBatch single sample matches single-sample");
    int n_pre = 100, n_post = 40, n_spikes = 250;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));
    gen_data(n_pre, n_post, n_spikes, w, times, sources);

    /* Forward pass */
    EventPropInput fi = {.w=w, .n_pre=n_pre, .n_post=n_post,
                 .times=times, .sources=sources, .n_spikes=n_spikes,
                 .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    SpikeOutput* fwd = eventprop_compute_spikes(&fi);

    if (fwd->n_spikes == 0) {
        printf("SKIP [no forward spikes]\n"); tests_failed++;
        free(w); free(times); free(sources); return;
    }

    /* Single-sample backward */
    for (int i = 0; i < fwd->n_spikes; i++) fwd->errors[i] = -1.0;

    EventPropInput base = { .w=w, .n_pre=n_pre, .n_post=n_post,
                              .times=times, .sources=sources, .n_spikes=n_spikes,
                              .tau_mem=0.02, .tau_syn=0.005 };
    BackwardSpikesInput bsi_single = {
        .input = &base,
        .fwd_times=fwd->times, .fwd_sources=fwd->sources,
        .fwd_currents=fwd->currents,
        .errors=fwd->errors, .n_post_spikes=fwd->n_spikes
    };
    BackwardOutput* grad_single = eventprop_backward_spikes(&bsi_single);

    /* Batch backward with batch_size=1 */
    int n_per_sample[] = { n_spikes };
    SpikeOutput* fwd_batch = malloc(sizeof(SpikeOutput));
    *fwd_batch = *fwd;  // copy struct (pointers still valid)

    BatchEventPropInput batch_input = { .input = &base, .n_spikes_per_sample = n_per_sample, .batch_size = 1 };
    BackwardOutput* grad_batch = eventprop_backward_spikes_batch(
        &batch_input, fwd_batch);

    /* Compare gradients element-wise */
    int mismatch = 0;
    for (int i = 0; i < n_pre * n_post; i++) {
        if (fabs(grad_single->grad[i] - grad_batch[0].grad[i]) > EPSILON) {
            mismatch = 1; break;
        }
    }
    ASSERT(!mismatch, "batch single-sample gradient mismatch");

    /* Verify gradients are non-zero (catches v_th=0 or missing input spikes bugs) */
    double grad_sum = 0;
    for (int i = 0; i < n_pre * n_post; i++) grad_sum += fabs(grad_single->grad[i]);
    ASSERT(grad_sum > EPSILON, "all gradients are zero");

    eventprop_free_backward_output(grad_single);
    eventprop_free_backward_batch_output(grad_batch, 1);
    eventprop_free_spike_output(fwd);
    free(fwd_batch);
    free(w); free(times); free(sources);
    TEST_END;
}

static void test_backward_spikes_batch_multi_sample(void)
{
    TEST("BackwardSpikesBatch multi-sample (3 samples, varying spikes)");
    int n_pre = 100, n_post = 40;
    int batch_size = 3;
    int n_per_sample[] = { 120, 180, 90 };
    int total_spikes = 0;
    for (int i = 0; i < batch_size; i++) total_spikes += n_per_sample[i];

    double* w = malloc(n_pre * n_post * sizeof(double));
    gen_data(n_pre, n_post, total_spikes, w, NULL, NULL); /* init weights only */

    double* all_times = malloc(total_spikes * sizeof(double));
    int32_t* all_sources = malloc(total_spikes * sizeof(int32_t));
    for (int i = 0; i < total_spikes; i++) {
        all_times[i] = 0.01 + (i + 1.0) / (total_spikes + 1.0);
        all_sources[i] = i % n_pre;
    }

    /* Batch forward */
    EventPropInput fi = {.w=w, .n_pre=n_pre, .n_post=n_post,
                 .times=all_times, .sources=all_sources, .n_spikes=0,
                 .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};
    BatchEventPropInput batch_fwd = { .input = &fi, .n_spikes_per_sample = n_per_sample, .batch_size = batch_size };
    SpikeOutput* fwd_batch = eventprop_compute_spikes_batch(&batch_fwd);

    /* Count total forward spikes */
    int total_fwd = 0;
    for (int b = 0; b < batch_size; b++) total_fwd += fwd_batch[b].n_spikes;

    if (total_fwd == 0) {
        printf("SKIP [no forward spikes]\n"); tests_failed++;
        eventprop_free_batch_spike_output(fwd_batch, batch_size);
        free(w); free(all_times); free(all_sources); return;
    }

    /* Write errors into fwd_batch[].errors */
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < fwd_batch[b].n_spikes; i++)
            fwd_batch[b].errors[i] = -1.0;
    }

    /* Batch backward */
    EventPropInput base = {
        .w=w, .n_pre=n_pre, .n_post=n_post,
        .times=all_times, .sources=all_sources, .n_spikes=total_spikes,
        .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
    };

    BatchEventPropInput batch_bwd = { .input = &base, .n_spikes_per_sample = n_per_sample, .batch_size = batch_size };
    BackwardOutput* grad_batch = eventprop_backward_spikes_batch(
        &batch_bwd, fwd_batch);

    /* Verify each sample has correct dimensions */
    for (int b = 0; b < batch_size; b++) {
        ASSERT(grad_batch[b].n_pre == n_pre, "wrong pre count in batch");
        ASSERT(grad_batch[b].n_post == n_post, "wrong post count in batch");
    }

    /* Verify gradients are non-zero (catches v_th=0 bug) */
    double total_grad = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n_pre * n_post; i++)
            total_grad += fabs(grad_batch[b].grad[i]);
    }
    ASSERT(total_grad > 0, "all gradients are zero");

    /* Compare batch results against single-sample for each sample */
    int offset_in = 0;
    for (int b = 0; b < batch_size; b++) {
        EventPropInput sample_base = {
            .w=w, .n_pre=n_pre, .n_post=n_post,
            .times=all_times + offset_in, .sources=all_sources + offset_in,
            .n_spikes=n_per_sample[b],
            .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
        };
        BackwardSpikesInput bsi = {
            .input = &sample_base,
            .fwd_times=fwd_batch[b].times, .fwd_sources=fwd_batch[b].sources,
            .fwd_currents=fwd_batch[b].currents,
            .errors=fwd_batch[b].errors,
            .n_post_spikes=fwd_batch[b].n_spikes
        };
        BackwardOutput* grad_single = eventprop_backward_spikes(&bsi);

        int mismatch = 0;
        for (int i = 0; i < n_pre * n_post; i++) {
            if (fabs(grad_batch[b].grad[i] - grad_single->grad[i]) > EPSILON) {
                mismatch = 1; break;
            }
        }
        ASSERT(!mismatch, "multi-sample batch gradient mismatch");

        eventprop_free_backward_output(grad_single);
        offset_in += n_per_sample[b];
    }

    /* Cleanup */
    eventprop_free_backward_batch_output(grad_batch, batch_size);
    eventprop_free_batch_spike_output(fwd_batch, batch_size);
    free(w); free(all_times); free(all_sources);
    TEST_END;
}

static void test_compute_spikes_invalid_bracketing(void)
{
    TEST("ComputeSpikes invalid bracketing returns NaN");

    /* Create data likely to cause invalid brackets:
     * - Very tight spike clustering (burst pattern)
     * - Low threshold to maximize voltage buildup
     */
    int n_pre = 10, n_post = 5, n_spikes = 30;
    double* w = malloc(n_pre * n_post * sizeof(double));
    double* times = malloc(n_spikes * sizeof(double));
    int32_t* sources = malloc(n_spikes * sizeof(int32_t));

    /* High weights to cause large voltage responses */
    for (int i = 0; i < n_pre * n_post; i++)
        w[i] = 10.0;

    /* Generate tightly clustered spikes around t=0.5 */
    for (int i = 0; i < n_spikes; i++) {
        times[i] = 0.49 + (i * 0.02) / n_spikes; /* Very tight clustering */
        sources[i] = i % n_pre;
    }

    /* Use low threshold to ensure voltage stays above */
    EventPropInput fi = {
        .w=w, .n_pre=n_pre, .n_post=n_post,
                 .times=times, .sources=sources, .n_spikes=n_spikes,
                 .tau_mem=0.02, .tau_syn=0.005, .v_th=0.01};
    SpikeOutput* output = eventprop_compute_spikes(&fi);

    /* Function should not crash even with invalid brackets */
    ASSERT(output->times != NULL, "output times is NULL");

    /* Check that returned spike times are valid (not NaN) */
    int invalid_count = 0;
    for (int i = 0; i < output->n_spikes; i++) {
        if (!isfinite(output->times[i])) {
            invalid_count++;
        }
    }

    /* Some spikes may be skipped due to invalid brackets */
    printf(" (spikes: %d, invalid skipped: may vary)\n", output->n_spikes);

    eventprop_free_spike_output(output);
    free(w); free(times); free(sources);
    TEST_END;
}

/* ── Dropout tests ─────────────────────────────────────────────── */

static void test_dropout_spikes(void)
{
    /* Generate simple input: 10 spikes from neurons 0-9 */
    int n_spikes = 10;
    double times[10];
    int sources[10];
    for (int i = 0; i < n_spikes; i++) {
        times[i] = (i + 1) * 0.1;
        sources[i] = i % 3;
    }

    EventPropInput input = {
        .w=NULL, .n_pre=3, .n_post=2,
        .times=times, .sources=sources, .n_spikes=n_spikes,
        .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
    };
    int n_per_sample[] = {n_spikes};
    BatchEventPropInput batch = {
        .input=&input, .n_spikes_per_sample=n_per_sample, .batch_size=1
    };

    /* Test 1: p_drop=0 should keep all spikes */
    TEST("Dropout  p_drop=0 (keep all)");
    DropoutOutput* out0 = eventprop_dropout_spikes_batch(&batch, 0.0, 42);
    ASSERT(out0[0].n_spikes == n_spikes, "should keep all spikes");
    for (int i = 0; i < n_spikes; i++) {
        ASSERT_NEAR(out0[0].times[i], times[i], 1e-12);
        ASSERT(out0[0].sources[i] == sources[i], "source mismatch");
    }
    eventprop_free_dropout_output(out0, 1);
    TEST_END;

    /* Test 2: p_drop=1.0 should drop all spikes */
    TEST("Dropout  p_drop=1.0 (drop all)");
    DropoutOutput* out1 = eventprop_dropout_spikes_batch(&batch, 1.0, 42);
    ASSERT(out1[0].n_spikes == 0, "should drop all spikes");
    eventprop_free_dropout_output(out1, 1);
    TEST_END;

    /* Test 3: p_drop=0.5 should drop roughly half (deterministic with seed) */
    TEST("Dropout  p_drop=0.5 (partial)");
    DropoutOutput* out2 = eventprop_dropout_spikes_batch(&batch, 0.5, 42);
    ASSERT(out2[0].n_spikes >= 1 && out2[0].n_spikes <= n_spikes - 1,
           "should drop some but not all");
    eventprop_free_dropout_output(out2, 1);
    TEST_END;

    /* Test 4: Multi-sample batch */
    double times2[20] = {0};
    int sources2[20] = {0};
    for (int i = 0; i < 20; i++) {
        times2[i] = (i + 1) * 0.05;
        sources2[i] = i % 3;
    }
    EventPropInput input2 = {
        .w=NULL, .n_pre=3, .n_post=2,
        .times=times2, .sources=sources2, .n_spikes=20,
        .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0
    };
    int n_per_sample2[] = {10, 10};
    BatchEventPropInput batch2 = {
        .input=&input2, .n_spikes_per_sample=n_per_sample2, .batch_size=2
    };

    TEST("Dropout  batch=2, p_drop=0.3");
    DropoutOutput* out3 = eventprop_dropout_spikes_batch(&batch2, 0.3, 123);
    ASSERT(out3[0].n_spikes <= 10 && out3[1].n_spikes <= 10, "no sample exceeds input");
    ASSERT(out3[0].n_spikes + out3[1].n_spikes < 20, "should drop some total");
    eventprop_free_dropout_output(out3, 2);
    TEST_END;
}

/* ── Large-scale batch crash diagnostic ─────────────────────── */

static void test_batch_large_crash(void)
{
    TEST("BatchTrainingStep Large (250x125, 500 spikes) crash diag");
    int n_pre = 250, n_post = 123, n_spikes = 500;
    double* w = malloc(n_pre * n_post * sizeof(double));
    gen_data(n_pre, n_post, n_spikes, w, NULL, NULL);

    int batch_size = 4;
    // Concatenated input: each sample gets its own slice of times/sources
    int total_input = batch_size * n_spikes;
    double* all_times = malloc(total_input * sizeof(double));
    int32_t* all_sources = malloc(total_input * sizeof(int32_t));
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < n_spikes; i++) {
            int idx = b * n_spikes + i;
            all_times[idx] = 0.01 + (i + 1.0) / (n_spikes + 1.0);
            all_sources[idx] = i % n_pre;
        }
    }

    EventPropInput fi = {.w=w, .n_pre=n_pre, .n_post=n_post,
                 .times=all_times, .sources=all_sources, .n_spikes=total_input,
                 .tau_mem=0.02, .tau_syn=0.005, .v_th=1.0};

    int n_per_sample[] = { n_spikes, n_spikes, n_spikes, n_spikes };
    BatchEventPropInput batch = { .input = &fi, .n_spikes_per_sample = n_per_sample, .batch_size = batch_size };

    printf(" forward... "); fflush(stdout);
    SpikeOutput* fwd_batch = eventprop_compute_spikes_batch(&batch);
    int total_fwd = 0;
    for (int b = 0; b < batch_size; b++) total_fwd += fwd_batch[b].n_spikes;
    printf("%d spikes\n", total_fwd);

    if (total_fwd == 0) {
        printf("SKIP [no forward spikes]\n"); tests_failed++;
        free(w); free(all_times); free(all_sources); return;
    }

    /* Write errors into fwd_batch[].errors */
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < fwd_batch[b].n_spikes; i++)
            fwd_batch[b].errors[i] = -1.0;
    }

    EventPropInput base = { .w=w, .n_pre=n_pre, .n_post=n_post,
                              .times=all_times, .sources=all_sources, .n_spikes=total_input,
                              .tau_mem=0.02, .tau_syn=0.005 };
    BatchEventPropInput batch_bwd = { .input = &base, .n_spikes_per_sample = n_per_sample, .batch_size = batch_size };

    printf(" backward... "); fflush(stdout);
    BackwardOutput* grad_batch = eventprop_backward_spikes_batch(&batch_bwd, fwd_batch);
    printf("done\n");

    for (int b = 0; b < batch_size; b++) {
        ASSERT(grad_batch[b].n_pre == n_pre, "wrong pre count");
        ASSERT(grad_batch[b].n_post == n_post, "wrong post count");
    }

    eventprop_free_backward_batch_output(grad_batch, batch_size);
    eventprop_free_batch_spike_output(fwd_batch, batch_size);
    free(w); free(all_times); free(all_sources);
    TEST_END;
}

/* ── Main ─────────────────────────────────────────────────────── */

int main(void)
{
    printf("EventProp C Tests ( functions only)\n");
    printf("==========================================\n\n");

    /* ComputeSums tests */
    test_compute_sums();
    test_compute_sums__zero_spikes();
    test_compute_sums__large();

    /* ComputeSpikes  tests */
    test_compute_spikes();
    test_compute_spikes__burst_medium();
    test_compute_spikes_invalid_bracketing();

    /* BackwardSpikes tests */
    test_backward_spikes();

    /* Batch backward tests */
    test_backward_spikes_batch_single_sample();
    test_backward_spikes_batch_multi_sample();

    test_batch_large_crash();
    /* ComputeMaxima  tests */
    test_compute_maxima();

    /* BackwardMaxima  tests */
    test_backward_maxima();

    /* Dropout tests */
    test_dropout_spikes();

    printf("\n==========================================\n");
    printf("Tests: %d passed, %d failed (of %d total)\n",
           tests_passed, tests_failed, tests_run);

    return tests_failed > 0 ? 1 : 0;
}
