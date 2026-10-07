/**
 * @file eventprop_backward.c
 * @brief Backward pass: adjoint (lambda) sweep through spike times.
 */
#include "eventprop.h"
#include "eventprop_internal.h"
#include "eventprop_simd.h"
#include "matrixview.inc"
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>


static int cmp_merged_desc(const void* a, const void* b) {
    const double ta = ((const MergedSpike*)a)->time;
    const double tb = ((const MergedSpike*)b)->time;
    return (ta < tb) - (ta > tb);
}

static void free_backward_spikes_ctx(const BackwardSpikesCtx* ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->n_post_alloc; i++) {
        free(ctx->jump_time[i]);
        free(ctx->jump_mag[i]);
    }
    free(ctx->merged);
    free(ctx->all_errors);
    free(ctx->lambda_v);
    free(ctx->lambda_i);
    free(ctx->jump_time);
    free(ctx->jump_mag);
    free(ctx->jump_count);
    free(ctx->jump_cap);
}

BackwardOutput eventprop_backward_spikes(const EventPropInput input, const SpikeOutput spike){

    BackwardOutput result = (BackwardOutput){ 
        .error_no = eventprop_validate_input(input),
        .n_pre = input.n_pre, 
        .n_post = input.n_post,   
        .n_spikes = input.n_spikes
    };
    BackwardSpikesCtx ctx = {0};

    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    /* Validate output spike count */
    if (spike.n_spikes < 0) {
        result.error_no     = EVENTPROP_NEGATIVE_SPIKES;
        goto cleanup_and_return;
    }

    /* Validate output spike arrays */
    if (spike.n_spikes > 0 &&
        (!spike.times || !spike.sources || !spike.currents || !spike.errors)) {
        result.error_no = EVENTPROP_NULL_POINTER;
        goto cleanup_and_return;
    }

    int total = input.n_spikes + spike.n_spikes;

    /* Check for integer overflow in merged array size */
    if (input.n_spikes > INT_MAX - spike.n_spikes) {
        result.error_no     = EVENTPROP_ALLOC_BWD_MERGED; /* overflow guard */
        goto cleanup_and_return;
    }

    /* Validate output spike source bounds: post-sources index post-neurons */
    for (int i = 0; i < spike.n_spikes; ++i) {
        if (spike.sources[i] < 0 || spike.sources[i] >= input.n_post) {
            result.error_no     = EVENTPROP_INVALID_SOURCE;
            goto cleanup_and_return;
        }
    }

    /* Validate currents: must not equal v_th (division by zero in inv_ic) */
    for (int i = 0; i < spike.n_spikes; ++i) {
        if (spike.currents[i] == input.v_th) {
            result.error_no     = EVENTPROP_INVALID_CURRENT;
            goto cleanup_and_return;
        }
    }

    /* Allocate output */
    result.grad         = eventprop_try_calloc((size_t)(input.n_pre * input.n_post), sizeof(double));
    if (!result.grad) {
        result.error_no     = EVENTPROP_ALLOC_BWD_GRAD;
        goto cleanup_and_return;
    }
    
    result.spike_errors = eventprop_try_calloc((size_t)input.n_spikes, sizeof(double));
    if (!result.spike_errors) {
        result.error_no     = EVENTPROP_ALLOC_BWD_SPIKE_ERRORS;
        goto cleanup_and_return;
    }

    /* No output spikes to process */
    if (spike.n_spikes == 0) {
        goto cleanup_and_return;
    }

    ctx.merged = eventprop_try_malloc((size_t)total * sizeof(MergedSpike));
    if (!ctx.merged) {
        result.error_no     = EVENTPROP_ALLOC_BWD_MERGED;
        goto cleanup_and_return;
    }

    for (int i = 0; i < input.n_spikes; i++) {
        ctx.merged[i].time   = input.times[i];
        ctx.merged[i].idx    = i;
        ctx.merged[i].is_input = 1;
    }

    for (int i = 0; i < spike.n_spikes; i++) {
        ctx.merged[input.n_spikes+i].time     = spike.times[i];
        ctx.merged[input.n_spikes+i].idx      = i;
        ctx.merged[input.n_spikes+i].is_input = 0;
    }

    /* Sort descending by time: O(n log n) via qsort */
    qsort(ctx.merged, (size_t)total, sizeof(MergedSpike), cmp_merged_desc);

    ctx.all_errors = eventprop_try_calloc((size_t)total, sizeof(double));
    if (!ctx.all_errors) {
        result.error_no     = EVENTPROP_ALLOC_BWD_ALL_ERRORS;
        goto cleanup_and_return;
    }

    for (int i = 0; i < spike.n_spikes; i++) {
        ctx.all_errors[input.n_spikes + i] = spike.errors[i];
    }

    /* ── Allocate intermediate buffers (sequential checks) ─────────────── */

    ctx.lambda_v = eventprop_try_calloc((size_t)input.n_post, sizeof(double));
    if (!ctx.lambda_v) {
        result.error_no     = EVENTPROP_ALLOC_BWD_LAMBDA_V;
        goto cleanup_and_return;
    }
   
    ctx.lambda_i = eventprop_try_malloc((size_t)input.n_post * sizeof(double));
    if (!ctx.lambda_i) {
        result.error_no     = EVENTPROP_ALLOC_BWD_LAMBDA_I;
        goto cleanup_and_return;
    }

    /* Per-neuron jump storage (matching Eigen's vector<vector<pair<double,double>>> structure) */
    ctx.jump_time  = eventprop_try_calloc((size_t)input.n_post, sizeof(double*));
    if (!ctx.jump_time) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_TIME;
        goto cleanup_and_return;
    }
    
    ctx.jump_mag   = eventprop_try_calloc((size_t)input.n_post, sizeof(double*));
    if (!ctx.jump_mag) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_MAG;
        goto cleanup_and_return;
    }
    
    ctx.jump_count = eventprop_try_calloc((size_t)input.n_post, sizeof(int));
    if (!ctx.jump_count) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_COUNT;
        goto cleanup_and_return;
    }
    
    ctx.jump_cap   = eventprop_try_calloc((size_t)input.n_post, sizeof(int));
    if (!ctx.jump_cap) {
        result.error_no     = EVENTPROP_ALLOC_BWD_JUMP_CAP;
        goto cleanup_and_return;
    }

    /* Allocate per-neuron buffers — track how many succeeded for safe cleanup */
    ctx.n_post_alloc = 0;
    for (int i = 0; i < input.n_post; i++) {
        ctx.jump_cap[i] = 8;  /* initial capacity */
        ctx.jump_time[i] = eventprop_try_malloc((size_t)ctx.jump_cap[i] * sizeof(double));
        if (!ctx.jump_time[i]) {
            result.error_no     = EVENTPROP_ALLOC_BWD_PER_NEURON;
            goto cleanup_and_return;
        }
        ctx.jump_mag[i]  = eventprop_try_malloc((size_t)ctx.jump_cap[i] * sizeof(double));
        if (!ctx.jump_mag[i]) {
            free(ctx.jump_time[i]);
            result.error_no     = EVENTPROP_ALLOC_BWD_PER_NEURON;
            goto cleanup_and_return;
        }
        ctx.n_post_alloc++;
    }

    MutableMatrixView grad = mv_mut(result.grad, input.n_pre, input.n_post);
    MatrixView w           = mv(input.w, input.n_pre, input.n_post);

    const double largest_time = ctx.merged[0].time;
    double previous_t = largest_time;
    const double k_bwd = input.tau_mem / (input.tau_mem - input.tau_syn);

    for (int s = 0; s < total; s++) {
        int spike_idx = ctx.merged[s].idx, is_input = ctx.merged[s].is_input;
        double spike_time = ctx.merged[s].time;
        int spike_source = is_input ? input.sources[spike_idx] : spike.sources[spike_idx];

        double t_bwd = largest_time - spike_time;
        double dt = previous_t - t_bwd;
        scale_inplace_simd(ctx.lambda_v, exp(dt / input.tau_mem), input.n_post);

        if (is_input) {
            /* Compute lambda_i for ALL post-synaptic neurons (matching Eigen structure) */
            memset(ctx.lambda_i, 0, (size_t)input.n_post * sizeof(double));
            for (int nrn = 0; nrn < input.n_post; nrn++) {
                for (int j = 0; j < ctx.jump_count[nrn]; j++) {
                    double dt_jump = t_bwd - ctx.jump_time[nrn][j];
                    ctx.lambda_i[nrn] += ctx.jump_mag[nrn][j] * k_bwd *
                        (exp(-dt_jump / input.tau_mem) - exp(-dt_jump / input.tau_syn));
                }
            }

            vadd_scaled_simd(mv_row_mut(grad, spike_source), ctx.lambda_i,
                             -input.tau_syn, input.n_post);

            /* Compute dot product: w[row] · (lambda_v - lambda_i)
               Use stack allocation for small n_post, heap otherwise. */
            double stack_diff[BACKWARD_DIFF_STACK_SIZE];
            int use_stack = (input.n_post <= BACKWARD_DIFF_STACK_SIZE);
            double* diff = use_stack ? stack_diff : eventprop_try_malloc((size_t)input.n_post * sizeof(double));

            vsub_simd(diff, ctx.lambda_v, ctx.lambda_i, input.n_post);
            double outbound = dot_product_simd(mv_row(w, spike_source), diff, input.n_post);
            ctx.all_errors[spike_idx] += outbound;

            if (!use_stack) free(diff);

            /* Write spike error to alloc buffer */
            result.spike_errors[spike_idx] = ctx.all_errors[spike_idx];
        } else {
            double current = spike.currents[spike_idx];
            double inv_ic = 1.0 / (current - input.v_th);
            int nrn = spike_source;  /* post-spike: spike_source is the post-synaptic neuron */
            double post_err = ctx.all_errors[input.n_spikes + spike_idx];

            /* Record jump for this neuron (with capacity management) */
            if (ctx.jump_count[nrn] >= ctx.jump_cap[nrn]) {
                if (ctx.jump_cap[nrn] > INT_MAX / 2) {   /* *2 would overflow int */
                    result.error_no = EVENTPROP_DIMENSION_TOO_LARGE;
                    goto cleanup_and_return;
                }
                const int old_cap = ctx.jump_cap[nrn];
                ctx.jump_cap[nrn] = old_cap * 2;
                const size_t jump_size = (size_t)ctx.jump_cap[nrn] * sizeof(double);

                double* new_time = eventprop_try_realloc(ctx.jump_time[nrn], jump_size);
                if (!new_time) {
                    ctx.jump_cap[nrn] = old_cap;   /* buffer unchanged */
                    result.error_no = EVENTPROP_ALLOC_BWD_PER_NEURON;
                    goto cleanup_and_return;
                }
                /* Adopt immediately: realloc may have moved the block, which */
                /* frees the old one and would leave ctx.jump_time[nrn] dangling. */
                ctx.jump_time[nrn] = new_time;

                double* new_mag = eventprop_try_realloc(ctx.jump_mag[nrn], jump_size);
                if (!new_mag) {
                    ctx.jump_cap[nrn] = old_cap;   /* mag buffer still old_cap */
                    result.error_no = EVENTPROP_ALLOC_BWD_PER_NEURON;
                    goto cleanup_and_return;
                }
                ctx.jump_mag[nrn] = new_mag;
            }
            ctx.jump_time[nrn][ctx.jump_count[nrn]] = t_bwd;
            ctx.jump_mag[nrn][ctx.jump_count[nrn]]  = inv_ic * (input.v_th * ctx.lambda_v[nrn] + post_err);
            ctx.jump_count[nrn]++;

            ctx.lambda_v[nrn] = (current / (current - input.v_th)) * ctx.lambda_v[nrn] + inv_ic * post_err;
        }
        previous_t = t_bwd;
    }

cleanup_and_return:
    free_backward_spikes_ctx(&ctx);
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_backward_output(result);
        result.grad = NULL;          /* already freed — keep a caller-side free() safe */
        result.spike_errors = NULL;
    }
    return result;
}
