/**
 * @file eventprop_spikes.c
 * @brief Forward pass: threshold-crossing root finding (TOMS748/Brent).
 */
#include "eventprop.h"
#include "eventprop_internal.h"
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#endif
#include <omp.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

static inline double compute_v_inline(const double t, const int i, const int spike_count,
    const double* post_spike_times, const int neuron_offset,
    const double* sum0, const double* sum1, const int target_nrn,
    const EventPropInput input, const double k_prefactor)
{
    double mem = k_prefactor * (sum1[i * input.n_post + target_nrn] * exp(-t / input.tau_mem)
                    - sum0[i * input.n_post + target_nrn] * exp(-t / input.tau_syn));
    for (int s = 0; s < spike_count; s++) {
        const double st = post_spike_times[neuron_offset + s];
        if (st <= t) {  /* Include all spikes where st <= t, don't break early */
            mem -= input.v_th * exp(-(t - st) / input.tau_mem);
        }
    }
    return mem;
}

static inline double compute_v_factored(const double t, const int i, const int spike_count,
    const double* post_spike_times, const int neuron_offset,
    const double* sum0, const double* sum1, const int target_nrn,const  int n_post,
    const double k_prefactor, const double tau_mem_inv, const double tau_syn_inv,
    const double v_th, const double* post_spike_exp_st)
{
    const double e_t = exp(-(t) * tau_mem_inv);
    double mem = k_prefactor * (sum1[i * n_post + target_nrn] * e_t
                    - sum0[i * n_post + target_nrn] * exp(-(t) * tau_syn_inv));
    if (post_spike_exp_st) {
        for (int s = 0; s < spike_count; s++) {
            const double st = post_spike_times[neuron_offset + s];
            if (st <= t) {
                mem -= v_th * e_t * post_spike_exp_st[s];
            }
        }
    } else {
        for (int s = 0; s < spike_count; s++) {
            const double st = post_spike_times[neuron_offset + s];
            if (st <= t) {
                mem -= v_th * exp(-(t - st) * tau_mem_inv);
            }
        }
    }
    return mem;
}

static double compute_tmax(const int i, const int n_spikes, const double* times,
                           const int spike_count, const double* post_spike_times, const int neuron_offset,
                           const double* sum0, const double* sum1,
                           const int target_nrn, const int n_post,
                           const double tmax_prefactor, const double tmax_summand, const double tau_mem, const double v_th)
{
    if (i == 0) return 0.0;

    const double t_input = (i == n_spikes) ? INFINITY : times[i];
    double t_before = (i > 0) ? times[i - 1] : 0.0;

    for (int s = 0; s < spike_count; s++) {
        const double st = post_spike_times[neuron_offset + s];
        if (st > t_input) break;
        if (st > t_before) t_before = st;
    }

    const double sum0_elem = sum0[i * n_post + target_nrn];
    const double sum1_elem = sum1[i * n_post + target_nrn];

    if (sum0_elem <= 0 || sum1_elem <= 0) return INFINITY;

    double sum1_adj = sum1_elem;
    for (int s = 0; s < spike_count; s++) {
        const double st = post_spike_times[neuron_offset + s];
        if (st > t_input) break;
        sum1_adj += -v_th * exp(st / tau_mem);
    }

    const double tmax = tmax_prefactor * (tmax_summand + log(sum1_adj / sum0_elem));

    if (t_before <= tmax && (i == n_spikes || tmax <= t_input))
        return tmax;
    return INFINITY;
}

static double compute_i(const double t, const int n_spikes, const double* times,
                        const double* sum0, const int target_nrn, const int n_post, const double tau_syn)
{
    int idx = 0;
    for (; idx < n_spikes; idx++) {
        if (times[idx] > t) break;
    }
    return sum0[idx * n_post + target_nrn] * exp(-t / tau_syn);
}

static int cmp_spike_asc(const void* a, const void* b) {
    const double ta = ((const SpikeEntry*)a)->time;
    const double tb = ((const SpikeEntry*)b)->time;
    return (ta > tb) - (ta < tb);
}

static double toms748_solve(const RootContext* ctx, double a, double b) {

    double fa = VOLTAGE(ctx, a), fb = VOLTAGE(ctx, b);
    if (!isfinite(fa) || !isfinite(fb)) return a;

    /* Precondition: bracket must contain sign change */
    if (fa * fb > 0) {
        return NAN;
    }

    if (a > b) { double tmp=a; a=b; b=tmp; tmp=fa; fa=fb; fb=tmp; }

    double c = a, fc = fa, d = 0.0;
    int iter = 0;
    while (++iter <= TOMS748_MAX_ITER) {
        const double m = 0.5 * (b - a);
        if (fabs(m) <= TOMS748_TOL || fb == 0.0)
            return b;

        double s = 0.0;
        int useBisection = 1;

        if (fabs(fc) > fabs(fb) && c != a && c != b) {
            const double qba = (fb - fa) / (b - a);
            const double qcb = (fc - fb) / (c - b);
            const double qcbba = (qcb - qba) / (c - a);

            if (fabs(qcbba) > TOMS748_NEAR_ZERO) {
                const double denom = qba + qcb;
                if (fabs(denom) > TOMS748_NEAR_ZERO) {
                    s = b - fb / (qba + qcb * (b - a) * qcbba / denom);
                    const int inBracket = (s > a && s < b);
                    const double mVal = 0.5 * (b - a);
                    const int smallStep = fabs(s - b) < mVal;
                    const int prevWasBisect = fabs(d - mVal) < TOMS748_TOL * 100;
                    if (inBracket && smallStep && !prevWasBisect)
                        useBisection = 0;
                }
            }
        }

        if (useBisection) s = a + m;

        const double f_oldB = fb;   // save before updating - avoids redundant fn call below
        const double prevB = b; b = s; fb = VOLTAGE(ctx, b);
        if (!isfinite(fb)) return prevB;

        if (fabs(fb) <= TOMS748_TOL || fabs(b - prevB) <= TOMS748_TOL * fmax(1.0, fabs(prevB)))
            return b;

        /* Reuse known value instead of re-evaluating fn(ctx, prevB) */
        if (fa * fb < 0.0) { c = prevB; fc = f_oldB; }
        else if (fb * fc < 0.0) { a = prevB; fa = f_oldB; }
        else { a = prevB; fa = f_oldB; }

        if (!isfinite(fa) || !isfinite(fc)) return prevB;
        d = b - prevB;
    }
    return b;
}

/** Grow both times and currents buffers if needed.
 *  Returns: EVENTPROP_OK on success, EVENTPROP_ALLOC_NEURON_BUF on failure. */
static inline EventPropError grow_spike_buffers(double** p_times,
                                                double** p_currents,
                                                int* cap,
                                                int count)
{
    if (count < (*cap)) return EVENTPROP_OK;

    (*cap) *= 3;
    const size_t new_size = (size_t)(*cap) * sizeof(double);

    /* Sequential realloc: undo times if currents fails */
    double* new_times = realloc(*p_times, new_size);
    if (!new_times) return EVENTPROP_ALLOC_NEURON_BUF;

    double* new_currents = realloc(*p_currents, new_size);
    if (!new_currents) {
        free(new_times);  /* rollback */
        return EVENTPROP_ALLOC_NEURON_BUF;
    }

    *p_times = new_times;
    *p_currents = new_currents;
    return EVENTPROP_OK;
}

/** Try to find a spike root in [t_lower, t_upper] and record it.
 *
 *  Returns:
 *    0  — spike found and recorded
 *   -1  — allocation error (caller should abort)
 *    1  — no valid spike in bracket
 */
static inline int try_find_root_spike(const RootContext* rctx_base,
                                      const EventPropInput input,
                                      double t_lower,
                                      double t_upper,
                                      NeuronState* state)
{
    /* Precompute exp(st * tau_mem_inv) for factorization */
    double* post_exp_st = NULL;
    if (state->count > 0) {
        post_exp_st = eventprop_try_malloc((size_t)state->count * sizeof(double));
        if (!post_exp_st) return -1;

        const double tau_mem_inv = 1.0 / input.tau_mem;
        for (int s = 0; s < state->count; s++) {
            post_exp_st[s] = exp(state->p_times[s] * tau_mem_inv);
        }
    }

    RootContext rctx = *rctx_base;
    /* Caller sets rctx_base.i to the current interval index */
    rctx.spike_count = state->count;
    rctx.post_spike_exp_st = post_exp_st;

    const double t_post = toms748_solve(&rctx, t_lower, t_upper);
    free(post_exp_st);

    /* Validate root: must be strictly inside bracket */
    if (!(t_post > t_lower && t_post < t_upper)) return 1;

    /* Grow buffers if needed */
    if (grow_spike_buffers(&state->p_times, &state->p_currents,
                           &state->cap, state->count) != EVENTPROP_OK)
        return -1;

    /* Record spike */
    state->p_times[state->count] = t_post;
    state->p_currents[state->count] = compute_i(t_post,
                                                 input.n_spikes,
                                                 input.times,
                                                 rctx_base->sum0,
                                                 rctx_base->target_nrn,
                                                 rctx_base->n_post,
                                                 input.tau_syn);
    state->count++;
    return 0;
}

/* ── Collect per-neuron spikes into sorted SpikeOutput (shared by v1 and v2) ── */
static inline EventPropError collect_spikes_result(ComputeSpikesCtx* ctx,
                                                   int n_post)
{
    /* Count total spikes */
    int total = 0;
    for (int n = 0; n < n_post; n++)
        total += ctx->spike_counts[n];

    if (total == 0) return EVENTPROP_OK;  /* no spikes, nothing to do */

    /* Collect into sortable entries */
    SpikeEntry* entries = eventprop_try_malloc((size_t)total * sizeof(SpikeEntry));
    if (!entries) return EVENTPROP_ALLOC_ENTRIES;

    int idx = 0;
    for (int n = 0; n < n_post; n++) {
        const int cnt = ctx->spike_counts[n];
        const double* pt = ctx->post_times_arr[n];
        const double* pc = ctx->post_currents_arr[n];
        for (int k = 0; k < cnt; k++) {
            entries[idx].time    = pt[k];
            entries[idx].source  = n;
            entries[idx].current = pc[k];
            idx++;
        }
    }

    qsort(entries, (size_t)total, sizeof(SpikeEntry), cmp_spike_asc);

    /* Allocate output arrays */
    double* times    = eventprop_try_malloc((size_t)total * sizeof(double));
    if (!times) { free(entries); return EVENTPROP_ALLOC_RESULT_TIMES; }

    int* sources     = eventprop_try_malloc((size_t)total * sizeof(int));
    if (!sources) { free(times); free(entries); return EVENTPROP_ALLOC_RESULT_SOURCES; }

    double* currents = eventprop_try_malloc((size_t)total * sizeof(double));
    if (!currents) { free(sources); free(times); free(entries);
                     return EVENTPROP_ALLOC_RESULT_CURRENTS; }

    double* errors   = eventprop_try_calloc((size_t)total, sizeof(double));
    if (!errors) { free(currents); free(sources); free(times); free(entries);
                   return EVENTPROP_ALLOC_RESULT_ERRORS; }

    /* Copy sorted data into output */
    for (int i = 0; i < total; i++) {
        times[i]    = entries[i].time;
        sources[i]  = entries[i].source;
        currents[i] = entries[i].current;
    }

    /* Store in ctx so caller can assign to result */
    ctx->result_times   = times;
    ctx->result_sources = sources;
    ctx->result_currents = currents;
    ctx->result_errors  = errors;
    ctx->result_n_spikes = total;

    free(entries);
    return EVENTPROP_OK;
}

/* ── Allocate per-neuron dynamic spike buffers (used by v2) ── */
static inline EventPropError allocate_neuron_buffers(ComputeSpikesCtx* ctx,
                                                     const EventPropInput input)
{
    const size_t sz = (size_t)input.n_post;

    ctx->post_times_arr = eventprop_try_calloc(sz, sizeof(double*));
    if (!ctx->post_times_arr) return EVENTPROP_ALLOC_TIMES_ARR;

    ctx->post_currents_arr = eventprop_try_calloc(sz, sizeof(double*));
    if (!ctx->post_currents_arr) return EVENTPROP_ALLOC_CURRENTS_ARR;

    ctx->spike_caps = eventprop_try_calloc(sz, sizeof(int));
    if (!ctx->spike_caps) return EVENTPROP_ALLOC_SPIKE_CAPS;

    ctx->spike_counts = eventprop_try_calloc(sz, sizeof(int));
    if (!ctx->spike_counts) return EVENTPROP_ALLOC_SPIKE_COUNTS;

    if (input.n_spikes > INT_MAX / 3)   /* 3 * n_spikes would overflow int */
        return EVENTPROP_DIMENSION_TOO_LARGE;
    const size_t initial_cap = (size_t)3 * (size_t)input.n_spikes + 1;
    int failed = 0;
    for (int n = 0; n < input.n_post; n++) {
        ctx->post_times_arr[n]    = eventprop_try_calloc(initial_cap, sizeof(double));
        if (!ctx->post_times_arr[n]) { failed = n + 1; break; }
        ctx->post_currents_arr[n] = eventprop_try_calloc(initial_cap, sizeof(double));
        if (!ctx->post_currents_arr[n]) { failed = n + 1; break; }
        ctx->spike_caps[n] = (int)initial_cap;
    }

    if (failed > 0) {
        /* Undo every buffer the loop owns, and clear the slot so the caller's */
        /* context free does not see a dangling pointer. */
        for (int n = 0; n < failed; n++) {
            free(ctx->post_times_arr[n]);
            ctx->post_times_arr[n] = NULL;
            free(ctx->post_currents_arr[n]);   /* NULL when its allocation failed */
            ctx->post_currents_arr[n] = NULL;
        }
        return EVENTPROP_ALLOC_NEURON_BUF;
    }

    return EVENTPROP_OK;
}

/** Generate candidate upper bounds for interval i.
 *  Returns the number of candidates (0, 1, or 2). */
static inline int generate_bracket_candidates(const RootContext* rctx,
                                              int count,
                                              const double* p_times,
                                              const EventPropInput input,
                                              double tmax_prefactor,
                                              double tmax_summand,
                                              double t_before,
                                              double candidates[2])
{
    int n_candidates = 0;

    if (rctx->i == input.n_spikes) {
        /* ── Last interval: one candidate — analytical tmax from adjusted sums ── */
        double s0 = rctx->sum0[input.n_spikes * input.n_post + rctx->target_nrn];
        double s1 = rctx->sum1[input.n_spikes * input.n_post + rctx->target_nrn];
        for (int s = 0; s < count; s++)
            s1 += -input.v_th * exp(p_times[s] / input.tau_mem);

        if (s0 > 0) {
            const double tmax = tmax_prefactor * (tmax_summand + log(s1 / s0));
            /* Validate bracket: tmax must be >= t_before (matches compute_tmax bounds check) */
            if (!isinf(tmax) && t_before <= tmax) {
                const double v = compute_v_inline(tmax, rctx->i, count, p_times, 0,
                                            rctx->sum0, rctx->sum1, rctx->target_nrn, input, rctx->k_prefactor);
                if (v > input.v_th + SPIKE_THRESHOLD_EPS)
                    candidates[n_candidates++] = tmax;
            }
        }
    } else {
        /* ── Regular interval: up to 2 candidates ── */
        const double t_input = input.times[rctx->i];

        /* Candidate 1: analytical tmax */
        const double tmax = compute_tmax(rctx->i, input.n_spikes, input.times, count, p_times, 0,
                                   rctx->sum0, rctx->sum1, rctx->target_nrn, rctx->n_post,
                                   tmax_prefactor, tmax_summand, input.tau_mem, input.v_th);
        if (!isinf(tmax)) {
            const double v = compute_v_inline(tmax, rctx->i, count, p_times, 0,
                                        rctx->sum0, rctx->sum1, rctx->target_nrn, input, rctx->k_prefactor);
            if (v > input.v_th + SPIKE_THRESHOLD_EPS)
                candidates[n_candidates++] = tmax;
        }

        /* Candidate 2: voltage at input spike time */
        const double v_at_input = compute_v_inline(t_input, rctx->i, count, p_times, 0,
                                             rctx->sum0, rctx->sum1, rctx->target_nrn, input, rctx->k_prefactor);
        if (v_at_input > input.v_th + SPIKE_THRESHOLD_EPS)
            candidates[n_candidates++] = t_input;
    }

    return n_candidates;
}

static void free_compute_spikes_ctx(const ComputeSpikesCtx* ctx) {
    eventprop_free_sums_output(ctx->sums);
    /* Each array is guarded separately: a partial allocation failure can leave */
    /* one of them NULL while the other is populated. */
    if (ctx->post_times_arr) {
        for (int n = 0; n < ctx->n_post; n++)
            free(ctx->post_times_arr[n]);
        free(ctx->post_times_arr);
    }
    if (ctx->post_currents_arr) {
        for (int n = 0; n < ctx->n_post; n++)
            free(ctx->post_currents_arr[n]);
        free(ctx->post_currents_arr);
    }
    free(ctx->spike_caps);
    free(ctx->spike_counts);
    free(ctx->entries);
}

SpikeOutput eventprop_compute_spikes(const EventPropInput input)
{
    /* Context tracks all heap allocations for unified clean-up */
    ComputeSpikesCtx ctx = { .n_post = input.n_post };
    SpikeOutput result = (SpikeOutput){ .error_no = eventprop_validate_input(input) };

    if (result.error_no != EVENTPROP_OK)
        goto cleanup_and_return;

    const double k_prefactor = input.tau_syn / (input.tau_mem - input.tau_syn);
    const double tmax_prefactor = 1.0 / (1.0 / input.tau_mem - 1.0 / input.tau_syn);
    const double tmax_summand = log(input.tau_syn / input.tau_mem);
    const double tau_mem_inv = 1.0 / input.tau_mem;
    const double tau_syn_inv = 1.0 / input.tau_syn;

    /* Compute sums */
    ctx.sums = eventprop_compute_sums(input);
    if (ctx.sums.error_no != EVENTPROP_OK) {
        result.error_no = ctx.sums.error_no;
        goto cleanup_and_return;
    }

    const double* sum0 = ctx.sums.sum0;
    const double* sum1 = ctx.sums.sum1;

    /* Allocate per-neuron dynamic buffers */
    result.error_no = allocate_neuron_buffers(&ctx, input);
    if (result.error_no != EVENTPROP_OK)
        goto cleanup_and_return;

    /* Shared error flag for OpenMP region — atomic: multiple threads may write it.
     * Relaxed is sufficient: only the constant value 1 is ever stored, and the
     * implicit barrier at the end of the parallel region publishes it to the master. */
    atomic_int fwd_alloc_error = ATOMIC_VAR_INIT(0);

    int target_nrn;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (target_nrn = 0; target_nrn < input.n_post; target_nrn++) {

        double* p_times    = ctx.post_times_arr[target_nrn];
        double* p_currents = ctx.post_currents_arr[target_nrn];
        int cap = ctx.spike_caps[target_nrn];
        int count = 0;
        int finished = 0, processed = 0;

        /* Root context — constant fields set once per neuron */
        RootContext rctx_base = (RootContext){
            .sum0 = sum0, .sum1 = sum1,
            .post_spike_times = p_times, .neuron_offset = 0,
            .target_nrn = target_nrn, .n_post = input.n_post,
            .k_prefactor = k_prefactor, .tau_mem_inv = tau_mem_inv,
            .tau_syn_inv = tau_syn_inv, .v_th = input.v_th,
        };

        while (!finished) {
            int found = 0;
            for (int i = processed; i <= input.n_spikes; i++) {

                /* ── Compute t_lower (bracket start) ── */
                const double t_input = (i == input.n_spikes) ? INFINITY : input.times[i];

                double t_before;
                if (i == input.n_spikes) {
                    /* Last interval: lower bound is last post-spike or 0 */
                    t_before = count > 0 ? p_times[count - 1] : 0.0;
                } else {
                    /* Regular interval: latest of (prev input time, post spikes before current input) */
                    t_before = (i > 0) ? input.times[i - 1] : 0.0;
                    for (int s = 0; s < count; s++) {
                        double st = p_times[s];
                        if (st > t_input) break;
                        if (st > t_before) t_before = st;
                    }
                }

                /* ── Generate candidate upper bounds ── */
                rctx_base.i = i;
                double candidates[2];
                int n_candidates = generate_bracket_candidates(
                    &rctx_base, count, p_times,
                    input, tmax_prefactor, tmax_summand, t_before, candidates);

                /* ── Try each candidate bracket with shared solver ── */
                for (int c = 0; c < n_candidates; c++) {
                    NeuronState state = { .p_times = p_times,
                                          .p_currents = p_currents,
                                          .cap = cap, .count = count };

                    const int rc = try_find_root_spike(&rctx_base, input, t_before,
                                                 candidates[c], &state);

                    if (rc == -1) {
                        atomic_store_explicit(&fwd_alloc_error, 1, memory_order_relaxed);
                        finished = 1;
                        break;
                    }
                    if (rc == 0) {
                        /* Spike found — update local state from NeuronState */
                        p_times    = state.p_times;
                        p_currents = state.p_currents;
                        cap        = state.cap;
                        count      = state.count;
                        found = 1;
                        break;  /* short-circuit: don't try remaining candidates */
                    }
                }

                if (found) {
                    processed = i;
                    break;  /* re-scan this interval next while iteration */
                }
                if (i == input.n_spikes) {
                    finished = 1;  /* exhausted last interval with no spike */
                }
            }

            if (!found)
                processed = (processed + 1 < input.n_spikes) ? processed + 1 : input.n_spikes;
        }

        /* Write back updated pointers */
        ctx.post_times_arr[target_nrn]   = p_times;
        ctx.post_currents_arr[target_nrn] = p_currents;
        ctx.spike_caps[target_nrn]       = cap;
        ctx.spike_counts[target_nrn]     = count;
    }

    if (atomic_load_explicit(&fwd_alloc_error, memory_order_acquire)) {
        result.n_spikes = 0;
        result.error_no = EVENTPROP_ALLOC_NEURON_BUF;
        goto cleanup_and_return;
    }

    /* Collect, sort, and copy output */
    const EventPropError err = collect_spikes_result(&ctx, input.n_post);
    if (err != EVENTPROP_OK) {
        result.error_no = err;
        goto cleanup_and_return;
    }

    result.n_spikes   = ctx.result_n_spikes;
    result.times      = ctx.result_times;
    result.sources    = ctx.result_sources;
    result.currents   = ctx.result_currents;
    result.errors     = ctx.result_errors;

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_spike_output(result);
        result.times = NULL;      /* already freed — keep a caller-side free() safe */
        result.sources = NULL;
        result.currents = NULL;
        result.errors = NULL;
    }
    free_compute_spikes_ctx(&ctx);
    return result;
}
