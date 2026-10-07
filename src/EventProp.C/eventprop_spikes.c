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

/* Initial per-thread scratch row size; the row grows x3 through the */
/* injection shim and is reused across neurons, so it ends up at the size */
/* the busiest neuron on that thread needs. */
#define EVENTPROP_INITIAL_ROW_CAP 8

/* ------------------------------------------------------------------ */
/* Parallel width                                                      */
/* ------------------------------------------------------------------ */
/* The sweep allocates scratch per thread, so the number of allocations */
/* a call performs depends on the OpenMP thread count. Tests that walk */
/* allocation indices (fault injection) need that count pinned; see */
/* eventprop_test_set_max_threads. */
static int max_threads_override = 0;   /* 0 = use the OpenMP runtime */

void eventprop_test_set_max_threads(int n)
{
    max_threads_override = (n > 0) ? n : 0;
}

static int effective_max_threads(void)
{
    if (max_threads_override > 0) return max_threads_override;
#ifdef _OPENMP
    const int rt = omp_get_max_threads();
    return (rt > 0) ? rt : 1;
#else
    return 1;
#endif
}

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

    const int old_cap = *cap;
    if (old_cap > INT_MAX / 3) {          /* *cap * 3 would overflow int */
        return EVENTPROP_DIMENSION_TOO_LARGE;
    }
    *cap = old_cap * 3;
    const size_t new_size = (size_t)(*cap) * sizeof(double);

    double* new_times = eventprop_try_realloc(*p_times, new_size);
    if (!new_times) {
        *cap = old_cap;                   /* buffer unchanged — keep cap truthful */
        return EVENTPROP_ALLOC_NEURON_BUF;
    }

    /* Never roll back with free(): realloc may have moved the block, which */
    /* frees the old one and leaves *p_times dangling. Hand the grown buffer to */
    /* the caller instead — it owns it through the thread slot. */
    double* new_currents = eventprop_try_realloc(*p_currents, new_size);
    if (!new_currents) {
        *p_times = new_times;
        *cap = old_cap;                   /* currents still holds only old_cap */
        return EVENTPROP_ALLOC_NEURON_BUF;
    }

    *p_times = new_times;
    *p_currents = new_currents;
    return EVENTPROP_OK;
}

/** Append one neuron's spikes to this thread's entry list (grows x2).
 *  Returns 0 on success, -1 on allocation failure. */
static inline int append_entries(SweepThread* th,
                                 const int source,
                                 const int count,
                                 const double* times,
                                 const double* currents)
{
    if (count == 0) return 0;

    if (th->n_entries + count > th->entries_cap) {
        int cap = th->entries_cap;
        while (th->n_entries + count > cap) {
            if (cap > INT_MAX / 2) return -1;   /* cap * 2 would overflow */
            cap *= 2;
        }
        SpikeEntry* grown = eventprop_try_realloc(th->entries,
                                                  (size_t)cap * sizeof(SpikeEntry));
        if (!grown) return -1;
        th->entries = grown;
        th->entries_cap = cap;
    }

    for (int k = 0; k < count; k++) {
        th->entries[th->n_entries].time    = times[k];
        th->entries[th->n_entries].source  = source;
        th->entries[th->n_entries].current = currents[k];
        th->n_entries++;
    }
    return 0;
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
                                                   const int total)
{
    if (total == 0) return EVENTPROP_OK;  /* no spikes, nothing to do */

    /* Sort each thread's own run, then merge the runs straight into the */
    /* output arrays: a concatenated staging buffer would cost a second */
    /* full-size allocation, which dominates when spikes are dense. */
    for (int t = 0; t < ctx->n_threads; t++) {
        SweepThread* run = &ctx->threads[t];
        if (run->n_entries > 1)
            qsort(run->entries, (size_t)run->n_entries,
                  sizeof(SpikeEntry), cmp_spike_asc);
    }

    /* Allocate output arrays — the runs stay owned by ctx on every path. */
    double* times    = eventprop_try_malloc((size_t)total * sizeof(double));
    if (!times) return EVENTPROP_ALLOC_RESULT_TIMES;

    int* sources     = eventprop_try_malloc((size_t)total * sizeof(int));
    if (!sources) { free(times); return EVENTPROP_ALLOC_RESULT_SOURCES; }

    double* currents = eventprop_try_malloc((size_t)total * sizeof(double));
    if (!currents) { free(sources); free(times);
                     return EVENTPROP_ALLOC_RESULT_CURRENTS; }

    double* errors   = eventprop_try_calloc((size_t)total, sizeof(double));
    if (!errors) { free(currents); free(sources); free(times);
                   return EVENTPROP_ALLOC_RESULT_ERRORS; }

    int* cursor      = eventprop_try_calloc((size_t)ctx->n_threads, sizeof(int));
    if (!cursor) { free(errors); free(currents); free(sources); free(times);
                   return EVENTPROP_ALLOC_THREAD_SCRATCH; }

    /* k-way merge over the sorted runs (k is the thread count, so a linear */
    /* scan of the run heads is cheaper than a heap). */
    for (int i = 0; i < total; i++) {
        int best = -1;
        for (int t = 0; t < ctx->n_threads; t++) {
            const SweepThread* run = &ctx->threads[t];
            if (cursor[t] >= run->n_entries) continue;
            if (best < 0 ||
                run->entries[cursor[t]].time < ctx->threads[best].entries[cursor[best]].time)
                best = t;
        }
        const SpikeEntry* e = &ctx->threads[best].entries[cursor[best]];
        times[i]    = e->time;
        sources[i]  = e->source;
        currents[i] = e->current;
        cursor[best]++;
    }
    free(cursor);

    /* Store in ctx so caller can assign to result */
    ctx->result_times   = times;
    ctx->result_sources = sources;
    ctx->result_currents = currents;
    ctx->result_errors  = errors;
    ctx->result_n_spikes = total;

    return EVENTPROP_OK;
}

/* ── Per-neuron spike buffers are gone: one scratch row per thread plus a */
/*    per-thread entry list replaces the n_post x (3*n_spikes+1) slab. ── */

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
    if (ctx->threads) {
        /* Slots are calloc'd, so unset slots hold NULL and are safe to free. */
        for (int t = 0; t < ctx->n_threads; t++) {
            free(ctx->threads[t].scratch_times);
            free(ctx->threads[t].scratch_currents);
            free(ctx->threads[t].entries);
        }
        free(ctx->threads);
    }
}

SpikeOutput eventprop_compute_spikes(const EventPropInput input)
{
    /* Context tracks all heap allocations for unified clean-up */
    ComputeSpikesCtx ctx = {0};
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

    /* One scratch row per thread, not one slab per neuron: the row starts */
    /* small and grows x3 through the injection shim, so peak memory tracks */
    /* the busiest neuron instead of 3 * n_spikes * n_post. */
    if (input.n_spikes > INT_MAX / 3) {
        result.error_no = EVENTPROP_DIMENSION_TOO_LARGE;
        goto cleanup_and_return;
    }
    int row_cap = 3 * input.n_spikes + 1;
    if (row_cap > EVENTPROP_INITIAL_ROW_CAP) row_cap = EVENTPROP_INITIAL_ROW_CAP;

    /* Slots and the region width must agree: the region is given an explicit */
    /* num_threads clause below, so omp_get_thread_num() can never exceed */
    /* n_slots - 1 even if the runtime spawns fewer threads than the default. */
    const int n_slots = effective_max_threads();
    ctx.n_threads = n_slots;
    ctx.threads = eventprop_try_calloc((size_t)n_slots, sizeof(SweepThread));
    if (!ctx.threads) {
        result.error_no = EVENTPROP_ALLOC_THREAD_SCRATCH;
        goto cleanup_and_return;
    }

    /* Shared error flags for the parallel region — atomic: multiple threads */
    /* may write them. Relaxed is sufficient: only the constant 1 is ever */
    /* stored, and the implicit barrier at the region exit publishes it. */
    atomic_int fwd_alloc_error = ATOMIC_VAR_INIT(0);

#ifdef _OPENMP
#pragma omp parallel num_threads(n_slots)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        SweepThread* th = &ctx.threads[tid];
        th->row_cap = row_cap;
        th->entries_cap = 64;
        th->entries = eventprop_try_calloc((size_t)th->entries_cap, sizeof(SpikeEntry));
        th->scratch_times = eventprop_try_calloc((size_t)row_cap, sizeof(double));
        th->scratch_currents = eventprop_try_calloc((size_t)row_cap, sizeof(double));

        /* A thread that cannot get its scratch sets the flag but still enters */
        /* the loop: an omp for must be encountered by every thread. */
        const int ok = (th->entries && th->scratch_times && th->scratch_currents) ? 1 : 0;
        if (!ok) atomic_store_explicit(&fwd_alloc_error, 1, memory_order_relaxed);

        double* p_times    = th->scratch_times;
        double* p_currents = th->scratch_currents;
        int cap              = row_cap;

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int target_nrn = 0; target_nrn < input.n_post; target_nrn++) {
            if (!ok) continue;

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
                        /* A realloc may have moved the buffers even on failure — */
                        /* sync so the row pointers below cannot dangle. */
                        p_times    = state.p_times;
                        p_currents = state.p_currents;
                        cap        = state.cap;
                        count      = state.count;
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
                        /* Buffers may have been moved by realloc: re-point the */
                        /* cached view the voltage helpers read. */
                        rctx_base.post_spike_times = p_times;
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

            /* The row may have been moved by realloc: hand the new pointers */
            /* back to the slot so the context free cannot double-free. */
            th->scratch_times = p_times;
            th->scratch_currents = p_currents;
            th->row_cap = cap;

            /* Hand this neuron's spikes to the thread's entry list. */
            if (append_entries(th, target_nrn, count, p_times, p_currents) != 0) {
                atomic_store_explicit(&fwd_alloc_error, 1, memory_order_relaxed);
            }
        }
    }

    if (atomic_load_explicit(&fwd_alloc_error, memory_order_acquire)) {
        result.n_spikes = 0;
        result.error_no = EVENTPROP_ALLOC_THREAD_SCRATCH;
        goto cleanup_and_return;
    }

    /* No concatenated buffer: collect_spikes_result merges the per-thread */
    /* runs into the output arrays, which keeps peak memory at one full-size */
    /* allocation instead of two. */
    int total = 0;
    for (int t = 0; t < ctx.n_threads; t++)
        total += ctx.threads[t].n_entries;

    /* Collect, sort, and copy output */
    const EventPropError err = collect_spikes_result(&ctx, total);
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
