/**
 * @file eventprop_diag.c
 * @brief Diagnostic helpers — voltage traces and lambda-I traces.
 */

#include "eventprop_diag.h"
#include "eventprop.h"
#include "eventprop_internal.h"
#include "matrixview.inc"  /* inline MatrixView helpers */
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Public output-freeing functions live in eventprop_memory.c. */

VoltageTraceOutput eventprop_compute_voltage_trace(const VoltageTraceInput input)
{
    VoltageTraceOutput result = (VoltageTraceOutput){0};

    /* Delegate shared validation to existing function */
    EventPropError err = eventprop_validate_input(input.input);
    if (err != EVENTPROP_OK) {
        result.error_no = err;
        return result;
    }

    /* Diag-specific checks */
    if (input.dt <= 0 || input.t_max <= 0) {
        result.error_no = EVENTPROP_DIAG_TIMESTEP_INVALID;
        return result;
    }
    if (input.target_nrn_idx < 0 || input.target_nrn_idx >= input.input.n_post) {
        result.error_no = EVENTPROP_DIAG_TARGET_OUT_OF_RANGE;
        return result;
    }
    if (input.n_post_spikes > 0 && (!input.fwd_times || !input.fwd_sources)) {
        result.error_no = EVENTPROP_NULL_POINTER;
        return result;
    }
    const int n_t = (int)(input.t_max / input.dt);
    if (n_t <= 0 || n_t > EVENTPROP_MAX_DIAG_SAMPLES) {
        result.error_no = EVENTPROP_DIAG_TRACE_TOO_LONG;
        return result;
    }
    const EventPropInput* fwd = &input.input;  // local pointer for readability in loop

    result.v_trace = eventprop_try_calloc((size_t)n_t, sizeof(double));
    if (!result.v_trace) {
        result.error_no = EVENTPROP_ALLOC_DIAG_V_TRACE;
        return result;
    }

    result.ts = eventprop_try_calloc((size_t)n_t, sizeof(double));
    if (!result.ts) {
        free(result.v_trace);
        result.v_trace = NULL;
        result.error_no = EVENTPROP_ALLOC_DIAG_TS;
        return result;
    }
    result.n_t = n_t;  // only set after both allocations succeed

    const double k_prefactor = fwd->tau_syn / (fwd->tau_mem - fwd->tau_syn);
    MatrixView w = mv(fwd->w, fwd->n_pre, fwd->n_post);

    for (int ti = 0; ti < n_t; ti++) {
        double t = input.dt * ti;
        result.ts[ti] = t;
        double mem = 0.0;

        /* Pre-synaptic contribution */
        for (int i = 0; i < fwd->n_spikes; i++) {
            if (fwd->times[i] > t) break;
            mem += *mv_at(w, fwd->sources[i], input.target_nrn_idx) *
                   k_prefactor * (exp(-(t - fwd->times[i]) / fwd->tau_mem) -
                                  exp(-(t - fwd->times[i]) / fwd->tau_syn));
        }

        /* Post-synaptic contribution (spike reset) */
        for (int i = 0; i < input.n_post_spikes; i++) {
            if (input.fwd_times[i] > t) break;
            if (input.fwd_sources[i] == input.target_nrn_idx)
                mem -= fwd->v_th * exp(-(t - input.fwd_times[i]) / fwd->tau_mem);
        }

        result.v_trace[ti] = mem;
    }

    return result;
}

/**
 * Adjoint lambda_i at one query time — a port of the reference `compute_lambda_i`.
 *
 * Sweeps the post-spike list backwards from the last spike, then sums the target neuron's
 * jumps through the backward kernel. Two details are what make this the reference's number
 * rather than one shaped like it:
 *
 *   - the decay is applied to lambda_v for EVERY neuron at EVERY spike, so the target's
 *     adjoint depends on when the other neurons fired and the sweep cannot skip them; and
 *   - spike positions, and the kernel elapsed time, are measured against the last spike of
 *     the whole list: a jump sits at largest_time - spike_time and is evaluated at
 *     largest_time - t, which is what makes its argument the true age of the jump at t.
 *
 * At a spike the jump is 1/(current - v_th) * (v_th * lambda_v + error) and lambda_v is then
 * recursed with the same denominator — the rule eventprop_backward.c uses, which is why the
 * currents come from eventprop_rebuild_post_currents rather than a second opinion of them.
 *
 * Scratch is reused across query times; resetting lambda_v and restarting the jump count is
 * what the reference's per-query reallocation amounts to.
 */
static double lambda_i_at_time(double t,
                               const EventPropInput* in,
                               const double* fwd_times,
                               const int* fwd_sources,
                               const double* post_errors,
                               const double* currents,
                               int n_post_spikes,
                               int target_nrn_idx,
                               double largest_time,
                               double* lambda_v,
                               double* jump_t,
                               double* jump_m) {
    memset(lambda_v, 0, (size_t)in->n_post * sizeof(double));
    int n_jumps = 0;
    double previous_t = largest_time;

    for (int i = n_post_spikes - 1; i >= 0; i--) {
        const double spike_time = fwd_times[i];
        if (spike_time < t) break;
        const int src = fwd_sources[i];
        const double current = currents[i];
        const double inv_ic = 1.0 / (current - in->v_th);

        const double t_bwd = largest_time - spike_time;
        const double decay = exp((previous_t - t_bwd) / in->tau_mem);
        for (int j = 0; j < in->n_post; j++) {
            lambda_v[j] *= decay;
        }

        /* The jump reads lambda_v[src] before this spike's own update, as the reference does. */
        if (src == target_nrn_idx) {
            jump_t[n_jumps] = t_bwd;
            jump_m[n_jumps] = inv_ic * (in->v_th * lambda_v[src] + post_errors[i]);
            n_jumps++;
        }
        lambda_v[src] = current * inv_ic * lambda_v[src] + inv_ic * post_errors[i];
        previous_t = t_bwd;
    }

    const double k_bwd = in->tau_mem / (in->tau_mem - in->tau_syn);
    const double t_bwd_from_t = largest_time - t;
    double lambda_i = 0.0;
    for (int j = 0; j < n_jumps; j++) {
        const double dt_jump = t_bwd_from_t - jump_t[j];
        lambda_i += jump_m[j] * k_bwd *
                    (exp(-dt_jump / in->tau_mem) - exp(-dt_jump / in->tau_syn));
    }
    return lambda_i;
}

/* Scratch the adjoint sweep needs: the rebuilt currents, the cursor row that rebuilds them,
 * the per-neuron membrane adjoint, and the traced neuron's jump list. One preparation routine
 * backs both the trace and the pointwise query, so the two entry points cannot disagree about
 * what is valid or about how the currents are obtained. */
typedef struct LambdaSweep {
    double* currents;
    double* cur_sum;
    double* lambda_v;
    double* jumps;          /* [0, n) positions, [n, 2n) magnitudes — one allocation */
    double  largest_time;
    int     n;              /* post-spike count; 0 means "nothing to sweep", all-zero result */
} LambdaSweep;

static void lambda_sweep_free(LambdaSweep* s) {
    free(s->currents);
    free(s->cur_sum);
    free(s->lambda_v);
    free(s->jumps);
    memset(s, 0, sizeof(*s));
}

/**
 * Validate, rebuild the post-spike currents, and size the scratch.
 *
 * On failure, whatever was allocated remains in *s for the caller to release with
 * lambda_sweep_free; the returned code says which check or allocation site failed.
 */
static EventPropError lambda_sweep_prepare(const LambdaITraceInput* in, LambdaSweep* s)
{
    /* Delegate shared validation to existing function */
    EventPropError err = eventprop_validate_input(in->input);
    if (err != EVENTPROP_OK) {
        return err;
    }

    /* Validate tau values (division by zero in the backward kernel prefactor) */
    if (in->input.tau_mem <= 0 || in->input.tau_syn <= 0 ||
        in->input.tau_mem == in->input.tau_syn) {
        return EVENTPROP_INVALID_TAU;
    }

    if (in->target_nrn_idx < 0 || in->target_nrn_idx >= in->input.n_post) {
        return EVENTPROP_DIAG_TARGET_OUT_OF_RANGE;
    }

    /* The sweep reads a source and an error per spike. */
    if (in->n_post_spikes > 0 && (!in->fwd_times || !in->fwd_sources || !in->post_errors)) {
        return EVENTPROP_NULL_POINTER;
    }

    /* The sweep walks the list backwards and stops at the first spike earlier than the query
     * time, so an unsorted list silently under-counts instead of failing, and a source index
     * indexes lambda_v. The reference requires both and checks neither; C is the diagnostic. */
    for (int i = 0; i < in->n_post_spikes; i++) {
        if (i > 0 && in->fwd_times[i] < in->fwd_times[i - 1]) {
            return EVENTPROP_UNSORTED_SPIKE_TIMES;
        }
        if (in->fwd_sources[i] < 0 || in->fwd_sources[i] >= in->input.n_post) {
            return EVENTPROP_INVALID_SOURCE;
        }
    }

    /* No post spikes means no jumps to sum: a legitimate empty sweep, not an error. */
    if (in->n_post_spikes == 0) {
        return EVENTPROP_OK;
    }

    const int n     = in->n_post_spikes;
    const int n_post = in->input.n_post;
    s->n = n;
    s->currents = eventprop_try_calloc((size_t)n, sizeof(double));
    if (!s->currents) {
        return EVENTPROP_ALLOC_DIAG_CURRENTS;
    }
    s->cur_sum = eventprop_try_calloc((size_t)n_post, sizeof(double));
    if (!s->cur_sum) {
        return EVENTPROP_ALLOC_DIAG_CUR_SUM;
    }
    s->lambda_v = eventprop_try_calloc((size_t)n_post, sizeof(double));
    if (!s->lambda_v) {
        return EVENTPROP_ALLOC_DIAG_LAMBDA_V;
    }
    /* Worst case every post spike belongs to the target, so the jump arrays are sized
     * n_post_spikes; sweeping backwards only ever visits a prefix, so they cannot overflow. */
    s->jumps = eventprop_try_calloc((size_t)2 * (size_t)n, sizeof(double));
    if (!s->jumps) {
        return EVENTPROP_ALLOC_DIAG_LAMBDA_JUMPS;
    }
    s->largest_time = in->fwd_times[n - 1];

    /* Same ascending cursor the backward pass uses — see eventprop_currents.c. */
    eventprop_rebuild_post_currents(&in->input, in->fwd_times, in->fwd_sources, n,
                                    s->currents, s->cur_sum);

    /* The jump denominator is current - v_th, so a spike sitting exactly on threshold would
     * divide by zero; the backward pass rejects it the same way rather than emitting an
     * infinite jump. */
    for (int i = 0; i < n; i++) {
        if (s->currents[i] == in->input.v_th) {
            return EVENTPROP_INVALID_CURRENT;
        }
    }
    return EVENTPROP_OK;
}

LambdaITraceOutput eventprop_compute_lambda_i_trace(const LambdaITraceInput input)
{
    LambdaITraceOutput result = (LambdaITraceOutput){0};
    LambdaSweep s = {0};

    /* Validate dt and t_max */
    if (input.dt <= 0 || input.t_max <= 0) {
        result.error_no = EVENTPROP_DIAG_TIMESTEP_INVALID;
        return result;
    }

    const int n_t = (int)(input.t_max / input.dt);
    if (n_t <= 0 || n_t > EVENTPROP_MAX_DIAG_SAMPLES) {
        result.error_no = EVENTPROP_DIAG_TRACE_TOO_LONG;
        return result;
    }

    result.error_no = lambda_sweep_prepare(&input, &s);
    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    result.lambda_trace = eventprop_try_calloc((size_t)n_t, sizeof(double));
    if (!result.lambda_trace) {
        result.error_no = EVENTPROP_ALLOC_DIAG_LAMBDA;
        goto cleanup_and_return;
    }

    result.ts = eventprop_try_calloc((size_t)n_t, sizeof(double));
    if (!result.ts) {
        result.error_no = EVENTPROP_ALLOC_DIAG_TS;
        goto cleanup_and_return;
    }
    result.n_t = n_t;  // only set after both allocations succeed

    for (int ti = 0; ti < n_t; ti++) {
        const double t = input.dt * ti;
        result.ts[ti] = t;
        if (s.n == 0) {
            continue;  /* calloc'd zeros are the answer for an empty sweep */
        }
        result.lambda_trace[ti] = lambda_i_at_time(t, &input.input, input.fwd_times,
                                                   input.fwd_sources, input.post_errors,
                                                   s.currents, s.n, input.target_nrn_idx,
                                                   s.largest_time, s.lambda_v, s.jumps,
                                                   s.jumps + s.n);
    }

cleanup_and_return:
    lambda_sweep_free(&s);
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_lambda_i_trace_output(result);
        result.lambda_trace = NULL;
        result.ts           = NULL;
        result.n_t          = 0;
    }
    return result;
}

LambdaIAtTimeOutput eventprop_compute_lambda_i_at_time(const LambdaITraceInput input, double t)
{
    LambdaIAtTimeOutput result = (LambdaIAtTimeOutput){0};
    LambdaSweep s = {0};

    /* The grid entry point samples this same sweep at dt * ti; a caller that wants one number
     * (the paper's figure-1 style query, or a cursor in a UI) gets exactly the value that grid
     * would have produced at the same t. input.t_max and input.dt are unused here — only the
     * sweep and the query time define the answer. */
    if (!isfinite(t) || t < 0) {
        result.error_no = EVENTPROP_DIAG_QUERY_TIME_INVALID;
        return result;
    }

    result.error_no = lambda_sweep_prepare(&input, &s);
    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }
    if (s.n == 0) {
        goto cleanup_and_return;  /* no jumps to sum: lambda_i is zero */
    }

    result.lambda_i = lambda_i_at_time(t, &input.input, input.fwd_times, input.fwd_sources,
                                       input.post_errors, s.currents, s.n,
                                       input.target_nrn_idx, s.largest_time, s.lambda_v,
                                       s.jumps, s.jumps + s.n);

cleanup_and_return:
    lambda_sweep_free(&s);
    if (result.error_no != EVENTPROP_OK) {
        result.lambda_i = 0.0;
    }
    return result;
}
