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

LambdaITraceOutput eventprop_compute_lambda_i_trace(const LambdaITraceInput input)
{
    LambdaITraceOutput result = (LambdaITraceOutput){0};
    double* currents  = NULL;
    double* cur_sum   = NULL;
    double* lambda_v  = NULL;
    double* jumps     = NULL;

    /* Delegate shared validation to existing function */
    EventPropError err = eventprop_validate_input(input.input);
    if (err != EVENTPROP_OK) {
        result.error_no = err;
        return result;
    }

    /* Validate dt and t_max */
    if (input.dt <= 0 || input.t_max <= 0) {
        result.error_no = EVENTPROP_DIAG_TIMESTEP_INVALID;
        return result;
    }

    /* Validate tau values (division by zero in the backward kernel prefactor) */
    if (input.input.tau_mem <= 0 || input.input.tau_syn <= 0 ||
        input.input.tau_mem == input.input.tau_syn) {
        result.error_no = EVENTPROP_INVALID_TAU;
        return result;
    }

    if (input.target_nrn_idx < 0 || input.target_nrn_idx >= input.input.n_post) {
        result.error_no = EVENTPROP_DIAG_TARGET_OUT_OF_RANGE;
        return result;
    }

    /* Validate forward spike pointers — the sweep reads a source and an error per spike. */
    if (input.n_post_spikes > 0 && (!input.fwd_times || !input.fwd_sources || !input.post_errors)) {
        result.error_no = EVENTPROP_NULL_POINTER;
        return result;
    }

    /* The sweep walks the list backwards and stops at the first spike earlier than the query
     * time, so an unsorted list silently under-counts instead of failing, and a source index
     * indexes lambda_v. The reference requires both and checks neither; C is the diagnostic. */
    for (int i = 0; i < input.n_post_spikes; i++) {
        if (i > 0 && input.fwd_times[i] < input.fwd_times[i - 1]) {
            result.error_no = EVENTPROP_UNSORTED_SPIKE_TIMES;
            return result;
        }
        if (input.fwd_sources[i] < 0 || input.fwd_sources[i] >= input.input.n_post) {
            result.error_no = EVENTPROP_INVALID_SOURCE;
            return result;
        }
    }

    const int n_t = (int)(input.t_max / input.dt);
    if (n_t <= 0 || n_t > EVENTPROP_MAX_DIAG_SAMPLES) {
        result.error_no = EVENTPROP_DIAG_TRACE_TOO_LONG;
        return result;
    }

    result.lambda_trace = eventprop_try_calloc((size_t)n_t, sizeof(double));
    if (!result.lambda_trace) {
        result.error_no = EVENTPROP_ALLOC_DIAG_LAMBDA;
        return result;
    }

    result.ts = eventprop_try_calloc((size_t)n_t, sizeof(double));
    if (!result.ts) {
        result.error_no = EVENTPROP_ALLOC_DIAG_TS;
        goto cleanup_and_return;
    }
    result.n_t = n_t;  // only set after both allocations succeed

    /* Handle zero forward spikes — trace is all zeros (already calloc'd) */
    if (input.n_post_spikes == 0) {
        for (int ti = 0; ti < n_t; ti++) {
            result.ts[ti] = input.dt * ti;
        }
        return result;
    }

    /* Rebuild the synaptic current at each post spike with the same ascending cursor the
     * backward pass uses, then jump per query time. Worst case every post spike belongs to
     * the target, so the jump arrays are sized n_post_spikes; sweeping backwards only ever
     * visits a prefix of the list, so they cannot overflow.
     * jumps is one allocation: [0, n) holds positions, [n, 2n) holds magnitudes. */
    const int n_post = input.input.n_post;
    const int n      = input.n_post_spikes;
    currents = eventprop_try_calloc((size_t)n, sizeof(double));
    if (!currents) {
        result.error_no = EVENTPROP_ALLOC_DIAG_CURRENTS;
        goto cleanup_and_return;
    }
    cur_sum = eventprop_try_calloc((size_t)n_post, sizeof(double));
    if (!cur_sum) {
        result.error_no = EVENTPROP_ALLOC_DIAG_CUR_SUM;
        goto cleanup_and_return;
    }
    lambda_v = eventprop_try_calloc((size_t)n_post, sizeof(double));
    if (!lambda_v) {
        result.error_no = EVENTPROP_ALLOC_DIAG_LAMBDA_V;
        goto cleanup_and_return;
    }
    jumps = eventprop_try_calloc((size_t)2 * (size_t)n, sizeof(double));
    if (!jumps) {
        result.error_no = EVENTPROP_ALLOC_DIAG_LAMBDA_JUMPS;
        goto cleanup_and_return;
    }
    double* jump_m = jumps + n;

    eventprop_rebuild_post_currents(&input.input, input.fwd_times, input.fwd_sources, n,
                                    currents, cur_sum);

    /* The jump denominator is current - v_th, so a spike sitting exactly on threshold would
     * divide by zero; the backward pass rejects it the same way rather than emitting an
     * infinite jump. */
    for (int i = 0; i < n; i++) {
        if (currents[i] == input.input.v_th) {
            result.error_no = EVENTPROP_INVALID_CURRENT;
            goto cleanup_and_return;
        }
    }

    /* Constants independent of time step */
    const double largest_time = input.fwd_times[n - 1];

    for (int ti = 0; ti < n_t; ti++) {
        const double t = input.dt * ti;
        result.ts[ti] = t;
        result.lambda_trace[ti] = lambda_i_at_time(t, &input.input, input.fwd_times,
                                                   input.fwd_sources, input.post_errors,
                                                   currents, n, input.target_nrn_idx,
                                                   largest_time, lambda_v, jumps, jump_m);
    }

cleanup_and_return:
    free(currents);
    free(cur_sum);
    free(lambda_v);
    free(jumps);
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_lambda_i_trace_output(result);
        result.lambda_trace = NULL;
        result.ts           = NULL;
        result.n_t          = 0;
    }
    return result;
}
