/**
 * @file eventprop_diag.c
 * @brief Diagnostic helpers — voltage traces and lambda-I traces.
 */

#include "eventprop_diag.h"
#include "eventprop.h"
#include "matrixview.inc"  /* inline MatrixView helpers */
#include <stdlib.h>
#include <math.h>

/* Numerical stability epsilon for lambda-I backward current computation */
#define LAMBDA_I_VTH_EPS 0.001

void eventprop_free_lambda_i_trace_output(LambdaITraceOutput output)
{
    free(output.lambda_trace);
    free(output.ts);
}

void eventprop_free_voltage_trace_output(VoltageTraceOutput output)
{
    free(output.v_trace);
    free(output.ts);
}

VoltageTraceOutput eventprop_compute_voltage_trace(const VoltageTraceInput input)
{
    VoltageTraceOutput result;
    result.n_t       = 0;
    result.v_trace   = NULL;
    result.ts        = NULL;
    result.error_no  = EVENTPROP_OK;

    /* Validate pointers */
    if (!input.input) {
        result.error_no = EVENTPROP_NULL_POINTER;
        return result;
    }
    const EventPropInput* fwd = input.input;

    /* Validate dt and t_max */
    if (input.dt <= 0 || input.t_max <= 0) {
        result.error_no = EVENTPROP_INVALID_DIMENSION;
        return result;
    }

    int n_t = (int)(input.t_max / input.dt);

    /* Validate tau values (division by zero in k_prefactor) */
    if (fwd->tau_mem <= 0 || fwd->tau_syn <= 0 || fwd->tau_mem == fwd->tau_syn) {
        result.error_no = EVENTPROP_INVALID_TAU;
        return result;
    }

    /* Validate target neuron index */
    if (input.target_nrn_idx < 0 || input.target_nrn_idx >= fwd->n_post) {
        result.error_no = EVENTPROP_INVALID_SOURCE;
        return result;
    }

    /* Validate forward spike pointers */
    if (fwd->n_spikes > 0 && (!fwd->times || !fwd->sources)) {
        result.error_no = EVENTPROP_NULL_POINTER;
        return result;
    }
    if (input.n_post_spikes > 0 && (!input.fwd_times || !input.fwd_sources)) {
        result.error_no = EVENTPROP_NULL_POINTER;
        return result;
    }

    result.n_t = n_t;
    result.v_trace = calloc(n_t, sizeof(double));
    if (!result.v_trace) {
        result.error_no = EVENTPROP_ALLOC_DIAG_V_TRACE;
        return result;
    }

    result.ts = calloc(n_t, sizeof(double));
    if (!result.ts) {
        free(result.v_trace);
        result.v_trace = NULL;
        result.error_no = EVENTPROP_ALLOC_DIAG_TS;
        return result;
    }

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

LambdaITraceOutput eventprop_compute_lambda_i_trace(const LambdaITraceInput input)
{
    LambdaITraceOutput result;
    result.n_t          = 0;
    result.lambda_trace = NULL;
    result.ts           = NULL;
    result.error_no     = EVENTPROP_OK;

    /* Validate dt and t_max */
    if (input.dt <= 0 || input.t_max <= 0) {
        result.error_no = EVENTPROP_INVALID_DIMENSION;
        return result;
    }

    int n_t = (int)(input.t_max / input.dt);

    /* Validate tau values (division by zero in k_bwd_prefactor) */
    if (input.tau_mem <= 0 || input.tau_syn <= 0 || input.tau_mem == input.tau_syn) {
        result.error_no = EVENTPROP_INVALID_TAU;
        return result;
    }

    /* Validate v_th (division in jump computation) */
    if (input.v_th <= 0) {
        result.error_no = EVENTPROP_INVALID_DIMENSION;
        return result;
    }

    /* Validate forward spike pointers */
    if (input.n_post_spikes > 0 && !input.fwd_times) {
        result.error_no = EVENTPROP_NULL_POINTER;
        return result;
    }

    result.n_t = n_t;
    result.lambda_trace = calloc(n_t, sizeof(double));
    if (!result.lambda_trace) {
        result.error_no = EVENTPROP_ALLOC_DIAG_LAMBDA;
        return result;
    }

    result.ts = calloc(n_t, sizeof(double));
    if (!result.ts) {
        free(result.lambda_trace);
        result.lambda_trace = NULL;
        result.error_no     = EVENTPROP_ALLOC_DIAG_TS;
        return result;
    }

    for (int ti = 0; ti < n_t; ti++) {
        double t = input.dt * ti;
        result.ts[ti] = t;

        if (input.n_post_spikes == 0) {
            result.lambda_trace[ti] = 0.0;
            continue;
        }

        const double largest_time = input.fwd_times[input.n_post_spikes - 1];

        double lambda_i = 0.0;
        double t_bwd_from_t = largest_time - t;
        const double k_bwd_prefactor = input.tau_mem / (input.tau_mem - input.tau_syn);

        /* Collect jumps from post-synaptic spikes after time t */
        for (int i = input.n_post_spikes - 1; i >= 0; i--) {
            if (input.fwd_times[i] < t) break;
            double dt_jump = t_bwd_from_t - (largest_time - input.fwd_times[i]);
            lambda_i += (1.0 / (input.v_th + LAMBDA_I_VTH_EPS)) * k_bwd_prefactor *
                        (exp(-dt_jump / input.tau_mem) - exp(-dt_jump / input.tau_syn));
        }

        result.lambda_trace[ti] = lambda_i;
    }

    return result;
}
