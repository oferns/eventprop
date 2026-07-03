/**
 * @file eventprop_diag.c
 * @brief Diagnostic helpers — voltage traces and lambda-I traces.
 */

#include "eventprop_diag.h"
#include "eventprop.h"
#include "matrixview.inc"  /* inline MatrixView helpers */
#include <stdlib.h>
#include <math.h>

void eventprop_free_lambda_i_trace_output(LambdaITraceOutput* output)
{
    if (!output) return;
    free(output->lambda_trace);
    free(output->ts);
}

void eventprop_free_voltage_trace_output(VoltageTraceOutput* output)
{
    if (!output) return;
    free(output->v_trace);
    free(output->ts);
}

VoltageTraceOutput eventprop_compute_voltage_trace(const VoltageTraceInput input)
{
    const EventPropInput* fwd = input.input;
    int n_t = (int)(input.t_max / input.dt);

    VoltageTraceOutput result;
    result.n_t = n_t;
    result.v_trace = calloc(n_t, sizeof(double));
    result.ts      = calloc(n_t, sizeof(double));

    double k_prefactor = fwd->tau_syn / (fwd->tau_mem - fwd->tau_syn);
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
    int n_t = (int)(input.t_max / input.dt);

    LambdaITraceOutput result;
    result.n_t = n_t;
    result.lambda_trace = calloc(n_t, sizeof(double));
    result.ts           = calloc(n_t, sizeof(double));

    for (int ti = 0; ti < n_t; ti++) {
        double t = input.dt * ti;
        result.ts[ti] = t;

        if (input.n_post_spikes == 0) {
            result.lambda_trace[ti] = 0.0;
            continue;
        }

        double largest_time = input.fwd_times[input.n_post_spikes - 1];

        double lambda_i = 0.0;
        double t_bwd_from_t = largest_time - t;
        double k_bwd_prefactor = input.tau_mem / (input.tau_mem - input.tau_syn);

        /* Collect jumps from post-synaptic spikes after time t */
        for (int i = input.n_post_spikes - 1; i >= 0; i--) {
            if (input.fwd_times[i] < t) break;
            double dt_jump = t_bwd_from_t - (largest_time - input.fwd_times[i]);
            lambda_i += (1.0 / (input.v_th + 0.001)) * k_bwd_prefactor *
                        (exp(-dt_jump / input.tau_mem) - exp(-dt_jump / input.tau_syn));
        }

        result.lambda_trace[ti] = lambda_i;
    }

    return result;
}
