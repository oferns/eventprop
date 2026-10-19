/**
 * @file eventprop_maxima.c
 * @brief Membrane-potential maxima and voltage-based backward pass.
 */
#include "eventprop.h"
#include "matrixview.inc"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

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

MaximaOutput eventprop_compute_maxima(const EventPropInput input)
{
    MaximaOutput result = (MaximaOutput){ .error_no = eventprop_validate_input_no_v_th(input) } ;
    SumsOutput sums     = {0};

    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    size_t sz = (size_t)input.n_post;
    /* Allocate output - always exactly n_post elements */
    result.times   = eventprop_try_calloc(sz, sizeof(double));
    if (!result.times) {
        result.error_no = EVENTPROP_ALLOC_MAXIMA_TIMES;
        goto cleanup_and_return;
    }
    result.values  = eventprop_try_calloc(sz, sizeof(double));
    if (!result.values) {
        result.error_no = EVENTPROP_ALLOC_MAXIMA_VALUES;
        goto cleanup_and_return;
    }
    result.errors  = eventprop_try_calloc(sz, sizeof(double));
    if (!result.errors) {
        result.error_no = EVENTPROP_ALLOC_MAXIMA_ERRORS;
        goto cleanup_and_return;
    }

    result.n_post = input.n_post;

    /* Initialize maxima to "not found" */
    for (int i = 0; i < input.n_post; i++) {
        result.times[i]   = NAN;
        result.values[i]  = 0.0;
    }

    /* Compute sums */
    sums = eventprop_compute_sums(input);
    if (sums.error_no != EVENTPROP_OK) {
        result.error_no = sums.error_no;   /* propagate from compute_sums */
        goto cleanup_and_return;
    }

    int n_rows = input.n_spikes + 1;
    double* sum0 = sums.sum0;
    double* sum1 = sums.sum1;

    /* Find maxima for each post-synaptic neuron */
    const double k_pref = input.tau_syn / (input.tau_mem - input.tau_syn);
    const double tmax_pref = 1.0 / (1.0/input.tau_mem - 1.0/input.tau_syn);
    const double tmax_sum = log(input.tau_syn / input.tau_mem);

    /* No spikes → nothing to compute, return NaN/0 (matches Eigen) */
    if (input.n_spikes == 0) {
        goto cleanup_and_return;
    }

    MatrixView s0 = mv(sum0, n_rows, input.n_post);
    MatrixView s1 = mv(sum1, n_rows, input.n_post);

    int target_nrn;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    // ReSharper disable once CppJoinDeclarationAndAssignment
    for (target_nrn = 0; target_nrn < input.n_post; target_nrn++) {
        for (int i = 0; i <= input.n_spikes; i++) {
            double t_input = (i == input.n_spikes) ? INFINITY : input.times[i];
            double s0v = *mv_at(s0, i, target_nrn);
            double s1v = *mv_at(s1, i, target_nrn);

            /* Analytical tmax from log(s1v/s0v) */
            if (s0v > 0 && s1v > 0) {
                double tmax = tmax_pref * (tmax_sum + log(s1v / s0v));

                if (!isinf(tmax) && (i == 0 || input.times[i-1] <= tmax) &&
                    (i == input.n_spikes || tmax <= t_input)) {
                    double vmax = k_pref * (s1v * exp(-tmax/input.tau_mem) -
                                            s0v * exp(-tmax/input.tau_syn));
                    if (vmax > result.values[target_nrn]) {
                        result.values[target_nrn] = vmax;
                        result.times[target_nrn]  = tmax;
                    }
                }
            }

            /* Also check voltage at input spike time */
            if (t_input < INFINITY && (s0v != 0 || s1v != 0)) {
                double v_at_input = k_pref * (s1v * exp(-t_input/input.tau_mem) -
                                              s0v * exp(-t_input/input.tau_syn));
                if (v_at_input > result.values[target_nrn]) {
                    result.values[target_nrn] = v_at_input;
                    result.times[target_nrn]  = t_input;
                }
            }
        }
    }

cleanup_and_return:
    eventprop_free_sums_output(sums);
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_maxima_output(result);
        result.times = NULL;   /* already freed — keep a caller-side free() safe */
        result.values = NULL;
        result.errors = NULL;
    }
    return result;
}

BackwardOutput eventprop_backward_maxima(const EventPropInput input,
                                         const double* maxima_times,
                                         const double* maxima_errors) {

    BackwardOutput result = (BackwardOutput) { .error_no = eventprop_validate_input_no_v_th(input) };

    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    /* Validate maxima pointers */
    if (!maxima_times || !maxima_errors) {
        result.error_no     = EVENTPROP_NULL_POINTER;
        goto cleanup_and_return;
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

    result.n_pre = input.n_pre;
    result.n_post = input.n_post;
    result.n_spikes = input.n_spikes;

    if (input.n_spikes == 0 || input.n_post == 0) {
        goto cleanup_and_return;
    }

    const double largest_time = input.times[input.n_spikes - 1];
    const double k_bwd_prefactor = input.tau_mem / (input.tau_mem - input.tau_syn);

    /* Precompute inverse time constants */
    const double tau_mem_inv = 1.0 / input.tau_mem;
    const double tau_syn_inv = 1.0 / input.tau_syn;

    /* Sweep backward through input spikes */
    for (int spike_idx = input.n_spikes - 1; spike_idx >= 0; spike_idx--) {
        const double t_bwd = largest_time - input.times[spike_idx];

        for (int max_idx = 0; max_idx < input.n_post; max_idx++) {
            if (isnan(maxima_times[max_idx]))
                continue;

            double t_vmax_bwd = largest_time - maxima_times[max_idx];
            if (t_vmax_bwd > t_bwd)
                continue;

            double dt_mem = -(t_bwd - t_vmax_bwd) * tau_mem_inv;
            double dt_syn = -(t_bwd - t_vmax_bwd) * tau_syn_inv;

            double lambda_v = -exp(dt_mem) * maxima_errors[max_idx] * tau_mem_inv;
            double k_bwd_val = k_bwd_prefactor * (exp(dt_mem) - exp(dt_syn));
            double lambda_i = -k_bwd_val * maxima_errors[max_idx] * tau_mem_inv;

            int source_nrn = input.sources[spike_idx];
            result.grad[source_nrn * input.n_post + max_idx] -= input.tau_syn * lambda_i;

            /* Spike error weighted by connection strength (matching Eigen/pybind11) */
            result.spike_errors[spike_idx] += input.w[source_nrn * input.n_post + max_idx] * (lambda_v - lambda_i);
        }
    }

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_backward_output(result);
        result.grad = NULL;          /* already freed — keep a caller-side free() safe */
        result.spike_errors = NULL;
    }
    return result;
}
