/**
 * @file eventprop_sums.c
 * @brief Pre-computed exponential-weighted cumulative sums.
 */
#include "eventprop.h"
#include "matrixview.inc"
#include "eventprop_simd.h"
#include <math.h>
#include <stdlib.h>

SumsOutput eventprop_compute_sums(const EventPropInput input)
{
    SumsOutput result = (SumsOutput){ .error_no = eventprop_validate_input(input) };
    double* exp_mem = NULL, *exp_syn = NULL;

    result.n_rows = input.n_spikes + 1;
    result.n_post = input.n_post;

    if (result.error_no != EVENTPROP_OK){
        goto cleanup_and_return;
    }

    size_t count = (size_t)result.n_rows * (size_t)input.n_post;
    
    result.sum0 = calloc(count, sizeof(double));
    if(!result.sum0) {
        result.error_no = EVENTPROP_ALLOC_SUM0;
        goto cleanup_and_return;
    }

    result.sum1 = calloc(count, sizeof(double));
    if (!result.sum1) {
        result.error_no = EVENTPROP_ALLOC_SUM1;
        goto cleanup_and_return;
    }

    if (input.n_spikes == 0) {
        goto cleanup_and_return;
    }

    /* Pre-compute exponential lookup tables */
    exp_mem = calloc((size_t)input.n_spikes, sizeof(double));
    if (!exp_mem) {
        result.error_no = EVENTPROP_ALLOC_EXP_MEM;
        goto cleanup_and_return;
    }
    exp_syn = calloc((size_t)input.n_spikes, sizeof(double));
    if (!exp_syn) {
        result.error_no = EVENTPROP_ALLOC_EXP_SYN;
        goto cleanup_and_return;
    }

    for (int i = 0; i < input.n_spikes; i++) {
        exp_mem[i] = exp(input.times[i] / input.tau_mem);
        exp_syn[i] = exp(input.times[i] / input.tau_syn);
    }

    /* Accumulate sums using SIMD-accelerated FMA */
    MutableMatrixView s0 = mv_mut(result.sum0, result.n_rows, input.n_post);
    MutableMatrixView s1 = mv_mut(result.sum1, result.n_rows, input.n_post);
    MatrixView w         = mv(input.w, input.n_pre, input.n_post);

    for (int i = 0; i < input.n_spikes; i++) {
        int src = input.sources[i];
        fma_row_simd(mv_row_mut(s0, i + 1), mv_row_mut(s0, i), mv_row(w, src), exp_syn[i], input.n_post);
        fma_row_simd(mv_row_mut(s1, i + 1), mv_row_mut(s1, i), mv_row(w, src), exp_mem[i], input.n_post);
    }

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_sums_output(result);
    }
    free(exp_mem);
    free(exp_syn);
    return result;
}
