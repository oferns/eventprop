/**
 * @file eventprop_currents.c
 * @brief Rebuilds the synaptic current at each post-synaptic spike.
 *
 * Extracted from the backward sweep so the lambda-I diagnostic can use the same rule the
 * training kernel uses, rather than a second opinion of it. The two had drifted:
 * eventprop_diag.c weights each adjoint jump by a fixed 1/(v_th + eps), while the backward
 * pass uses the reference's 1/(current - v_th) * (v_th * lambda_v + error) — and nothing
 * caught it, because the parity test named LambdaITrace never calls the Eigen side. One
 * implementation of "what is the current at this spike" is what keeps that from recurring.
 *
 * The current at a post-spike is exactly sum0[k][nrn] * exp(-t / tau_syn), k the number of
 * input spikes at or before t, so the array the forward pass once handed over is redundant:
 * one cursor over both spike lists rebuilds it for n_post bytes of scratch plus one double
 * per output spike.
 *
 * Accumulated ascending and scaled once at the end, which is how the reference gets it. The
 * sweep itself runs time-reversed, and a row kept rebased at the sweep time would look
 * tempting — every term stays under its weight — but dropping a spike that dominates the row
 * takes six of the row's digits away at the benchmark sizes, and the gradient then drifts
 * 1e-5 relative. Ascending is the only stable direction.
 */

#include "eventprop_internal.h"
#include "eventprop_simd.h"
#include "matrixview.inc"

#include <math.h>

void eventprop_rebuild_post_currents(const EventPropInput* input,
                                     const double* post_times,
                                     const int* post_sources,
                                     int n_post_spikes,
                                     double* currents,
                                     double* row_scratch) {
    /* row_scratch is a running synaptic sum, so it must enter zeroed; the caller owns it
     * (the backward pass keeps one calloc'd row in its context and reuses it). */
    MatrixView w = mv(input->w, input->n_pre, input->n_post);
    double* row = row_scratch;
    int k = 0;
    for (int i = 0; i < n_post_spikes; i++) {
        while (k < input->n_spikes && input->times[k] <= post_times[i]) {
            fma_row_simd(row, row, mv_row(w, input->sources[k]),
                         exp(input->times[k] / input->tau_syn), input->n_post);
            k++;
        }
        currents[i] = row[post_sources[i]] * exp(-post_times[i] / input->tau_syn);
    }
}
