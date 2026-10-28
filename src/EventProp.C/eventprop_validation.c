/**
 * @file eventprop_validation.c
 * @brief Input validation and error-message helpers.
 */
#include "eventprop.h"
#include "eventprop_internal.h"   /* declares eventprop_validate_post_spikes, defined below */
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/**
 * Structural validation — every field every entry point reads, except v_th.
 *
 * Use this for entry points that never read the threshold voltage: ComputeSums,
 * ComputeMaxima and backward_maxima. Those algorithms are defined for any v_th,
 * and the reference accepts v_th == 0 there, so demanding a positive one would
 * reject a perfectly valid call. Entry points that do read v_th (the spike
 * search, the adjoint sweep, the diagnostic traces) call eventprop_validate_input
 * instead, which layers the threshold check on top of this one.
 */
EventPropError eventprop_validate_input_no_v_th(const EventPropInput input)
{
    /* ── Dimension checks ─────────────────────────────────────── */
    if (input.n_pre <= 0 || input.n_post <= 0)
        return EVENTPROP_INVALID_DIMENSION;
    if (input.n_pre > EVENTPROP_MAX_DIM || input.n_post > EVENTPROP_MAX_DIM)
        return EVENTPROP_DIMENSION_TOO_LARGE;
    /* n_pre * n_post can overflow int even when both pass MAX_DIM */
    if ((size_t)input.n_pre * (size_t)input.n_post > (size_t)INT_MAX)
        return EVENTPROP_DIMENSION_TOO_LARGE;

    /* ── Spike count ──────────────────────────────────────────── */
    if (input.n_spikes < 0)
        return EVENTPROP_NEGATIVE_SPIKES;

    /* ── Tau bounds and proximity ─────────────────────────────── */
    /* Negated comparisons so NaN taus are rejected (NaN <= 0 would pass). */
    if (!(input.tau_mem > 0.0) || !(input.tau_syn > 0.0))
        return EVENTPROP_INVALID_TAU;
    if (input.tau_mem < EVENTPROP_MIN_TAU || input.tau_syn < EVENTPROP_MIN_TAU)
        return EVENTPROP_TAU_TOO_SMALL;
    if (input.tau_mem > EVENTPROP_MAX_TAU || input.tau_syn > EVENTPROP_MAX_TAU)
        return EVENTPROP_TAU_TOO_LARGE;
    if (input.tau_mem == input.tau_syn)
        return EVENTPROP_INVALID_TAU;

    /* Relative difference: |tau_mem - tau_syn| / min(tau) < 1e-6 → k_prefactor blows up */
    {
        const double min_tau = fmin(input.tau_mem, input.tau_syn);
        const double rel_diff = fabs(input.tau_mem - input.tau_syn) / min_tau;
        if (rel_diff < EVENTPROP_TAU_REL_DIFF_MIN)
            return EVENTPROP_TAU_TOO_CLOSE;
    }

    /* ── Pointer checks ───────────────────────────────────────── */
    if (input.w == NULL)
        return EVENTPROP_NULL_POINTER;
    if (input.n_spikes > 0) {
        if (input.times == NULL || input.sources == NULL)
            return EVENTPROP_NULL_POINTER;
    } else {
        /* n_spikes == 0: times/sources may be NULL, nothing to iterate */
        return EVENTPROP_OK;
    }

    /* ── Per-spike checks (single pass) ──────────────────────── */
    double max_time = -1.0;   /* sentinel: will become >= 0 on first spike */
    double prev_time = -1.0;  /* ascending-order enforcement */
    for (int i = 0; i < input.n_spikes; i++) {
        if (input.sources[i] < 0 || input.sources[i] >= input.n_pre)
            return EVENTPROP_INVALID_SOURCE;

        const double t = input.times[i];
        if (!isfinite(t))
            return EVENTPROP_NONFINITE_SPIKE_TIME;
        if (i > 0 && t < prev_time)
            return EVENTPROP_UNSORTED_SPIKE_TIMES;
        prev_time = t;
        if (t < 0.0)
            return EVENTPROP_NEGATIVE_SPIKE_TIME;
        if (t > EVENTPROP_MAX_SPIKE_TIME)
            return EVENTPROP_EXCESSIVE_SPIKE_TIME;
        if (max_time < 0.0 || t > max_time)
            max_time = t;
    }

    /* ── Cross-field: times vs tau overflow check ─────────────── */
    /* exp(max_time / min_tau) must stay finite (< INF at ~709.78). */
    {
        const double min_tau = fmin(input.tau_mem, input.tau_syn);
        if (max_time > 0.0 && max_time / min_tau >= EVENTPROP_EXP_OVERFLOW_THRESHOLD)
            return EVENTPROP_TIMES_TAU_MISMATCH;
    }

    return EVENTPROP_OK;
}

/** ── Full validation ───────────────────────────────────────────────── */

EventPropError eventprop_validate_input(const EventPropInput input)
{
    const EventPropError structural = eventprop_validate_input_no_v_th(input);
    if (structural != EVENTPROP_OK)
        return structural;

    /* Reject a NaN threshold explicitly. The negated comparison below is
     * NaN-correct only under strict FP semantics: MSVC's /fp:fast assumes no NaN
     * and rewrites `v_th > 0.0` to `v_th >= 0.0`, so the negation becomes
     * `v_th <= 0.0` — false for NaN — and a NaN threshold is accepted (observed:
     * ValidateInput.NanVthRejectedAsInvalidVth fails, validate returns OK).
     *
     * isnan() is a bit-pattern classification (significand nonzero, exponent
     * all-ones), not an ordering relation, so the finite-math assumptions in
     * /fp:fast have nothing to rewrite. The fast-math set MSVC documents
     * (-ffp-contract=fast, -freciprocal-math, -fno-signed-zeros) does not touch
     * classification intrinsics.
     *
     * This is a no-op on GCC/Clang, whose fast-math set here deliberately omits
     * -ffinite-math-only, so the negated comparison already survives NaN. Keeping
     * the explicit test means correctness travels with the source: it does not
     * depend on a compile flag set in a CMakeLists.txt that a reader of this file
     * cannot see. Pinned by ValidateInput.NanVthRejectedAsInvalidVth — note that
     * test can only FAIL on MSVC, since no other compiler here applies the flag
     * that breaks the guard. */
    if (isnan(input.v_th) || !(input.v_th > 0.0))
        return EVENTPROP_INVALID_VTH;

    return EVENTPROP_OK;
}

/** ── Caller-supplied post-synaptic spike lists ───────────────────────── */

/**
 * Validate a post-synaptic spike list handed back by the caller.
 *
 * The forward pass returns times/sources; the backward pass and both diagnostic traces take
 * such a list as an argument. Every consumer then walks it with a single cursor moving forward
 * (eventprop_rebuild_post_currents) or backward with an early break (the adjoint sweep), so an
 * unsorted list is not a valid-but-different input — it rebuilds the wrong synaptic currents or
 * under-counts adjoint jumps and still returns EVENTPROP_OK. The sources index per-neuron arrays
 * (lambda_v, the current-rebuild row), so an out-of-range index writes out of bounds rather than
 * merely computing a wrong number.
 *
 * So the rules are the ones every spike list in this library already has — non-decreasing,
 * finite, non-negative — with the source bound taken against n_post instead of n_pre. The Eigen
 * reference requires all of it and checks none of it, which is why it silently returns plausible
 * numbers on such input while C reports a code; this is the strict side of that shared contract,
 * matching eventprop_backward_spikes' handling of post_spike_errors' spike list.
 *
 * A zero-length list may pass NULL arrays: there is nothing to read, and the empty sweep is a
 * legitimate case (a neuron that never fires), not an error.
 */
EventPropError eventprop_validate_post_spikes(const double* times,
                                              const int* sources,
                                              int n_spikes,
                                              int n_post)
{
    if (n_spikes < 0)
        return EVENTPROP_NEGATIVE_SPIKES;
    if (n_spikes == 0)
        return EVENTPROP_OK;
    if (times == NULL || sources == NULL)
        return EVENTPROP_NULL_POINTER;

    /* Negated / classification checks first, in the order the per-spike pass in
     * eventprop_validate_input_no_v_th uses, so the two lists fail the same way. */
    for (int i = 0; i < n_spikes; i++) {
        const double t = times[i];
        if (!isfinite(t))
            return EVENTPROP_NONFINITE_SPIKE_TIME;
        if (t < 0.0)
            return EVENTPROP_NEGATIVE_SPIKE_TIME;
        if (i > 0 && t < times[i - 1])
            return EVENTPROP_UNSORTED_SPIKE_TIMES;
        if (sources[i] < 0 || sources[i] >= n_post)
            return EVENTPROP_INVALID_SOURCE;
    }

    return EVENTPROP_OK;
}

/** ── Error string helper ─────────────────────────────────────────────── */

const char* eventprop_error_string(const int error_no)
{
    switch (error_no) {
        case EVENTPROP_OK:                      return "OK";
        case EVENTPROP_INVALID_DIMENSION:       return "Invalid dimension: n_pre or n_post must be positive";
        case EVENTPROP_NEGATIVE_SPIKES:         return "Negative spike count";
        case EVENTPROP_NULL_POINTER:            return "NULL pointer: w, times, or sources is NULL when required";
        case EVENTPROP_INVALID_SOURCE:          return "Invalid source neuron index (out of [0, n_pre) range)";
        case EVENTPROP_INVALID_TAU:             return "Invalid tau: tau_mem and tau_syn must be positive and distinct";
        case EVENTPROP_INVALID_CURRENT:         return "Invalid current: spike current equals threshold (division by zero)";
        case EVENTPROP_NEGATIVE_SPIKE_TIME:     return "Negative spike time";
        case EVENTPROP_EXCESSIVE_SPIKE_TIME:    return "Spike time exceeds maximum allowed value";
        case EVENTPROP_TAU_TOO_SMALL:           return "Tau too small (below minimum threshold)";
        case EVENTPROP_TAU_TOO_LARGE:           return "Tau too large (above maximum threshold)";
        case EVENTPROP_INVALID_VTH:             return "Invalid spike threshold: v_th must be positive";
        case EVENTPROP_DIMENSION_TOO_LARGE:     return "Dimension exceeds maximum allowed size";
        case EVENTPROP_TIMES_TAU_MISMATCH:      return "Spike time and tau mismatch would cause exponential overflow";
        case EVENTPROP_TAU_TOO_CLOSE:           return "tau_mem and tau_syn are too close together";
        case EVENTPROP_BATCHSIZE_ZERO_OR_NEG:   return "Batch size must be positive";
        case EVENTPROP_ALLOC_SUM0:              return "Allocation failed: sum0 array";
        case EVENTPROP_ALLOC_SUM1:              return "Allocation failed: sum1 array";
        case EVENTPROP_ALLOC_EXP_MEM:           return "Allocation failed: exp_mem lookup table";
        case EVENTPROP_ALLOC_EXP_SYN:           return "Allocation failed: exp_syn lookup table";
        case EVENTPROP_ALLOC_TIMES_ARR:         return "Allocation failed: post_times array";
        case EVENTPROP_ALLOC_CURRENTS_ARR:      return "Allocation failed: post_currents array";
        case EVENTPROP_ALLOC_SPIKE_CAPS:        return "Allocation failed: spike_caps array";
        case EVENTPROP_ALLOC_SPIKE_COUNTS:      return "Allocation failed: spike_counts array";
        case EVENTPROP_ALLOC_NEURON_BUF:        return "Allocation failed: forward-pass scratch row growth";
        case EVENTPROP_ALLOC_ENTRIES:           return "Allocation failed: entries (sorted collection)";
        case EVENTPROP_ALLOC_RESULT_TIMES:      return "Allocation failed: result spike times";
        case EVENTPROP_ALLOC_RESULT_SOURCES:    return "Allocation failed: result spike sources";
        case EVENTPROP_ALLOC_RESULT_CURRENTS:   return "Allocation failed: result spike currents";
        case EVENTPROP_ALLOC_RESULT_ERRORS:     return "Allocation failed: result errors buffer";
        case EVENTPROP_ALLOC_MAXIMA_TIMES:      return "Allocation failed: maxima times array";
        case EVENTPROP_ALLOC_MAXIMA_VALUES:     return "Allocation failed: maxima values array";
        case EVENTPROP_ALLOC_MAXIMA_ERRORS:     return "Allocation failed: maxima errors array";
        case EVENTPROP_ALLOC_BWD_GRAD:          return "Allocation failed: backward gradient matrix";
        case EVENTPROP_ALLOC_BWD_SPIKE_ERRORS:  return "Allocation failed: backward spike errors buffer";
        case EVENTPROP_ALLOC_BWD_MERGED:        return "Allocation failed: backward merged spike array";
        case EVENTPROP_ALLOC_BWD_ALL_ERRORS:    return "Allocation failed: backward all_errors buffer";
        case EVENTPROP_ALLOC_BWD_LAMBDA_V:      return "Allocation failed: backward lambda_v array";
        case EVENTPROP_ALLOC_BWD_LAMBDA_I:      return "Allocation failed: backward lambda_i array";
        case EVENTPROP_ALLOC_BWD_JUMP_TIME:     return "Allocation failed: backward jump_time array";
        case EVENTPROP_ALLOC_BWD_JUMP_MAG:      return "Allocation failed: backward jump_mag array";
        case EVENTPROP_ALLOC_BWD_JUMP_COUNT:    return "Allocation failed: backward jump_count array";
        case EVENTPROP_ALLOC_BWD_JUMP_CAP:      return "Allocation failed: backward jump_cap array";
        case EVENTPROP_ALLOC_BWD_PER_NEURON:    return "Allocation failed: per-neuron backward buffer";
        case EVENTPROP_ALLOC_BATCH_OUTPUTS:     return "Allocation failed: batch outputs array";
        case EVENTPROP_ALLOC_BATCH_GRAD:        return "Allocation failed: accumulated gradient pointer";
        case EVENTPROP_ALLOC_DROPOUT_SAMPLE:    return "Allocation failed: dropout per-sample buffer";
        case EVENTPROP_ALLOC_DROPOUT_OFFSETS:   return "Allocation failed: dropout offsets scratch buffer";
        case EVENTPROP_ALLOC_DIAG_V_TRACE:      return "Allocation failed: voltage trace array";
        case EVENTPROP_ALLOC_DIAG_TS:           return "Allocation failed: time steps array";
        case EVENTPROP_ALLOC_DIAG_LAMBDA:       return "Allocation failed: lambda-I trace array";
        case EVENTPROP_BATCH_EMPTY:             return "Batch input was empty or NULL";
        case EVENTPROP_BATCH_PARTIAL_ACCUMULATION:
                                                return "Gradient sum is partial — one or more samples failed and were excluded";
        case EVENTPROP_NONFINITE_SPIKE_TIME:    return "Spike time is not finite (NaN)";
        case EVENTPROP_DIAG_TIMESTEP_INVALID:   return "Invalid trace timestep: dt and t_max must be positive";
        case EVENTPROP_DIAG_TARGET_OUT_OF_RANGE: return "Trace target neuron index out of [0, n_post) range";
        case EVENTPROP_DIAG_TRACE_TOO_LONG:     return "Trace would exceed maximum number of time samples";
        case EVENTPROP_INVALID_DROPOUT_PROB:    return "Invalid dropout probability: p_drop must be in [0, 1]";
        case EVENTPROP_NEGATIVE_SAMPLE_COUNT:   return "Negative per-sample spike count (n_per_sample[i] < 0)";
        case EVENTPROP_BATCH_DIMENSION_MISMATCH: return "Sample gradient dimensions do not match requested n_pre/n_post";
        case EVENTPROP_UNSORTED_SPIKE_TIMES:    return "Spike times are not sorted in ascending order";
        case EVENTPROP_ALLOC_THREAD_SCRATCH:    return "Allocation failed: forward-pass thread scratch";
        case EVENTPROP_ALLOC_BWD_CUR_SUM:       return "Allocation failed: backward-pass running current";
        case EVENTPROP_ALLOC_DIAG_CURRENTS:     return "Allocation failed: lambda-I post-spike currents";
        case EVENTPROP_ALLOC_DIAG_CUR_SUM:      return "Allocation failed: lambda-I running synaptic sum";
        case EVENTPROP_ALLOC_DIAG_LAMBDA_V:     return "Allocation failed: lambda-I membrane adjoint array";
        case EVENTPROP_ALLOC_DIAG_LAMBDA_JUMPS: return "Allocation failed: lambda-I jump arrays";
        case EVENTPROP_DIAG_QUERY_TIME_INVALID: return "Invalid lambda-I query time (negative or non-finite)";
        case EVENTPROP_ALLOC_BWD_CURRENTS:      return "Allocation failed: backward-pass post-spike currents";
        case EVENTPROP_ALLOC_LOSS_ERRORS:       return "Allocation failed: regression dL/dmaxima array";
        case EVENTPROP_ALLOC_LOSS_DECODER_GRAD: return "Allocation failed: regression dL/da array";
        case EVENTPROP_INVALID_HUBER_DELTA:     return "Invalid Huber scale: huber_delta must be positive, or non-finite for least squares";
        case EVENTPROP_NONFINITE_TARGET:        return "Regression target is NaN or infinite";
        case EVENTPROP_NONFINITE_DECODER:       return "Readout parameter is NaN or infinite (decoder weight or bias)";
        case EVENTPROP_NONFINITE_PREDICTION:    return "Readout overflowed to a non-finite value (weights, bias or maxima too large)";
        default:                                return "Unknown error";
    }
}
