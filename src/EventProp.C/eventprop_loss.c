/**
 * @file eventprop_loss.c
 * @brief Continuous readout (linear decoder over voltage maxima) and its losses.
 *
 * One sample routine backs all three entry points — the allocating single-sample call, the
 * in-place single-sample call and the batch call. That is not a convenience: the skip rule,
 * the Huber branch and the accumulate-not-zero rule are each a policy, and two copies of a
 * policy is how a batch ends up computing a different loss from the single-sample path that
 * every unit test exercises.
 */

#include "eventprop_loss.h"
#include "eventprop.h"
#include "eventprop_internal.h"
#include <stdlib.h>   /* size_t */
#include <math.h>

/* Public output-freeing functions live in eventprop_memory.c. */

/**
 * Loss and chain factor for a residual, shared by every entry point.
 *
 * Smooth (Bessel) Huber of scale delta is
 *
 *     L(r)   = delta^2 (sqrt(1 + r^2/delta^2) - 1)
 *     dL/dr  = r / sqrt(1 + r^2/delta^2)
 *
 * Multiplying out delta^2 * q^2 = r^2 rewrites the loss as r^2 / (sqrt(1 + q^2) + 1), which
 * is what is computed here. The algebra is not decoration: as written, `delta^2 * (sqrt(1 +
 * q^2) - 1)` loses every significant digit to cancellation for a small residual — precisely
 * the regime a converged model lives in — and `delta^2` overflows before a large scale does
 * any good. `hypot(1, q)` is the overflow-safe spelling of sqrt(1 + q^2).
 *
 * Plain least squares is the delta -> infinity limit of the same expression: sqrt(1 + q^2)
 * becomes 1, the loss becomes r^2/2 and the derivative becomes r. So `use_mse` picks the
 * denominator and nothing else, and the two losses cannot drift apart.
 *
 * The caller has already rejected NaN and non-positive delta; a non-finite delta (only +Inf
 * can reach here) selects the least-squares branch.
 */
static void residual_terms(double r, double delta, int use_mse,
                           double* loss, double* dloss_dr)
{
    const double denom = use_mse ? 1.0 : hypot(1.0, r / delta);
    *loss     = (r * r) / (denom + 1.0);
    *dloss_dr = r / denom;
}

/**
 * Readout, loss and gradients for one sample.
 *
 * `errors_out` (length n_post) receives dL/dmaxima[n] for every neuron — 0 for a skipped one,
 * so a reused buffer cannot carry a stale error forward. `decoder_grad_accum` and
 * `bias_grad_accum` are accumulated into and may be NULL.
 *
 * Skip rule: a neuron whose maxima value is not finite never peaked, so it carries no
 * readout signal. It is excluded from y_hat and given a zero error, and counted. It is not an
 * error — a silent neuron is a legitimate sample, and `eventprop_backward_maxima` skips those
 * neurons by their NaN maxima time anyway; writing 0 here just keeps the array honest for
 * whoever reads it back.
 */
static MaximaRegressionResult
maxima_regression_sample(const double* values, const double* decoder, double bias,
                         double target, double huber_delta, int n_post,
                         double* errors_out, double* decoder_grad_accum,
                         double* bias_grad_accum)
{
    MaximaRegressionResult result = { .error_no = EVENTPROP_OK,
                                      .prediction = 0.0,
                                      .loss = 0.0,
                                      .residual_derivative = 0.0,
                                      .n_skipped = 0 };

    if (!values || !decoder || !errors_out) {
        result.error_no = EVENTPROP_NULL_POINTER;
        return result;
    }
    if (n_post <= 0) {
        result.error_no = EVENTPROP_INVALID_DIMENSION;
        return result;
    }
    if (isnan(target) || isinf(target)) {
        result.error_no = EVENTPROP_NONFINITE_TARGET;
        return result;
    }
    /* NaN delta would otherwise fall quietly into one of the two branches below, and a
     * negative scale is a different loss from the one the caller named. */
    if (isnan(huber_delta) || huber_delta <= 0.0) {
        result.error_no = EVENTPROP_INVALID_HUBER_DELTA;
        return result;
    }
    if (isnan(bias) || isinf(bias)) {
        result.error_no = EVENTPROP_NONFINITE_DECODER;
        return result;
    }
    const int use_mse = !isfinite(huber_delta);

    /* Pass 1 — the readout. Every decoder entry must be finite, including entries for
     * neurons that are silent in this sample: a NaN in the decoder is a corrupted parameter
     * vector, not a property of the datum, and refusing it here is the difference between a
     * clear error and a gradient that is silently NaN two steps later. */
    double y_hat = bias;
    int skipped = 0;
    for (int n = 0; n < n_post; ++n) {
        if (!isfinite(decoder[n])) {
            result.error_no = EVENTPROP_NONFINITE_DECODER;
            return result;
        }
        if (!isfinite(values[n])) {
            ++skipped;
            continue;
        }
        y_hat += decoder[n] * values[n];
    }
    if (!isfinite(y_hat)) {
        /* Weights and maxima are individually finite but the sum is not: the parameter
         * vector has overflowed. Reporting a loss of inf with inf gradients would let a
         * training loop take an inf step and call it divergence later. */
        result.error_no = EVENTPROP_NONFINITE_PREDICTION;
        return result;
    }

    const double residual = y_hat - target;
    double loss, dloss_dr;
    residual_terms(residual, huber_delta, use_mse, &loss, &dloss_dr);

    /* Pass 2 — chain the single scalar dL/dy_hat onto each neuron.
     *   dL/dmaxima[n] = dL/dy_hat * a[n]
     *   dL/da[n]      = dL/dy_hat * maxima[n]      (accumulated: a is shared by the batch)
     *   dL/db         = dL/dy_hat                  (accumulated likewise) */
    for (int n = 0; n < n_post; ++n) {
        if (!isfinite(values[n])) {
            errors_out[n] = 0.0;
            continue;
        }
        errors_out[n] = dloss_dr * decoder[n];
        if (decoder_grad_accum) {
            decoder_grad_accum[n] += dloss_dr * values[n];
        }
    }
    if (bias_grad_accum) {
        *bias_grad_accum += dloss_dr;
    }

    result.prediction = y_hat;
    result.loss       = loss;
    result.n_skipped  = skipped;
    result.residual_derivative = dloss_dr;
    return result;
}

MaximaRegressionOutput eventprop_maxima_regression(MaximaRegressionInput input)
{
    MaximaRegressionOutput result = { .error_no = EVENTPROP_OK,
                                      .prediction = 0.0,
                                      .loss = 0.0,
                                      .residual_derivative = 0.0,
                                      .errors = NULL,
                                      .decoder_grad = NULL,
                                      .bias_grad = 0.0,
                                      .n_post = 0,
                                      .n_skipped = 0 };

    /* A refused output reports n_post == 0 alongside NULL arrays, so a caller that reads the
     * struct without reading error_no cannot size a loop to an array that is not there. That
     * is why n_post is assigned on the success path rather than at the top. */
    if (!input.maxima_values || !input.decoder) {
        result.error_no = EVENTPROP_NULL_POINTER;
        goto cleanup_and_return;
    }
    if (input.n_post <= 0) {
        result.error_no = EVENTPROP_INVALID_DIMENSION;
        goto cleanup_and_return;
    }

    result.errors = eventprop_try_calloc((size_t)input.n_post, sizeof(double));
    if (!result.errors) {
        result.error_no = EVENTPROP_ALLOC_LOSS_ERRORS;
        goto cleanup_and_return;
    }
    /* Second allocation, checked before it is attempted, per the never-two-allocations-in-a-
     * row rule: had this been written the other way round a failure here would have leaked
     * the array above while returning an output whose errors pointer looked usable. */
    result.decoder_grad = eventprop_try_calloc((size_t)input.n_post, sizeof(double));
    if (!result.decoder_grad) {
        result.error_no = EVENTPROP_ALLOC_LOSS_DECODER_GRAD;
        goto cleanup_and_return;
    }

    /* Both arrays are calloc'd, so accumulating into them is assignment. One routine for
     * every entry point is the point of this file; the zero-initialisation is what makes
     * accumulate == assign rather than a second code path. */
    const MaximaRegressionResult sample =
        maxima_regression_sample(input.maxima_values, input.decoder, input.bias,
                                 input.target, input.huber_delta, input.n_post,
                                 result.errors, result.decoder_grad, &result.bias_grad);
    result.error_no = sample.error_no;
    if (result.error_no != EVENTPROP_OK) {
        goto cleanup_and_return;
    }

    result.prediction = sample.prediction;
    result.loss       = sample.loss;
    result.n_post     = input.n_post;
    result.n_skipped  = sample.n_skipped;
    /* The chain factor the errors above were built from, taken from the one routine that
     * computed it rather than recomputed here — a second copy of that expression is a
     * convention free to disagree with the first. */
    result.residual_derivative = sample.residual_derivative;

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) {
        eventprop_free_maxima_regression_output(result);
        result.errors       = NULL;
        result.decoder_grad = NULL;
        result.prediction   = 0.0;
        result.loss         = 0.0;
        result.n_post       = 0;
        result.n_skipped    = 0;
    }
    return result;
}

MaximaRegressionResult eventprop_maxima_regression_into(MaximaOutput maxima,
                                                        const double* decoder,
                                                        double bias,
                                                        double target,
                                                        double huber_delta)
{
    return maxima_regression_sample(maxima.values, decoder, bias, target, huber_delta,
                                    maxima.n_post, maxima.errors, NULL, NULL);
}

EventPropError eventprop_maxima_regression_batch(const MaximaBatch maxima_batch,
                                                 const double* decoder,
                                                 double bias,
                                                 const double* targets,
                                                 double huber_delta,
                                                 double* predictions,
                                                 double* losses,
                                                 int* n_skipped,
                                                 double* decoder_grad_accum,
                                                 double* bias_grad_accum)
{
    if (!maxima_batch.outputs || !decoder || !targets) {
        return EVENTPROP_NULL_POINTER;
    }
    if (maxima_batch.batch_size < 0) {
        return EVENTPROP_INVALID_DIMENSION;
    }
    if (maxima_batch.batch_size == 0) {
        return EVENTPROP_OK;  /* nothing to do; not an error */
    }

    /* The decoder is one array shared by every sample, so every sample must have the same
     * width. Without this check a short batch would read the decoder past its end rather
     * than fail — the same reasoning as EVENTPROP_BATCH_DIMENSION_MISMATCH in the forward
     * batch path. */
    const int n_post = maxima_batch.outputs[0].n_post;
    if (n_post <= 0) {
        return EVENTPROP_INVALID_DIMENSION;
    }
    for (int i = 1; i < maxima_batch.batch_size; ++i) {
        if (maxima_batch.outputs[i].n_post != n_post) {
            return EVENTPROP_BATCH_DIMENSION_MISMATCH;
        }
    }

    double bias_accum_local = 0.0;
    double* bias_target = bias_grad_accum ? bias_grad_accum : &bias_accum_local;

    for (int i = 0; i < maxima_batch.batch_size; ++i) {
        const MaximaOutput* m = &maxima_batch.outputs[i];
        if (!m->errors) {
            return EVENTPROP_NULL_POINTER;
        }
        const MaximaRegressionResult r =
            maxima_regression_sample(m->values, decoder, bias, targets[i], huber_delta,
                                     n_post, m->errors, decoder_grad_accum, bias_target);
        if (r.error_no != EVENTPROP_OK) {
            /* Earlier samples have already written their errors. Leaving them is deliberate:
             * they are correct for their own samples, a caller that ignores the return value
             * gets a gradient of the shape it asked for, and half-written errors are no worse
             * than the garbage the caller would otherwise read. The return value is the
             * contract. */
            return r.error_no;
        }
        if (predictions) { predictions[i] = r.prediction; }
        if (losses)      { losses[i]      = r.loss; }
        if (n_skipped)   { n_skipped[i]   = r.n_skipped; }
    }
    return EVENTPROP_OK;
}
