/**
 * @file eventprop_loss.h
 * @brief Loss functions and continuous readouts built on the library's maxima pass.
 *
 * The core library computes gradients *given* an error signal: `eventprop_backward_maxima`
 * takes `maxima_errors`, and `eventprop_backward_spikes` takes `post_spike_errors`, both
 * documented as caller-owned. What it has never shipped is anything that produces those
 * errors, so each binding has been writing loss maths for itself — and the managed side has
 * written none at all. This header moves that arithmetic behind the C boundary so that
 * Cython and P/Invoke share one implementation of a loss rather than two interpretations of
 * a convention.
 *
 * The readout here is continuous: a linear decoder over the post-synaptic neurons' voltage
 * maxima, `y_hat = bias + Σ_n a[n] · maxima[n]`, trained against a real-valued target. Maxima
 * are the natural continuous quantity the library already produces — they are the peak of
 * the membrane trajectory and involve no threshold, so unlike a spike-count or first-spike
 * readout they exist for every neuron whether or not it fired.
 *
 * There is no counterpart to this in the read-only reference library, so there is no C-vs-
 * Eigen parity test for it. Correctness rests on finite-difference checks of the derivative
 * (src/EventProp.C.Tests/test/maxima_regression_loss.cpp) and on managed-vs-native parity in
 * src/EventProp.Tests, where the managed implementation and the P/Invoke one must agree.
 */

#ifndef EVENTPROP_LOSS_H
#define EVENTPROP_LOSS_H

#include "eventprop.h"
#include "eventprop_batch.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Input for one sample of the maxima regression readout.
 *
 * Deliberately does NOT embed EventPropInput: computing a loss reads no weights, no input
 * spikes and no time constants, and a struct that embedded them would force every caller to
 * keep a forward-pass input alive just to ask for a gradient. The dimension that matters is
 * stated directly and cross-checked against `MaximaOutput.n_post` by the batch entry point.
 *
 * `huber_delta` selects the loss. Any non-finite value (±Inf) means plain least squares;
 * NaN is refused rather than silently treated as one or the other. A finite positive delta
 * gives the smooth (Bessel) Huber loss with scale delta. The two are the same expression
 * evaluated with a different denominator — see `maxima_regression_sample` — so switching
 * between them cannot introduce a second, drifting convention.
 */
typedef struct MaximaRegressionInput {
    const double* maxima_values;       /* Per-neuron voltage maxima, from MaximaOutput.values
                                        * (borrowed, length n_post). A non-finite entry means
                                        * that neuron never peaked; see the skip rule on
                                        * MaximaRegressionOutput.n_skipped. */
    const double* decoder;             /* Readout weights a[n_post] (borrowed). Shared by
                                        * every sample of a batch: it is part of the model,
                                        * not of the sample. */
    double bias;                       /* Readout bias b */
    double target;                     /* Observed value y */
    double huber_delta;                /* Non-finite => least squares; > 0 => smooth Huber */
    int n_post;                        /* Number of post-synaptic neurons (length of the two
                                        * arrays above) */
} MaximaRegressionInput;

/**
 * Output of the allocating single-sample entry point.
 *
 * `errors` is laid out exactly as `eventprop_backward_maxima` wants its `maxima_errors`
 * argument, so the handoff is
 *
 *     MaximaOutput maxima = eventprop_compute_maxima(input);
 *     MaximaRegressionOutput loss = eventprop_maxima_regression(...);
 *     BackwardOutput grad = eventprop_backward_maxima(input, maxima.times, loss.errors);
 *
 * `error_no` is the first field, matching the Output structs in eventprop.h rather than the
 * diagnostic ones, because this struct is part of the training path.
 */
typedef struct MaximaRegressionOutput {
    EventPropError error_no;           /* Error code (EVENTPROP_OK on success) */
    double prediction;                 /* y_hat = bias + Σ a[n] · maxima[n] */
    double loss;                       /* L(y_hat, target) */
    double residual_derivative;        /* dL/dy_hat — the chain factor every error below is
                                        * built from; exposed because a caller that wants a
                                        * mean over a batch divides this, not the errors */
    double* errors;                    /* dL/dmaxima[n] (malloc'd, length n_post) — feed to
                                        * eventprop_backward_maxima */
    double* decoder_grad;              /* dL/da[n] for this sample (malloc'd, length n_post) */
    double bias_grad;                  /* dL/db for this sample */
    int n_post;                        /* Length of the two arrays above */
    int n_skipped;                     /* Neurons with a non-finite maxima value: excluded
                                        * from y_hat, error written as 0. See the skip rule. */
} MaximaRegressionOutput;

/**
 * Output of the in-place and batch entry points. Holds no heap memory, so it has no free
 * function — the errors it computes live in the `MaximaOutput.errors` array that the library
 * already allocated for the caller to write.
 */
typedef struct MaximaRegressionResult {
    EventPropError error_no;           /* Error code (EVENTPROP_OK on success) */
    double prediction;                 /* y_hat for this sample */
    double loss;                       /* L(y_hat, target) */
    double residual_derivative;        /* dL/dy_hat — the single scalar every error in this
                                        * sample was built from. Exposed because a caller
                                        * that wants a mean over a batch scales this, and
                                        * recovering it from errors[] would mean dividing by
                                        * a decoder entry that may legitimately be 0. */
    int n_skipped;                     /* Neurons skipped for a non-finite maxima value */
} MaximaRegressionResult;

/** Free the arrays of a MaximaRegressionOutput. Takes the struct by value, as every free
 *  function in this library does. Safe to call on an output that never allocated. */
void eventprop_free_maxima_regression_output(MaximaRegressionOutput output);

/**
 * Readout and loss for one sample, with freshly allocated `errors` and `decoder_grad`.
 *
 * Gradients are SUMS over the neurons of one sample, not means. Nothing here divides by a
 * batch size: the batch entry point accumulates, and the caller decides whether it wants a
 * summed or a mean gradient. A library that silently averaged half the time is how a
 * learning rate ends up coupled to a batch size.
 */
MaximaRegressionOutput eventprop_maxima_regression(MaximaRegressionInput input);

/**
 * Write this sample's errors straight into `maxima.errors` — the array
 * `eventprop_compute_maxima` allocated for exactly this purpose.
 *
 * Zero allocation, which is what makes it usable inside a training loop: a step that
 * mallocs per sample pays an allocator round trip for ~n_post floats.
 *
 * It computes NO readout gradients — it has no parameters to put them in. A caller that
 * trains `decoder` and `bias` wants `eventprop_maxima_regression_batch`, which sums them
 * across the batch in one call; a caller with a fixed readout wants this, and a caller that
 * needs one sample's error written on its own (its own target, its own loss scale, or a
 * re-run of a single sample) wants this too.
 *
 * `maxima.errors` is overwritten for every neuron of this one sample, including the skipped
 * ones (written 0), so a reused MaximaOutput cannot carry a previous step's error into this
 * one. Other samples are not touched: a batch in which only sample 2 has been through this
 * function has sample 2's errors and everyone else's, which are whatever the caller last
 * wrote there — the forward pass zeroes them, a reused batch does not.
 */
MaximaRegressionResult eventprop_maxima_regression_into(MaximaOutput maxima,
                                                        const double* decoder,
                                                        double bias,
                                                        double target,
                                                        double huber_delta);

/**
 * The same computation over a whole `MaximaBatch`, in place.
 *
 * Writes `outputs[i].errors` for every sample, accumulates the readout gradients across
 * samples, and optionally reports per-sample predictions and losses. Zero allocation.
 *
 * `decoder_grad_accum` and `bias_grad_accum` are ACCUMULATED into, never zeroed — matching
 * the layer contract that Backward() adds to existing gradients and ZeroGrad() is the only
 * thing that clears them. Either may be NULL when the caller does not train the readout.
 *
 * Every sample must have the same `n_post`; a mismatch is EVENTPROP_BATCH_DIMENSION_MISMATCH
 * rather than a decoder array read past its end.
 *
 * A sample whose maxima are all non-finite is NOT an error: it contributes 0 to every
 * gradient and is counted in `n_skipped` if that array is supplied. Silently dropping it
 * would be the failure mode — a training run on a dataset where a third of the samples make
 * the neuron silent is a different experiment from one where none do, and it should look
 * like one.
 *
 * This function does not touch `maxima_batch.has_errors`. That flag describes the forward
 * pass that produced the batch; a programming error in a loss call is reported by the return
 * value, and overloading the forward pass's flag with it would make the batch report a
 * failure it did not have.
 *
 * `predictions`, `losses` and `n_skipped` may each be NULL.
 */
EventPropError eventprop_maxima_regression_batch(const MaximaBatch maxima_batch,
                                                 const double* decoder,
                                                 double bias,
                                                 const double* targets,
                                                 double huber_delta,
                                                 double* predictions,
                                                 double* losses,
                                                 int* n_skipped,
                                                 double* decoder_grad_accum,
                                                 double* bias_grad_accum);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_LOSS_H */
