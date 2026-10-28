namespace CW.EventProp;

/// <summary>
/// Managed implementation of the continuous readout and regression losses — the C library's
/// <c>eventprop_loss.h</c> expressed in spans, so that <c>EventProp</c> and
/// <c>EventProp.Native</c> can be held to the same numbers by a parity test.
/// </summary>
/// <remarks>
/// <para>
/// This is deliberately a mirror rather than an interpretation. Every policy that the C
/// header documents is reproduced here with the same arithmetic, because the parity tests
/// compare them digit for digit and a difference in either implementation is a bug in one of
/// them:
/// </para>
/// <list type="bullet">
/// <item><description>A non-finite maxima value means that neuron never peaked: excluded from
/// the readout, zero error, counted in <see cref="RegressionResult.NSkipped"/>. Not an
/// error.</description></item>
/// <item><description>Least squares and smooth Huber are one expression,
/// <c>r² / (sqrt(1 + q²) + 1)</c>, with least squares the δ→∞ branch. The naive
/// <c>δ²(sqrt(1+q²) − 1)</c> is exactly zero in floating point for small residuals.</description></item>
/// <item><description>Gradients are sums over one sample; nothing divides by a batch
/// size.</description></item>
/// </list>
/// <para>
/// Note the one asymmetry that is not a divergence: the single-sample call <b>overwrites</b>
/// its output spans, because the C single-sample call returns freshly zeroed arrays that the
/// binding then copies. Accumulation across samples is the batch path's job, mirroring
/// <c>eventprop_maxima_regression_batch</c>.
/// </para>
/// </remarks>
public static class RegressionLoss
{
    /// <summary>Pass as <c>huberDelta</c> for plain least squares, as the C library does with
    /// any non-finite scale. A finite positive value selects the smooth Huber loss.</summary>
    public const double LeastSquares = double.PositiveInfinity;

    /// <summary>One sample's readout, loss and gradients.</summary>
    public readonly record struct RegressionResult(
        double Prediction, double Loss, double ResidualDerivative, double BiasGrad, int NSkipped);

    /// <summary>
    /// Readout, loss and per-neuron gradients for one sample.
    /// </summary>
    /// <param name="maximaValues">Per-neuron voltage maxima. A non-finite entry is a neuron
    /// that never peaked and carries no readout signal.</param>
    /// <param name="decoder">Readout weights a[n]; every entry must be finite, including
    /// those for silent neurons, because a NaN here is a corrupted parameter vector rather
    /// than a property of the datum.</param>
    /// <param name="errors">Overwritten with dL/dmaxima, length ≥ maximaValues.Length.</param>
    /// <param name="decoderGrad">Overwritten with dL/da, length ≥ maximaValues.Length.</param>
    public static RegressionResult MaximaRegression(ReadOnlySpan<double> maximaValues,
                                                    ReadOnlySpan<double> decoder,
                                                    double bias,
                                                    double target,
                                                    double huberDelta,
                                                    Span<double> errors,
                                                    Span<double> decoderGrad)
    {
        if (maximaValues.IsEmpty)
            throw new ArgumentException("Maxima values must not be empty", nameof(maximaValues));
        if (decoder.Length != maximaValues.Length)
            throw new ArgumentException(
                $"decoder has {decoder.Length} entries, maxima values has {maximaValues.Length}",
                nameof(decoder));
        if (errors.Length < maximaValues.Length)
            throw new ArgumentException(
                $"errors must hold {maximaValues.Length} entries, got {errors.Length}", nameof(errors));
        if (decoderGrad.Length < maximaValues.Length)
            throw new ArgumentException(
                $"decoderGrad must hold {maximaValues.Length} entries, got {decoderGrad.Length}",
                nameof(decoderGrad));
        if (double.IsNaN(target) || double.IsInfinity(target))
            throw new ArgumentException("Regression target is NaN or infinite", nameof(target));
        if (double.IsNaN(huberDelta) || huberDelta <= 0.0)
            throw new ArgumentException(
                "huberDelta must be positive, or non-finite for least squares", nameof(huberDelta));
        if (double.IsNaN(bias) || double.IsInfinity(bias))
            throw new ArgumentException("Readout bias is NaN or infinite", nameof(bias));

        bool leastSquares = !double.IsFinite(huberDelta);

        double prediction = bias;
        int skipped = 0;
        for (int n = 0; n < maximaValues.Length; n++)
        {
            if (!double.IsFinite(decoder[n]))
                throw new ArgumentException(
                    $"Readout weight a[{n}] is NaN or infinite", nameof(decoder));
            if (!double.IsFinite(maximaValues[n]))
            {
                skipped++;
                continue;
            }
            prediction += decoder[n] * maximaValues[n];
        }
        if (!double.IsFinite(prediction))
            throw new ArgumentException(
                "Readout overflowed to a non-finite value (weights, bias or maxima too large)",
                nameof(decoder));

        double residual = prediction - target;
        /* double.Hypot is the overflow-safe spelling of sqrt(1 + q²), matching C's hypot. */
        double denominator = leastSquares
            ? 1.0
            : double.Hypot(1.0, residual / huberDelta);
        double loss = residual * residual / (denominator + 1.0);
        double dLossDPrediction = residual / denominator;

        for (int n = 0; n < maximaValues.Length; n++)
        {
            if (!double.IsFinite(maximaValues[n]))
            {
                errors[n] = 0.0;
                decoderGrad[n] = 0.0;
                continue;
            }
            errors[n] = dLossDPrediction * decoder[n];
            decoderGrad[n] = dLossDPrediction * maximaValues[n];
        }

        return new RegressionResult(prediction, loss, dLossDPrediction, dLossDPrediction, skipped);
    }

    /// <summary>
    /// The batch form: per-sample readouts with the readout gradients SUMMED across samples
    /// into caller-owned buffers, which are accumulated into rather than overwritten.
    /// </summary>
    /// <remarks>
    /// Per-sample errors go into <paramref name="errorsBySample"/>, laid out sample-major with
    /// <paramref name="nPost"/> entries per sample — the shape the backward pass consumes.
    /// </remarks>
    public static void MaximaRegressionBatch(ReadOnlySpan<double> maximaValuesBySample,
                                             int nPost,
                                             ReadOnlySpan<double> decoder,
                                             double bias,
                                             ReadOnlySpan<double> targets,
                                             double huberDelta,
                                             Span<double> errorsBySample,
                                             Span<double> predictions,
                                             Span<double> losses,
                                             Span<int> nSkipped,
                                             Span<double> decoderGradAccum,
                                             ref double biasGradAccum)
    {
        if (nPost <= 0)
            throw new ArgumentException("n_post must be positive", nameof(nPost));
        int batch = targets.Length;
        if (maximaValuesBySample.Length < batch * nPost)
            throw new ArgumentException(
                $"maxima values hold {(double)maximaValuesBySample.Length / nPost} samples, " +
                $"the batch has {batch}", nameof(maximaValuesBySample));
        if (decoder.Length != nPost)
            throw new ArgumentException(
                $"decoder has {decoder.Length} entries, the layer has {nPost}", nameof(decoder));
        if (errorsBySample.Length < batch * nPost)
            throw new ArgumentException("errorsBySample is too short", nameof(errorsBySample));
        if (decoderGradAccum.Length < nPost)
            throw new ArgumentException("decoderGradAccum is too short", nameof(decoderGradAccum));

        Span<double> gradScratch = new double[nPost];
        for (int i = 0; i < batch; i++)
        {
            var values = maximaValuesBySample.Slice(i * nPost, nPost);
            var errors = errorsBySample.Slice(i * nPost, nPost);
            var result = MaximaRegression(values, decoder, bias, targets[i], huberDelta,
                                          errors, gradScratch);
            if (predictions.Length > i) predictions[i] = result.Prediction;
            if (losses.Length > i) losses[i] = result.Loss;
            if (nSkipped.Length > i) nSkipped[i] = result.NSkipped;
            for (int n = 0; n < nPost; n++) decoderGradAccum[n] += gradScratch[n];
            biasGradAccum += result.BiasGrad;
        }
    }
}
