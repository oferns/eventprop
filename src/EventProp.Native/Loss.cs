using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>
/// P/Invoke bindings for eventprop_loss.h — the continuous readout over voltage maxima and
/// the regression losses built on it.
/// </summary>
/// <remarks>
/// The point of this being in the library rather than in each binding is that P/Invoke and
/// Cython compute the same gradient from the same code. The readout is
/// <c>y_hat = bias + Σ a[n] · maxima[n]</c>; the array this writes into <c>errors</c> is
/// <c>dL/dmaxima</c>, which is exactly what <c>eventprop_backward_maxima</c> takes, so a
/// regression step is ComputeMaximaBatch → RegressInto → BackwardMaximaBatch with no
/// loss arithmetic on the managed side at all.
///
/// Gradients are sums and ACCUMULATE into caller-owned buffers; nothing here divides by a
/// batch size or zeroes a buffer, matching both the C contract and <c>Layer.Backward()</c>.
/// </remarks>
public static unsafe partial class Loss
{
    private const string Lib = "eventprop_c_api";

    /// <summary>Least-squares scale. Pass this (or any non-finite value) for plain squared error;
    /// a finite positive value selects the smooth Huber loss with that scale.</summary>
    public const double LeastSquares = double.PositiveInfinity;

    // ── Native structs (match C layout exactly; sizes are checked at AbiCheck) ──

    /// <summary>Readout and loss inputs for one sample. Plain data, no ownership.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MaximaRegressionInput
    {
        public IntPtr MaximaValues;   // const double*
        public IntPtr Decoder;        // const double*
        public double Bias;
        public double Target;
        public double HuberDelta;
        public int NPost;
    }

    /// <summary>Allocating single-sample result. The two pointers are C-owned; free with
    /// <see cref="eventprop_free_maxima_regression_output"/>.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MaximaRegressionOutput
    {
        public Api.EventPropError ErrorNo;
        public double Prediction;
        public double Loss;
        public double ResidualDerivative;
        public IntPtr Errors;         // double* (malloc'd, nPost)
        public IntPtr DecoderGrad;    // double* (malloc'd, nPost)
        public double BiasGrad;
        public int NPost;
        public int NSkipped;

        /// <summary>Zero-copy span over dL/dmaxima, valid until this output is freed.</summary>
        public readonly Span<double> ErrorsSpan =>
            new((void*)Errors, Math.Max(0, NPost));

        /// <summary>Zero-copy span over dL/da, valid until this output is freed.</summary>
        public readonly Span<double> DecoderGradSpan =>
            new((void*)DecoderGrad, Math.Max(0, NPost));
    }

    /// <summary>Result of the in-place and batch paths. Holds no memory, so there is nothing to free.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MaximaRegressionResult
    {
        public Api.EventPropError ErrorNo;
        public double Prediction;
        public double Loss;
        public double ResidualDerivative;
        public int NSkipped;
    }

    // ── Raw imports ────────────────────────────────────────────────

    [LibraryImport(Lib)]
    public static partial MaximaRegressionOutput eventprop_maxima_regression(MaximaRegressionInput input);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_maxima_regression_output(MaximaRegressionOutput output);

    [LibraryImport(Lib)]
    public static partial MaximaRegressionResult eventprop_maxima_regression_into(
        Batch.MaximaEntry maxima, double* decoder, double bias, double target, double huberDelta);

    [LibraryImport(Lib)]
    public static partial Api.EventPropError eventprop_maxima_regression_batch(
        Batch.MaximaBatch maximaBatch, double* decoder, double bias, double* targets,
        double huberDelta, double* predictions, double* losses, int* nSkipped,
        double* decoderGradAccum, double* biasGradAccum);

    // ── Friendly surface ───────────────────────────────────────────

    /// <summary>One sample's readout and loss.</summary>
    /// <param name="Prediction">y_hat = bias + Σ a[n] · maxima[n].</param>
    /// <param name="Loss">The loss value.</param>
    /// <param name="ResidualDerivative">dL/dy_hat — the single scalar every gradient below is built from.</param>
    /// <param name="BiasGrad">dL/db for this sample.</param>
    /// <param name="NSkipped">Neurons whose maxima were non-finite: no peak, so no signal.</param>
    public readonly record struct RegressionResult(
        double Prediction, double Loss, double ResidualDerivative, double BiasGrad, int NSkipped);

    /// <summary>
    /// Readout and loss for one sample, writing dL/dmaxima into <paramref name="errors"/> and
    /// dL/da into <paramref name="decoderGrad"/>.
    /// </summary>
    /// <remarks>
    /// Buffer lengths are checked before the call: passing a short <c>errors</c> span would
    /// otherwise leave the library's own array as the only place the gradient went, and the
    /// caller would read an untouched buffer rather than an error.
    /// </remarks>
    public static RegressionResult Regress(ReadOnlySpan<double> maximaValues,
                                           ReadOnlySpan<double> decoder,
                                           double bias,
                                           double target,
                                           double huberDelta,
                                           Span<double> errors,
                                           Span<double> decoderGrad)
    {
        Validate(maximaValues, decoder);
        if (errors.Length < maximaValues.Length)
            throw new ArgumentException(
                $"errors must hold {maximaValues.Length} entries, got {errors.Length}", nameof(errors));
        if (decoderGrad.Length < maximaValues.Length)
            throw new ArgumentException(
                $"decoderGrad must hold {maximaValues.Length} entries, got {decoderGrad.Length}",
                nameof(decoderGrad));

        double biasGrad = 0.0;
        int nSkipped = 0;
        double prediction = 0.0, loss = 0.0, dLossDPrediction = 0.0;

        fixed (double* pValues = maximaValues)
        fixed (double* pDecoder = decoder)
        {
            var input = new MaximaRegressionInput
            {
                MaximaValues = (IntPtr)pValues,
                Decoder = (IntPtr)pDecoder,
                Bias = bias,
                Target = target,
                HuberDelta = huberDelta,
                NPost = maximaValues.Length,
            };

            MaximaRegressionOutput output = eventprop_maxima_regression(input);
            try
            {
                if (output.ErrorNo != Api.EventPropError.Ok)
                    throw new InvalidOperationException(
                        $"Regression readout failed: {Api.GetErrorString(output.ErrorNo)}");

                output.ErrorsSpan.CopyTo(errors);
                output.DecoderGradSpan.CopyTo(decoderGrad);
                prediction = output.Prediction;
                loss = output.Loss;
                dLossDPrediction = output.ResidualDerivative;
                biasGrad = output.BiasGrad;
                nSkipped = output.NSkipped;
            }
            finally
            {
                eventprop_free_maxima_regression_output(output);
            }
        }

        return new RegressionResult(prediction, loss, dLossDPrediction, biasGrad, nSkipped);
    }

    /// <summary>
    /// Write one sample's errors straight into a maxima entry's own <c>errors</c> buffer —
    /// the array the library allocated for exactly this — with no managed allocation at all.
    /// </summary>
    /// <remarks>
    /// This is what a training loop calls between <c>ComputeMaximaBatch</c> and
    /// <c>BackwardMaximaBatch</c>. The readout gradients are not computed here: use
    /// <see cref="RegressBatch"/> when the decoder is being trained too.
    /// </remarks>
    public static RegressionResult RegressInto(Batch.MaximaEntry maxima,
                                               ReadOnlySpan<double> decoder,
                                               double bias,
                                               double target,
                                               double huberDelta)
    {
        if (decoder.IsEmpty)
            throw new ArgumentException("Decoder must not be empty", nameof(decoder));

        fixed (double* pDecoder = decoder)
        {
            MaximaRegressionResult result = eventprop_maxima_regression_into(
                maxima, pDecoder, bias, target, huberDelta);
            if (result.ErrorNo != Api.EventPropError.Ok)
                throw new InvalidOperationException(
                    $"Regression readout failed: {Api.GetErrorString(result.ErrorNo)}");

            return new RegressionResult(result.Prediction, result.Loss,
                                        result.ResidualDerivative, 0.0, result.NSkipped);
        }
    }

    /// <summary>
    /// The whole batch: write each sample's errors into its maxima buffer, and accumulate the
    /// readout gradients across samples into caller-owned buffers.
    /// </summary>
    /// <remarks>
    /// <paramref name="decoderGradAccum"/> and <paramref name="biasGradAccum"/> are added to,
    /// never overwritten — <c>Backward()</c> accumulates in this library, and
    /// <c>ZeroGrad()</c> is the only thing that clears. Pass spans of length nPost and 1.
    /// Predictions, losses and skip counts are optional (empty spans).
    /// </remarks>
    public static void RegressBatch(Batch.MaximaBatch maximaBatch,
                                    ReadOnlySpan<double> decoder,
                                    double bias,
                                    ReadOnlySpan<double> targets,
                                    double huberDelta,
                                    Span<double> predictions,
                                    Span<double> losses,
                                    Span<int> nSkipped,
                                    Span<double> decoderGradAccum,
                                    ref double biasGradAccum)
    {
        if (decoder.IsEmpty)
            throw new ArgumentException("Decoder must not be empty", nameof(decoder));
        if (targets.Length < maximaBatch.BatchSize)
            throw new ArgumentException(
                $"targets must hold {maximaBatch.BatchSize} entries, got {targets.Length}",
                nameof(targets));
        if (decoderGradAccum.Length < decoder.Length)
            throw new ArgumentException(
                $"decoderGradAccum must hold {decoder.Length} entries, got {decoderGradAccum.Length}",
                nameof(decoderGradAccum));

        /* The bias gradient accumulator lives on the stack: the alternative is a one-element
         * array per call, and this sits in the training loop. */
        double* pGradB = stackalloc double[1];
        pGradB[0] = biasGradAccum;

        fixed (double* pDecoder = decoder)
        fixed (double* pTargets = targets)
        fixed (double* pPredictions = predictions)
        fixed (double* pLosses = losses)
        fixed (int* pSkipped = nSkipped)
        fixed (double* pGradA = decoderGradAccum)
        {
            Api.EventPropError error = eventprop_maxima_regression_batch(
                maximaBatch, pDecoder, bias, pTargets, huberDelta,
                pPredictions, pLosses, pSkipped, pGradA, pGradB);
            if (error != Api.EventPropError.Ok)
                throw new InvalidOperationException(
                    $"Regression readout failed for the batch: {Api.GetErrorString(error)}");
        }

        biasGradAccum = pGradB[0];
    }

    private static void Validate(ReadOnlySpan<double> maximaValues, ReadOnlySpan<double> decoder)
    {
        if (maximaValues.IsEmpty)
            throw new ArgumentException("Maxima values must not be empty", nameof(maximaValues));
        if (decoder.Length != maximaValues.Length)
            throw new ArgumentException(
                $"decoder has {decoder.Length} entries, maxima values has {maximaValues.Length}",
                nameof(decoder));
    }
}
