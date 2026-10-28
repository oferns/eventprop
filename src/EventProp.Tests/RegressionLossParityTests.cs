using CW.EventProp;
using Native = CW.EventProp.Native;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// Managed (<c>EventProp.RegressionLoss</c>) vs native (<c>eventprop_loss.h</c> through
/// <c>EventProp.Native.Loss</c>) parity for the continuous readout and regression losses.
/// </summary>
/// <remarks>
/// <para>
/// This loss exists because a regression target cannot be expressed as a first-spike or
/// maxima ordering problem, so there is no function in the Eigen reference library or the
/// Python reference to compare it against. The parity guarantee that replaces
/// "matches the reference" is <b>managed ↔ native</b>: two independent implementations of the
/// documented formulas, which must agree to the same tolerance every other parity test uses.
/// A disagreement means one of them stopped implementing the header.
/// </para>
/// <para>
/// The cases below are the ones where an independent implementation is most likely to drift,
/// not the easy middle: a silent neuron (skip rule), a residual small enough that the naive
/// Huber spelling rounds to zero, a residual large enough that it cancels, a population with
/// nothing to read out, and the accumulation rule across a batch.
/// </para>
/// </remarks>
public class RegressionLossParityTests
{
    /// <summary>Same band as the other parity suites: fails only if both bounds are exceeded.</summary>
    private const double AbsTolerance = 1e-12;
    private const double RelTolerance = 1e-10;

    private const double TauMem = 0.02;
    private const double TauSyn = 0.005;
    private const double VTh = 1.0;

    /// <summary>SyncAssert carries no message; a parity failure has to say which side moved.</summary>
    private static void Check(bool condition, string because)
    {
        if (!condition) throw new InvalidOperationException(because);
    }

    private static void AssertClose(double actual, double expected, string what)
    {
        double diff = Math.Abs(actual - expected);
        double scale = Math.Max(Math.Abs(actual), Math.Abs(expected));
        Check(diff <= AbsTolerance || diff <= RelTolerance * scale,
              $"{what}: managed {actual} vs native {expected}");
    }

    private static void AssertSequencesClose(ReadOnlySpan<double> managed,
                                             ReadOnlySpan<double> native, string what)
    {
        Check(managed.Length == native.Length, $"{what}: length mismatch");
        for (int i = 0; i < managed.Length; i++)
            AssertClose(managed[i], native[i], $"{what}[{i}]");
    }

    /// <summary>Maxima values from a real forward pass, so the readout sees the distribution
    /// it will actually see (values near v_th, not round numbers).</summary>
    private static double[] NativeMaximaValues(int sample)
    {
        double[] w = [6.0, 5.0, 4.0, 3.5, 2.5, 5.5];
        double[] times = [0.010, 0.018, 0.026, 0.033];
        int[] sources = [0, 1, 2, 0];

        using var output = Native.Api.ComputeMaxima(w, 3, 2, times, sources, TauMem, TauSyn);
        Check(output.ErrorNo == Native.Api.EventPropError.Ok,
              $"forward pass failed for sample {sample}: {output.ErrorNo}");
        return output.Values.ToArray();
    }

    [Test]
    public void ManagedAndNativeAgree_LeastSquares()
    {
        double[] values = NativeMaximaValues(0);
        double[] decoder = [0.7, -1.3];

        double[] mErrors = new double[values.Length], mGrad = new double[values.Length];
        var m = RegressionLoss.MaximaRegression(values, decoder, 0.25, 1.0,
                                                RegressionLoss.LeastSquares, mErrors, mGrad);

        double[] nErrors = new double[values.Length], nGrad = new double[values.Length];
        var n = Native.Loss.Regress(values, decoder, 0.25, 1.0,
                                    Native.Loss.LeastSquares, nErrors, nGrad);

        AssertClose(m.Prediction, n.Prediction, "prediction");
        AssertClose(m.Loss, n.Loss, "loss");
        AssertClose(m.ResidualDerivative, n.ResidualDerivative, "dL/dy_hat");
        AssertClose(m.BiasGrad, n.BiasGrad, "dL/db");
        AssertSequencesClose(mErrors, nErrors, "dL/dmaxima");
        AssertSequencesClose(mGrad, nGrad, "dL/da");
        Check(m.NSkipped == n.NSkipped, $"skip count: managed {m.NSkipped} vs native {n.NSkipped}");
    }

    [Test]
    public void ManagedAndNativeAgree_Huber_SmallAndLargeResiduals()
    {
        double[] values = NativeMaximaValues(1);
        double[] decoder = [0.7, -1.3];
        double bias = 0.25;

        /* The residual, not the target, decides which branch of the loss you are in. Anchoring
         * the targets to the readout's own prediction is how you reach the regime where the
         * textbook Huber spelling, delta^2 * (sqrt(1 + q^2) - 1), is exactly zero in floating
         * point: r = 1e-12 makes q ~ 1e-11 and 1 + q^2 rounds back to 1. */
        double prediction = bias;
        for (int n = 0; n < values.Length; n++) prediction += decoder[n] * values[n];
        double[] offsets = [1e-12, -3e-9, 1e-7, 0.4, -60.0];

        foreach (double offset in offsets)
        {
            double target = prediction - offset;

            foreach (double delta in new[] { 0.05, 1.0 })
            {
                double[] mErrors = new double[values.Length], mGrad = new double[values.Length];
                var m = RegressionLoss.MaximaRegression(values, decoder, bias, target, delta,
                                                        mErrors, mGrad);

                double[] nErrors = new double[values.Length], nGrad = new double[values.Length];
                var n = Native.Loss.Regress(values, decoder, bias, target, delta,
                                            nErrors, nGrad);

                AssertClose(m.Loss, n.Loss, $"loss (offset {offset}, delta {delta})");
                AssertClose(m.ResidualDerivative, n.ResidualDerivative,
                            $"dL/dy_hat (offset {offset}, delta {delta})");
                AssertSequencesClose(mErrors, nErrors, $"errors (offset {offset}, delta {delta})");
                AssertSequencesClose(mGrad, nGrad, $"dL/da (offset {offset}, delta {delta})");
                AssertClose(m.Prediction, n.Prediction, $"prediction (offset {offset})");

                if (Math.Abs(offset) <= 1e-7)
                {
                    /* Quadratic, and non-zero: dL/dy_hat ~ r, loss ~ r^2/2. Zero on either side
                     * means that implementation lost the curvature, and every gradient through
                     * a nearly-correct sample is then silently zeroed. */
                    double expectedLoss = offset * offset / 2.0;
                    Check(m.Loss > 0.0 && n.Loss > 0.0,
                          $"near-zero-residual loss underflowed to zero " +
                          $"(managed {m.Loss}, native {n.Loss}, expected ~{expectedLoss})");
                    AssertClose(m.Loss, expectedLoss, $"managed loss curvature (delta {delta})");
                    AssertClose(n.Loss, expectedLoss, $"native loss curvature (delta {delta})");
                    AssertClose(m.ResidualDerivative, offset,
                                $"managed dL/dy_hat near zero (delta {delta})");
                }
            }
        }
    }

    [Test]
    public void ManagedAndNativeAgree_WhenNeuronsAreSilent()
    {
        double[] decoder = [0.7, -1.3, 0.4, 2.0];
        double[][] cases =
        [
            [double.NaN, 0.9, 1.1, 0.7],
            [0.8, 1.0, double.PositiveInfinity, 1.2],
            [double.NaN, double.NaN, double.NaN, double.NaN],   // nothing to read out
        ];

        foreach (double[] values in cases)
        {
            double[] mErrors = new double[values.Length], mGrad = new double[values.Length];
            var m = RegressionLoss.MaximaRegression(values, decoder, 0.1, 0.6,
                                                    RegressionLoss.LeastSquares, mErrors, mGrad);

            double[] nErrors = new double[values.Length], nGrad = new double[values.Length];
            var n = Native.Loss.Regress(values, decoder, 0.1, 0.6,
                                        Native.Loss.LeastSquares, nErrors, nGrad);

            AssertClose(m.Prediction, n.Prediction, "prediction (silent)");
            AssertClose(m.Loss, n.Loss, "loss (silent)");
            AssertSequencesClose(mErrors, nErrors, "errors (silent)");
            AssertSequencesClose(mGrad, nGrad, "dL/da (silent)");
            Check(m.NSkipped == n.NSkipped,
                  $"skip count (silent): managed {m.NSkipped} vs native {n.NSkipped}");

            /* A silent neuron contributes nothing on either side — the skip rule is not merely
             * counted, it is zeroed, or a stale buffer entry becomes phantom gradient. */
            for (int i = 0; i < values.Length; i++)
            {
                if (!double.IsFinite(values[i]))
                {
                    Check(mErrors[i] == 0.0, $"managed error at silent neuron {i} is nonzero");
                    Check(nErrors[i] == 0.0, $"native error at silent neuron {i} is nonzero");
                    Check(mGrad[i] == 0.0, $"managed dL/da at silent neuron {i} is nonzero");
                    Check(nGrad[i] == 0.0, $"native dL/da at silent neuron {i} is nonzero");
                }
            }
        }
    }

    [Test]
    public void ManagedAndNativeBatchPathsAgree_AndAccumulate()
    {
        double[] w = [6.0, 5.0, 4.0, 3.5, 2.5, 5.5];
        double[][] sampleTimes =
        [
            [0.010, 0.018, 0.026, 0.033],
            [0.012, 0.021, 0.029, 0.040],
        ];
        int[][] sampleSources =
        [
            [0, 1, 2, 0],
            [0, 1, 2, 0],
        ];
        double[] decoder = [0.7, -1.3];
        double[] targets = [0.9, -0.4];

        /* Per-sample maxima, read through the single-sample entry point. The batch paths on
         * both sides are then expected to reproduce these, which makes a batch-vs-single
         * divergence reportable rather than silently averaged in. */
        double[] valuesFlat = new double[4];
        double[] singlePred = new double[2], singleLoss = new double[2];
        for (int i = 0; i < 2; i++)
        {
            double[] flatTimes = [.. sampleTimes[i]];
            int[] flatSources = [.. sampleSources[i]];
            using var one = Native.Api.ComputeMaxima(w, 3, 2, flatTimes, flatSources,
                                                     TauMem, TauSyn);
            Check(one.ErrorNo == Native.Api.EventPropError.Ok,
                  $"single-sample forward failed for sample {i}: {one.ErrorNo}");
            double[] values = one.Values.ToArray();
            Check(values.Length == 2, $"sample {i} produced {values.Length} maxima, expected 2");
            values.CopyTo(valuesFlat.AsSpan(i * 2, 2));

            double[] err = new double[2], grad = new double[2];
            var r = Native.Loss.Regress(values, decoder, 0.25, targets[i],
                                        Native.Loss.LeastSquares, err, grad);
            singlePred[i] = r.Prediction;
            singleLoss[i] = r.Loss;
        }

        double[] flatAllTimes = [.. sampleTimes[0], .. sampleTimes[1]];
        int[] flatAllSources = [.. sampleSources[0], .. sampleSources[1]];
        int[] nPerSample = [4, 4];

        using var batch = Native.Batch.ComputeMaximaBatch(
            w, 3, 2, flatAllTimes, flatAllSources, nPerSample, TauMem, TauSyn);
        Check(batch.ErrorNo == Native.Api.EventPropError.Ok,
              $"batch forward failed: {batch.ErrorNo}");

        double[] nPredictions = new double[2], nLosses = new double[2], nGradA = new double[2];
        int[] nSkipped = new int[2];
        double nGradB = 0.0;
        Native.Loss.RegressBatch(batch, decoder, 0.25, targets, Native.Loss.LeastSquares,
                                 nPredictions, nLosses, nSkipped, nGradA, ref nGradB);

        /* The batch readout must agree with the per-sample readout on the same values — a
         * property of the C batch path, checked here because the managed comparison below
         * would otherwise absorb the difference. */
        AssertSequencesClose(nPredictions, singlePred, "native batch vs native per-sample predictions");
        AssertSequencesClose(nLosses, singleLoss, "native batch vs native per-sample losses");

        double[] mErrors = new double[4], mPred = new double[2], mLoss = new double[2], mGradA = new double[2];
        int[] mSkipped = new int[2];
        double mGradB = 0.0;
        RegressionLoss.MaximaRegressionBatch(valuesFlat, 2, decoder, 0.25, targets,
                                             RegressionLoss.LeastSquares,
                                             mErrors, mPred, mLoss, mSkipped, mGradA, ref mGradB);

        AssertSequencesClose(mPred, nPredictions, "batch predictions");
        AssertSequencesClose(mLoss, nLosses, "batch losses");
        AssertSequencesClose(mGradA, nGradA, "batch dL/da");
        AssertClose(mGradB, nGradB, "batch dL/db");
        Check(mSkipped.SequenceEqual(nSkipped), "batch skip counts differ");

        /* Accumulation, not overwrite — the library's gradient rule, on both sides. A binding
         * that zeroed the buffers would silently turn a summed gradient into the last
         * mini-batch's. */
        double[] beforeA = nGradA.ToArray();
        double beforeB = nGradB;
        Native.Loss.RegressBatch(batch, decoder, 0.25, targets, Native.Loss.LeastSquares,
                                 nPredictions, nLosses, nSkipped, nGradA, ref nGradB);
        AssertSequencesClose(nGradA, new[] { 2 * beforeA[0], 2 * beforeA[1] },
                             "dL/da after a second pass");
        AssertClose(nGradB, 2 * beforeB, "dL/db after a second pass");
    }

    [Test]
    public void BothSidesRejectTheSameBadInputs()
    {
        double[] values = NativeMaximaValues(2);
        double[] decoder = [0.7, -1.3];
        double[] errors = new double[values.Length];
        double[] grad = new double[values.Length];

        /* Non-finite decoder: a corrupted parameter vector, refused even when the matching
         * neuron is silent, so a poisoned weight cannot hide behind a silent one. */
        RejectsBoth("non-finite decoder", () =>
            RegressionLoss.MaximaRegression(values, [double.NaN, -1.3], 0.1, 0.5,
                                            RegressionLoss.LeastSquares, errors, grad),
            () => Native.Loss.Regress(values, [double.NaN, -1.3], 0.1, 0.5,
                                      Native.Loss.LeastSquares, errors, grad));

        RejectsBoth("non-finite target", () =>
            RegressionLoss.MaximaRegression(values, decoder, 0.1, double.NaN,
                                            RegressionLoss.LeastSquares, errors, grad),
            () => Native.Loss.Regress(values, decoder, 0.1, double.NaN,
                                      Native.Loss.LeastSquares, errors, grad));

        RejectsBoth("non-positive huber delta", () =>
            RegressionLoss.MaximaRegression(values, decoder, 0.1, 0.5, 0.0, errors, grad),
            () => Native.Loss.Regress(values, decoder, 0.1, 0.5, 0.0, errors, grad));

        RejectsBoth("NaN huber delta", () =>
            RegressionLoss.MaximaRegression(values, decoder, 0.1, 0.5, double.NaN, errors, grad),
            () => Native.Loss.Regress(values, decoder, 0.1, 0.5, double.NaN, errors, grad));

        RejectsBoth("non-finite bias", () =>
            RegressionLoss.MaximaRegression(values, decoder, double.PositiveInfinity, 0.5,
                                            RegressionLoss.LeastSquares, errors, grad),
            () => Native.Loss.Regress(values, decoder, double.PositiveInfinity, 0.5,
                                      Native.Loss.LeastSquares, errors, grad));

        /* Length mismatches are caught managed-side, before the native call is reached. */
        RejectsBoth("decoder length mismatch", () =>
            RegressionLoss.MaximaRegression(values, [0.7], 0.1, 0.5,
                                            RegressionLoss.LeastSquares, errors, grad),
            () => Native.Loss.Regress(values, [0.7], 0.1, 0.5,
                                      Native.Loss.LeastSquares, errors, grad));
    }

    private static void RejectsBoth(string what, Action managed, Action native)
    {
        bool managedThrew = Throws(managed);
        bool nativeThrew = Throws(native);
        Check(managedThrew, $"{what}: the managed implementation accepted it");
        Check(nativeThrew, $"{what}: the native binding accepted it");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }
}
