using CW.EventProp;
using Native = CW.EventProp.Native;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// Managed (<c>CW.EventProp.LILayer</c>) vs native (<c>CW.EventProp.Native.LILayer</c>) parity for
/// the layer-level continuous readout, <c>Regression(...)</c>.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="RegressionLossParityTests"/> holds the two <b>per-sample primitives</b> to each
/// other. That is necessary but not sufficient: it says nothing about whether the derivative each
/// backend writes actually reaches the buffers that backend's <c>Backward()</c> then reads. The two
/// layers obtain their maxima differently for exactly this reason — the native layer cannot cache
/// the C <see cref="MaximaVector"/>-equivalent (a ref struct) and recomputes it inside
/// <c>Backward()</c>, copying the managed <c>MaximaData.Errors</c> into it, while the managed layer
/// reads its own arrays directly. A regression step that wrote into the wrong copy would keep the
/// primitive-level tests green and train nothing.
/// </para>
/// <para>
/// So the assertions below run through <c>Backward()</c> and compare <c>Gradient</c>, which is the
/// only quantity that traverses both the readout and the marshalling on each side.
/// </para>
/// </remarks>
public class RegressionLayerParityTests
{
    /// <summary>Same band as the other parity suites.</summary>
    private const double Tolerance = 1e-9;

    private const int NPre = 12;
    private const int NPost = 5;
    private const double TauMem = 0.02;
    private const double TauSyn = 0.005;

    private static void Check(bool condition, string because)
    {
        if (!condition) throw new InvalidOperationException(because);
    }

    /// <summary>Deterministic input, same generator as the layer parity suite.</summary>
    private static (double[] w, double[] times, int[] sources) MakeInput(int nSpikes, int seed)
    {
        var rng = new Random(seed);
        var w = new double[NPre * NPost];
        for (int i = 0; i < w.Length; i++) w[i] = 0.5 + rng.NextDouble() * 1.5;

        var times = new double[nSpikes];
        var sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = (i + 1) * 1.0 / (nSpikes + 1);
            sources[i] = rng.Next(NPre);
        }
        return (w, times, sources);
    }

    private static LILayerParameters Params() => new(NPost, NPre, TauMem, TauSyn);

    /// <summary>One or two samples, both backends, same numbers.</summary>
    private static (SpikesVector Managed, SpikesVector Native) MakeBatch(
        int samples, int spikesPerSample, int seed)
    {
        var managed = new SpikesVector();
        var native = new SpikesVector();
        for (int s = 0; s < samples; s++)
        {
            var (_, times, sources) = MakeInput(spikesPerSample, seed: seed + s);
            managed.Add(new SpikesData(times, sources));
            native.Add(new SpikesData(times, sources));
        }
        return (managed, native);
    }

    private static double[] Targets(int count, int seed)
    {
        var rng = new Random(seed);
        var t = new double[count];
        for (int i = 0; i < count; i++) t[i] = rng.NextDouble() * 2.0 - 0.5;
        return t;
    }

    private static void AssertStepMatches(MaximaRegressionStep m, MaximaRegressionStep n, string because)
    {
        Check(m.Predictions.Length == n.Predictions.Length, $"{because}: prediction count differs");
        Check(m.Losses.Length == n.Losses.Length, $"{because}: loss count differs");
        for (int i = 0; i < m.Predictions.Length; i++)
        {
            SyncAssert.That(n.Predictions[i]).IsCloseTo(m.Predictions[i], Tolerance);
            SyncAssert.That(n.Losses[i]).IsCloseTo(m.Losses[i], Tolerance);
            SyncAssert.That(n.Skipped[i]).IsEqualTo(m.Skipped[i]);
        }
        Check(n.DecoderGrad.Length == m.DecoderGrad.Length, $"{because}: decoderGrad length differs");
        for (int i = 0; i < m.DecoderGrad.Length; i++)
            SyncAssert.That(n.DecoderGrad[i]).IsCloseTo(m.DecoderGrad[i], Tolerance);
        for (int i = 0; i < m.BiasGrad.Length; i++)
            SyncAssert.That(n.BiasGrad[i]).IsCloseTo(m.BiasGrad[i], Tolerance);
    }

    private static void AssertGradientsMatch(double[] managed, double[] native, string because)
    {
        Check(managed.Length == native.Length, $"{because}: gradient length differs");
        double maxAbs = 0.0;
        for (int i = 0; i < managed.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(managed[i]));
        Check(maxAbs > 1e-12,
              $"{because}: both gradients are identically zero, so comparing them proves nothing");

        for (int i = 0; i < managed.Length; i++)
            SyncAssert.That(native[i]).IsCloseTo(managed[i], Tolerance);
    }

    [Test]
    public void Regression_NativeVsManaged_LeastSquares()
    {
        var (w, _, _) = MakeInput(NPre * NPost, seed: 5);
        var (mIn, nIn) = MakeBatch(samples: 2, spikesPerSample: 40, seed: 3);
        var targets = Targets(2, seed: 8);
        var decoder = new double[NPost];
        for (int i = 0; i < NPost; i++) decoder[i] = 0.4 - 0.2 * i;

        using var managed = new CW.EventProp.LILayer(Params(), w);
        using var native = new Native.LILayer(Params(), w);

        managed.Forward(mIn);
        native.Forward(nIn);

        var mStep = managed.Regression(decoder, 0.15, targets);
        var nStep = native.Regression(decoder, 0.15, targets);
        AssertStepMatches(mStep, nStep, "least squares");
        Check(mStep.Losses[0] > 0.0, "a zero loss would make every comparison below vacuous");

        managed.Backward();
        native.Backward();
        AssertGradientsMatch(managed.Gradient, native.Gradient, "least squares weight gradient");
    }

    [Test]
    public void Regression_NativeVsManaged_SmoothHuber()
    {
        var (w, _, _) = MakeInput(NPre * NPost, seed: 6);
        var (mIn, nIn) = MakeBatch(samples: 2, spikesPerSample: 40, seed: 3);
        var targets = Targets(2, seed: 8);
        var decoder = new double[NPost];
        for (int i = 0; i < NPost; i++) decoder[i] = 0.4 - 0.2 * i;

        using var managed = new CW.EventProp.LILayer(Params(), w);
        using var native = new Native.LILayer(Params(), w);

        managed.Forward(mIn);
        native.Forward(nIn);

        // Both scales matter: a small delta exercises the saturation the smooth form only
        // approaches, a large one has to stay numerically indistinguishable from least squares.
        foreach (double delta in new[] { 1e-3, 0.5, 50.0 })
        {
            var mStep = managed.Regression(decoder, 0.15, targets, delta);
            var nStep = native.Regression(decoder, 0.15, targets, delta);
            AssertStepMatches(mStep, nStep, $"huber delta {delta}");
        }

        managed.Backward();
        native.Backward();
        AssertGradientsMatch(managed.Gradient, native.Gradient, "huber weight gradient");
    }

    [Test]
    public void Regression_NativeVsManaged_AccumulatesIntoCallerBuffers()
    {
        var (w, _, _) = MakeInput(NPre * NPost, seed: 9);
        var (mIn, nIn) = MakeBatch(samples: 2, spikesPerSample: 40, seed: 3);
        var targets = Targets(2, seed: 8);
        var decoder = new double[NPost];
        for (int i = 0; i < NPost; i++) decoder[i] = 0.3 + 0.1 * i;

        using var managed = new CW.EventProp.LILayer(Params(), w);
        using var native = new Native.LILayer(Params(), w);

        managed.Forward(mIn);
        native.Forward(nIn);

        var mGrad = new double[NPost];
        var mBias = new double[1];
        var nGrad = new double[NPost];
        var nBias = new double[1];

        managed.Regression(decoder, 0.1, targets, double.PositiveInfinity, mGrad, mBias);
        native.Regression(decoder, 0.1, targets, double.PositiveInfinity, nGrad, nBias);
        var firstM = (double[])mGrad.Clone();
        var firstN = (double[])nGrad.Clone();
        Check(Math.Abs(firstM[0]) > 0.0 && Math.Abs(firstN[0]) > 0.0,
              "the readout produced no decoder gradient on at least one side");

        var mStep2 = managed.Regression(decoder, 0.1, targets, double.PositiveInfinity, mGrad, mBias);
        var nStep2 = native.Regression(decoder, 0.1, targets, double.PositiveInfinity, nGrad, nBias);

        for (int i = 0; i < NPost; i++)
        {
            Check(Math.Abs(mGrad[i] - 2 * firstM[i]) < 1e-12, $"managed decoderGrad[{i}] did not accumulate");
            Check(Math.Abs(nGrad[i] - 2 * firstN[i]) < 1e-12, $"native decoderGrad[{i}] did not accumulate");
        }
        Check(Math.Abs(mBias[0]) > 0.0 && Math.Abs(nBias[0]) > 0.0, "bias gradient was never written");
        AssertStepMatches(mStep2, nStep2, "accumulated step");
    }

    [Test]
    public async Task Regression_BothBackendsRefuseWithoutForward()
    {
        var (w, _, _) = MakeInput(NPre * NPost, seed: 4);
        var decoder = new double[NPost];
        var targets = Targets(1, seed: 2);

        using var managed = new CW.EventProp.LILayer(Params(), w);
        using var native = new Native.LILayer(Params(), w);

        var mEx = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        { var _ = managed.Regression(decoder, 0.0, targets); await Task.CompletedTask; });
        var nEx = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        { var _ = native.Regression(decoder, 0.0, targets); await Task.CompletedTask; });

        // The refusal has to say what to do about it; an InvalidOperationException that could mean
        // any of the layer's four states is a poor error to hit in a training loop at 2am.
        Check(mEx!.Message.Contains("Forward()"), $"managed refusal must name Forward(): {mEx.Message}");
        Check(nEx!.Message.Contains("Forward()"), $"native refusal must name Forward(): {nEx.Message}");
    }

    [Test]
    public async Task Regression_BothBackendsRefuseAWrongLengthDecoder()
    {
        var (w, _, _) = MakeInput(NPre * NPost, seed: 4);
        var (mIn, nIn) = MakeBatch(samples: 1, spikesPerSample: 30, seed: 3);
        var targets = Targets(1, seed: 2);

        using var managed = new CW.EventProp.LILayer(Params(), w);
        using var native = new Native.LILayer(Params(), w);
        managed.Forward(mIn);
        native.Forward(nIn);

        await Assert.ThrowsAsync<ArgumentException>(async () =>
        { var _ = managed.Regression(new double[NPost + 1], 0.0, targets); await Task.CompletedTask; });
        await Assert.ThrowsAsync<ArgumentException>(async () =>
        { var _ = native.Regression(new double[NPost + 1], 0.0, targets); await Task.CompletedTask; });
        // Targets are checked as "at least one per sample", the same span rule as
        // Loss.RegressBatch, so a longer buffer is legal and the tail is ignored. Asserted rather
        // than assumed: a silent off-by-one in how many entries are consumed would train on the
        // wrong targets and read as a convergence problem, not a bug.
        var longTargets = Targets(1, seed: 2);
        var padded = new double[longTargets.Length + 3];
        Array.Copy(longTargets, padded, longTargets.Length);
        padded[longTargets.Length] = 9.0;

        var mFull = managed.Regression(new double[NPost], 0.0, longTargets);
        var nFull = native.Regression(new double[NPost], 0.0, longTargets);
        var mPad = managed.Regression(new double[NPost], 0.0, padded);
        var nPad = native.Regression(new double[NPost], 0.0, padded);
        AssertStepMatches(mFull, mPad, "managed: trailing targets ignored");
        AssertStepMatches(nFull, nPad, "native: trailing targets ignored");
        AssertStepMatches(mPad, nPad, "padded targets");
    }

    [Test]
    public void Regression_AZeroReadoutLeavesTheWeightsNoGradient()
    {
        // dL/dmaxima = dL/dy_hat · decoder, so decoder = 0 means the weights cannot move. Both
        // backends must agree that nothing happens; a side that produced a gradient here would be
        // reading stale error buffers rather than the ones just written.
        var (w, _, _) = MakeInput(NPre * NPost, seed: 21);
        var (mIn, nIn) = MakeBatch(samples: 2, spikesPerSample: 40, seed: 3);
        var targets = Targets(2, seed: 8);

        using var managed = new CW.EventProp.LILayer(Params(), w);
        using var native = new Native.LILayer(Params(), w);
        managed.Forward(mIn);
        native.Forward(nIn);

        managed.Regression(new double[NPost], 0.3, targets);
        native.Regression(new double[NPost], 0.3, targets);
        managed.Backward();
        native.Backward();

        double maxM = 0.0, maxN = 0.0;
        for (int i = 0; i < w.Length; i++)
        {
            maxM = Math.Max(maxM, Math.Abs(managed.Gradient[i]));
            maxN = Math.Max(maxN, Math.Abs(native.Gradient[i]));
        }
        Check(maxM == 0.0, $"managed gradient must be exactly zero, got {maxM}");
        Check(maxN == 0.0, $"native gradient must be exactly zero, got {maxN}");
    }

    [Test]
    public void Regression_SilentNeuronsAreSkippedIdentically()
    {
        // A read-out neuron driven only by inhibitory weights never peaks, so its maximum is
        // non-finite. Both backends must classify it the same way — including the loss, which is
        // where a disagreeing skip rule would silently change what is being optimised.
        var (w, _, _) = MakeInput(NPre * NPost, seed: 13);
        for (int i = 0; i < NPre; i++) w[i * NPost + (NPost - 1)] = -Math.Abs(w[i * NPost + (NPost - 1)]);

        var (mIn, nIn) = MakeBatch(samples: 1, spikesPerSample: 40, seed: 3);
        var targets = Targets(1, seed: 8);
        var decoder = new double[NPost];
        for (int i = 0; i < NPost; i++) decoder[i] = 0.5;

        using var managed = new CW.EventProp.LILayer(Params(), w);
        using var native = new Native.LILayer(Params(), w);

        managed.Forward(mIn);
        native.Forward(nIn);

        var mStep = managed.Regression(decoder, 0.2, targets);
        var nStep = native.Regression(decoder, 0.2, targets);
        AssertStepMatches(mStep, nStep, "silent neuron");

        managed.Backward();
        native.Backward();
        AssertGradientsMatch(managed.Gradient, native.Gradient, "silent neuron weight gradient");
    }

    [Test]
    public void Regression_SGDStepsKeepBothBackendsInParityAndReduceLoss()
    {
        // Not a benchmark and not a claim about convergence: three steps, checked to go downhill
        // on both backends while the two weight vectors stay within parity of each other. It is
        // the only test here that would notice one backend training on stale errors.
        var (w0, _, _) = MakeInput(NPre * NPost, seed: 21);
        var (mIn, nIn) = MakeBatch(samples: 2, spikesPerSample: 40, seed: 3);
        var targets = Targets(2, seed: 8);
        // Deliberately non-zero: dL/dmaxima = dL/dy_hat · decoder, so an all-zero readout leaves
        // the weights with no gradient at all and only the bias ever trains. Initializing a decoder
        // to zeros is the natural thing to type and a silent way to freeze the layer.
        var decoderM = new double[NPost];
        var decoderN = new double[NPost];
        for (int i = 0; i < NPost; i++) decoderM[i] = decoderN[i] = 0.25 + 0.1 * i;
        double biasM = 0.0, biasN = 0.0;

        using var managed = new CW.EventProp.LILayer(Params(), (double[])w0.Clone());
        using var native = new Native.LILayer(Params(), (double[])w0.Clone());

        const double lrW = 0.02;
        const double lrReadout = 0.02;
        double firstM = 0.0, lastM = 0.0, lastN = 0.0;

        for (int step = 0; step < 3; step++)
        {
            managed.ZeroGrad();
            native.ZeroGrad();
            managed.Forward(mIn);
            native.Forward(nIn);

            var mGrad = new double[NPost];
            var mBias = new double[1];
            var nGrad = new double[NPost];
            var nBias = new double[1];
            var mStep = managed.Regression(decoderM, biasM, targets, double.PositiveInfinity, mGrad, mBias);
            var nStep = native.Regression(decoderN, biasN, targets, double.PositiveInfinity, nGrad, nBias);
            AssertStepMatches(mStep, nStep, $"sgd step {step}");

            managed.Backward();
            native.Backward();
            AssertGradientsMatch(managed.Gradient, native.Gradient, $"sgd step {step} gradient");

            for (int i = 0; i < w0.Length; i++)
            {
                managed.Weights[i] -= lrW * managed.Gradient[i];
                native.Weights[i] -= lrW * native.Gradient[i];
            }
            for (int i = 0; i < NPost; i++)
            {
                decoderM[i] -= lrReadout * mGrad[i];
                decoderN[i] -= lrReadout * nGrad[i];
            }
            biasM -= lrReadout * mBias[0];
            biasN -= lrReadout * nBias[0];

            double lossM = 0.0, lossN = 0.0;
            foreach (double v in mStep.Losses) lossM += v;
            foreach (double v in nStep.Losses) lossN += v;
            if (step == 0) firstM = lossM;
            lastM = lossM;
            lastN = lossN;
        }

        Check(lastM < firstM, $"managed loss did not fall: {firstM} → {lastM}");
        Check(lastN < firstM, $"native loss did not fall: {firstM} → {lastN}");
        SyncAssert.That(lastN).IsCloseTo(lastM, Tolerance);
    }
}
