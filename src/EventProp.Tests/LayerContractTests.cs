using CW.EventProp;
using Native = CW.EventProp.Native;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// The <see cref="Layer"/> contract, exercised against every backend.
///
/// Step 1 made the managed and native (P/Invoke) layers implement the same abstract
/// <see cref="Layer"/> base. This suite runs one scenario through both, so any future
/// backend has to satisfy the same contract, and so the two implementations cannot drift.
///
/// Each test is parameterized over <see cref="Backends"/>. Backend-specific plumbing
/// (writing loss derivatives, reading maxima) is hidden behind <see cref="LayerBackend"/>.
/// </summary>
public class LayerContractTests
{
    private const double Tolerance = 1e-9;

    // ── Backend descriptor ─────────────────────────────────────────────

    /// <summary>
    /// One implementation of the layer contract: a factory for the layer plus the
    /// backend-specific ways to read output and write loss derivatives.
    /// </summary>
    public sealed record LayerBackend(
        string Name,
        Func<LIFLayerParameters, double[], Layer> CreateLif,
        Func<LILayerParameters, double[], Layer> CreateLi,
        Action<Layer, int, int, double> SetSpikeError,
        Func<Layer, MaximaVector?> GetMaxima)
    {
        public override string ToString() => Name;
    }

    /// <summary>Managed backend — errors live on the output SpikesData / MaximaData.</summary>
    public static readonly LayerBackend Managed = new(
        "Managed",
        (p, w) => new CW.EventProp.LIFLayer(p, w),
        (p, w) => new CW.EventProp.LILayer(p, w),
        (layer, b, i, v) => ((CW.EventProp.LIFLayer)layer).Output[b].SetError(i, v),
        layer => ((CW.EventProp.LILayer)layer).MaximaBatch);

    /// <summary>Native backend — the layer exposes SetError / MaximaBatch directly.</summary>
    public static readonly LayerBackend NativeBackend = new(
        "Native",
        (p, w) => new Native.LIFLayer(p, w),
        (p, w) => new Native.LILayer(p, w),
        (layer, b, i, v) => ((Native.LIFLayer)layer).SetError(b, i, v),
        layer => ((Native.LILayer)layer).MaximaBatch);

    public static IEnumerable<LayerBackend> AllBackends() => [Managed, NativeBackend];

    // ── Shared fixtures ────────────────────────────────────────────────

    // Weights strong enough that clustered input drives neurons past v_th = 1.0.
    // The benchmark suites use 0.5..2.0 (realistic, and 0 spikes is fine for timing);
    // contract tests need the layer to actually fire, so the gradient is non-zero and
    // the parity/determinism checks are not vacuous.
    private static double[] MakeWeights(int nIn, int n, int seed)
    {
        var rng = new Random(seed);
        var w = new double[nIn * n];
        for (int i = 0; i < w.Length; i++) w[i] = 8.0 + rng.NextDouble() * 7.0;
        return w;
    }

    // Spike times clustered in a 30 ms window (a few tau_mem) so they summate at the
    // membrane; spread over [0,1) they decay between arrivals and nothing fires.
    private static SpikesVector MakeInput(int nSpikes, int nPre, int seed)
    {
        var rng = new Random(seed);
        var times = new double[nSpikes];
        var sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = 0.01 + (i + 1) * 0.03 / (nSpikes + 1);
            sources[i] = rng.Next(nPre);
        }
        return new SpikesVector([new SpikesData(times, sources)]);
    }

    private static SpikesVector MakeBatch(int batch, int nSpikes, int nPre, int seed)
    {
        var vector = new SpikesVector(batch);
        for (int b = 0; b < batch; b++)
            vector.Add(MakeInput(nSpikes, nPre, seed + b)[0]);
        return vector;
    }

    // ── Contract: shape and determinism ────────────────────────────────

    [Test]
    public void Gradient_HasExpectedShape()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 12, n = 6;
                var w = MakeWeights(nIn, n, seed: 1);
                using var layer = backend.CreateLif(new LIFLayerParameters(n, nIn), w);

                SyncAssert.That(layer.NIn).IsEqualTo(nIn);
                SyncAssert.That(layer.NOut).IsEqualTo(n);
                SyncAssert.That(layer.Gradient.Length).IsEqualTo(nIn * n);
        }
    }

    [Test]
    public void Forward_IsDeterministic()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 15, n = 7, nSpikes = 50;
                var w = MakeWeights(nIn, n, seed: 2);
                var input = MakeInput(nSpikes, nIn, seed: 3);

                using var a = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                using var b = backend.CreateLif(new LIFLayerParameters(n, nIn), w);

                a.Forward(input);
                b.Forward(input);

                SyncAssert.That(b.Output[0].NSpikes).IsEqualTo(a.Output[0].NSpikes);
                for (int i = 0; i < a.Output[0].NSpikes; i++)
                {
                    SyncAssert.That(b.Output[0].Times[i]).IsCloseTo(a.Output[0].Times[i], Tolerance);
                    SyncAssert.That(b.Output[0].Sources[i]).IsEqualTo(a.Output[0].Sources[i]);
                }
        }
    }

    [Test]
    public void Backward_IsDeterministic()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 15, n = 7, nSpikes = 50;
                var w = MakeWeights(nIn, n, seed: 4);
                var input = MakeInput(nSpikes, nIn, seed: 5);

                using var a = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                using var b = backend.CreateLif(new LIFLayerParameters(n, nIn), w);

                foreach (var layer in new[] { a, b })
                {
                    layer.Forward(input);
                    for (int i = 0; i < layer.Output[0].NSpikes; i++)
                        backend.SetSpikeError(layer, 0, i, 1.0);
                    layer.Backward();
                }

                for (int i = 0; i < w.Length; i++)
                    SyncAssert.That(b.Gradient[i]).IsCloseTo(a.Gradient[i], Tolerance);
        }
    }

    [Test]
    public void ZeroGrad_ClearsGradient()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 12, n = 6, nSpikes = 40;
                var w = MakeWeights(nIn, n, seed: 6);
                var input = MakeInput(nSpikes, nIn, seed: 7);

                using var layer = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                layer.Forward(input);
                for (int i = 0; i < layer.Output[0].NSpikes; i++)
                    backend.SetSpikeError(layer, 0, i, 1.0);
                layer.Backward();

                layer.ZeroGrad();
                foreach (double g in layer.Gradient)
                    SyncAssert.That(g).IsEqualTo(0.0);
        }
    }

    [Test]
    public void Backward_AccumulatesAcrossCalls()
    {
        foreach (var backend in AllBackends())
        {
                // Backward accumulates into Gradient (Python semantics: the training loop
                // calls zero_grad() separately). Running it twice on the same input must
                // double the gradient, not overwrite it.
                // Strong weights + dense clustered input so the layer actually fires —
                // a silent layer has an all-zero gradient and the doubling check is vacuous.
                const int nIn = 3, n = 5;
                var w = new double[nIn * n];
                for (int i = 0; i < w.Length; i++) w[i] = 15.0;
                var times = new double[] { 0.01, 0.02, 0.03, 0.04, 0.05 };
                var sources = new int[] { 0, 1, 2, 0, 1 };
                var input = new SpikesVector([new SpikesData(times, sources)]);

                using var layer = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                layer.Forward(input);
                SyncAssert.That(layer.Output[0].NSpikes > 0).IsTrue();
                for (int i = 0; i < layer.Output[0].NSpikes; i++)
                    backend.SetSpikeError(layer, 0, i, 1.0);

                layer.ZeroGrad();
                layer.Backward();
                var first = layer.Gradient.ToArray();
                layer.Backward();

                for (int i = 0; i < first.Length; i++)
                    SyncAssert.That(layer.Gradient[i]).IsCloseTo(2.0 * first[i], Tolerance);
                // Guard against a vacuous all-zero gradient making the doubling check trivial.
                SyncAssert.That(first.Any(g => g != 0.0)).IsTrue();
        }
    }

    // ── Contract: weights ──────────────────────────────────────────────

    [Test]
    public void Weights_RoundTripThroughSpans()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 3, n = 2;
                double[] known = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                using var layer = backend.CreateLif(new LIFLayerParameters(n, nIn), known);

                double[] readBack = layer.GetWeights().ToArray();
                SyncAssert.That(readBack.Length).IsEqualTo(known.Length);
                for (int i = 0; i < known.Length; i++)
                    SyncAssert.That(readBack[i]).IsEqualTo(known[i]);

                double[] replacement = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
                layer.SetWeights(replacement);
                double[] readBack2 = layer.GetWeights().ToArray();
                for (int i = 0; i < replacement.Length; i++)
                    SyncAssert.That(readBack2[i]).IsEqualTo(replacement[i]);
        }
    }

    [Test]
    public void Constructor_RejectsWrongWeightLength()
    {
        foreach (var backend in AllBackends())
        {
                var parameters = new LIFLayerParameters(N: 3, NIn: 2);
                var badWeights = new double[] { 1.0, 2.0, 3.0 };

                bool threw = false;
                try { backend.CreateLif(parameters, badWeights).Dispose(); }
                catch (ArgumentException) { threw = true; }
                SyncAssert.That(threw).IsTrue();
        }
    }

    [Test]
    public void SetWeights_RejectsWrongLength()
    {
        foreach (var backend in AllBackends())
        {
                using var layer = backend.CreateLif(new LIFLayerParameters(N: 3, NIn: 2), new double[6]);
                bool threw = false;
                try { layer.SetWeights(new double[] { 1.0, 2.0 }); }
                catch (ArgumentException) { threw = true; }
                SyncAssert.That(threw).IsTrue();
        }
    }

    // ── Contract: lifecycle errors ─────────────────────────────────────

    [Test]
    public void Backward_BeforeForward_Throws()
    {
        foreach (var backend in AllBackends())
        {
                using var layer = backend.CreateLif(new LIFLayerParameters(N: 5, NIn: 3), MakeWeights(3, 5, 10));
                bool threw = false;
                try { layer.Backward(); }
                catch (InvalidOperationException) { threw = true; }
                SyncAssert.That(threw).IsTrue();
        }
    }

    [Test]
    public void Output_BeforeForward_Throws()
    {
        foreach (var backend in AllBackends())
        {
                using var layer = backend.CreateLif(new LIFLayerParameters(N: 5, NIn: 3), MakeWeights(3, 5, 11));
                bool threw = false;
                try { _ = layer.Output; }
                catch (InvalidOperationException) { threw = true; }
                SyncAssert.That(threw).IsTrue();
        }
    }

    [Test]
    public void Forward_AfterForward_ReplacesOutput()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 10, n = 5;
                var w = MakeWeights(nIn, n, seed: 12);
                using var layer = backend.CreateLif(new LIFLayerParameters(n, nIn), w);

                layer.Forward(MakeInput(40, nIn, seed: 13));
                int firstCount = layer.Output[0].NSpikes;

                // A batch of 3 samples must replace the single-sample output.
                layer.Forward(MakeBatch(3, 40, nIn, seed: 13));
                SyncAssert.That(layer.Output.Count).IsEqualTo(3);
                SyncAssert.That(firstCount >= 0).IsTrue();
        }
    }

    // ── Contract: batch semantics ──────────────────────────────────────

    [Test]
    public void Forward_Batch_MatchesPerSample()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 14, n = 6, nSpikes = 40, batch = 3;
                var w = MakeWeights(nIn, n, seed: 14);

                using var batchLayer = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                batchLayer.Forward(MakeBatch(batch, nSpikes, nIn, seed: 15));
                SyncAssert.That(batchLayer.Output.Count).IsEqualTo(batch);

                for (int b = 0; b < batch; b++)
                {
                    using var single = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                    single.Forward(MakeInput(nSpikes, nIn, seed: 15 + b));

                    int nOut = single.Output[0].NSpikes;
                    SyncAssert.That(batchLayer.Output[b].NSpikes).IsEqualTo(nOut);
                    for (int i = 0; i < nOut; i++)
                    {
                        SyncAssert.That(batchLayer.Output[b].Times[i]).IsCloseTo(single.Output[0].Times[i], Tolerance);
                        SyncAssert.That(batchLayer.Output[b].Sources[i]).IsEqualTo(single.Output[0].Sources[i]);
                    }
                }
        }
    }

    [Test]
    public void Backward_BatchGradient_IsSumOfPerSampleGradients()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 12, n = 6, nSpikes = 35, batch = 3;
                var w = MakeWeights(nIn, n, seed: 16);

                using var batchLayer = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                var batchInput = MakeBatch(batch, nSpikes, nIn, seed: 17);
                batchLayer.Forward(batchInput);
                for (int b = 0; b < batch; b++)
                    for (int i = 0; i < batchLayer.Output[b].NSpikes; i++)
                        backend.SetSpikeError(batchLayer, b, i, 1.0);
                batchLayer.Backward();

                var summed = new double[w.Length];
                for (int b = 0; b < batch; b++)
                {
                    using var single = backend.CreateLif(new LIFLayerParameters(n, nIn), w);
                    single.Forward(MakeInput(nSpikes, nIn, seed: 17 + b));
                    for (int i = 0; i < single.Output[0].NSpikes; i++)
                        backend.SetSpikeError(single, 0, i, 1.0);
                    single.Backward();
                    for (int i = 0; i < w.Length; i++) summed[i] += single.Gradient[i];
                }

                for (int i = 0; i < w.Length; i++)
                    SyncAssert.That(batchLayer.Gradient[i]).IsCloseTo(summed[i], Tolerance);
        }
    }

    // ── Contract: LI (maxima loss) ─────────────────────────────────────

    [Test]
    public void Li_Forward_ExposesMaximaPerNeuron()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 12, n = 6, nSpikes = 45;
                var w = MakeWeights(nIn, n, seed: 18);
                using var layer = backend.CreateLi(new LILayerParameters(n, nIn), w);

                layer.Forward(MakeInput(nSpikes, nIn, seed: 19));
                var maxima = backend.GetMaxima(layer)!;
                SyncAssert.That(maxima.Count).IsEqualTo(1);
                SyncAssert.That(maxima[0].NNeurons).IsEqualTo(n);
        }
    }

    [Test]
    public void Li_Backward_IsDeterministic()
    {
        foreach (var backend in AllBackends())
        {
                const int nIn = 12, n = 6, nSpikes = 45;
                var w = MakeWeights(nIn, n, seed: 20);

                using var a = backend.CreateLi(new LILayerParameters(n, nIn), w);
                using var b = backend.CreateLi(new LILayerParameters(n, nIn), w);
                var input = MakeInput(nSpikes, nIn, seed: 21);

                foreach (var layer in new[] { a, b })
                {
                    layer.Forward(input);
                    var maxima = backend.GetMaxima(layer)!;
                    for (int i = 0; i < maxima[0].NNeurons; i++) maxima[0].SetError(i, 1.0);
                    layer.Backward();
                }

                for (int i = 0; i < w.Length; i++)
                    SyncAssert.That(b.Gradient[i]).IsCloseTo(a.Gradient[i], Tolerance);
        }
    }

    // ── Cross-backend parity: the contract must be numerically equivalent ──

    [Test]
    public void Lif_Gradient_MatchesBetweenBackends()
    {
        const int nIn = 18, n = 8, nSpikes = 55;
        var w = MakeWeights(nIn, n, seed: 22);
        var input = MakeInput(nSpikes, nIn, seed: 23);

        var (gradient, spikes) = RunLif(Managed, w, input, n);
        var (nativeGradient, nativeSpikes) = RunLif(NativeBackend, w, input, n);

        SyncAssert.That(spikes > 0).IsTrue();
        SyncAssert.That(gradient.Any(g => g != 0.0)).IsTrue();
        SyncAssert.That(nativeSpikes).IsEqualTo(spikes);
        for (int i = 0; i < w.Length; i++)
            SyncAssert.That(nativeGradient[i]).IsCloseTo(gradient[i], Tolerance);
    }

    [Test]
    public void Li_Gradient_MatchesBetweenBackends()
    {
        const int nIn = 18, n = 8, nSpikes = 55;
        var w = MakeWeights(nIn, n, seed: 24);
        var input = MakeInput(nSpikes, nIn, seed: 25);

        var (gradient, neurons) = RunLi(Managed, w, input);
        var (nativeGradient, nativeNeurons) = RunLi(NativeBackend, w, input);

        SyncAssert.That(nativeNeurons).IsEqualTo(neurons);
        for (int i = 0; i < w.Length; i++)
            SyncAssert.That(nativeGradient[i]).IsCloseTo(gradient[i], Tolerance);
    }

    [Test]
    public void Lif_BatchGradient_MatchesBetweenBackends()
    {
        const int nIn = 14, n = 6, nSpikes = 40, batch = 3;
        var w = MakeWeights(nIn, n, seed: 26);
        var input = MakeBatch(batch, nSpikes, nIn, seed: 27);

        var (gradient, totalSpikes) = RunLifBatch(Managed, w, input, batch);
        var (nativeGradient, nativeTotal) = RunLifBatch(NativeBackend, w, input, batch);

        SyncAssert.That(totalSpikes > 0).IsTrue();
        SyncAssert.That(gradient.Any(g => g != 0.0)).IsTrue();
        SyncAssert.That(nativeTotal).IsEqualTo(totalSpikes);
        for (int i = 0; i < w.Length; i++)
            SyncAssert.That(nativeGradient[i]).IsCloseTo(gradient[i], Tolerance);
    }

    // ── Contract: chains ───────────────────────────────────────────────

    [Test]
    public void Chain_ForwardBackward_WorksPerBackend()
    {
        foreach (var backend in AllBackends())
        {
            const int nIn = 8, nMid = 6, nOut = 4;
            var w1 = MakeWeights(nIn, nMid, seed: 30);
            var w2 = MakeWeights(nMid, nOut, seed: 31);

            using var layerB = backend.CreateLif(new LIFLayerParameters(nMid, nIn), w1);
            using var layerA = backend.CreateLif(new LIFLayerParameters(nOut, nMid), w2);
            var chain = Layer.Chain(layerA, layerB);

            chain.Forward(MakeInput(40, nIn, seed: 32));
            SyncAssert.That(chain.Output.Count).IsEqualTo(1);
            SyncAssert.That(chain.Intermediate.Count).IsEqualTo(1);
            chain.Backward();

            SyncAssert.That(layerA.Gradient.Length).IsEqualTo(nMid * nOut);
            SyncAssert.That(layerB.Gradient.Length).IsEqualTo(nIn * nMid);
        }
    }

    [Test]
    public void Chain_MixedBackends_ProducesSameOutput()
    {
        // A managed layer feeding a native layer (and vice versa) must give the same
        // spikes and gradients as the same chain with both layers on one backend.
        const int nIn = 10, nMid = 6, nOut = 4;
        var w1 = MakeWeights(nIn, nMid, seed: 33);
        var w2 = MakeWeights(nMid, nOut, seed: 34);
        var input = MakeInput(45, nIn, seed: 35);

        var (homogeneous, homoSpikes) = RunChain(Managed, Managed, w1, w2, input, nMid, nOut);
        var (mixed, mixedSpikes) = RunChain(Managed, NativeBackend, w1, w2, input, nMid, nOut);

        SyncAssert.That(mixedSpikes).IsEqualTo(homoSpikes);
        for (int i = 0; i < w2.Length; i++)
            SyncAssert.That(mixed[i]).IsCloseTo(homogeneous[i], Tolerance);
    }

    // ── Helpers for cross-backend runs ─────────────────────────────────

    private static (double[] gradient, int spikes) RunChain(
        LayerBackend outer, LayerBackend inner, double[] w1, double[] w2,
        SpikesVector input, int nMid, int nOut)
    {
        using var layerB = inner.CreateLif(new LIFLayerParameters(nMid, w1.Length / nMid), w1);
        using var layerA = outer.CreateLif(new LIFLayerParameters(nOut, nMid), w2);
        var chain = Layer.Chain(layerA, layerB);

        chain.Forward(input);
        int spikes = 0;
        for (int b = 0; b < chain.Output.Count; b++)
        {
            spikes += chain.Output[b].NSpikes;
            for (int i = 0; i < chain.Output[b].NSpikes; i++)
                outer.SetSpikeError(layerA, b, i, 1.0);
        }
        chain.Backward();
        return (layerA.Gradient.ToArray(), spikes);
    }

    private static (double[] gradient, int spikes) RunLif(LayerBackend backend, double[] w, SpikesVector input, int n)
    {
        using var layer = backend.CreateLif(new LIFLayerParameters(n, w.Length / n), w);
        layer.Forward(input);
        int spikes = 0;
        for (int b = 0; b < layer.Output.Count; b++)
        {
            spikes += layer.Output[b].NSpikes;
            for (int i = 0; i < layer.Output[b].NSpikes; i++)
                backend.SetSpikeError(layer, b, i, 1.0);
        }
        layer.Backward();
        return (layer.Gradient.ToArray(), spikes);
    }

    private static (double[] gradient, int neurons) RunLi(LayerBackend backend, double[] w, SpikesVector input)
    {
        const int n = 8;
        using var layer = backend.CreateLi(new LILayerParameters(n, w.Length / n), w);
        layer.Forward(input);
        var maxima = backend.GetMaxima(layer)!;
        for (int i = 0; i < maxima[0].NNeurons; i++) maxima[0].SetError(i, 1.0);
        layer.Backward();
        return (layer.Gradient.ToArray(), maxima[0].NNeurons);
    }

    private static (double[] gradient, int spikes) RunLifBatch(LayerBackend backend, double[] w, SpikesVector input, int batch)
    {
        const int n = 6;
        using var layer = backend.CreateLif(new LIFLayerParameters(n, w.Length / n), w);
        layer.Forward(input);
        int spikes = 0;
        for (int b = 0; b < batch; b++)
        {
            spikes += layer.Output[b].NSpikes;
            for (int i = 0; i < layer.Output[b].NSpikes; i++)
                backend.SetSpikeError(layer, b, i, 1.0);
        }
        layer.Backward();
        return (layer.Gradient.ToArray(), spikes);
    }
}
