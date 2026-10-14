using CW.EventProp;
using Native = CW.EventProp.Native;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// Parity between the native (P/Invoke) layer classes and the managed layers.
///
/// The native layers exist so benchmarks compare like-for-like; these tests pin that
/// the native LIFLayer/LILayer produce the same spikes, maxima, and gradients as the
/// managed ones for identical weights and input.
/// </summary>
public class NativeLayerParityTests
{
    private const double Tolerance = 1e-9;

    private static (double[] w, double[] times, int[] sources) MakeInput(
        int nPre, int nPost, int nSpikes, int seed)
    {
        var rng = new Random(seed);
        var w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++) w[i] = 0.5 + rng.NextDouble() * 1.5;

        var times = new double[nSpikes];
        var sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = (i + 1) * 1.0 / (nSpikes + 1);
            sources[i] = rng.Next(nPre);
        }
        return (w, times, sources);
    }

    [Test]
    public void LIFLayer_NativeVsManaged_SpikesAndGradientMatch()
    {
        const int nPre = 20, nPost = 8, nSpikes = 60;
        const double tauMem = 0.02, tauSyn = 0.005, vTh = 1.0;
        var (w, times, sources) = MakeInput(nPre, nPost, nSpikes, seed: 7);

        var managedInput = new SpikesVector([new SpikesData(times, sources)]);
        var nativeInput = new SpikesVector([new SpikesData(times, sources)]);

        var managed = new CW.EventProp.LIFLayer(
            new LIFLayerParameters(nPost, nPre, tauMem, tauSyn, vTh), w);
        using var native = new Native.LIFLayer(
            new LIFLayerParameters(nPost, nPre, tauMem, tauSyn, vTh), w);

        managed.Forward(managedInput);
        native.Forward(nativeInput);

        int mSpikes = managed.Output[0].NSpikes;
        int nSpikesOut = native.Output[0].NSpikes;
        SyncAssert.That(nSpikesOut).IsEqualTo(mSpikes);
        for (int i = 0; i < mSpikes; i++)
        {
            SyncAssert.That(native.Output[0].Times[i]).IsCloseTo(managed.Output[0].Times[i], Tolerance);
            SyncAssert.That(native.Output[0].Sources[i]).IsEqualTo(managed.Output[0].Sources[i]);
        }

        for (int i = 0; i < mSpikes; i++) managed.Output[0].SetError(i, 1.0);
        for (int i = 0; i < nSpikesOut; i++) native.SetError(0, i, 1.0);

        managed.Backward();
        native.Backward();

        for (int i = 0; i < w.Length; i++)
            SyncAssert.That(native.Gradient[i]).IsCloseTo(managed.Gradient[i], Tolerance);
    }

    [Test]
    public void LILayer_NativeVsManaged_MaximaAndGradientMatch()
    {
        const int nPre = 20, nPost = 8, nSpikes = 60;
        const double tauMem = 0.02, tauSyn = 0.005;
        var (w, times, sources) = MakeInput(nPre, nPost, nSpikes, seed: 11);

        var managedInput = new SpikesVector([new SpikesData(times, sources)]);
        var nativeInput = new SpikesVector([new SpikesData(times, sources)]);

        var managed = new CW.EventProp.LILayer(
            new LILayerParameters(nPost, nPre, tauMem, tauSyn), w);
        using var native = new Native.LILayer(
            new LILayerParameters(nPost, nPre, tauMem, tauSyn), w);

        managed.Forward(managedInput);
        native.Forward(nativeInput);

        var mMax = managed.MaximaBatch![0];
        var nMax = native.MaximaBatch![0];
        SyncAssert.That(nMax.NNeurons).IsEqualTo(mMax.NNeurons);
        for (int i = 0; i < mMax.NNeurons; i++)
        {
            SyncAssert.That(nMax.Values[i]).IsCloseTo(mMax.Values[i], Tolerance);
        }

        for (int i = 0; i < mMax.NNeurons; i++) mMax.SetError(i, 1.0);
        for (int i = 0; i < nMax.NNeurons; i++) nMax.SetError(i, 1.0);

        managed.Backward();
        native.Backward();

        for (int i = 0; i < w.Length; i++)
            SyncAssert.That(native.Gradient[i]).IsCloseTo(managed.Gradient[i], Tolerance);
    }

    [Test]
    public void LIFLayer_NativeBatch_ForwardMatchesPerSample()
    {
        // Batch of 3 samples through the native layer; each sample's spikes must match
        // the managed layer run one sample at a time.
        const int nPre = 15, nPost = 6, nSpikes = 40;
        const double tauMem = 0.02, tauSyn = 0.005, vTh = 1.0;

        var samples = new (double[] t, int[] s)[3];
        for (int b = 0; b < 3; b++)
        {
            var (_, t, s) = MakeInput(nPre, nPost, nSpikes, seed: 100 + b);
            samples[b] = (t, s);
        }
        var w = MakeInput(nPre, nPost, nSpikes, seed: 5).w;

        var managedBatch = new SpikesVector([
            new SpikesData(samples[0].t, samples[0].s),
            new SpikesData(samples[1].t, samples[1].s),
            new SpikesData(samples[2].t, samples[2].s),
        ]);
        var nativeBatch = new SpikesVector([
            new SpikesData(samples[0].t, samples[0].s),
            new SpikesData(samples[1].t, samples[1].s),
            new SpikesData(samples[2].t, samples[2].s),
        ]);

        var managed = new CW.EventProp.LIFLayer(
            new LIFLayerParameters(nPost, nPre, tauMem, tauSyn, vTh), w);
        using var native = new Native.LIFLayer(
            new LIFLayerParameters(nPost, nPre, tauMem, tauSyn, vTh), w);

        managed.Forward(managedBatch);
        native.Forward(nativeBatch);

        for (int b = 0; b < 3; b++)
        {
            int n = managed.Output[b].NSpikes;
            SyncAssert.That(native.Output[b].NSpikes).IsEqualTo(n);
            for (int i = 0; i < n; i++)
            {
                SyncAssert.That(native.Output[b].Times[i]).IsCloseTo(managed.Output[b].Times[i], Tolerance);
                SyncAssert.That(native.Output[b].Sources[i]).IsEqualTo(managed.Output[b].Sources[i]);
            }
        }
    }
}
