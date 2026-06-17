using CW.EventProp;
using Native = CW.EventProp.Native;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

public class NativeVsManagedEdgeTests
{
    private const double Tolerance = 1e-10;

    [Test]
    public void ComputeSpikes_NativeVsManaged_SimultaneousInputSpikeTimes()
    {
        int nPre = 2, nPost = 1;
        double[] w = [15.0, 10.0];
        double[] times = [0.05, 0.05];
        int[] sources = [0, 1];

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 8);
        var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        using var nativeOutput = Native.Api.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005);

        SyncAssert.That(csResult.NSpikes).IsEqualTo(nativeOutput.NSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            SyncAssert.That(csResult.Times[i]).IsCloseTo(nativeOutput.Times[i], Tolerance);
            SyncAssert.That(csResult.Sources[i]).IsEqualTo(nativeOutput.Sources[i]);
        }
    }

    [Test]
    public void ComputeSpikes_NativeVsManaged_RepeatedSameSource()
    {
        int nPre = 3, nPost = 1;
        double[] w = [5.0, 0.0, 0.0];
        double[] times = [0.01, 0.02, 0.03, 0.04, 0.05];
        int[] sources = [0, 0, 0, 0, 0];

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 16);
        var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        using var nativeOutput = Native.Api.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005);

        SyncAssert.That(csResult.NSpikes).IsEqualTo(nativeOutput.NSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            SyncAssert.That(csResult.Times[i]).IsCloseTo(nativeOutput.Times[i], Tolerance);
            SyncAssert.That(csResult.Sources[i]).IsEqualTo(nativeOutput.Sources[i]);
        }
    }

    [Test]
    public void ComputeSpikes_NativeVsManaged_NearZeroThreshold()
    {
        int nPre = 1, nPost = 1;
        double[] w = [5.0];
        double vTh = 0.001;

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 2048);
        var csResult = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, vTh, 0.02, 0.005, buffers);

        using var nativeOutput = Native.Api.ComputeSpikes(w, nPre, nPost, [0.01], [0], vTh, 0.02, 0.005);

        SyncAssert.That(csResult.NSpikes).IsEqualTo(nativeOutput.NSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            SyncAssert.That(csResult.Times[i]).IsCloseTo(nativeOutput.Times[i], Tolerance);
            SyncAssert.That(csResult.Sources[i]).IsEqualTo(nativeOutput.Sources[i]);
        }
    }

    [Test]
    public void ComputeSpikes_NativeVsManaged_VeryHighThreshold()
    {
        int nPre = 2, nPost = 1;
        double[] w = [50.0, 40.0];
        double vTh = 100.0;

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 8);
        var csResult = EventPropMath.ComputeSpikes(w, [0.01, 0.02], [0, 1], nPost, vTh, 0.02, 0.005, buffers);

        using var nativeOutput = Native.Api.ComputeSpikes(w, nPre, nPost, [0.01, 0.02], [0, 1], vTh, 0.02, 0.005);

        SyncAssert.That(csResult.NSpikes).IsEqualTo(nativeOutput.NSpikes);
    }

    [Test]
    public void ComputeSpikes_NativeVsManaged_ShorterTauSyn()
    {
        int nPre = 1, nPost = 1;
        double[] w = [20.0];

        // Shorter tau_syn
        using var buffersShort = new ScratchBuffers();
        buffersShort.InitPostSpikeBuffers(nPost, 64);
        var csShort = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.003, buffersShort);
        using var nativeShort = Native.Api.ComputeSpikes(w, nPre, nPost, [0.01], [0], 1.0, 0.02, 0.003);
        Console.WriteLine($"Short tau_syn: C={nativeShort.NSpikes}, Managed={csShort.NSpikes}");
        SyncAssert.That(csShort.NSpikes).IsEqualTo(nativeShort.NSpikes);

        // Longer tau_syn
        using var buffersLong = new ScratchBuffers();
        buffersLong.InitPostSpikeBuffers(nPost, 64);
        var csLong = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.01, buffersLong);
        using var nativeLong = Native.Api.ComputeSpikes(w, nPre, nPost, [0.01], [0], 1.0, 0.02, 0.01);
        Console.WriteLine($"Long tau_syn: C={nativeLong.NSpikes}, Managed={csLong.NSpikes}");
        SyncAssert.That(csLong.NSpikes).IsEqualTo(nativeLong.NSpikes);
    }

    [Test]
    public void ComputeSpikes_NativeVsManaged_MultipleIntervals()
    {
        int nPre = 2, nPost = 1;
        double[] w = [3.0, 15.0];

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 64);
        var csResult = EventPropMath.ComputeSpikes(w, [0.01, 0.05], [0, 1], nPost, 1.0, 0.02, 0.005, buffers);
        using var nativeOutput = Native.Api.ComputeSpikes(w, nPre, nPost, [0.01, 0.05], [0, 1], 1.0, 0.02, 0.005);
        Console.WriteLine($"Multiple intervals: C={nativeOutput.NSpikes}, Managed={csResult.NSpikes}");
        for (int i = 0; i < nativeOutput.NSpikes; i++)
            Console.WriteLine($"  C[{i}]={nativeOutput.Times[i]:F6}");
        for (int i = 0; i < csResult.NSpikes; i++)
            Console.WriteLine($"  M[{i}]={csResult.Times[i]:F6}");
        SyncAssert.That(csResult.NSpikes).IsEqualTo(nativeOutput.NSpikes);
    }

    // ========================================================================
    // BackwardSpikes Edge Cases
    // ========================================================================

    [Test]
    public void BackwardSpikes_NativeVsManaged_PostSpikeEqualsInputTime()
    {
        int nPre = 2, nPost = 1;
        double[] w = [10.0, 8.0];
        double[] inputTimes = [0.05, 0.10];
        int[] sources = [0, 1];

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 8);
        var csSpikes = EventPropMath.ComputeSpikes(w, inputTimes, sources, nPost, 1.0, 0.02, 0.005, buffers);

        double[] postTimes = [inputTimes[0]];
        int[] postSourcesArr = [0];
        double[] postCurrents = [1.0];
        double[] postErrors = [1.0];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, sources, new SpikesData(postTimes, postSourcesArr, postCurrents) { Errors = postErrors }, nPost, 1.0, 0.02, 0.005, csGradient, buffers);

        using var nativeOutput = Native.Api.BackwardSpikes(inputTimes, sources, postTimes, postSourcesArr, postCurrents, postErrors, 1, w, nPre, nPost, 1.0, 0.02, 0.005);

        for (int i = 0; i < nPre * nPost; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeOutput.Gradient[i], Tolerance);
    }

    [Test]
    public void BackwardSpikes_NativeVsManaged_ManyPostSpikesPerInput()
    {
        int nPre = 2, nPost = 3;
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++) w[i] = 15.0 + i;

        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] sources = [0, 1, 0];

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 16);
        var csSpikes = EventPropMath.ComputeSpikes(w, inputTimes, sources, nPost, 1.0, 0.02, 0.005, buffers);

        for (int i = 0; i < csSpikes.NSpikes; i++)
            csSpikes.SetError(i, 1.0 + i * 0.1);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, sources, csSpikes, nPost, 1.0, 0.02, 0.005, csGradient, buffers);

        using var nativeOutput = Native.Api.BackwardSpikes(inputTimes, sources, csSpikes.Times[..csSpikes.NSpikes], csSpikes.Sources[..csSpikes.NSpikes], csSpikes.Currents[..csSpikes.NSpikes], csSpikes.Errors[..csSpikes.NSpikes], csSpikes.NSpikes, w, nPre, nPost, 1.0, 0.02, 0.005);

        for (int i = 0; i < nPre * nPost; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeOutput.Gradient[i], Tolerance);
    }

    [Test]
    public void BackwardSpikes_NativeVsManaged_SameSourceMultipleTimes()
    {
        int nPre = 3, nPost = 1;
        double[] w = [5.0, 0.0, 0.0];

        double[] inputTimes = [0.01, 0.02, 0.03, 0.04, 0.05];
        int[] sources = [0, 0, 0, 0, 0];

        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 16);
        var csSpikes = EventPropMath.ComputeSpikes(w, inputTimes, sources, nPost, 1.0, 0.02, 0.005, buffers);

        for (int i = 0; i < csSpikes.NSpikes; i++)
            csSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, sources, csSpikes, nPost, 1.0, 0.02, 0.005, csGradient, buffers);

        using var nativeOutput = Native.Api.BackwardSpikes(inputTimes, sources, csSpikes.Times[..csSpikes.NSpikes], csSpikes.Sources[..csSpikes.NSpikes], csSpikes.Currents[..csSpikes.NSpikes], csSpikes.Errors[..csSpikes.NSpikes], csSpikes.NSpikes, w, nPre, nPost, 1.0, 0.02, 0.005);

        for (int i = 0; i < nPre * nPost; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeOutput.Gradient[i], Tolerance);
    }

    // ========================================================================
    // BackwardMaxima Edge Cases
    // ========================================================================

    [Test]
    public void BackwardMaxima_NativeVsManaged_MixedNaNAndValidMaxima()
    {
        int nPre = 2, nPost = 3;
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++) w[i] = 5.0 + i;

        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] sources = [0, 1, 0];

        double[] maximaTimes = [0.5, double.NaN, 0.3];
        double[] maximaValues = [1.5, 0.0, 0.8];
        double[] maximaErrors = [0.1, 0.0, -0.2];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardMaxima(inputTimes, sources, maximaTimes, maximaValues, maximaErrors, w, nPost, 0.02, 0.005, csGradient);

        using var nativeOutput = Native.Api.BackwardMaxima(inputTimes, sources, maximaTimes, maximaValues, maximaErrors, 3, w, nPre, nPost, 0.02, 0.005);

        for (int i = 0; i < nPre * nPost; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeOutput.Gradient[i], Tolerance);
    }
}
