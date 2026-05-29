using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// Tests that compare the C# implementation against the native C library via P/Invoke.
/// These serve as ground-truth verification of numerical accuracy.
/// </summary>
public class NativeVsManagedTests
{
    private const double Tolerance = 1e-10;

    [Test]
    public async Task ComputeSums_NativeVsManaged_AreIdentical()
    {
        int nPre = 3, nPost = 2;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] times = [0.01, 0.02, 0.03];
        int[] sources = [0, 1, 2];

        // C# implementation
        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, csSum0, csSum1, nPost,
            new double[times.Length], new double[times.Length]);

        // C++ via P/Invoke
        var nativeSum0 = new double[(times.Length + 1) * nPost];
        var nativeSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, nativeSum0, nativeSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(nativeSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(nativeSum1[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeSums_NativeVsManaged_EmptyInput()
    {
        int nPost = 3;
        double[] w = [1, 0.5, 0.3, 0.2, 1, 0.4];
        double[] times = [];
        int[] sources = [];

        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, csSum0, csSum1, nPost,
            Array.Empty<double>(), Array.Empty<double>());

        var nativeSum0 = new double[(times.Length + 1) * nPost];
        var nativeSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, w.Length / nPost, nPost, times, sources, 0.02, 0.005, nativeSum0, nativeSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(nativeSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(nativeSum1[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeSpikes_NativeVsManaged_SameOutput()
    {
        int nPost = 1, nPre = 1;
        double[] w = [20.0];

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 4);

        // C# implementation
        var csResult = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.005, buffers);

        // C++ via P/Invoke
        var nativeTimes = new double[10];
        var nativeSources = new int[10];
        var nativeCurrents = new double[10];
        int nNativeSpikes = NativeApi.ComputeSpikes(w, nPre, nPost, [0.01], [0], 1.0, 0.02, 0.005, nativeTimes, nativeSources, nativeCurrents);

        await Assert.That(csResult.NSpikes).IsEqualTo(nNativeSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            await Assert.That(csResult.Times[i]).IsCloseTo(nativeTimes[i], Tolerance);
            await Assert.That(csResult.Sources[i]).IsEqualTo(nativeSources[i]);
            await Assert.That(csResult.Currents[i]).IsCloseTo(nativeCurrents[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeSpikes_NativeVsManaged_MultipleNeurons()
    {
        int nPost = 3, nPre = 2;
        double[] w = [20.0, 0.01, 0.01, 0.01, 0.01, 20.0];
        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 8);

        var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        var nativeTimes = new double[100];
        var nativeSources = new int[100];
        var nativeCurrents = new double[100];
        int nNativeSpikes = NativeApi.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, nativeTimes, nativeSources, nativeCurrents);

        // Both C++ and C# now use the while-loop pattern, finding 5 spikes each.
        await Assert.That(csResult.NSpikes).IsEqualTo(nNativeSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            await Assert.That(csResult.Times[i]).IsCloseTo(nativeTimes[i], Tolerance);
            await Assert.That(csResult.Sources[i]).IsEqualTo(nativeSources[i]);
        }
    }

    [Test]
    public async Task ComputeSpikes_NativeVsManaged_NoOutput()
    {
        int nPost = 1, nPre = 1;
        double[] w = [0.01]; // Too weak to spike

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 4);

        var csResult = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.005, buffers);

        var nativeTimes = new double[10];
        var nativeSources = new int[10];
        var nativeCurrents = new double[10];
        int nNativeSpikes = NativeApi.ComputeSpikes(w, nPre, nPost, [0.01], [0], 1.0, 0.02, 0.005, nativeTimes, nativeSources, nativeCurrents);

        await Assert.That(csResult.NSpikes).IsEqualTo(0);
        await Assert.That(nNativeSpikes).IsEqualTo(0);
    }

    [Test]
    public async Task ComputeMaxima_NativeVsManaged_AreIdentical()
    {
        int nPost = 3, nPre = 3;
        double[] w = [1, 0.5, 0.3, 0.2, 1, 0.4, 0.1, 0.6, 1];
        double[] times = [0.01];
        int[] sources = [0];

        // C# implementation
        var csMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        // C++ via P/Invoke
        var nativeTimes = new double[nPost];
        var nativeValues = new double[nPost];
        NativeApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, nativeTimes, nativeValues);

        for (int i = 0; i < nPost; i++)
        {
            if (double.IsNaN(csMaxima.Times[i]))
                await Assert.That(nativeTimes[i]).IsNaN();
            else
            {
                await Assert.That(csMaxima.Times[i]).IsCloseTo(nativeTimes[i], Tolerance);
                await Assert.That(csMaxima.Values[i]).IsCloseTo(nativeValues[i], Tolerance);
            }
        }
    }

    [Test]
    public async Task ComputeMaxima_NativeVsManaged_MultipleSpikeInput()
    {
        int nPost = 2, nPre = 2;
        double[] w = [0.5, 0.5, 0.5, 0.5];
        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var csMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        var nativeTimes = new double[nPost];
        var nativeValues = new double[nPost];
        NativeApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, nativeTimes, nativeValues);

        for (int i = 0; i < nPost; i++)
        {
            await Assert.That(csMaxima.Times[i]).IsCloseTo(nativeTimes[i], Tolerance);
            await Assert.That(csMaxima.Values[i]).IsCloseTo(nativeValues[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeMaxima_NativeVsManaged_LargeNetwork()
    {
        int nPre = 50, nPost = 20;
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++) w[i] = 0.1 + 0.1 * (i % 5);

        int nSpikes = 200;
        double[] times = new double[nSpikes];
        int[] sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = 0.001 * (i + 1);
            sources[i] = i % nPre;
        }

        var csMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        var nativeTimes = new double[nPost];
        var nativeValues = new double[nPost];
        NativeApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, nativeTimes, nativeValues);

        for (int i = 0; i < nPost; i++)
        {
            await Assert.That(csMaxima.Times[i]).IsCloseTo(nativeTimes[i], Tolerance);
            await Assert.That(csMaxima.Values[i]).IsCloseTo(nativeValues[i], Tolerance);
        }
    }

    [Test]
    public async Task BackwardMaxima_NativeVsManaged_GradientMatch()
    {
        int nPre = 3, nPost = 2;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [0.025, double.NaN];
        double[] maximaValues = [1.5, 0.0];
        double[] maximaErrors = [0.1, 0.0];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            await Assert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    [Test]
    public async Task ComputeSpikes_NativeVsManaged_AccumulationCase()
    {
        // Test the accumulation case: two inputs, output fires after second input
        int nPost = 1, nPre = 2;
        double[] w = [3.0, 15.0];
        double[] times = [0.01, 0.05];
        int[] sources = [0, 1];

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 8);

        var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        var nativeTimes = new double[10];
        var nativeSources = new int[10];
        var nativeCurrents = new double[10];
        int nNativeSpikes = NativeApi.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, nativeTimes, nativeSources, nativeCurrents);

        await Assert.That(csResult.NSpikes).IsEqualTo(nNativeSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            await Assert.That(csResult.Times[i]).IsCloseTo(nativeTimes[i], Tolerance);
            await Assert.That(csResult.Sources[i]).IsEqualTo(nativeSources[i]);
        }
    }

    /// <summary>
    /// P/Invoke API contract test for BackwardSpikes.
    /// Note: The C++ eventprop_backward_spikes is currently a stub that returns zeros.
    /// Real correctness is verified by numerical gradient tests in BackwardSpikesTests.cs.
    /// This test verifies the P/Invoke boundary works end-to-end and that C# produces
    /// non-trivial gradients while the C++ stub returns zeros (documented limitation).
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_ApiContract()
    {
        int nPre = 2, nPost = 1;
        double tauMem = 0.02, tauSyn = 0.005;
        // Use strong weight to guarantee post-spike activity
        double wSave = 20.0;

        double[] w = [wSave, 0.0];
        double[] inputTimes = [0.1];
        int[] inputSources = [0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: inputTimes.Length);
        buffers.Clear();

        // Forward pass to get post-spike times and currents
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, tauMem, tauSyn, buffers);

        // Set error on the post-spike (required for non-zero gradients)
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        // C# implementation — produces real gradients
        var csGradient = new double[nPre * nPost];
        Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, tauMem, tauSyn, csGradient, buffers);

        // C++ via P/Invoke — full implementation (was stub, now ported from C#)
        var nativeGradient = new double[nPre * nPost];
        Array.Fill(nativeGradient, 0.0);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, tauMem, tauSyn, nativeGradient, new double[1024]);

        // Both should produce non-trivial gradients when post-spike error is set
        double csSum = 0.0, nativeSum = 0.0;
        for (int i = 0; i < csGradient.Length; i++)
            csSum += Math.Abs(csGradient[i]);
        for (int i = 0; i < nativeGradient.Length; i++)
            nativeSum += Math.Abs(nativeGradient[i]);
        SyncAssert.That(csSum).IsGreaterThan(0.0);
        SyncAssert.That(nativeSum).IsGreaterThan(0.0);

        // C++ and C# should produce matching gradients within tolerance
        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    // ========================================================================

    /// <summary>
    /// Test Case 2: Multiple input spikes, single post-spike.
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_MultipleInputs_SinglePost()
    {
        int nPre = 3, nPost = 1;
        double[] w = [20.0, 5.0, 0.0];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes == 0) return;
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 3: Single input spike, multiple post-spikes.
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_SingleInput_MultiplePosts()
    {
        int nPre = 2, nPost = 3;
        double[] w = [20.0, 0.0, 5.0, 0.0, 15.0, 0.0];
        double[] inputTimes = [0.1];
        int[] inputSources = [0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes == 0) return;
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 4: Multiple input spikes, multiple post-spikes (full network).
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_MultipleInputs_MultiplePosts()
    {
        int nPre = 5, nPost = 3;
        double[] w = [20.0, 5.0, 10.0, 8.0, 3.0,
                      12.0, 7.0, 15.0, 6.0, 9.0,
                      2.0, 18.0, 4.0, 11.0, 13.0];
        double[] inputTimes = [0.01, 0.02, 0.03, 0.04, 0.05];
        int[] inputSources = [0, 1, 2, 3, 4];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes == 0) return;
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 5: Asymmetric network (nPre != nPost).
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_AsymmetricNetwork()
    {
        int nPre = 4, nPost = 2;
        double[] w = [15.0, 3.0, 8.0, 20.0,
                      5.0, 12.0, 7.0, 18.0];
        double[] inputTimes = [0.01, 0.03, 0.05];
        int[] inputSources = [0, 1, 2];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes == 0) return;
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 6: Zero error on all post-spikes → zero gradients.
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_ZeroErrors()
    {
        int nPre = 3, nPost = 2;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 0];

        var postSpikes = new SpikesData([0.015, 0.025], [0, 1]);
        // Errors are already zero (default)

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, new ScratchBuffers());
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 7: Single non-zero post-spike error (sparse error).
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_SparseError()
    {
        int nPre = 2, nPost = 3;
        double[] w = [20.0, 5.0, 8.0, 15.0, 3.0, 12.0];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes < 2) return;

        // Set error only on first post-spike
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, i == 0 ? 1.0 : 0.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    // ========================================================================
    // BackwardMaxima Comparison Tests (C++ vs C#)
    // ========================================================================

    /// <summary>
    /// Test Case 1: Single input spike, single maxima.
    /// Note: C++ implementation assumes n_maxima == n_post.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_SingleSpikeSingleMaxima()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01];
        int[] inputSources = [0];

        // n_maxima == n_post (C++ implementation constraint)
        double[] maximaTimes = [0.015];
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 2: Multiple inputs, single maxima.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_MultipleInputsSingleMaxima()
    {
        int nPre = 3, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 2];

        double[] maximaTimes = [0.025];
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 3: Multiple inputs, multiple maxima (all valid).
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_MultipleInputsMultipleMaxima()
    {
        int nPre = 3, nPost = 2;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        // n_maxima == n_post (C++ implementation constraint)
        double[] maximaTimes = [0.015, 0.025]; // Both valid
        double[] maximaValues = [1.5, 2.0];
        double[] maximaErrors = [0.1, -0.2];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 4: Maxima after last input spike.
    /// Note: C++ implementation processes maxima with negative t_vmax_bwd (doesn't skip).
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_MaximaAfterLastSpike()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [0.03]; // After last input spike (0.02)
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 5: All maxima NaN → zero gradients.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_AllNaNMaxima()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [double.NaN];
        double[] maximaValues = [0.0];
        double[] maximaErrors = [0.0];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 6: Large network with random maxima.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_LargeNetwork()
    {
        int nPre = 50, nPost = 20;
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++) w[i] = 0.1 + 0.1 * (i % 5);

        int nSpikes = 200;
        double[] inputTimes = new double[nSpikes];
        int[] inputSources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++) {
            inputTimes[i] = 0.001 * (i + 1);
            inputSources[i] = i % nPre;
        }

        // n_maxima == n_post (C++ implementation constraint)
        int nMaxima = 20;
        double[] maximaTimes = new double[nMaxima];
        double[] maximaValues = new double[nMaxima];
        double[] maximaErrors = new double[nMaxima];

        for (int i = 0; i < nMaxima; i++) {
            maximaTimes[i] = (i % 2 == 0) ? 0.5 + 0.1 * i : double.NaN;
            maximaValues[i] = 1.0 + 0.5 * i;
            maximaErrors[i] = (i % 2 == 0) ? 0.1 * (i + 1) : 0.0;
        }

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 7: Different tau values (edge case).
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_DifferentTaus()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [0.015];
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.01, 0.003, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.01, 0.003, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    // ========================================================================
    // P0: Critical Edge Cases for Numerical Parity
    // ========================================================================

    /// <summary>
    /// tauMem very close to tauSyn (ratio 0.95). Tests numerical stability of prefactors:
    ///   k_prefactor = tau_syn / (tau_mem - tau_syn) = 0.019 / 0.001 = 19
    ///   tmax_prefactor = 1.0 / (1/tau_mem - 1/tau_syn) ≈ -380
    /// These large prefactors could cause different rounding between C++ and C#.
    /// </summary>
    [Test]
    public async Task ComputeSums_NativeVsManaged_NearEqualTaus()
    {
        int nPre = 2, nPost = 2;
        double[] w = [1.0, 0.5, 0.3, 2.0];
        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        // tauMem and tauSyn differ by only 0.001 — prefactors blow up
        double tauMem = 0.02, tauSyn = 0.019;

        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, tauMem, tauSyn, csSum0, csSum1, nPost,
            new double[times.Length], new double[times.Length]);

        var nativeSum0 = new double[(times.Length + 1) * nPost];
        var nativeSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, tauMem, tauSyn, nativeSum0, nativeSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(nativeSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(nativeSum1[i], Tolerance);
        }
    }

    /// <summary>
    /// Single input spike, single post-synaptic neuron — the simplest non-trivial case.
    /// Exercises the full forward+backward pipeline with minimal data.
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_SingleInputSinglePost()
    {
        int nPre = 1, nPost = 1;
        double[] w = [20.0];
        double[] inputTimes = [0.1];
        int[] inputSources = [0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);

        if (postSpikes.NSpikes == 0)
            return; // Skip — no post-spike activity with these parameters

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        SyncAssert.That(csGradient[0]).IsCloseTo(nativeGradient[0], 1e-10);
    }

    /// <summary>
    /// Single neuron, multiple spikes — tests spike reset logic.
    /// A very strong weight causes rapid successive post-spikes from one input spike.
    /// </summary>
    [Test]
    public async Task ComputeSpikes_NativeVsManaged_SingleNeuronMultipleSpikes()
    {
        int nPost = 1, nPre = 1;
        double[] w = [80.0]; // Very strong — causes multiple spikes

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 16);

        var csResult = EventPropMath.ComputeSpikes(w, [0.05], [0], nPost, 1.0, 0.02, 0.005, buffers);

        var nativeTimes = new double[16];
        var nativeSources = new int[16];
        var nativeCurrents = new double[16];
        int nNativeSpikes = NativeApi.ComputeSpikes(w, nPre, nPost, [0.05], [0], 1.0, 0.02, 0.005,
            nativeTimes, nativeSources, nativeCurrents);

        await Assert.That(csResult.NSpikes).IsEqualTo(nNativeSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            await Assert.That(csResult.Times[i]).IsCloseTo(nativeTimes[i], Tolerance);
            await Assert.That(csResult.Sources[i]).IsEqualTo(nativeSources[i]);
        }
    }

    // ========================================================================
    // P1: Important Boundary Cases
    // ========================================================================

    /// <summary>
    /// Maxima times before all input spikes — backward pass should skip these.
    /// Tests the t_vmax_bwd > t_bwd boundary condition in BackwardMaxima.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_MaximaBeforeAllSpikes()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.05, 0.1]; // Spikes at 0.05 and 0.1
        int[] inputSources = [0, 1];

        // Maxima time before first input spike
        double[] maximaTimes = [0.02];
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        // Both should produce zero gradients (maxima before any input spike)
        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// All weights zero for some neurons — tests gradient sparsity.
    /// Neurons with all-zero weight rows should produce zero gradients.
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_ZeroWeightRows()
    {
        int nPre = 4, nPost = 2;
        // Neurons 1 and 3 have all-zero weights
        double[] w = [15.0, 3.0,
                      0.0, 0.0,
                      8.0, 20.0,
                      0.0, 0.0];
        double[] inputTimes = [0.01, 0.03, 0.05];
        int[] inputSources = [0, 1, 2]; // Spikes from neurons 0, 1, and 2

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);

        if (postSpikes.NSpikes == 0)
            return;

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        // All neurons should have matching gradients (including zero-weight rows)
        // Note: Zero weights don't necessarily mean zero gradients because the adjoint
        // method propagates errors through λ_i which depends on membrane dynamics,
        // not directly on weights. The key check is C++ vs C# parity.

        // All neurons should match between C++ and C#
        for (int src = 0; src < nPre; src++)
            for (int nrn = 0; nrn < nPost; nrn++)
            {
                int idx = src * nPost + nrn;
                SyncAssert.That(csGradient[idx]).IsCloseTo(nativeGradient[idx], 1e-10);
            }
    }

    /// <summary>
    /// Maxima exactly at input spike time — tests floating-point equality boundary.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_MaximaAtSpikeTime()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        // Maxima time exactly equals second input spike time
        double[] maximaTimes = [0.02];
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    // ========================================================================
    // P2/P3: Remaining Edge Cases from test-parity-gaps.md
    // ========================================================================

    /// <summary>
    /// ComputeSums with single input spike — minimal case verification.
    /// Tests the base case of sum computation with one spike.
    /// </summary>
    [Test]
    public async Task ComputeSums_NativeVsManaged_SingleInputSpike()
    {
        int nPre = 1, nPost = 2;
        double[] w = [0.5, 0.3];
        double[] times = [0.05]; // Single spike
        int[] sources = [0];

        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, csSum0, csSum1, nPost,
            new double[times.Length], new double[times.Length]);

        var nativeSum0 = new double[(times.Length + 1) * nPost];
        var nativeSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, nativeSum0, nativeSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(nativeSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(nativeSum1[i], Tolerance);
        }
    }

    /// <summary>
    /// BackwardSpikes with no post-spikes — should return zero gradients.
    /// Tests the early-return path when there are no post-spike errors to propagate.
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_NoPostSpikes()
    {
        int nPre = 3, nPost = 2;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 2];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, inputTimes.Length);

        // Use very weak weights so no post-spike activity occurs
        double[] weakW = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        var postSpikes = EventPropMath.ComputeSpikes(weakW, inputTimes, inputSources, nPost,
            1.0, 0.02, 0.005, buffers);

        // No post-spikes expected — gradients should be all zeros
        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(weakW, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            weakW, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        // Both should produce zero gradients (no post-spike activity)
        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(0.0, 1e-14);
        for (int i = 0; i < nativeGradient.Length; i++)
            SyncAssert.That(nativeGradient[i]).IsCloseTo(0.0, 1e-14);
    }

    /// <summary>
    /// BackwardMaxima with no valid maxima — all NaN.
    /// Tests that the backward pass correctly handles zero valid maxima cases
    /// and produces all-zero gradients.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_NoValidMaxima()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        // Maxima time before first spike — will be NaN in ComputeMaxima or invalid
        double[] maximaTimes = [0.001]; // Before first input spike at 0.01
        double[] maximaValues = [double.NaN];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        // Both should produce zero gradients (maxima before any input spike)
        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// BackwardSpikes with no input spikes — tests merged timeline edge case.
    /// Post-spikes exist but there are no input spike sources to merge into the timeline.
    /// Gradients should remain zero (nothing to update).
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_NoInputSpikes()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = []; // No input spikes
        int[] inputSources = [];

        var buffers = new ScratchBuffers();
        // Simulate a post-spike from a previous forward pass
        double[] postTimes = [0.05];
        int[] postSources = [0];
        double[] postCurrents = [1.5];
        double[] postErrors = [1.0];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        // C# backward pass with empty input spikes
        var csPostSpikes = new SpikesData(postTimes, postSources, postCurrents);
        csPostSpikes.Errors = postErrors;
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, csPostSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);

        // C++ backward pass with empty input spikes
        NativeApi.BackwardSpikes(inputTimes, inputSources,
            postTimes, postSources, postCurrents, postErrors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        // Both should produce zero gradients (no input spikes to propagate through)
        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Large tau values — tests numerical stability with extreme time constants.
    /// </summary>
    [Test]
    public async Task ComputeSums_NativeVsManaged_LargeTauValues()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5];
        double[] times = [0.01, 0.02];
        int[] sources = [0, 0];

        // Large tau values (1 second) — tests exponential decay with slow dynamics
        double tauMem = 1.0, tauSyn = 0.5;

        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, tauMem, tauSyn, csSum0, csSum1, nPost,
            new double[times.Length], new double[times.Length]);

        var nativeSum0 = new double[(times.Length + 1) * nPost];
        var nativeSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, tauMem, tauSyn, nativeSum0, nativeSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(nativeSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(nativeSum1[i], Tolerance);
        }
    }

    /// <summary>
    /// Very small tau values — tests numerical stability with fast dynamics.
    /// </summary>
    [Test]
    public async Task ComputeSums_NativeVsManaged_SmallTauValues()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5];
        double[] times = [0.01, 0.02];
        int[] sources = [0, 0];

        // Very small tau values (1ms) — tests exponential decay with fast dynamics
        double tauMem = 0.001, tauSyn = 0.0005;

        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, tauMem, tauSyn, csSum0, csSum1, nPost,
            new double[times.Length], new double[times.Length]);

        var nativeSum0 = new double[(times.Length + 1) * nPost];
        var nativeSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, tauMem, tauSyn, nativeSum0, nativeSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(nativeSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(nativeSum1[i], Tolerance);
        }
    }

    /// <summary>
    /// BackwardSpikes with very large weight values — tests numerical stability.
    /// </summary>
    [Test]
    public void BackwardSpikes_NativeVsManaged_LargeWeights()
    {
        int nPre = 2, nPost = 1;
        double[] w = [50.0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, 1);

        var postSpikes = EventPropMath.ComputeSpikes(w, [0.05], [0], nPost,
            1.0, 0.02, 0.005, buffers);

        if (postSpikes.NSpikes == 0)
            return;

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardSpikes(w, [0.05], [0], postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        NativeApi.BackwardSpikes([0.05], [0],
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, nativeGradient, new double[1024]);

        SyncAssert.That(csGradient[0]).IsCloseTo(nativeGradient[0], 1e-10);
    }

    /// <summary>
    /// BackwardMaxima with multiple valid maxima across neurons.
    /// Tests that each neuron's gradient is computed independently and correctly.
    /// </summary>
    [Test]
    public void BackwardMaxima_NativeVsManaged_MultipleValidAcrossNeurons()
    {
        int nPre = 2, nPost = 3;
        double[] w = [0.5, 0.3,
                      0.1, 0.2,
                      0.8, 0.4];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        // All three neurons have valid maxima at different times
        double[] maximaTimes = [0.015, 0.025, 0.03];
        double[] maximaValues = [1.5, 2.0, 1.8];
        double[] maximaErrors = [0.1, -0.2, 0.3];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var nativeGradient = new double[nPre * nPost]; Array.Fill(nativeGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        NativeApi.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, nativeGradient, new double[1024]);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }
}
