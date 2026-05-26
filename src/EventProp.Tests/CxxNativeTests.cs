using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// Tests that compare the C# implementation against the native C++ library via P/Invoke.
/// These serve as ground-truth verification of numerical accuracy.
/// </summary>
public class CxxNativeTests
{
    private const double Tolerance = 1e-10;

    [Test]
    public async Task ComputeSums_CppVsCs_AreIdentical()
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
        var cppSum0 = new double[(times.Length + 1) * nPost];
        var cppSum1 = new double[(times.Length + 1) * nPost];
        CxxNative.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, cppSum0, cppSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(cppSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(cppSum1[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeSums_CppVsCs_EmptyInput()
    {
        int nPost = 3;
        double[] w = [1, 0.5, 0.3, 0.2, 1, 0.4];
        double[] times = [];
        int[] sources = [];

        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, csSum0, csSum1, nPost,
            Array.Empty<double>(), Array.Empty<double>());

        var cppSum0 = new double[(times.Length + 1) * nPost];
        var cppSum1 = new double[(times.Length + 1) * nPost];
        CxxNative.ComputeSums(w, w.Length / nPost, nPost, times, sources, 0.02, 0.005, cppSum0, cppSum1);

        for (int i = 0; i < csSum0.Length; i++)
        {
            await Assert.That(csSum0[i]).IsCloseTo(cppSum0[i], Tolerance);
            await Assert.That(csSum1[i]).IsCloseTo(cppSum1[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeSpikes_CppVsCs_SameOutput()
    {
        int nPost = 1, nPre = 1;
        double[] w = [20.0];

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 4);

        // C# implementation
        var csResult = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.005, buffers);

        // C++ via P/Invoke
        var cppTimes = new double[10];
        var cppSources = new int[10];
        var cppCurrents = new double[10];
        int nCppSpikes = CxxNative.ComputeSpikes(w, nPre, nPost, [0.01], [0], 1.0, 0.02, 0.005, cppTimes, cppSources, cppCurrents);

        await Assert.That(csResult.NSpikes).IsEqualTo(nCppSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            await Assert.That(csResult.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(csResult.Sources[i]).IsEqualTo(cppSources[i]);
            await Assert.That(csResult.Currents[i]).IsCloseTo(cppCurrents[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeSpikes_CppVsCs_MultipleNeurons()
    {
        int nPost = 3, nPre = 2;
        double[] w = [20.0, 0.01, 0.01, 0.01, 0.01, 20.0];
        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 8);

        var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        var cppTimes = new double[100];
        var cppSources = new int[100];
        var cppCurrents = new double[100];
        int nCppSpikes = CxxNative.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, cppTimes, cppSources, cppCurrents);

        // Both C++ and C# now use the while-loop pattern, finding 5 spikes each.
        await Assert.That(csResult.NSpikes).IsEqualTo(nCppSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            await Assert.That(csResult.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(csResult.Sources[i]).IsEqualTo(cppSources[i]);
        }
    }

    [Test]
    public async Task ComputeSpikes_CppVsCs_NoOutput()
    {
        int nPost = 1, nPre = 1;
        double[] w = [0.01]; // Too weak to spike

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 4);

        var csResult = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.005, buffers);

        var cppTimes = new double[10];
        var cppSources = new int[10];
        var cppCurrents = new double[10];
        int nCppSpikes = CxxNative.ComputeSpikes(w, nPre, nPost, [0.01], [0], 1.0, 0.02, 0.005, cppTimes, cppSources, cppCurrents);

        await Assert.That(csResult.NSpikes).IsEqualTo(0);
        await Assert.That(nCppSpikes).IsEqualTo(0);
    }

    [Test]
    public async Task ComputeMaxima_CppVsCs_AreIdentical()
    {
        int nPost = 3, nPre = 3;
        double[] w = [1, 0.5, 0.3, 0.2, 1, 0.4, 0.1, 0.6, 1];
        double[] times = [0.01];
        int[] sources = [0];

        // C# implementation
        var csMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        // C++ via P/Invoke
        var cppTimes = new double[nPost];
        var cppValues = new double[nPost];
        CxxNative.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cppTimes, cppValues);

        for (int i = 0; i < nPost; i++)
        {
            if (double.IsNaN(csMaxima.Times[i]))
                await Assert.That(cppTimes[i]).IsNaN();
            else
            {
                await Assert.That(csMaxima.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
                await Assert.That(csMaxima.Values[i]).IsCloseTo(cppValues[i], Tolerance);
            }
        }
    }

    [Test]
    public async Task ComputeMaxima_CppVsCs_MultipleSpikeInput()
    {
        int nPost = 2, nPre = 2;
        double[] w = [0.5, 0.5, 0.5, 0.5];
        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var csMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        var cppTimes = new double[nPost];
        var cppValues = new double[nPost];
        CxxNative.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cppTimes, cppValues);

        for (int i = 0; i < nPost; i++)
        {
            await Assert.That(csMaxima.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(csMaxima.Values[i]).IsCloseTo(cppValues[i], Tolerance);
        }
    }

    [Test]
    public async Task ComputeMaxima_CppVsCs_LargeNetwork()
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

        var cppTimes = new double[nPost];
        var cppValues = new double[nPost];
        CxxNative.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cppTimes, cppValues);

        for (int i = 0; i < nPost; i++)
        {
            await Assert.That(csMaxima.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(csMaxima.Values[i]).IsCloseTo(cppValues[i], Tolerance);
        }
    }

    [Test]
    public async Task BackwardMaxima_CppVsCs_GradientMatch()
    {
        int nPre = 3, nPost = 2;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [0.025, double.NaN];
        double[] maximaValues = [1.5, 0.0];
        double[] maximaErrors = [0.1, 0.0];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            await Assert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    [Test]
    public async Task ComputeSpikes_CppVsCs_AccumulationCase()
    {
        // Test the accumulation case: two inputs, output fires after second input
        int nPost = 1, nPre = 2;
        double[] w = [3.0, 15.0];
        double[] times = [0.01, 0.05];
        int[] sources = [0, 1];

        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, 8);

        var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        var cppTimes = new double[10];
        var cppSources = new int[10];
        var cppCurrents = new double[10];
        int nCppSpikes = CxxNative.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, cppTimes, cppSources, cppCurrents);

        await Assert.That(csResult.NSpikes).IsEqualTo(nCppSpikes);
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            await Assert.That(csResult.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(csResult.Sources[i]).IsEqualTo(cppSources[i]);
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
    public void BackwardSpikes_CppVsCs_ApiContract()
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
        var cppGradient = new double[nPre * nPost];
        Array.Fill(cppGradient, 0.0);
        CxxNative.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, tauMem, tauSyn, cppGradient);

        // Both should produce non-trivial gradients when post-spike error is set
        double csSum = 0.0, cppSum = 0.0;
        for (int i = 0; i < csGradient.Length; i++)
            csSum += Math.Abs(csGradient[i]);
        for (int i = 0; i < cppGradient.Length; i++)
            cppSum += Math.Abs(cppGradient[i]);
        SyncAssert.That(csSum).IsGreaterThan(0.0);
        SyncAssert.That(cppSum).IsGreaterThan(0.0);

        // C++ and C# should produce matching gradients within tolerance
        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    // ========================================================================

    /// <summary>
    /// Test Case 2: Multiple input spikes, single post-spike.
    /// </summary>
    [Test]
    public void BackwardSpikes_CppVsCs_MultipleInputs_SinglePost()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        CxxNative.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 3: Single input spike, multiple post-spikes.
    /// </summary>
    [Test]
    public void BackwardSpikes_CppVsCs_SingleInput_MultiplePosts()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        CxxNative.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 4: Multiple input spikes, multiple post-spikes (full network).
    /// </summary>
    [Test]
    public void BackwardSpikes_CppVsCs_MultipleInputs_MultiplePosts()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        CxxNative.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 5: Asymmetric network (nPre != nPost).
    /// </summary>
    [Test]
    public void BackwardSpikes_CppVsCs_AsymmetricNetwork()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        CxxNative.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 6: Zero error on all post-spikes → zero gradients.
    /// </summary>
    [Test]
    public void BackwardSpikes_CppVsCs_ZeroErrors()
    {
        int nPre = 3, nPost = 2;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 0];

        var postSpikes = new SpikesData([0.015, 0.025], [0, 1]);
        // Errors are already zero (default)

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, new ScratchBuffers());
        CxxNative.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 7: Single non-zero post-spike error (sparse error).
    /// </summary>
    [Test]
    public void BackwardSpikes_CppVsCs_SparseError()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);
        CxxNative.BackwardSpikes(inputTimes, inputSources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    // ========================================================================
    // BackwardMaxima Comparison Tests (C++ vs C#)
    // ========================================================================

    /// <summary>
    /// Test Case 1: Single input spike, single maxima.
    /// Note: C++ implementation assumes n_maxima == n_post.
    /// </summary>
    [Test]
    public void BackwardMaxima_CppVsCs_SingleSpikeSingleMaxima()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 2: Multiple inputs, single maxima.
    /// </summary>
    [Test]
    public void BackwardMaxima_CppVsCs_MultipleInputsSingleMaxima()
    {
        int nPre = 3, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2, 0.8, 0.4];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 2];

        double[] maximaTimes = [0.025];
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 3: Multiple inputs, multiple maxima (all valid).
    /// </summary>
    [Test]
    public void BackwardMaxima_CppVsCs_MultipleInputsMultipleMaxima()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 4: Maxima after last input spike.
    /// Note: C++ implementation processes maxima with negative t_vmax_bwd (doesn't skip).
    /// </summary>
    [Test]
    public void BackwardMaxima_CppVsCs_MaximaAfterLastSpike()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [0.03]; // After last input spike (0.02)
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 5: All maxima NaN → zero gradients.
    /// </summary>
    [Test]
    public void BackwardMaxima_CppVsCs_AllNaNMaxima()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [double.NaN];
        double[] maximaValues = [0.0];
        double[] maximaErrors = [0.0];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 6: Large network with random maxima.
    /// </summary>
    [Test]
    public void BackwardMaxima_CppVsCs_LargeNetwork()
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
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }

    /// <summary>
    /// Test Case 7: Different tau values (edge case).
    /// </summary>
    [Test]
    public void BackwardMaxima_CppVsCs_DifferentTaus()
    {
        int nPre = 2, nPost = 1;
        double[] w = [0.5, 0.3, 0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        double[] maximaTimes = [0.015];
        double[] maximaValues = [1.5];
        double[] maximaErrors = [0.1];

        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);

        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.01, 0.003, csGradient);
        CxxNative.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.01, 0.003, cppGradient);

        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(cppGradient[i], 1e-10);
    }
}
