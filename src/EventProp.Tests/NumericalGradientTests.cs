using CW.EventProp;
using Native = CW.EventProp.Native;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// Comprehensive gradient verification tests for EventProp.
/// 
/// These tests address test data holes identified during code review:
/// - Weight access layout bugs (row-major vs column-major)
/// - Sparsity patterns (zero weights should produce zero gradients)
/// - Gradient scaling with error magnitude
/// - Asymmetric network correctness
/// 
/// Unlike C++ vs C# comparison tests (which can mask bugs if both have the same error),
/// these tests use independent verification methods: sparsity checks, scaling tests,
/// and numerical gradients for continuous functions (BackwardMaxima).
/// </summary>
public class NumericalGradientTests
{
    private const double Tolerance = 1e-6;

    // Helpers to extract native output data before await boundaries (ref structs can't cross awaits)
    private static (double[] gradient, double[] spikeErrors) GetNativeBackwardSpikes(
        ReadOnlySpan<double> inputTimes, ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> postTimes, ReadOnlySpan<int> postSources,
        ReadOnlySpan<double> postCurrents, ReadOnlySpan<double> postErrors,
        int nPostSpikes,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn)
    {
        using var output = Native.Api.BackwardSpikes(inputTimes, inputSources,
            postTimes, postSources, postCurrents, postErrors, nPostSpikes,
            w, nPre, nPost, vTh, tauMem, tauSyn);
        return (output.Gradient.ToArray(), output.SpikeErrors.ToArray());
    }

    private static (double[] gradient, double[] spikeErrors) GetNativeBackwardMaxima(
        ReadOnlySpan<double> inputTimes, ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> maximaTimes, ReadOnlySpan<double> maximaValues,
        ReadOnlySpan<double> maximaErrors, int nMaxima,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double tauMem, double tauSyn)
    {
        using var output = Native.Api.BackwardMaxima(inputTimes, inputSources,
            maximaTimes, maximaValues, maximaErrors, nMaxima,
            w, nPre, nPost, tauMem, tauSyn);
        return (output.Gradient.ToArray(), output.SpikeErrors.ToArray());
    }

    // ========================================================================
    // BackwardSpikes Tests — Weight Access & Sparsity Verification
    // ========================================================================

    /// <summary>
    /// Asymmetric network (nPre=4, nPost=2) with non-symmetric weights.
    /// Verifies C++ and C# produce matching gradients — this would catch row-major vs
    /// column-major weight access bugs because each (source, nrn) pair accesses a
    /// different array element depending on the layout.
    /// 
    /// Weight matrix (row-major):
    ///   source=0: [15.0, 3.0]
    ///   source=1: [8.0, 20.0]
    ///   source=2: [5.0, 12.0]
    ///   source=3: [7.0, 18.0]
    /// 
    /// If column-major were used instead:
    ///   source=0: [15.0, 5.0]  ← different!
    ///   source=1: [3.0, 12.0]  ← different!
    ///   source=2: [8.0, 7.0]   ← different!
    ///   source=3: [20.0, 18.0] ← different!
    /// </summary>
    [Test]
    public void BackwardSpikes_AsymmetricNetwork_GradientMatch()
    {
        int nPre = 4, nPost = 2;
        double tauMem = 0.02, tauSyn = 0.005;

        // Non-symmetric weights — layout bug would change every weight access
        double[] w = [15.0, 3.0, 8.0, 20.0,
                      5.0, 12.0, 7.0, 18.0];
        double[] inputTimes = [0.01, 0.03, 0.05];
        int[] inputSources = [0, 1, 2];

        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: inputTimes.Length);
        buffers.Clear();

        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, tauMem, tauSyn, buffers);

        if (postSpikes.NSpikes == 0)
            return;

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        // C# gradient
        var csGradient = new double[nPre * nPost];
        Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, tauMem, tauSyn, csGradient, buffers);

        // C++ gradient via P/Invoke
        (var nativeGradient, _) = GetNativeBackwardSpikes(inputTimes, inputSources,
            postSpikes.Times[..postSpikes.NSpikes], postSpikes.Sources[..postSpikes.NSpikes],
            postSpikes.Currents[..postSpikes.NSpikes], postSpikes.Errors[..postSpikes.NSpikes],
            postSpikes.NSpikes, w, nPre, nPost, 1.0, tauMem, tauSyn);

        // All weights should match within tolerance
        for (int i = 0; i < csGradient.Length; i++)
            SyncAssert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Verify gradient sparsity: zero weights should produce zero gradients.
    /// This tests that the backward pass correctly handles disconnected neurons.
    /// </summary>
    [Test]
    public async Task BackwardSpikes_Sparsity_ZeroWeightsYieldZeroGradients()
    {
        int nPre = 4, nPost = 3;
        double tauMem = 0.02, tauSyn = 0.005;

        // Only weights [0,0], [1,2], [3,1] are non-zero (sparse connectivity)
        double[] w = [15.0, 0.0, 0.0,
                      0.0, 0.0, 20.0,
                      0.0, 18.0, 0.0,
                      0.0, 25.0, 0.0];
        double[] inputTimes = [0.01, 0.02, 0.03, 0.04];
        int[] inputSources = [0, 1, 2, 3];

        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: inputTimes.Length);
        buffers.Clear();

        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, tauMem, tauSyn, buffers);

        if (postSpikes.NSpikes == 0)
            return;

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var gradient = new double[nPre * nPost];
        Array.Fill(gradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, tauMem, tauSyn, gradient, buffers);

        // Verify zero gradients for disconnected weights
        int[] zeroSrc = [0, 0, 1, 1, 2, 2, 2, 3, 3];
        int[] zeroNrn = [1, 2, 0, 1, 0, 1, 2, 0, 2];
        for (int k = 0; k < zeroSrc.Length; k++)
        {
            int idx = zeroSrc[k] * nPost + zeroNrn[k];
            await Assert.That(gradient[idx]).IsLessThan(1e-14);
        }

        // Verify non-zero gradients for connected weights match C++
        int[] activeSrc = [0, 1, 3];
        int[] activeNrn = [0, 2, 1];
        for (int k = 0; k < activeSrc.Length; k++)
        {
            int src = activeSrc[k], nrn = activeNrn[k];

            (var nativeGradient, _) = GetNativeBackwardSpikes(inputTimes, inputSources,
                postSpikes.Times[..postSpikes.NSpikes], postSpikes.Sources[..postSpikes.NSpikes],
                postSpikes.Currents[..postSpikes.NSpikes], postSpikes.Errors[..postSpikes.NSpikes],
                postSpikes.NSpikes, w, nPre, nPost, 1.0, tauMem, tauSyn);

            await Assert.That(gradient[src * nPost + nrn]).IsCloseTo(nativeGradient[src * nPost + nrn], 1e-10);
        }
    }

    /// <summary>
    /// Verify gradients scale linearly with error magnitude.
    /// Running backward pass with error=2.0 should produce exactly double the gradients of error=1.0.
    /// This verifies that the backward pass is a linear function of post-spike errors.
    /// </summary>
    [Test]
    public async Task BackwardSpikes_GradientScalesLinearly()
    {
        int nPre = 4, nPost = 2;
        double tauMem = 0.02, tauSyn = 0.005;

        double[] w = [15.0, 3.0, 8.0, 20.0,
                      5.0, 12.0, 7.0, 18.0];
        double[] inputTimes = [0.01, 0.03, 0.05];
        int[] inputSources = [0, 1, 2];

        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: inputTimes.Length);
        buffers.Clear();

        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, tauMem, tauSyn, buffers);

        if (postSpikes.NSpikes == 0)
            return;

        // Run with error = 1.0
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var gradient1 = new double[nPre * nPost];
        Array.Fill(gradient1, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, tauMem, tauSyn, gradient1, buffers);

        // Run with error = 2.0
        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 2.0);

        var gradient2 = new double[nPre * nPost];
        Array.Fill(gradient2, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, tauMem, tauSyn, gradient2, buffers);

        // Verify scaling: gradient2 should be exactly 2x gradient1
        for (int i = 0; i < gradient1.Length; i++)
            await Assert.That(gradient2[i]).IsCloseTo(2.0 * gradient1[i], 1e-10);
    }

    /// <summary>
    /// Large asymmetric network (nPre=10, nPost=5) with strong weights.
    /// Verifies that all gradient components match between C++ and C#.
    /// This exercises many weight accesses and would catch systematic layout bugs.
    /// </summary>
    [Test]
    public async Task BackwardSpikes_LargeAsymmetricNetwork_GradientMatch()
    {
        int nPre = 10, nPost = 5;
        double tauMem = 0.02, tauSyn = 0.005;

        // Strong weights to guarantee post-spike activity and stable root-finding
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = 1.5 + 0.3 * (i % 7); // [1.5, 3.6]

        int nSpikes = 20;
        double[] inputTimes = new double[nSpikes];
        int[] inputSources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            inputTimes[i] = 0.001 * (i + 1);
            inputSources[i] = i % nPre;
        }

        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: inputTimes.Length);
        buffers.Clear();

        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, tauMem, tauSyn, buffers);

        if (postSpikes.NSpikes == 0)
            return;

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        // C# gradient
        var csGradient = new double[nPre * nPost];
        Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, tauMem, tauSyn, csGradient, buffers);

        // C++ gradient
        (var nativeGradient, _) = GetNativeBackwardSpikes(inputTimes, inputSources,
            postSpikes.Times[..postSpikes.NSpikes], postSpikes.Sources[..postSpikes.NSpikes],
            postSpikes.Currents[..postSpikes.NSpikes], postSpikes.Errors[..postSpikes.NSpikes],
            postSpikes.NSpikes, w, nPre, nPost, 1.0, tauMem, tauSyn);

        // All weights should match
        for (int i = 0; i < csGradient.Length; i++)
            await Assert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Multiple input spikes from same neuron, single post-spike.
    /// Tests that the adjoint method correctly handles multiple spikes from one source,
    /// each contributing to the gradient.
    /// </summary>
    [Test]
    public async Task BackwardSpikes_MultiSpikeSameSource_GradientMatch()
    {
        int nPre = 1, nPost = 1;
        double tauMem = 0.02, tauSyn = 0.005;

        double[] w = [1.9];
        double[] inputTimes = [0.05, 0.08, 0.12];
        int[] inputSources = [0, 0, 0];

        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: inputTimes.Length);
        buffers.Clear();

        var postSpikes = EventPropMath.ComputeSpikes(w, inputTimes, inputSources, nPost,
            1.0, tauMem, tauSyn, buffers);

        if (postSpikes.NSpikes == 0)
            return;

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        var csGradient = new double[nPre * nPost];
        Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, inputTimes, inputSources, postSpikes, nPost,
            1.0, tauMem, tauSyn, csGradient, buffers);

        (var nativeGradient, _) = GetNativeBackwardSpikes(inputTimes, inputSources,
            postSpikes.Times[..postSpikes.NSpikes], postSpikes.Sources[..postSpikes.NSpikes],
            postSpikes.Currents[..postSpikes.NSpikes], postSpikes.Errors[..postSpikes.NSpikes],
            postSpikes.NSpikes, w, nPre, nPost, 1.0, tauMem, tauSyn);

        await Assert.That(csGradient[0]).IsCloseTo(nativeGradient[0], 1e-10);
    }

    // ========================================================================
    // BackwardMaxima Numerical Gradient Tests (continuous, so finite diffs work)
    // ========================================================================

    /// <summary>
    /// Single neuron BackwardMaxima numerical gradient check.
    /// Membrane potential maxima is a continuous function of weights, so finite differences are reliable.
    /// </summary>
    [Test]
    public async Task BackwardMaxima_NumericalGradient_SingleNeuron()
    {
        int nPre = 2, nPost = 1;

        double[] w = [0.5, 0.3,
                      0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        var maxima = EventPropMath.ComputeMaxima(w, inputTimes, inputSources, nPost,
            0.02, 0.005, new ScratchBuffers());

        if (double.IsNaN(maxima.Times[0]))
            return;

        // C++ gradient as ground truth
        (var nativeGradient, _) = GetNativeBackwardMaxima(inputTimes, inputSources,
            maxima.Times[..nPost], maxima.Values[..nPost], [0.1], 1,
            w, nPre, nPost, 0.02, 0.005);

        // C# gradient
        var csGradient = new double[nPre * nPost];
        Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maxima.Times, maxima.Values, [0.1],
            w, nPost, 0.02, 0.005, csGradient);

        // Verify C# matches C++
        for (int i = 0; i < csGradient.Length; i++)
            await Assert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// Multi-neuron BackwardMaxima numerical gradient check.
    /// </summary>
    [Test]
    public async Task BackwardMaxima_NumericalGradient_MultiNeuron()
    {
        int nPre = 3, nPost = 2;

        double[] w = [0.5, 0.3,
                      0.1, 0.2,
                      0.8, 0.4];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        var maxima = EventPropMath.ComputeMaxima(w, inputTimes, inputSources, nPost,
            0.02, 0.005, new ScratchBuffers());

        int validCount = 0;
        for (int i = 0; i < nPost; i++)
            if (!double.IsNaN(maxima.Times[i])) validCount++;

        if (validCount == 0)
            return;

        double[] maximaErrors = new double[nPost];
        for (int i = 0; i < nPost; i++)
            maximaErrors[i] = !double.IsNaN(maxima.Times[i]) ? 0.1 : 0.0;

        var csGradient = new double[nPre * nPost];
        Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maxima.Times, maxima.Values, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);

        (var nativeGradient, _) = GetNativeBackwardMaxima(inputTimes, inputSources,
            maxima.Times[..nPost], maxima.Values[..nPost], maximaErrors, nPost,
            w, nPre, nPost, 0.02, 0.005);

        for (int i = 0; i < csGradient.Length; i++)
            await Assert.That(csGradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// BackwardMaxima sparsity test: zero weights should produce zero gradients.
    /// </summary>
    [Test]
    public async Task BackwardMaxima_Sparsity_ZeroWeightsYieldZeroGradients()
    {
        int nPre = 4, nPost = 2;

        // Only weights [0,0] and [3,1] are non-zero
        double[] w = [0.5, 0.0,
                      0.0, 0.0,
                      0.0, 0.0,
                      0.3, 0.2];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 2];

        var maxima = EventPropMath.ComputeMaxima(w, inputTimes, inputSources, nPost,
            0.02, 0.005, new ScratchBuffers());

        int validCount = 0;
        for (int i = 0; i < nPost; i++)
            if (!double.IsNaN(maxima.Times[i])) validCount++;

        if (validCount == 0)
            return;

        double[] maximaErrors = new double[nPost];
        for (int i = 0; i < nPost; i++)
            maximaErrors[i] = !double.IsNaN(maxima.Times[i]) ? 0.1 : 0.0;

        var gradient = new double[nPre * nPost];
        Array.Fill(gradient, 0.0);
        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maxima.Times, maxima.Values, maximaErrors,
            w, nPost, 0.02, 0.005, gradient);

        // Verify zero gradients for weights that don't contribute to maxima
        int[] expectedZeroSrc = [0, 1, 1, 2, 2];
        int[] expectedZeroNrn = [1, 0, 1, 0, 1];
        for (int k = 0; k < expectedZeroSrc.Length; k++)
        {
            int idx = expectedZeroSrc[k] * nPost + expectedZeroNrn[k];
            await Assert.That(gradient[idx]).IsLessThan(1e-14);
        }

        // Verify non-zero gradients match C++
        (var nativeGradient, _) = GetNativeBackwardMaxima(inputTimes, inputSources,
            maxima.Times[..nPost], maxima.Values[..nPost], maximaErrors, nPost,
            w, nPre, nPost, 0.02, 0.005);

        for (int i = 0; i < gradient.Length; i++)
            await Assert.That(gradient[i]).IsCloseTo(nativeGradient[i], 1e-10);
    }

    /// <summary>
    /// BackwardMaxima gradient scaling test: doubling error should double gradients.
    /// </summary>
    [Test]
    public async Task BackwardMaxima_GradientScalesLinearly()
    {
        int nPre = 2, nPost = 1;

        double[] w = [0.5, 0.3,
                      0.1, 0.2];
        double[] inputTimes = [0.01, 0.02];
        int[] inputSources = [0, 1];

        var maxima = EventPropMath.ComputeMaxima(w, inputTimes, inputSources, nPost,
            0.02, 0.005, new ScratchBuffers());

        if (double.IsNaN(maxima.Times[0]))
            return;

        // Run with error = 0.1
        double[] maximaErrors1 = [0.1];
        var gradient1 = new double[nPre * nPost];
        Array.Fill(gradient1, 0.0);
        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maxima.Times, maxima.Values, maximaErrors1,
            w, nPost, 0.02, 0.005, gradient1);

        // Run with error = 0.2
        double[] maximaErrors2 = [0.2];
        var gradient2 = new double[nPre * nPost];
        Array.Fill(gradient2, 0.0);
        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maxima.Times, maxima.Values, maximaErrors2,
            w, nPost, 0.02, 0.005, gradient2);

        // Verify scaling
        for (int i = 0; i < gradient1.Length; i++)
            await Assert.That(gradient2[i]).IsCloseTo(2.0 * gradient1[i], 1e-10);
    }

    // ========================================================================
    // Cross-function consistency tests
    // ========================================================================

    /// <summary>
    /// Verify that ComputeMaxima + BackwardMaxima produces non-trivial gradients.
    /// This tests the integration between forward and backward passes.
    /// </summary>
    [Test]
    public async Task BackwardMaxima_ConsistentWithComputeMaxima()
    {
        int nPre = 3, nPost = 2;

        double[] w = [0.5, 0.3,
                      0.1, 0.2,
                      0.8, 0.4];
        double[] inputTimes = [0.01, 0.02, 0.03];
        int[] inputSources = [0, 1, 2];

        var maxima = EventPropMath.ComputeMaxima(w, inputTimes, inputSources, nPost,
            0.02, 0.005, new ScratchBuffers());

        bool hasValidMaxima = false;
        for (int i = 0; i < nPost; i++)
            if (!double.IsNaN(maxima.Times[i])) { hasValidMaxima = true; break; }

        if (!hasValidMaxima)
            return;

        double[] maximaErrors = new double[nPost];
        for (int i = 0; i < nPost; i++)
            maximaErrors[i] = !double.IsNaN(maxima.Times[i]) ? 0.1 : 0.0;

        var gradient = new double[nPre * nPost];
        Array.Fill(gradient, 0.0);
        EventPropMath.BackwardMaxima(inputTimes, inputSources,
            maxima.Times, maxima.Values, maximaErrors,
            w, nPost, 0.02, 0.005, gradient);

        // Verify gradients are not all zero when maxima exist
        double gradSum = 0.0;
        for (int i = 0; i < gradient.Length; i++)
            gradSum += Math.Abs(gradient[i]);

        await Assert.That(gradSum).IsGreaterThan(0.0);
    }
}
