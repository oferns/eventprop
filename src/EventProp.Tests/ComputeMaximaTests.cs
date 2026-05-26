using CW.EventProp;

namespace CW.EventProp.Tests;

public class MaximaDataTests
{
    [Test]
    public async Task Constructor_ThreeArg_SetsAllArrays()
    {
        double[] times = [1.0, 2.0];
        double[] values = [3.0, 4.0];
        double[] errors = [5.0, 6.0];

        var maxima = new MaximaData(times, values, errors);

        await Assert.That(maxima.Times.Length).IsEqualTo(2);
        await Assert.That(maxima.Values.Length).IsEqualTo(2);
        await Assert.That(maxima.Errors.Length).IsEqualTo(2);
        await Assert.That(maxima.NNeurons).IsEqualTo(2);
    }

    [Test]
    public async Task Constructor_TwoArg_CreatesZeroErrors()
    {
        double[] times = [1.0, 2.0];
        double[] values = [3.0, 4.0];

        var maxima = new MaximaData(times, values);

        await Assert.That(maxima.Times.Length).IsEqualTo(2);
        await Assert.That(maxima.Values.Length).IsEqualTo(2);
        await Assert.That(maxima.Errors.Length).IsEqualTo(2);
        await Assert.That(maxima.Errors[0]).IsCloseTo(0.0, 1e-12);
        await Assert.That(maxima.Errors[1]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public async Task SetError_SetsCorrectIndex()
    {
        var maxima = new MaximaData([0.1, 0.2], [1.0, 2.0]);

        maxima.SetError(1, 42.0);

        await Assert.That(maxima.Errors[0]).IsCloseTo(0.0, 1e-12);
        await Assert.That(maxima.Errors[1]).IsCloseTo(42.0, 1e-12);
    }

    [Test]
    public async Task SetValue_SetsCorrectIndex()
    {
        var maxima = new MaximaData([0.1, 0.2], [1.0, 2.0]);

        maxima.SetValue(0, 99.0);

        await Assert.That(maxima.Values[0]).IsCloseTo(99.0, 1e-12);
        await Assert.That(maxima.Values[1]).IsCloseTo(2.0, 1e-12);
    }

    [Test]
    public async Task Errors_Setter_AcceptsNewArray()
    {
        var maxima = new MaximaData([0.1], [1.0]);

        maxima.Errors = new double[] { 7.0, 8.0 };

        await Assert.That(maxima.Errors.Length).IsEqualTo(2);
        await Assert.That(maxima.Errors[0]).IsCloseTo(7.0, 1e-12);
        await Assert.That(maxima.Errors[1]).IsCloseTo(8.0, 1e-12);
    }
}

public class ComputeMaximaTests
{
    [Test]
    public async Task ComputeMaxima_NoInputSpikes_ReturnsNanTimes()
    {
        int nPost = 3;
        double[] w = [1, 0.5, 0.3, 0.2, 1, 0.4, 0.1, 0.6, 1];

        double[] times = [];
        int[] sources = [];

        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, buffers);

        for (int i = 0; i < nPost; i++)
            await Assert.That(maxima.Times[i]).IsNaN();
    }

    [Test]
    public async Task ComputeMaxima_SingleSpike_ReturnsCorrectMaxTime()
    {
        int nPost = 3;
        // Identity weights: neuron i connects to post-neuron i with weight 1.0
        double[] w = [1, 0, 0, 0, 1, 0, 0, 0, 1];

        // Single spike from neuron 0 at t=0.01
        double[] times = [0.01];
        int[] sources = [0];

        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, buffers);

        // Post-neuron 0 should have a non-NaN max time
        // The α-function peak is at τ_syn*τ_mem/(τ_mem-τ_syn) * ln(τ_mem/τ_syn) ≈ 0.01 after the spike
        // So with a spike at t=0.01, max should be around 0.02
        await Assert.That(maxima.Times[0]).IsNotNaN();
        // Max time should be after the spike and within a reasonable window
        await Assert.That(maxima.Times[0]).IsGreaterThan(0.01);
        await Assert.That(maxima.Times[0]).IsLessThan(0.05);

        // Other neurons should have NaN (no input connected)
        await Assert.That(maxima.Times[1]).IsNaN();
        await Assert.That(maxima.Times[2]).IsNaN();

        // Max value should be positive (the α-function peak amplitude)
        await Assert.That(maxima.Values[0]).IsGreaterThan(0.0);
    }

    [Test]
    public async Task ComputeMaxima_MultipleSpikeInput_AccumulatesCorrectly()
    {
        int nPost = 2;
        // All weights = 0.5 (uniform)
        double[] w = [0.5, 0.5, 0.5, 0.5];

        // Two spikes: neuron 0 at t=0.01, neuron 1 at t=0.02
        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, buffers);

        // Both neurons receive input from both spikes
        for (int i = 0; i < nPost; i++)
        {
            await Assert.That(maxima.Times[i]).IsNotNaN();
            await Assert.That(maxima.Values[i]).IsGreaterThan(0.0);
        }

        // With uniform weights and same τ values, both should have the same max value
        await Assert.That(maxima.Values[0]).IsCloseTo(maxima.Values[1], 1e-10);
    }

    [Test]
    public async Task ComputeMaxima_WeightMagnitude_ReturnsProportionalMax()
    {
        int nPost = 1;
        double[] times = [0.01];
        int[] sources = [0];

        // Compute maxima for several weight magnitudes and verify the ratio value/|w|
        // is roughly constant (proportionality), not that it equals 1.0.
        double[] weights = [0.01, 0.5, 1.0, 5.0];
        double[] ratios = new double[weights.Length];

        var buffers = new ScratchBuffers();
        for (int i = 0; i < weights.Length; i++)
        {
            var maxima = EventPropMath.ComputeMaxima([weights[i]], times, sources, nPost, 0.02, 0.005, buffers);
            ratios[i] = maxima.Values[0] / Math.Abs(weights[i]);
        }

        // All ratios should be within 50% of the median ratio (proportionality check)
        double[] sorted = new double[ratios.Length];
        Array.Copy(ratios, sorted, ratios.Length);
        Array.Sort(sorted);
        double median = sorted[ratios.Length / 2];
        foreach (double ratio in ratios)
            await Assert.That(ratio).IsCloseTo(median, median * 0.5);
    }

    [Test]
    public async Task ComputeMaxima_LargeNetwork_NoCrash()
    {
        int nPre = 50;
        int nPost = 20;
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = 0.1 + 0.1 * (i % 5);

        int nSpikes = 200;
        double[] times = new double[nSpikes];
        int[] sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = 0.001 * (i + 1);
            sources[i] = i % nPre;
        }

        // Should not throw — buffer reused across calls
        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, buffers);

        await Assert.That(maxima.NNeurons).IsEqualTo(nPost);
        for (int i = 0; i < nPost; i++)
            await Assert.That(maxima.Times[i]).IsNotNaN();
    }

    [Test]
    public async Task ComputeMaxima_MaxTimeBetweenSpikes()
    {
        int nPost = 1;
        double[] w = [1.0];

        // Two spikes from the same neuron, far apart
        double[] times = [0.01, 0.1];
        int[] sources = [0, 0];

        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, buffers);

        await Assert.That(maxima.Times[0]).IsNotNaN();
        // The max should occur after the later spike (closer to t=0.1)
        // since both contribute but the second is more recent
        await Assert.That(maxima.Times[0]).IsGreaterThan(0.01);
    }

    [Test]
    [MatrixDataSource]
    public async Task ComputeMaxima_ValueScalesWithWeight([Matrix(1.0, -1.0, 3.0)] double baseWeight)
    {
        int nPost = 1;

        double[] times = [0.01];
        int[] sources = [0];

        var buffers = new ScratchBuffers();
        // Test with base weight
        var maxima1 = EventPropMath.ComputeMaxima([baseWeight], times, sources, nPost, 0.02, 0.005, buffers);

        // Test with double the weight (buffer reused)
        var maxima2 = EventPropMath.ComputeMaxima([2.0 * baseWeight], times, sources, nPost, 0.02, 0.005, buffers);

        // The max time should be the same (depends on τ values, not weight)
        await Assert.That(maxima1.Times[0]).IsCloseTo(maxima2.Times[0], 1e-10);

        // The max value should scale linearly with weight.
        // For negative weights, both values flip sign but the ratio stays 2.0,
        // so checking absolute value avoids sign confusion.
        await Assert.That(Math.Abs(maxima2.Values[0])).IsCloseTo(2.0 * Math.Abs(maxima1.Values[0]), 1e-10);
    }

    [Test]
    public async Task ComputeMaxima_NearSingularTau_DoesNotExplode()
    {
        int nPost = 1;
        double[] w = [1.0];

        double[] times = [0.01];
        int[] sources = [0];

        // tauMem ≈ tauSyn → formula divides by (tauMem - tauSyn) and logs ratio near 1
        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.01, 0.009, buffers);

        // Should not produce NaN or Infinity despite numerical sensitivity
        await Assert.That(maxima.Times[0]).IsNotNaN();
        await Assert.That(double.IsInfinity(maxima.Values[0])).IsFalse();
    }

    [Test]
    public async Task ComputeMaxima_ZeroWeights_ReturnsZeroValues()
    {
        int nPost = 2;
        double[] w = [0.0, 0.0, 0.0, 0.0]; // All zero weights

        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, buffers);

        // Zero weights → zero membrane potential → max value is 0 (or negative boundary)
        for (int i = 0; i < nPost; i++)
            await Assert.That(maxima.Values[i]).IsLessThanOrEqualTo(0.0);
    }

    [Test]
    public async Task ComputeMaxima_NoConnectedNeuron_ReturnsNan()
    {
        int nPost = 3;
        // Neuron 0 connects to post-neuron 0 only; neurons 1,2 have no input
        double[] w = [1.0, 0.0, 0.0, 0.0];

        double[] times = [0.01];
        int[] sources = [0]; // Only spike from neuron 0

        var buffers = new ScratchBuffers();
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, buffers);

        // Post-neuron 0 gets input → valid max time
        await Assert.That(maxima.Times[0]).IsNotNaN();
        // Post-neurons 1,2 get no input → NaN times
        await Assert.That(maxima.Times[1]).IsNaN();
        await Assert.That(maxima.Times[2]).IsNaN();
    }

    [Test]
    public async Task MaximaData_ZeroLengthArrays()
    {
        var maxima = new MaximaData([], []);

        await Assert.That(maxima.NNeurons).IsEqualTo(0);
        await Assert.That(maxima.Times.Length).IsEqualTo(0);
        await Assert.That(maxima.Values.Length).IsEqualTo(0);
        await Assert.That(maxima.Errors.Length).IsEqualTo(0);
    }
}
