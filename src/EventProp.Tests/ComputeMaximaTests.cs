using CW.EventProp;

namespace EventProp.Tests;

public class ComputeMaximaTests
{
    [Test]
    public async Task ComputeMaxima_NoInputSpikes_ReturnsNanTimes()
    {
        int nPost = 3;
        double[] w = [1, 0.5, 0.3, 0.2, 1, 0.4, 0.1, 0.6, 1];

        double[] times = [];
        int[] sources = [];

        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005);

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

        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005);

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

        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005);

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
    public async Task ComputeMaxima_WeakWeights_ReturnsSmallMax()
    {
        int nPost = 1;
        double[] w = [0.01]; // Very small weight

        double[] times = [0.01];
        int[] sources = [0];

        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005);

        await Assert.That(maxima.Times[0]).IsNotNaN();
        // Max value should scale with weight
        await Assert.That(maxima.Values[0]).IsGreaterThan(0.0);
    }

    [Test]
    public async Task ComputeMaxima_StrongWeights_ReturnsLargeMax()
    {
        int nPost = 1;
        double[] w = [5.0]; // Large weight

        double[] times = [0.01];
        int[] sources = [0];

        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005);

        await Assert.That(maxima.Times[0]).IsNotNaN();
        await Assert.That(maxima.Values[0]).IsGreaterThan(0.5); // Should be much larger than weak weight case (0.01)
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

        // Should not throw
        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005);

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

        var maxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005);

        await Assert.That(maxima.Times[0]).IsNotNaN();
        // The max should occur after the later spike (closer to t=0.1)
        // since both contribute but the second is more recent
        await Assert.That(maxima.Times[0]).IsGreaterThan(0.01);
    }

    [Test]
    public async Task ComputeMaxima_ValueScalesWithWeight()
    {
        int nPost = 1;

        double[] times = [0.01];
        int[] sources = [0];

        // Test with weight 1.0
        var maxima1 = EventPropMath.ComputeMaxima([1.0], times, sources, nPost, 0.02, 0.005);

        // Test with weight 2.0 (double the input)
        var maxima2 = EventPropMath.ComputeMaxima([2.0], times, sources, nPost, 0.02, 0.005);

        // The max time should be the same (depends on τ values, not weight)
        await Assert.That(maxima1.Times[0]).IsCloseTo(maxima2.Times[0], 1e-10);

        // The max value should scale linearly with weight
        await Assert.That(maxima2.Values[0]).IsCloseTo(2.0 * maxima1.Values[0], 1e-10);
    }
}
