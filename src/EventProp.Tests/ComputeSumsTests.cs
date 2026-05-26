using CW.EventProp;

namespace CW.EventProp.Tests;

public class ComputeSumsTests
{
    [Test]
    public async Task ComputeSums_IdentityWeights_ZeroInput_ReturnsZeroMatrices()
    {
        int nPost = 3;
        double[] w = [1, 0, 0, 0, 1, 0, 0, 0, 1];

        double[] times = [];
        int[] sources = [];

        var sum0 = new double[(times.Length + 1) * nPost];
        var sum1 = new double[(times.Length + 1) * nPost];
        var expInputMem = new double[times.Length];
        var expInputSyn = new double[times.Length];

        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, sum0, sum1, nPost,
            expInputMem.AsSpan(), expInputSyn.AsSpan());

        for (int i = 0; i < sum0.Length; i++)
            await Assert.That(sum0[i]).IsEqualTo(0.0);

        for (int i = 0; i < sum1.Length; i++)
            await Assert.That(sum1[i]).IsEqualTo(0.0);
    }

    [Test]
    public async Task ComputeSums_IdentityWeights_SingleSpike_ReturnsCorrectFirstRow()
    {
        int nPost = 3;
        double[] w = [1, 0, 0, 0, 1, 0, 0, 0, 1];

        double[] times = [0.01];
        int[] sources = [0];

        var sum0 = new double[(times.Length + 1) * nPost];
        var sum1 = new double[(times.Length + 1) * nPost];
        var expInputMem = new double[times.Length];
        var expInputSyn = new double[times.Length];

        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, sum0, sum1, nPost,
            expInputMem.AsSpan(), expInputSyn.AsSpan());
        for (int j = 0; j < nPost; j++)
        {
            await Assert.That(sum0[j]).IsEqualTo(0.0);
            await Assert.That(sum1[j]).IsEqualTo(0.0);
        }

        // Row 1: exp(t/tau_syn) * w[0,:] for sum0, exp(t/tau_mem) * w[0,:] for sum1
        double expSyn = Math.Exp(0.01 / 0.005);
        double expMem = Math.Exp(0.01 / 0.02);

        // Post neuron 0 gets the spike (identity weights)
        await Assert.That(sum0[nPost]).IsEqualTo(expSyn).Within(1e-12);
        await Assert.That(sum1[nPost]).IsEqualTo(expMem).Within(1e-12);

        // Other neurons get zero (identity weights)
        await Assert.That(sum0[1]).IsEqualTo(0.0);
        await Assert.That(sum0[2]).IsEqualTo(0.0);
    }

    [Test]
    public async Task ComputeSums_UniformWeights_CorrectAccumulation()
    {
        int nPost = 3;
        double[] w = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var sum0 = new double[(times.Length + 1) * nPost];
        var sum1 = new double[(times.Length + 1) * nPost];
        var expInputMem = new double[times.Length];
        var expInputSyn = new double[times.Length];

        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, sum0, sum1, nPost,
            expInputMem.AsSpan(), expInputSyn.AsSpan());
        for (int j = 0; j < nPost; j++)
        {
            await Assert.That(sum0[j]).IsEqualTo(0.0);
            await Assert.That(sum1[j]).IsEqualTo(0.0);
        }

        // Row 1: after spike from neuron 0 at t=0.01
        double expSyn1 = Math.Exp(0.01 / 0.005);
        double expMem1 = Math.Exp(0.01 / 0.02);
        for (int j = 0; j < nPost; j++)
        {
            await Assert.That(sum0[nPost + j]).IsEqualTo(0.5 * expSyn1).Within(1e-12);
            await Assert.That(sum1[nPost + j]).IsEqualTo(0.5 * expMem1).Within(1e-12);
        }

        // Row 2: after spike from neuron 1 at t=0.02
        double expSyn2 = Math.Exp(0.02 / 0.005);
        double expMem2 = Math.Exp(0.02 / 0.02);
        for (int j = 0; j < nPost; j++)
        {
            double expectedSum0 = 0.5 * expSyn1 + 0.5 * expSyn2;
            double expectedSum1 = 0.5 * expMem1 + 0.5 * expMem2;
            await Assert.That(sum0[2 * nPost + j]).IsEqualTo(expectedSum0).Within(1e-12);
            await Assert.That(sum1[2 * nPost + j]).IsEqualTo(expectedSum1).Within(1e-12);
        }
    }

    [Test]
    public async Task ComputeSums_LargeMatrix_NoCrash()
    {
        int nPre = 200;
        int nPost = 100;
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = 0.1;

        int nSpikes = 500;
        double[] times = new double[nSpikes];
        int[] sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = 0.001 * (i + 1);
            sources[i] = i % nPre;
        }

        var sum0 = new double[(nSpikes + 1) * nPost];
        var sum1 = new double[(nSpikes + 1) * nPost];
        var expInputMem = new double[nSpikes];
        var expInputSyn = new double[nSpikes];

        // Should not throw
        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, sum0, sum1, nPost,
            expInputMem.AsSpan(), expInputSyn.AsSpan());

        // Verify last row is non-zero (accumulated many spikes)
        double lastRowSum = 0.0;
        for (int j = 0; j < nPost; j++)
            lastRowSum += sum0[nSpikes * nPost + j];

        await Assert.That(lastRowSum).IsGreaterThan(0.0);
    }

    [Test]
    public async Task ComputeSums_NegativeWeights_ReturnsNegativeAccumulation()
    {
        int nPost = 2;
        // Negative (inhibitory) weights
        double[] w = [-0.5, -0.5, -0.5, -0.5];

        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var sum0 = new double[(times.Length + 1) * nPost];
        var sum1 = new double[(times.Length + 1) * nPost];
        var expInputMem = new double[times.Length];
        var expInputSyn = new double[times.Length];

        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, sum0, sum1, nPost,
            expInputMem.AsSpan(), expInputSyn.AsSpan());
        for (int i = nPost; i < sum0.Length; i++)
        {
            await Assert.That(sum0[i]).IsLessThan(0.0);
            await Assert.That(sum1[i]).IsLessThan(0.0);
        }
    }

    [Test]
    public async Task ComputeSums_SameSourceMultipleSpikes_AccumulatesCorrectly()
    {
        int nPost = 2;
        double[] w = [1.0, 1.0, 0.5, 0.5]; // neuron 0 has weight 1.0, neuron 1 has 0.5

        // Two spikes from the SAME source neuron at different times
        double[] times = [0.01, 0.02];
        int[] sources = [0, 0];

        var sum0 = new double[(times.Length + 1) * nPost];
        var sum1 = new double[(times.Length + 1) * nPost];
        var expInputMem = new double[times.Length];
        var expInputSyn = new double[times.Length];

        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, sum0, sum1, nPost,
            expInputMem.AsSpan(), expInputSyn.AsSpan());
        double expSyn1 = Math.Exp(0.01 / 0.005);
        double expMem1 = Math.Exp(0.01 / 0.02);
        for (int j = 0; j < nPost; j++)
        {
            await Assert.That(sum0[nPost + j]).IsEqualTo(expSyn1).Within(1e-12);
            await Assert.That(sum1[nPost + j]).IsEqualTo(expMem1).Within(1e-12);
        }

        // Row 2: after second spike from neuron 0 at t=0.02 — should accumulate
        double expSyn2 = Math.Exp(0.02 / 0.005);
        double expMem2 = Math.Exp(0.02 / 0.02);
        for (int j = 0; j < nPost; j++)
        {
            await Assert.That(sum0[2 * nPost + j]).IsEqualTo(expSyn1 + expSyn2).Within(1e-12);
            await Assert.That(sum1[2 * nPost + j]).IsEqualTo(expMem1 + expMem2).Within(1e-12);
        }
    }
}
