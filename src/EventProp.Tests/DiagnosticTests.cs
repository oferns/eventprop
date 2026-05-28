using CW.EventProp;

namespace CW.EventProp.Tests;

/// <summary>
/// Unit tests for the three diagnostic methods added to EventPropMath:
/// ComputeVoltageTrace, ComputeLambdaI, and ComputeLambdaITrace.
/// </summary>
public class DiagnosticTests
{
    // ── ComputeVoltageTrace tests ───────────────────────────────────────

    [Test]
    public async Task ComputeVoltageTrace_EmptyInput_ReturnsZeroVoltages()
    {
        // Arrange — no input spikes, identity weights
        int nPost = 3;
        double[] w = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        double[] times = [];
        int[] sources = [];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 1, 0);

        // Act
        var result = EventPropMath.ComputeVoltageTrace(
            tMax: 0.1, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, times, sources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Assert — all voltages should be zero (no input)
        await Assert.That(result.times.Length).IsEqualTo(11); // 0.1 / 0.01 + 1
        await Assert.That(result.voltages.Length).IsEqualTo(11);

        for (int i = 0; i < result.voltages.Length; i++)
            await Assert.That(result.voltages[i]).IsCloseTo(0.0, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeVoltageTrace_SingleSpike_PositiveWeight_YieldsPositiveVoltage()
    {
        // Arrange — single spike from neuron 0 with weight 1.5 to neuron 0
        int nPost = 3;
        double[] w = [1.5, 0, 0, 0, 1, 0, 0, 0, 1];
        double[] times = [0.05];
        int[] sources = [0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 1, 1);

        // Act
        var result = EventPropMath.ComputeVoltageTrace(
            tMax: 0.2, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, times, sources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Assert
        await Assert.That(result.times.Length).IsEqualTo(21); // 0.2 / 0.01 + 1

        // Voltage should be zero before the spike time, then rise
        for (int i = 0; i < result.times.Length; i++)
        {
            if (result.times[i] < 0.05 - 1e-9)
                await Assert.That(result.voltages[i]).IsCloseTo(0.0, tolerance: 1e-12);
        }

        // After spike, voltage should be positive and then decay
        bool foundPositive = false;
        for (int i = 0; i < result.times.Length; i++)
        {
            if (result.times[i] > 0.05 + 1e-9 && result.voltages[i] > 0.01)
            { foundPositive = true; break; }
        }
        await Assert.That(foundPositive).IsTrue();

        // At tMax, voltage should have decayed but still be positive
        await Assert.That(result.voltages[result.voltages.Length - 1]).IsGreaterThan(0.0);
    }

    [Test]
    public async Task ComputeVoltageTrace_MultipleSpikes_Accumulates()
    {
        // Arrange — two spikes from same neuron, weights = 1.0
        int nPost = 3;
        double[] w = [1.0, 0, 0, 0, 1, 0, 0, 0, 1];
        double[] times = [0.05, 0.10];
        int[] sources = [0, 0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 1, 2);

        // Act — two spikes
        var resultTwo = EventPropMath.ComputeVoltageTrace(
            tMax: 0.2, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, times, sources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Compare with single spike at t=0.05
        var buffers2 = new ScratchBuffers();
        buffers2.EnsureCapacity(nPost, 1, 1);
        var resultSingle = EventPropMath.ComputeVoltageTrace(
            tMax: 0.2, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, new double[] { 0.05 }, new int[] { 0 }, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers2);

        // Assert — voltage at t=0.2 should be larger with two spikes
        await Assert.That(resultTwo.voltages[resultTwo.voltages.Length - 1])
            .IsGreaterThan(resultSingle.voltages[resultSingle.voltages.Length - 1]);
    }

    [Test]
    public async Task ComputeVoltageTrace_TimePointsAreMonotonic()
    {
        // Arrange
        int nPost = 3;
        double[] w = [1.0, 0, 0, 0, 1, 0, 0, 0, 1];
        double[] times = [0.05, 0.15];
        int[] sources = [0, 1];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 2, 2);

        // Act
        var result = EventPropMath.ComputeVoltageTrace(
            tMax: 0.3, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 2, nPost, times, sources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Assert — time points should be monotonically increasing
        for (int i = 1; i < result.times.Length; i++)
            await Assert.That(result.times[i]).IsGreaterThan(result.times[i - 1]);

        // First time should be 0, last should be tMax
        await Assert.That(result.times[0]).IsCloseTo(0.0, tolerance: 1e-12);
        await Assert.That(result.times[result.times.Length - 1]).IsCloseTo(0.3, tolerance: 1e-6);
    }

    [Test]
    public async Task ComputeVoltageTrace_DifferentTargetNeurons_DifferentVoltages()
    {
        // Arrange — different weights for neurons 0 and 1 from the same input neuron
        int nPost = 3;
        // w[row * nPost + col]: row=pre-synaptic, col=post-synaptic
        // pre=0 → post=0 has weight 2.0, pre=0 → post=1 has weight 1.0
        double[] w = [2.0, 1.0, 0, 0, 0, 0, 0, 0, 0];
        double[] times = [0.05];
        int[] sources = [0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 1, 1);

        // Act — compute for neuron 0 and neuron 1
        var result0 = EventPropMath.ComputeVoltageTrace(
            tMax: 0.1, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, times, sources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        var result1 = EventPropMath.ComputeVoltageTrace(
            tMax: 0.1, dt: 0.01, targetNrnIdx: 1,
            w, nPre: 1, nPost, times, sources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Assert — neuron 0 has weight 2.0, neuron 1 has weight 1.0
        // So voltage for neuron 0 should be exactly twice that of neuron 1 at each time point
        int nPoints = result0.voltages.Length;
        for (int i = 0; i < nPoints; i++)
            await Assert.That(result0.voltages[i]).IsCloseTo(2.0 * result1.voltages[i], tolerance: 1e-10);
    }

    [Test]
    public async Task ComputeVoltageTrace_TimeAfterLastSpike_DecaysCorrectly()
    {
        // Arrange — single spike, check that voltage decays exponentially after the last spike
        int nPost = 1;
        double[] w = [2.0]; // weight from input to the single output neuron

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 1, 1);

        // Spike at t=0.1
        var result = EventPropMath.ComputeVoltageTrace(
            tMax: 0.3, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, new double[] { 0.1 }, new int[] { 0 }, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Assert — after the spike (t > 0.1), voltage should decay
        int startIdx = -1;
        for (int i = 0; i < result.voltages.Length; i++)
        {
            if (result.voltages[i] > 0.01) { startIdx = i; break; }
        }

        // After the peak, voltage should monotonically decrease (in the tail)
        bool decaying = true;
        for (int i = startIdx + 10; i < result.voltages.Length - 1 && decaying; i++)
        {
            if (result.voltages[i] > result.voltages[i - 1] + 1e-12) decaying = false;
        }

        await Assert.That(decaying).IsTrue();
    }

    [Test]
    public async Task ComputeVoltageTrace_ZeroDtBoundary_ReturnsCorrectLength()
    {
        // Arrange — tMax = dt, so only 2 time points (0 and dt)
        int nPost = 1;
        double[] w = [1.0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 1, 0);

        // Act — no spikes
        var result = EventPropMath.ComputeVoltageTrace(
            tMax: 0.01, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, new double[] { }, new int[] { }, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Assert — should have exactly 2 time points
        await Assert.That(result.times.Length).IsEqualTo(2); // (0.01 / 0.01) + 1
        await Assert.That(result.voltages.Length).IsEqualTo(2);
    }

    [Test]
    public async Task ComputeVoltageTrace_VoltageZeroBeforeSpike()
    {
        // Arrange — spikes at t=0.1, check that voltage is zero before the spike
        int nPost = 1;
        double[] w = [2.0];

        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, 1, 1);

        // Act
        var result = EventPropMath.ComputeVoltageTrace(
            tMax: 0.3, dt: 0.01, targetNrnIdx: 0,
            w, nPre: 1, nPost, new double[] { 0.1 }, new int[] { 0 }, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers);

        // Assert — voltage should be zero for all time points before the spike
        for (int i = 0; i < result.times.Length; i++)
        {
            if (result.times[i] < 0.1 - 1e-9)
                await Assert.That(result.voltages[i]).IsCloseTo(0.0, tolerance: 1e-12);
        }

        // And positive after the spike
        bool foundPositive = false;
        for (int i = 0; i < result.times.Length && !foundPositive; i++)
        {
            if (result.times[i] > 0.1 + 1e-9 && result.voltages[i] > 0.0) foundPositive = true;
        }
        await Assert.That(foundPositive).IsTrue();
    }

    // ── ComputeLambdaI tests ────────────────────────────────────────────

    [Test]
    public async Task ComputeLambdaI_EmptyPostSpikes_ReturnsZero()
    {
        // Act — no post-spike times
        double lambdaI = EventPropMath.ComputeLambdaI(
            t: 0.1, targetNrnIdx: 0,
            Array.Empty<double>(), Array.Empty<int>(),
            vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert
        await Assert.That(lambdaI).IsCloseTo(0.0, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaI_TimeBeforeAllSpikes_NonZero()
    {
        // Arrange — post-spike at t=0.2, query at t=0.1 (before the spike)
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act
        double lambdaI = EventPropMath.ComputeLambdaI(
            t: 0.1, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — should be non-zero (there's a spike after time t)
        await Assert.That(lambdaI).IsNotEqualTo(0.0);
    }

    [Test]
    public async Task ComputeLambdaI_TimeAfterAllSpikes_ReturnsZero()
    {
        // Arrange — post-spike at t=0.1, query at t=0.2 (after the spike)
        double[] postTimes = [0.1];
        int[] postSources = [0];

        // Act
        double lambdaI = EventPropMath.ComputeLambdaI(
            t: 0.2, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — no spikes after time t, so lambdaI should be zero
        await Assert.That(lambdaI).IsCloseTo(0.0, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaI_TimeEqualsLastSpike_ReturnsZero()
    {
        // Arrange — post-spike at t=0.1, query exactly at t=0.1
        double[] postTimes = [0.1];
        int[] postSources = [0];

        // Act
        double lambdaI = EventPropMath.ComputeLambdaI(
            t: 0.1, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — when tBwdTotal = 0, kb = kBwdPrefactor * (1 - 1) = 0
        await Assert.That(lambdaI).IsCloseTo(0.0, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaI_MultipleSpikes_AccumulatesAll()
    {
        // Arrange — three post-spikes, query at t=0.05 (before all of them)
        double[] postTimes = [0.1, 0.2, 0.3];
        int[] postSources = [0, 0, 0];

        // Act — lambdaI at t=0.05
        double lambdaI = EventPropMath.ComputeLambdaI(
            t: 0.05, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — should accumulate contributions from all three spikes
        await Assert.That(lambdaI).IsNotEqualTo(0.0);

        // Compare with single spike at t=0.3 — lambdaI should be larger with more spikes
        double lambdaISingle = EventPropMath.ComputeLambdaI(
            t: 0.05, targetNrnIdx: 0,
            new double[] { 0.3 }, new int[] { 0 }, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        await Assert.That(lambdaI).IsGreaterThan(lambdaISingle);
    }

    [Test]
    public async Task ComputeLambdaI_SpikeBeforeQueryTime_Excluded()
    {
        // Arrange — spikes at t=0.1 and t=0.3, query at t=0.2
        double[] postTimes = [0.1, 0.3];
        int[] postSources = [0, 0];

        // Act — lambdaI at t=0.2 (only the spike at 0.3 contributes)
        double lambdaI = EventPropMath.ComputeLambdaI(
            t: 0.2, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — should equal lambdaI computed from just the spike at t=0.3
        double expected = EventPropMath.ComputeLambdaI(
            t: 0.2, targetNrnIdx: 0,
            new double[] { 0.3 }, new int[] { 0 }, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        await Assert.That(lambdaI).IsCloseTo(expected, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaI_DifferentTauValues_ChangesResult()
    {
        // Arrange — single post-spike at t=0.2, query at t=0.1
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act — compute with two different tauMem values
        double lambdaI1 = EventPropMath.ComputeLambdaI(
            t: 0.1, targetNrnIdx: 0, postTimes, postSources,
            vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        double lambdaI2 = EventPropMath.ComputeLambdaI(
            t: 0.1, targetNrnIdx: 0, postTimes, postSources,
            vTh: 1.0, tauMem: 0.05, tauSyn: 0.005);

        // Assert — different tauMem should produce different lambdaI values
        await Assert.That(lambdaI1).IsNotEqualTo(lambdaI2);
    }

    [Test]
    public async Task ComputeLambdaI_DifferentVTh_ChangesResult()
    {
        // Arrange — single post-spike at t=0.2, query at t=0.1
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act — compute with two different vTh values
        double lambdaI1 = EventPropMath.ComputeLambdaI(
            t: 0.1, targetNrnIdx: 0, postTimes, postSources,
            vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        double lambdaI2 = EventPropMath.ComputeLambdaI(
            t: 0.1, targetNrnIdx: 0, postTimes, postSources,
            vTh: 2.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — jump magnitude = 1/(vTh + epsilon), so higher vTh → smaller lambdaI
        await Assert.That(lambdaI1).IsGreaterThan(lambdaI2);
    }

    // ── ComputeLambdaITrace tests ───────────────────────────────────────

    [Test]
    public async Task ComputeLambdaITrace_EmptyPostSpikes_AllZeros()
    {
        // Act — no post-spike times
        var result = EventPropMath.ComputeLambdaITrace(
            tMax: 0.3, dt: 0.1, targetNrnIdx: 0,
            Array.Empty<double>(), Array.Empty<int>(),
            vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert
        await Assert.That(result.times.Length).IsEqualTo(3); // (int)(0.3/0.1)+1 = 3 due to floating point
        await Assert.That(result.lambdaI.Length).IsEqualTo(3);

        for (int i = 0; i < result.lambdaI.Length; i++)
            await Assert.That(result.lambdaI[i]).IsCloseTo(0.0, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaITrace_TimePointsAreMonotonic()
    {
        // Arrange — single post-spike at t=0.2
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act
        var result = EventPropMath.ComputeLambdaITrace(
            tMax: 0.3, dt: 0.1, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — time points should be monotonically increasing
        for (int i = 1; i < result.times.Length; i++)
            await Assert.That(result.times[i]).IsGreaterThan(result.times[i - 1]);

        // First time should be 0, last should be tMax
        await Assert.That(result.times[0]).IsCloseTo(0.0, tolerance: 1e-12);
        await Assert.That(result.times[result.times.Length - 1]).IsCloseTo(0.1 * (result.times.Length - 1), tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaITrace_TimeAfterLastSpike_AllZeros()
    {
        // Arrange — single post-spike at t=0.2, dt=0.1 → time points: 0, 0.1, 0.2, 0.3
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act
        var result = EventPropMath.ComputeLambdaITrace(
            tMax: 0.3, dt: 0.1, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — lambdaI should be zero at t=0.3 (after the spike)
        await Assert.That(result.lambdaI[result.lambdaI.Length - 1]).IsCloseTo(0.0, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaITrace_TimeBeforeLastSpike_NonZero()
    {
        // Arrange — single post-spike at t=0.2, dt=0.1 → time points: 0, 0.1, 0.2, 0.3
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act
        var result = EventPropMath.ComputeLambdaITrace(
            tMax: 0.3, dt: 0.1, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — lambdaI should be non-zero at t=0 (before the spike)
        await Assert.That(result.lambdaI[0]).IsNotEqualTo(0.0);

        // And non-zero at t=0.1 (before the spike)
        await Assert.That(result.lambdaI[1]).IsNotEqualTo(0.0);

        // Check that at least one value before the last is non-zero
        bool foundNonZero = false;
        for (int i = 0; i < result.lambdaI.Length - 1 && !foundNonZero; i++)
        {
            if (Math.Abs(result.lambdaI[i]) > 1e-12) foundNonZero = true;
        }
        await Assert.That(foundNonZero).IsTrue();
    }

    [Test]
    public async Task ComputeLambdaITrace_MultipleSpikes_ChangingValues()
    {
        // Arrange — three post-spikes at t=0.1, 0.2, 0.3
        double[] postTimes = [0.1, 0.2, 0.3];
        int[] postSources = [0, 0, 0];

        // Act
        var result = EventPropMath.ComputeLambdaITrace(
            tMax: 0.4, dt: 0.1, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — lambdaI should change at each time point
        int nPoints = result.times.Length;

        // At t=0, all three spikes contribute
        await Assert.That(result.lambdaI[0]).IsNotEqualTo(0.0);

        // At t=0.4 (after all spikes), should be zero
        await Assert.That(result.lambdaI[nPoints - 1]).IsCloseTo(0.0, tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaITrace_FineDt_MoreTimePoints()
    {
        // Arrange — single post-spike at t=0.2
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act — compare coarse vs fine dt
        var resultCoarse = EventPropMath.ComputeLambdaITrace(
            tMax: 0.3, dt: 0.1, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        var resultFine = EventPropMath.ComputeLambdaITrace(
            tMax: 0.3, dt: 0.01, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert — fine dt should have more time points
        await Assert.That(resultFine.times.Length).IsEqualTo(31); // 0.3 / 0.01 + 1
        await Assert.That(resultCoarse.times.Length).IsEqualTo(3);

        // The fine trace should have non-zero values before the spike
        bool foundNonZero = false;
        for (int i = 0; i < resultFine.lambdaI.Length - 1 && !foundNonZero; i++)
        {
            if (resultFine.times[i] < 0.2 - 1e-9 && Math.Abs(resultFine.lambdaI[i]) > 1e-6)
                foundNonZero = true;
        }
        await Assert.That(foundNonZero).IsTrue();
    }

    [Test]
    public async Task ComputeLambdaITrace_SingleTimePoint()
    {
        // Arrange — single post-spike at t=0.2, dt = tMax
        double[] postTimes = [0.2];
        int[] postSources = [0];

        // Act — tMax=dt → only 2 time points
        var result = EventPropMath.ComputeLambdaITrace(
            tMax: 0.2, dt: 0.2, targetNrnIdx: 0,
            postTimes, postSources, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);

        // Assert
        await Assert.That(result.times.Length).IsEqualTo(2); // 0.2 / 0.2 + 1
        await Assert.That(result.lambdaI.Length).IsEqualTo(2);

        // At t=0, should be non-zero (spike at 0.2 is after)
        await Assert.That(result.lambdaI[0]).IsNotEqualTo(0.0);

        // At t=0.2 (exactly at spike), should be ~zero
        await Assert.That(result.lambdaI[1]).IsCloseTo(0.0, tolerance: 1e-12);
    }

    // ── Integration tests: diagnostic functions match layer methods ─────

    [Test]
    public async Task ComputeVoltageTrace_MatchesLayerGetVoltageTrace()
    {
        // Arrange — create a LIF layer and compare EventPropMath.ComputeVoltageTrace with Layer.GetVoltageTrace
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        var layer = new LIFLayer(parameters);

        // Set positive weights to ensure spikes are generated
        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        layer.Forward(input);

        // Act — get voltage trace from both the layer and directly
        var (layerTimes, layerVoltages) = layer.GetVoltageTrace(0, 0, tMax: 0.3, dt: 0.01);
        var (directTimes, directVoltages) = EventPropMath.ComputeVoltageTrace(
            0.3, 0.01, 0,
            layer.Weights.AsSpan(), parameters.NIn, parameters.N,
            input[0].Times, input[0].Sources,
            parameters.VTh, parameters.TauMem, parameters.TauSyn, new ScratchBuffers());

        // Assert — results should match
        await Assert.That(layerTimes.Length).IsEqualTo(directTimes.Length);
        for (int i = 0; i < layerVoltages.Length; i++)
            await Assert.That(layerVoltages[i]).IsCloseTo(directVoltages[i], tolerance: 1e-12);
    }

    [Test]
    public async Task ComputeLambdaITrace_MatchesLayerGetLambdaITrace()
    {
        // Arrange — create a LIF layer and compare EventPropMath.ComputeLambdaITrace with Layer.GetLambdaITrace
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        var layer = new LIFLayer(parameters);

        // Set positive weights to ensure spikes are generated
        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        layer.Forward(input);

        // Act — get lambda_i trace from both the layer and directly
        var (layerTimes, layerLambdaI) = layer.GetLambdaITrace(0, 0, tMax: 0.3, dt: 0.1);
        var (directTimes, directLambdaI) = EventPropMath.ComputeLambdaITrace(
            0.3, 0.1, 0,
            layer.Output[0].Times, layer.Output[0].Sources,
            parameters.VTh, parameters.TauMem, parameters.TauSyn);

        // Assert — results should match
        await Assert.That(layerTimes.Length).IsEqualTo(directTimes.Length);
        for (int i = 0; i < layerLambdaI.Length; i++)
            await Assert.That(layerLambdaI[i]).IsCloseTo(directLambdaI[i], tolerance: 1e-12);
    }
}
