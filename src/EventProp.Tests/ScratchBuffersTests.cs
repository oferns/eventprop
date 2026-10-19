using System;
using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

public class ScratchBuffersTests
{
    [Test]
    [MatrixDataSource]
    public async Task EnsureCapacity_NetworkSizes_AllocatesCorrectBuffers([Matrix("small", "large")] string size)
    {
        (int nNeurons, int nPre, int nSpikes) = size switch
        {
            "small" => (5, 10, 20),
            "large" => (100, 200, 500),
            _ => throw new ArgumentException($"Unknown size: {size}")
        };

        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: nNeurons, nPre: nPre, nSpikes: nSpikes);

        // Sum0 and Sum1 should be (nSpikes+1) * nNeurons
        int expectedSumSize = (nSpikes + 1) * nNeurons;
        await Assert.That(buffers.Sum0.Length).IsGreaterThanOrEqualTo(expectedSumSize);
        await Assert.That(buffers.Sum1.Length).IsGreaterThanOrEqualTo(expectedSumSize);

        // Lambda vectors should be nNeurons
        await Assert.That(buffers.LambdaV.Length).IsGreaterThanOrEqualTo(nNeurons);
        await Assert.That(buffers.LambdaI.Length).IsGreaterThanOrEqualTo(nNeurons);

        // TransposeW should be nPre * nNeurons
        await Assert.That(buffers.TransposeW.Length).IsGreaterThanOrEqualTo(nPre * nNeurons);

        // VoltageTrace and LambdaITrace are reserved for future algorithms (not currently allocated).

        // Spike buffers should be allocated
        await Assert.That(buffers.SpikeErrors.Length).IsGreaterThanOrEqualTo(nSpikes);
    }

    [Test]
    public async Task EnsureCapacity_GrowFromEmpty_AllocatesAllBuffers()
    {
        using var buffers = new ScratchBuffers();

        // Initially all arrays are empty
        await Assert.That(buffers.Sum0.Length).IsEqualTo(0);

        buffers.EnsureCapacity(nNeurons: 3, nPre: 4, nSpikes: 5);

        await Assert.That(buffers.Sum0.Length).IsGreaterThan(0);
        await Assert.That(buffers.Sum1.Length).IsGreaterThan(0);
        await Assert.That(buffers.LambdaV.Length).IsGreaterThan(0);
    }

    [Test]
    public async Task EnsureCapacity_ReuseExistingBuffers_WhenLargeEnough()
    {
        using var buffers = new ScratchBuffers();

        // First allocation
        buffers.EnsureCapacity(nNeurons: 10, nPre: 20, nSpikes: 50);
        int firstSum0Length = buffers.Sum0.Length;

        // Second allocation with smaller dimensions — should reuse
        buffers.EnsureCapacity(nNeurons: 5, nPre: 10, nSpikes: 20);

        // Should not have grown
        await Assert.That(buffers.Sum0.Length).IsEqualTo(firstSum0Length);
    }

    [Test]
    public async Task EnsureCapacity_GrowWhenNeeded()
    {
        using var buffers = new ScratchBuffers();

        // First allocation
        buffers.EnsureCapacity(nNeurons: 5, nPre: 10, nSpikes: 20);
        int firstSum0Length = buffers.Sum0.Length;

        // Second allocation with larger dimensions — should grow
        buffers.EnsureCapacity(nNeurons: 20, nPre: 40, nSpikes: 100);

        await Assert.That(buffers.Sum0.Length).IsGreaterThan(firstSum0Length);
    }

    [Test]
    public async Task Clear_ZerosAllRequiredBuffers()
    {
        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: 5, nPre: 10, nSpikes: 20);

        // Fill buffers with non-zero values
        for (int i = 0; i < buffers.Sum0.Length; i++)
            buffers.Sum0[i] = 1.0;
        for (int i = 0; i < buffers.LambdaV.Length; i++)
            buffers.LambdaV[i] = 2.0;

        buffers.Clear();

        // Sum0, Sum1, LambdaV, LambdaI should be zeroed
        for (int i = 0; i < buffers.Sum0.Length; i++)
            await Assert.That(buffers.Sum0[i]).IsCloseTo(0.0, 1e-12);
        for (int i = 0; i < buffers.Sum1.Length; i++)
            await Assert.That(buffers.Sum1[i]).IsCloseTo(0.0, 1e-12);
        for (int i = 0; i < buffers.LambdaV.Length; i++)
            await Assert.That(buffers.LambdaV[i]).IsCloseTo(0.0, 1e-12);
        for (int i = 0; i < buffers.LambdaI.Length; i++)
            await Assert.That(buffers.LambdaI[i]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public async Task Clear_DoesNotAffectTransposeBuffers()
    {
        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: 5, nPre: 10, nSpikes: 20);

        // Fill TransposeW with non-zero values
        for (int i = 0; i < buffers.TransposeW.Length; i++)
            buffers.TransposeW[i] = 3.14;

        buffers.Clear();

        // TransposeW should NOT be cleared (it's not in the Clear method)
        await Assert.That(buffers.TransposeW[0]).IsCloseTo(3.14, 1e-12);
    }

    [Test]
    public async Task Dispose_EmptiesAllBuffers()
    {
        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: 5, nPre: 10, nSpikes: 20);

        buffers.Dispose();

        await Assert.That(buffers.Sum0.Length).IsEqualTo(0);
        await Assert.That(buffers.Sum1.Length).IsEqualTo(0);
        await Assert.That(buffers.LambdaV.Length).IsEqualTo(0);
        await Assert.That(buffers.LambdaI.Length).IsEqualTo(0);
        await Assert.That(buffers.TransposeW.Length).IsEqualTo(0);
    }

    [Test]
    public async Task Dispose_MultipleCalls_DoesNotThrow()
    {
        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nNeurons: 5, nPre: 10, nSpikes: 20);

        buffers.Dispose();
        Exception? ex = null;
        try { buffers.Dispose(); }
        catch (Exception e) { ex = e; }

        await Assert.That(ex).IsNull();
    }

    [Test]
    public async Task EnsureCapacity_WithManyNeuronsPerSpike_CorrectlyEstimates()
    {
        using var buffers = new ScratchBuffers();

        // 10 neurons, 5 spikes — should estimate at least 4 per neuron
        buffers.EnsureCapacity(nNeurons: 10, nPre: 5, nSpikes: 5);

        await Assert.That(buffers.MaxSpikesPerNeuron).IsGreaterThanOrEqualTo(4);
    }

    [Test]
    public async Task EnsureCapacity_MultipleCallsWithIncreasingSizes()
    {
        using var buffers = new ScratchBuffers();

        int[] sizes = [5, 20, 100, 500];
        foreach (var n in sizes)
            buffers.EnsureCapacity(nNeurons: n, nPre: n * 2, nSpikes: n * 10);

        // Final allocation should have been for the largest size
        await Assert.That(buffers.Sum0.Length).IsGreaterThanOrEqualTo(501 * 500);
    }

    [Test]
    public void PostSpikeRow_GrowPreservesLiveSpikes()
    {
        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(1, 2);

        var row = buffers.PostSpikeRowAt(0);
        row.Times[0] = 1.25;
        row.Currents[0] = 3.5;
        row.ExpSt[0] = 0.75;

        row.Grow(1, 8);

        SyncAssert.That(row.Capacity).IsGreaterThanOrEqualTo(8);
        SyncAssert.That(row.Times[0]).IsCloseTo(1.25, 1e-12);
        SyncAssert.That(row.Currents[0]).IsCloseTo(3.5, 1e-12);
        SyncAssert.That(row.ExpSt[0]).IsCloseTo(0.75, 1e-12);
    }

    [Test]
    public void ScratchBuffers_IsIDisposable()
    {
        using var buffers = new ScratchBuffers();

        SyncAssert.That((object)buffers).IsInstanceOfType(typeof(IDisposable));
    }
}
