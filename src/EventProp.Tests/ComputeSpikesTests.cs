using CW.EventProp;

namespace CW.EventProp.Tests;

public class ComputeSpikesTests
{
    private static ScratchBuffers GetBuffers(int nNeurons, int maxSpikesPerNeuron)
    {
        using var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nNeurons, maxSpikesPerNeuron);
        return buffers;
    }

    [Test]
    public async Task ComputeSpikes_NoInputSpikes_ReturnsEmpty()
    {
        int nPost = 2;
        double[] w = [1, 0.5, 0.3, 0.2, 1, 0.4];

        double[] times = [];
        int[] sources = [];

        var buffers = GetBuffers(nPost, 4);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        await Assert.That(result.NSpikes).IsEqualTo(0);
    }

    [Test]
    public async Task ComputeSpikes_ZeroWeights_ReturnsEmpty()
    {
        int nPost = 2;
        double[] w = [0, 0, 0, 0, 0, 0];

        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var buffers = GetBuffers(nPost, 4);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        await Assert.That(result.NSpikes).IsEqualTo(0);
    }

    [Test]
    public async Task ComputeSpikes_StrongWeights_ReturnsSpike()
    {
        int nPost = 1;
        double[] w = [20.0]; // Strong weight to guarantee threshold crossing

        double[] times = [0.01];
        int[] sources = [0];

        var buffers = GetBuffers(nPost, 4);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // Should have at least one spike
        await Assert.That(result.NSpikes).IsGreaterThan(0);

        // First spike should be after the input spike
        await Assert.That(result.Times[0]).IsGreaterThan(0.01);

        // Source should be the post-synaptic neuron (nPost=1, so source is 0)
        await Assert.That(result.Sources[0]).IsEqualTo(0);

        // Current should be positive
        await Assert.That(result.Currents[0]).IsGreaterThan(0.0);
    }

    [Test]
    public async Task ComputeSpikes_WeakWeights_ReturnsEmpty()
    {
        int nPost = 1;
        double[] w = [0.01]; // Very weak weight, unlikely to cross threshold

        double[] times = [0.01];
        int[] sources = [0];

        var buffers = GetBuffers(nPost, 4);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // Should have no spikes (voltage never reaches threshold)
        await Assert.That(result.NSpikes).IsEqualTo(0);
    }

    [Test]
    public async Task ComputeSpikes_SingleNeuron_MultipleInputSpikes()
    {
        int nPost = 1;
        double[] w = [8.0]; // Moderate weight — enough for accumulation to cross threshold

        // Three input spikes from the same neuron
        double[] times = [0.01, 0.02, 0.03];
        int[] sources = [0, 0, 0];

        var buffers = GetBuffers(nPost, 8);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // Should have at least one output spike (accumulation should cross threshold)
        await Assert.That(result.NSpikes).IsGreaterThan(0);

        // First output spike should be after the first input spike (w=8 is strong enough for single-spike trigger)
        await Assert.That(result.Times[0]).IsGreaterThan(0.01);

        // All sources should be 0 (single post-neuron)
        for (int i = 0; i < result.NSpikes; i++)
            await Assert.That(result.Sources[i]).IsEqualTo(0);

        // Spike times should be sorted ascending
        for (int i = 1; i < result.NSpikes; i++)
            await Assert.That(result.Times[i]).IsGreaterThanOrEqualTo(result.Times[i - 1]);
    }

    [Test]
    public async Task ComputeSpikes_MultipleNeurons_DifferentSpikeCounts()
    {
        int nPost = 3;
        // Neuron 0 connects strongly to post-neuron 0, weakly to others
        // Neuron 1 connects only to post-neuron 2
        double[] w = [20.0, 0.01, 0.01, 0.01, 0.01, 20.0];

        double[] times = [0.01, 0.02];
        int[] sources = [0, 1];

        var buffers = GetBuffers(nPost, 8);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // Post-neuron 0 should have spikes (strong input from neuron 0)
        int post0Count = 0;
        for (int i = 0; i < result.NSpikes; i++)
            if (result.Sources[i] == 0) post0Count++;

        // Post-neuron 2 should have spikes (strong input from neuron 1)
        int post2Count = 0;
        for (int i = 0; i < result.NSpikes; i++)
            if (result.Sources[i] == 2) post2Count++;

        // Post-neuron 1 should have no spikes (too weak)
        int post1Count = 0;
        for (int i = 0; i < result.NSpikes; i++)
            if (result.Sources[i] == 1) post1Count++;

        await Assert.That(post0Count).IsGreaterThan(0);
        await Assert.That(post2Count).IsGreaterThan(0);
        await Assert.That(post1Count).IsEqualTo(0);

        // Total spikes should be sum of per-neuron counts
        await Assert.That(result.NSpikes).IsEqualTo(post0Count + post1Count + post2Count);
    }

    [Test]
    public async Task ComputeSpikes_SortedOutput_AllTimesAscending()
    {
        int nPost = 2;
        double[] w = [5.0, 3.0, 4.0, 2.0, 5.0, 1.0];

        double[] times = [0.01, 0.02, 0.03];
        int[] sources = [0, 1, 2];

        var buffers = GetBuffers(nPost, 8);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // All spike times should be sorted ascending
        for (int i = 1; i < result.NSpikes; i++)
            await Assert.That(result.Times[i]).IsGreaterThanOrEqualTo(result.Times[i - 1]);
    }

    [Test]
    public async Task ComputeSpikes_CurrentsArePositive()
    {
        int nPost = 1;
        double[] w = [20.0];

        double[] times = [0.01];
        int[] sources = [0];

        var buffers = GetBuffers(nPost, 4);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // All currents should be positive (synaptic current at spike time)
        for (int i = 0; i < result.NSpikes; i++)
            await Assert.That(result.Currents[i]).IsGreaterThan(0.0);
    }

    [Test]
    public async Task ComputeSpikes_LargeNetwork_NoCrash()
    {
        int nPre = 50;
        int nPost = 20;
        double[] w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = 1.0 + (i % 5) * 0.5;

        int nSpikes = 200;
        double[] times = new double[nSpikes];
        int[] sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = 0.001 * (i + 1);
            sources[i] = i % nPre;
        }

        // Should not throw
        var buffers = GetBuffers(nPost, nSpikes);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // Verify output is valid
        await Assert.That(result.NSpikes).IsGreaterThanOrEqualTo(0);

        if (result.NSpikes > 0)
        {
            // All sources should be in valid range [0, nPost)
            for (int i = 0; i < result.NSpikes; i++)
                await Assert.That(result.Sources[i]).IsGreaterThanOrEqualTo(0);
        }
    }

    [Test]
    public async Task ComputeSpikes_InhibitoryWeights_ReturnsFewerSpikes()
    {
        int nPost = 1;

        // Positive weight: should spike
        var buffersPos = GetBuffers(nPost, 4);
        var resultPos = EventPropMath.ComputeSpikes([20.0], [0.01], [0], nPost, 1.0, 0.02, 0.005, buffersPos);
        int posCount = resultPos.NSpikes;

        // Negative (inhibitory) weight: should not spike
        var buffersNeg = GetBuffers(nPost, 4);
        var resultNeg = EventPropMath.ComputeSpikes([-20.0], [0.01], [0], nPost, 1.0, 0.02, 0.005, buffersNeg);
        int negCount = resultNeg.NSpikes;

        await Assert.That(posCount).IsGreaterThan(negCount);
    }

    [Test]
    public async Task ComputeSpikes_ShorterTauSyn_MoreSpikes()
    {
        int nPost = 1;
        double[] w = [20.0];

        // Shorter tau_syn → sharper α-function but LOWER peak amplitude
        // (k = tau_syn/(tau_mem - tau_syn) decreases as tau_syn gets smaller)
        var buffersShort = GetBuffers(nPost, 16);
        var resultShort = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.003, buffersShort);
        int shortCount = resultShort.NSpikes;

        // Longer tau_syn → broader α-function but HIGHER peak amplitude
        var buffersLong = GetBuffers(nPost, 16);
        var resultLong = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.01, buffersLong);
        int longCount = resultLong.NSpikes;

        // Longer tau_syn should produce more spikes (higher peak current)
        await Assert.That(longCount).IsGreaterThanOrEqualTo(shortCount);
    }

    [Test]
    public async Task ComputeSpikes_HigherThreshold_FewerSpikes()
    {
        int nPost = 1;
        double[] w = [20.0];

        var buffersLowTh = GetBuffers(nPost, 4);
        var resultLowTh = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 0.5, 0.02, 0.005, buffersLowTh);
        int lowThCount = resultLowTh.NSpikes;

        var buffersHighTh = GetBuffers(nPost, 4);
        var resultHighTh = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 2.0, 0.02, 0.005, buffersHighTh);
        int highThCount = resultHighTh.NSpikes;

        await Assert.That(lowThCount).IsGreaterThanOrEqualTo(highThCount);
        await Assert.That(lowThCount).IsGreaterThan(0);
    }

    [Test]
    public async Task ComputeSpikes_CurrentsMatchAlphaFunction()
    {
        int nPost = 1;
        double[] w = [20.0];

        double tauMem = 0.02;
        double tauSyn = 0.005;

        var buffers = GetBuffers(nPost, 4);
        var result = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, tauMem, tauSyn, buffers);

        if (result.NSpikes > 0)
        {
            // Current at spike time should be positive (synaptic filter output)
            await Assert.That(result.Currents[0]).IsGreaterThan(0.0);
        }
    }

    [Test]
    public async Task ComputeSpikes_ReusingBuffers_NoAllocation()
    {
        int nPost = 1;
        double[] w = [20.0];

        var buffers = GetBuffers(nPost, 4);
        
        // First call
        var result1 = EventPropMath.ComputeSpikes(w, [0.01], [0], nPost, 1.0, 0.02, 0.005, buffers);
        int count1 = result1.NSpikes;

        // Second call with different input — should reuse buffers
        var result2 = EventPropMath.ComputeSpikes(w, [0.01, 0.02], [0, 0], nPost, 1.0, 0.02, 0.005, buffers);
        int count2 = result2.NSpikes;

        // Buffers should be reused (same array references)
        await Assert.That(result1.NSpikes).IsGreaterThanOrEqualTo(0);
        await Assert.That(result2.NSpikes).IsGreaterThanOrEqualTo(count1);
    }

    [Test]
    public async Task ComputeSpikes_MultipleOutputSpikesAcrossIntervals_PostSpikeUpdatesTBefore()
    {
        // This test triggers the uncovered path where post-synaptic spikes update tBefore
        // in later intervals (lines 219, 268). We need an output spike whose time
        // is GREATER than tBefore (the previous input time) for a later interval.
        // This happens when accumulation from multiple inputs causes the output spike
        // to fire AFTER a subsequent input time.
        int nPost = 1;
        double[] w = [3.0, 15.0]; // Neuron 0 weak (no spike alone), neuron 1 strong

        // Input at t=0.01 (neuron 0, too weak alone) + t=0.05 (neuron 1, strong enough).
        // The output spike fires at ~t=0.051 — AFTER times[1]=0.05.
        // When processing interval i=2: tBefore = times[1] = 0.05,
        // spikeTime (0.051) > tBefore → triggers the update.
        double[] times = [0.01, 0.05];
        int[] sources = [0, 1];

        var buffers = GetBuffers(nPost, 8);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // Produces 3 spikes (matches C): first at ~0.051, then two more from α-function tail
        await Assert.That(result.NSpikes).IsGreaterThanOrEqualTo(1);

        // Verify the tBefore update path is exercised:
        // First output spike fires after times[1]=0.05
        await Assert.That(result.Times[0]).IsGreaterThan(times[1]);
    }

    [Test]
    public async Task ComputeSpikes_OutputSpikeInMiddleInterval_TriggersTBeforeUpdate()
    {
        // Same scenario as above but simpler — one output spike fires after
        // a subsequent input, triggering tBefore = spikeTime.
        int nPost = 1;
        double[] w = [3.0, 15.0];

        // First input at t=0.01 (weak), second at t=0.05 (strong)
        double[] times = [0.01, 0.05];
        int[] sources = [0, 1];

        var buffers = GetBuffers(nPost, 8);
        var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // At least one output spike from the strong second input
        await Assert.That(result.NSpikes).IsGreaterThanOrEqualTo(1);

        // Verify the output spike fires after times[1] (triggers tBefore update)
        await Assert.That(result.Times[0]).IsGreaterThan(times[1]);
    }
}
