using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

// ============================================================
// Spike pattern benchmarks — poisson and burst patterns for Medium size
// These extend the existing uniform-only coverage with diverse input patterns.
// ============================================================

public abstract class Bench_Pattern_Base : BenchBase
{
    protected SpikesData? PostSpikes { get; set; }

    protected abstract void GeneratePattern(Random rng);

    public override void Setup()
    {
        // Don't call base.Setup() — we need custom pattern generation
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        for (int idx = 0; idx < W.Length; idx++)
            W[idx] = 0.5 + rng.NextDouble() * 1.5;

        Times = new double[NSpikes];
        Sources = new int[NSpikes];
        GeneratePattern(rng);

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);

        int sumRows = NSpikes + 1;
        Sum0Native = new double[sumRows * NPost];
        Sum1Native = new double[sumRows * NPost];

        int maxOutSpikes = NSpikes * NPost;
        OutTimesNative = new double[maxOutSpikes];
        OutSourcesNative = new int[maxOutSpikes];
        OutCurrentsNative = new double[maxOutSpikes];

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

        // Run forward pass once to get post-synaptic spikes
        PostSpikes = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        for (int idx = 0; idx < PostSpikes.NSpikes; idx++)
            PostSpikes.SetError(idx, 1.0);

        // Prepare native backward data
        int nPost = PostSpikes.NSpikes;
        PostTimesNative = new double[nPost];
        PostSourcesNative = new int[nPost];
        PostErrorsNative = new double[nPost];
        SpikeErrorsNative = new double[NSpikes];
        for (int idx = 0; idx < nPost; idx++)
        {
            PostTimesNative[idx] = PostSpikes.Times[idx];
            PostSourcesNative[idx] = PostSpikes.Sources[idx];
            PostErrorsNative[idx] = 1.0;
        }
    }

    [Benchmark(Baseline = true)]
    public void ComputeSums_Managed()
    {
        Buffers.Sum0.AsSpan(0, (NSpikes + 1) * NPost).Clear();
        Buffers.Sum1.AsSpan(0, (NSpikes + 1) * NPost).Clear();
        EventPropMath.ComputeSums(W, Times, Sources, TauMem, TauSyn,
            Buffers.Sum0, Buffers.Sum1, NPost,
            Buffers.ExpInputMem, Buffers.ExpInputSyn);
    }

    [Benchmark]
    public void ComputeSums_Native()
    {
        using var output = CApi.Api.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        // Access to prevent optimization
        var _ = output.Sum0[0];
    }

    [Benchmark]
    public SpikesData ComputeSpikes_Managed()
        => EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public int ComputeSpikes_Native()
    {
        using var output = CApi.Api.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        return output.NSpikes;
    }

    [Benchmark]
    public void BackwardSpikes_Managed()
    {
        Array.Clear(GradManaged);
        EventPropMath.BackwardSpikes(W, Times, Sources, PostSpikes!, NPost, VTh, TauMem, TauSyn, GradManaged, Buffers);
    }

    [Benchmark]
    public void BackwardSpikes_Native()
    {
        using var output = CApi.Api.BackwardSpikes(Times, Sources,
            PostTimesNative, PostSourcesNative, PostErrorsNative, PostSpikes!.NSpikes, W, NPre, NPost,
            VTh, TauMem, TauSyn);
        // Access to prevent optimization
        var _ = output.Gradient[0];
    }
}

// ============================================================
// Poisson spike pattern (exponential inter-spike intervals, ~200 Hz)
// ============================================================
public class Bench_Pattern_Poisson : Bench_Pattern_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    protected override void GeneratePattern(Random rng)
    {
        // Canonical Poisson @200 Hz — see SpikePatterns for the contract.
        SpikePatterns.Poisson(NPre, NSpikes, Times, Sources, rng);
    }
}

// ============================================================
// Burst spike pattern — canonical definition, see SpikePatterns
// ============================================================
public class Bench_Pattern_Burst : Bench_Pattern_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    protected override void GeneratePattern(Random rng)
    {
        SpikePatterns.Burst(NPre, NSpikes, Times, Sources, rng);
    }
}
