using BenchmarkDotNet.Attributes;
using CW.EventProp;

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
        PostCurrentsNative = new double[nPost];
        PostErrorsNative = new double[nPost];
        SpikeErrorsNative = new double[NSpikes];
        for (int idx = 0; idx < nPost; idx++)
        {
            PostTimesNative[idx] = PostSpikes.Times[idx];
            PostSourcesNative[idx] = PostSpikes.Sources[idx];
            PostCurrentsNative[idx] = PostSpikes.Currents[idx];
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
        using var output = NativeApi.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        // Access to prevent optimization
        var _ = output.Sum0[0];
    }

    [Benchmark]
    public SpikesData ComputeSpikes_Managed()
        => EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public int ComputeSpikes_Native()
    {
        using var output = NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
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
        using var output = NativeApi.BackwardSpikes(Times, Sources,
            PostTimesNative, PostSourcesNative, PostCurrentsNative,
            PostErrorsNative, PostSpikes!.NSpikes, W, NPre, NPost,
            VTh, TauMem, TauSyn);
        // Access to prevent optimization
        var _ = output.Grad[0];
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
        // Poisson: exponential inter-spike intervals
        double rate = 200.0; // Hz per neuron
        double dt = 1.0 / (rate * NSpikes); // time step

        for (int idx = 0; idx < NSpikes; idx++)
        {
            // Exponential distribution for inter-spike interval
            double u = rng.NextDouble();
            Times[idx] = -Math.Log(1.0 - u) / rate;
            Sources[idx] = rng.Next(NPre);

            // Ensure strictly increasing times
            if (idx > 0 && Times[idx] <= Times[idx - 1])
                Times[idx] = Times[idx - 1] + dt;
        }
    }
}

// ============================================================
// Burst spike pattern (clusters of 5 spikes with 10ms gaps)
// ============================================================
public class Bench_Pattern_Burst : Bench_Pattern_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    protected override void GeneratePattern(Random rng)
    {
        int burstSize = 5;
        int nBursts = NSpikes / burstSize; // 100 bursts
        double gapMs = 10.0; // ms between bursts

        for (int b = 0; b < nBursts; b++)
        {
            double burstStart = b * (burstSize + 1) * gapMs / 1000.0; // convert to seconds
            for (int s = 0; s < burstSize; s++)
            {
                int idx = b * burstSize + s;
                Times[idx] = burstStart + s * 0.5 / 1000.0; // 0.5ms within burst
                Sources[idx] = rng.Next(NPre);
            }
        }

        // Fill remaining spikes (if NSpikes not divisible by burstSize)
        for (int idx = nBursts * burstSize; idx < NSpikes; idx++)
        {
            Times[idx] = (idx + 1) * 1.0 / NSpikes;
            Sources[idx] = rng.Next(NPre);
        }
    }
}
