using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

// ============================================================
// Parameter sweep — isolate neuron count vs spike count effects
// Sweep A: fixed neurons (50x20), varying spikes (10/100/500/1000)
// Sweep B: fixed spikes (100), varying neurons (all sizes)
// ============================================================

[MemoryDiagnoser]
public abstract class Bench_TrainingLoopSweep_Base : BenchBase
{
    protected SpikesData? PostSpikes { get; set; }

    public override void Setup()
    {
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        for (int i = 0; i < W.Length; i++)
            W[i] = 0.5 + rng.NextDouble() * 1.5;

        Times = new double[NSpikes];
        Sources = new int[NSpikes];
        for (int i = 0; i < NSpikes; i++)
        {
            Times[i] = (i + 1) * 1.0 / (NSpikes + 1);
            Sources[i] = rng.Next(NPre);
        }

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);

        int sumRows = NSpikes + 1;
        Sum0Native = new double[sumRows * NPost];
        Sum1Native = new double[sumRows * NPost];

        int maxOutSpikes = NSpikes * NPost;
        OutTimesManaged = new double[maxOutSpikes];
        OutSourcesManaged = new int[maxOutSpikes];
        OutCurrentsManaged = new double[maxOutSpikes];
        OutErrorsManaged = new double[maxOutSpikes];

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

        // Run forward pass once to get post-synaptic spikes for native backward
        PostSpikes = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        for (int i = 0; i < PostSpikes.NSpikes; i++)
            PostSpikes.SetError(i, 1.0);

        int nPost = PostSpikes.NSpikes;
        PostTimesNative = new double[nPost];
        PostSourcesNative = new int[nPost];
        PostCurrentsNative = new double[nPost];
        PostErrorsNative = new double[nPost];
        for (int i = 0; i < nPost; i++)
        {
            PostTimesNative[i] = PostSpikes.Times[i];
            PostSourcesNative[i] = PostSpikes.Sources[i];
            PostCurrentsNative[i] = PostSpikes.Currents[i];
            PostErrorsNative[i] = 1.0;
        }
    }

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Array.Clear(GradManaged);

        int nOut = EventPropMath.ComputeSpikesInto(
            W, Times, Sources, NPost, VTh, TauMem, TauSyn,
            OutTimesManaged.AsSpan(),
            OutSourcesManaged.AsSpan(),
            OutCurrentsManaged.AsSpan(),
            Buffers);

        for (int i = 0; i < nOut; i++)
            OutErrorsManaged[i] = 1.0;

        EventPropMath.BackwardSpikesRaw(
            W, Times, Sources,
            OutTimesManaged.AsSpan(0, nOut),
            OutSourcesManaged.AsSpan(0, nOut),
            OutCurrentsManaged.AsSpan(0, nOut),
            OutErrorsManaged.AsSpan(0, nOut),
            NPost, VTh, TauMem, TauSyn,
            GradManaged.AsSpan(), Buffers);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(GradNative);

        using var output = CApi.Api.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        int nOut = output.NSpikes;

        using var backwardOutput = CApi.Api.BackwardSpikes(Times, Sources,
            output.Times, output.Sources, output.Currents,
            PostErrorsNative, nOut, W, NPre, NPost,
            VTh, TauMem, TauSyn);
    }
}

// --- Sweep A: fixed neurons (50x20), varying spike count ---

public class Bench_TrainingLoopSweepSpikes_10 : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 10); }

public class Bench_TrainingLoopSweepSpikes_100 : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); }

public class Bench_TrainingLoopSweepSpikes_500 : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 500); }

public class Bench_TrainingLoopSweepSpikes_1000 : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 1000); }

// --- Sweep B: fixed spikes (100), varying neuron count ---

public class Bench_TrainingLoopSweepNeurons_Tiny : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 100); }

public class Bench_TrainingLoopSweepNeurons_Small : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); }

public class Bench_TrainingLoopSweepNeurons_Medium : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 100); }

public class Bench_TrainingLoopSweepNeurons_Large : Bench_TrainingLoopSweep_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 100); }
