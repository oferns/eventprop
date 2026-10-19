using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

// ============================================================
// Training loop benchmarks — full forward + backward iteration
// Measures the combined cost of one training step.
// ============================================================

[MemoryDiagnoser]
public abstract class Bench_TrainingLoop_Base : BenchBase
{
    protected SpikesData? PostSpikes { get; set; }

    public override void Setup()
    {
        // Don't call base.Setup() — we need forward result for backward
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
        OutTimesNative = new double[maxOutSpikes];
        OutSourcesNative = new int[maxOutSpikes];
        OutCurrentsNative = new double[maxOutSpikes];

        // Zero-allocation output buffers for ComputeSpikesInto
        OutTimesManaged = new double[maxOutSpikes];
        OutSourcesManaged = new int[maxOutSpikes];
        OutCurrentsManaged = new double[maxOutSpikes];
        OutErrorsManaged = new double[maxOutSpikes];

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

        // Run forward pass once to get post-synaptic spikes
        PostSpikes = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        for (int i = 0; i < PostSpikes.NSpikes; i++)
            PostSpikes.SetError(i, 1.0);

        // Prepare native backward data
        int nPost = PostSpikes.NSpikes;
        PostTimesNative = new double[nPost];
        PostSourcesNative = new int[nPost];
        PostErrorsNative = new double[nPost];
        for (int i = 0; i < nPost; i++)
        {
            PostTimesNative[i] = PostSpikes.Times[i];
            PostSourcesNative[i] = PostSpikes.Sources[i];
            PostErrorsNative[i] = 1.0;
        }
    }

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Array.Clear(GradManaged);

        // Forward pass — zero-allocation with ComputeSpikesInto
        int nOut = EventPropMath.ComputeSpikesInto(
            W, Times, Sources, NPost, VTh, TauMem, TauSyn,
            OutTimesManaged.AsSpan(0, OutTimesManaged.Length),
            OutSourcesManaged.AsSpan(0, OutSourcesManaged.Length),
            OutCurrentsManaged.AsSpan(0, OutCurrentsManaged.Length),
            Buffers);

        // Set errors on output spikes
        for (int i = 0; i < nOut; i++)
            OutErrorsManaged[i] = 1.0;

        // Backward pass — zero-allocation with raw spans
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

        // Forward pass via P/Invoke into the C library (libeventprop_c_api).
        // NOT the Eigen reference — CApi.Api is the C binding. The C library
        // allocates the spike output on the native heap; SpikeOutput owns it and
        // frees it on Dispose (the `using` above).
        using var output = CApi.Api.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        int nOut = output.NSpikes;

        // Backward pass via native — use actual forward output
        using var backwardOutput = CApi.Api.BackwardSpikes(Times, Sources,
            output.Times, output.Sources,
            PostErrorsNative, nOut, W, NPre, NPost,
            VTh, TauMem, TauSyn);
    }
}

public class Bench_TrainingLoop_Tiny : Bench_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); }

public class Bench_TrainingLoop_Small : Bench_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); }

public class Bench_TrainingLoop_Medium : Bench_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); }

public class Bench_TrainingLoop_Large : Bench_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); }

public class Bench_TrainingLoop_XL : Bench_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (2000, 500, 5000); }
