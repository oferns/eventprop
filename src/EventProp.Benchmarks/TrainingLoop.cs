using BenchmarkDotNet.Attributes;
using CW.EventProp;

namespace EventProp.Benchmarks;

// ============================================================
// Training loop benchmarks — full forward + backward iteration
// Measures the combined cost of one training step.
// ============================================================

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

        // Forward pass (allocation in managed code)
        var fwdResult = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        for (int i = 0; i < fwdResult.NSpikes; i++)
            fwdResult.SetError(i, 1.0);

        // Backward pass (zero-allocation)
        EventPropMath.BackwardSpikes(W, Times, Sources, fwdResult, NPost, VTh, TauMem, TauSyn, GradManaged, Buffers);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(GradNative);

        // Forward pass via native (allocation in C++ Eigen)
        int maxOut = NSpikes * NPost;
        Array.Clear(OutTimesNative, 0, OutTimesNative.Length);
        Array.Clear(OutSourcesNative, 0, OutSourcesNative.Length);
        Array.Clear(OutCurrentsNative, 0, OutCurrentsNative.Length);

        using var output = Api.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        int nPost = output.NSpikes;

        // Backward pass via native
        using var backwardOutput = Api.BackwardSpikes(Times, Sources,
            OutTimesNative, OutSourcesNative, OutCurrentsNative,
            PostErrorsNative, nPost, W, NPre, NPost,  // Added nPost parameter
            VTh, TauMem, TauSyn);
    }

    [Benchmark]
    public void Mixed()
    {
        Array.Clear(GradManaged);

        // Forward pass via managed (zero-allocation with buffers)
        var fwdResult = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        for (int i = 0; i < fwdResult.NSpikes; i++)
            fwdResult.SetError(i, 1.0);

        // Backward pass via native (P/Invoke)
        using var output = Api.BackwardSpikes(Times, Sources,
            fwdResult.Times, fwdResult.Sources, fwdResult.Currents,
            fwdResult.Errors, fwdResult.NSpikes, W, NPre, NPost,  // Added nPost parameter
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
