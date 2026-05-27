using BenchmarkDotNet.Attributes;
using CW.EventProp;

namespace EventProp.Benchmarks;

// ============================================================
// Tiny and XL size benchmarks for ComputeSums, ComputeSpikes, BackwardSpikes
// These extend the existing Small/Medium/Large coverage.
// ============================================================

public abstract class Bench_TinyXL_Base : BenchBase
{
    protected SpikesData? PostSpikes { get; set; }

    public override void Setup()
    {
        // Don't call base.Setup() — we need forward result for backward
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        for (int idx = 0; idx < W.Length; idx++)
            W[idx] = 0.5 + rng.NextDouble() * 1.5;

        Times = new double[NSpikes];
        Sources = new int[NSpikes];
        for (int idx = 0; idx < NSpikes; idx++)
        {
            Times[idx] = (idx + 1) * 1.0 / (NSpikes + 1);
            Sources[idx] = rng.Next(NPre);
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
        for (int idx = 0; idx < PostSpikes.NSpikes; idx++)
            PostSpikes.SetError(idx, 1.0);

        // Prepare native backward data
        int nPost = PostSpikes.NSpikes;
        PostTimesNative = new double[nPost];
        PostSourcesNative = new int[nPost];
        PostCurrentsNative = new double[nPost];
        PostErrorsNative = new double[nPost];
        for (int idx = 0; idx < nPost; idx++)
        {
            PostTimesNative[idx] = PostSpikes.Times[idx];
            PostSourcesNative[idx] = PostSpikes.Sources[idx];
            PostCurrentsNative[idx] = PostSpikes.Currents[idx];
            PostErrorsNative[idx] = 1.0;
        }
    }
}

// ============================================================
// ComputeSums — Tiny (10, 5, 20) and XL (2000, 500, 5000)
// ============================================================
public class Bench_ComputeSums_Tiny : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Buffers.Sum0.AsSpan(0, (NSpikes + 1) * NPost).Clear();
        Buffers.Sum1.AsSpan(0, (NSpikes + 1) * NPost).Clear();
        EventPropMath.ComputeSums(W, Times, Sources, TauMem, TauSyn,
            Buffers.Sum0, Buffers.Sum1, NPost,
            Buffers.ExpInputMem, Buffers.ExpInputSyn);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(Sum0Native);
        Array.Clear(Sum1Native);
        NativeApi.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn,
            Sum0Native, Sum1Native);
    }
}

public class Bench_ComputeSums_XL : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (2000, 500, 5000);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Buffers.Sum0.AsSpan(0, (NSpikes + 1) * NPost).Clear();
        Buffers.Sum1.AsSpan(0, (NSpikes + 1) * NPost).Clear();
        EventPropMath.ComputeSums(W, Times, Sources, TauMem, TauSyn,
            Buffers.Sum0, Buffers.Sum1, NPost,
            Buffers.ExpInputMem, Buffers.ExpInputSyn);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(Sum0Native);
        Array.Clear(Sum1Native);
        NativeApi.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn,
            Sum0Native, Sum1Native);
    }
}

// ============================================================
// ComputeSpikes — Tiny and XL
// ============================================================
public class Bench_ComputeSpikes_Tiny : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20);

    [Benchmark(Baseline = true)]
    public SpikesData Managed()
        => EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public int Native()
    {
        Array.Clear(OutTimesNative, 0, OutTimesNative.Length);
        Array.Clear(OutSourcesNative, 0, OutSourcesNative.Length);
        Array.Clear(OutCurrentsNative, 0, OutCurrentsNative.Length);
        return NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn,
            OutTimesNative, OutSourcesNative, OutCurrentsNative);
    }
}

public class Bench_ComputeSpikes_XL : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (2000, 500, 5000);

    [Benchmark(Baseline = true)]
    public SpikesData Managed()
        => EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public int Native()
    {
        Array.Clear(OutTimesNative, 0, OutTimesNative.Length);
        Array.Clear(OutSourcesNative, 0, OutSourcesNative.Length);
        Array.Clear(OutCurrentsNative, 0, OutCurrentsNative.Length);
        return NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn,
            OutTimesNative, OutSourcesNative, OutCurrentsNative);
    }
}

// ============================================================
// BackwardSpikes — Tiny and XL (need forward result for backward)
// ============================================================
public class Bench_BackwardSpikes_Tiny : Bench_TinyXL_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Array.Clear(GradManaged);
        EventPropMath.BackwardSpikes(W, Times, Sources, PostSpikes!, NPost, VTh, TauMem, TauSyn, GradManaged, Buffers);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(GradNative);
        NativeApi.BackwardSpikes(Times, Sources, PostTimesNative, PostSourcesNative,
            PostCurrentsNative, PostErrorsNative, W, NPre, NPost, VTh, TauMem, TauSyn, GradNative);
    }
}

public class Bench_BackwardSpikes_XL : Bench_TinyXL_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (2000, 500, 5000);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Array.Clear(GradManaged);
        EventPropMath.BackwardSpikes(W, Times, Sources, PostSpikes!, NPost, VTh, TauMem, TauSyn, GradManaged, Buffers);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(GradNative);
        NativeApi.BackwardSpikes(Times, Sources, PostTimesNative, PostSourcesNative,
            PostCurrentsNative, PostErrorsNative, W, NPre, NPost, VTh, TauMem, TauSyn, GradNative);
    }
}
