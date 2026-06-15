using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

// ============================================================
// Layer-level benchmarks — Managed (CW.EventProp) vs Native (CW.EventProp.Native)
// Tests the full layer API: Forward() + Backward() via batch operations.
// Matches Python benchmark coverage (LIF forward/backward per size).
// ============================================================

public abstract class Bench_Layer_Base : BenchBase
{
    protected CW.EventProp.LIFLayer? ManagedLayer { get; set; }
    protected CApi.LIFLayer? NativeLayer { get; set; }

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

        // Create layers with identical weights
        var managedParams = new LIFLayerParameters(NPost, NPre, TauMem, TauSyn, VTh);
        var nativeParams = new CApi.LIFLayerParameters(NPost, NPre, TauMem, TauSyn, VTh);
        ManagedLayer = new CW.EventProp.LIFLayer(managedParams, W);
        NativeLayer = new CApi.LIFLayer(nativeParams, W);
    }

    [IterationCleanup]
    public void Cleanup()
    {
        NativeLayer?.Dispose();
    }
}

// ============================================================
// LIF Forward — Managed vs Native (batch=1, uniform pattern)
// ============================================================

[MemoryDiagnoser]
public class Bench_LIF_Forward_Small : Bench_Layer_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
    }
}

[MemoryDiagnoser]
public class Bench_LIF_Forward_Medium : Bench_Layer_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
    }
}

[MemoryDiagnoser]
public class Bench_LIF_Forward_Large : Bench_Layer_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
    }
}

// ============================================================
// LIF Backward — Managed vs Native (batch=1, uniform pattern)
// ============================================================

[MemoryDiagnoser]
public class Bench_LIF_Backward_Small : Bench_Layer_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
        var post = ManagedLayer.Output!;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
        var post = NativeLayer.Output!;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}

[MemoryDiagnoser]
public class Bench_LIF_Backward_Medium : Bench_Layer_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
        var post = ManagedLayer.Output!;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
        var post = NativeLayer.Output!;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}

[MemoryDiagnoser]
public class Bench_LIF_Backward_Large : Bench_Layer_Base
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
        var post = ManagedLayer.Output!;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
        var post = NativeLayer.Output!;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}
