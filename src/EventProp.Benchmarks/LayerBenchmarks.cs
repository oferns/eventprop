using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

// ============================================================
// Layer-level benchmarks — Managed (CW.EventProp) vs Native (CApi)
// Tests the full layer API: Forward() + Backward().
// ============================================================

public abstract class BenchLayerBase
{
    public int NPre { get; protected set; }
    public int NPost { get; protected set; }
    public int NSpikes { get; protected set; }

    protected const double TauMem = 0.02;
    protected const double TauSyn = 0.005;
    protected const double VTh = 1.0;

    protected double[] W { get; set; } = null!;
    protected double[] Times { get; set; } = null!;
    protected int[] Sources { get; set; } = null!;

    protected CW.EventProp.LIFLayer? ManagedLayer { get; set; }
    protected CApi.LIFLayer? NativeLayer { get; set; }

    protected abstract void Configure();

    [GlobalSetup]
    public virtual void Setup()
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
public class Bench_LIF_Forward_Small : BenchLayerBase
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
public class Bench_LIF_Forward_Medium : BenchLayerBase
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
public class Bench_LIF_Forward_Large : BenchLayerBase
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
public class Bench_LIF_Backward_Small : BenchLayerBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
        var post = ManagedLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
        var post = NativeLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}

[MemoryDiagnoser]
public class Bench_LIF_Backward_Medium : BenchLayerBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
        var post = ManagedLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
        var post = NativeLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}

[MemoryDiagnoser]
public class Bench_LIF_Backward_Large : BenchLayerBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var input = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        ManagedLayer!.Forward(input);
        var post = ManagedLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var input = new CApi.SpikesVector(new[] { new CApi.SpikesData(Times, Sources) });
        NativeLayer!.Forward(input);
        var post = NativeLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}
