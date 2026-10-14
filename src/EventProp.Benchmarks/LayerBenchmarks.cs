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

    // Pre-allocated input for reuse across iterations (zero-allocation benchmark)
    protected SpikesVector ManagedInput { get; set; } = null!;
    protected CW.EventProp.SpikesVector NativeInput { get; set; } = null!;

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
        var nativeParams = new CW.EventProp.LIFLayerParameters(NPost, NPre, TauMem, TauSyn, VTh);
        ManagedLayer = new CW.EventProp.LIFLayer(managedParams, W);
        NativeLayer = new CApi.LIFLayer(nativeParams, W);

        // Pre-allocate input vectors for reuse across iterations (zero-allocation benchmark)
        ManagedInput = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        NativeInput = new CW.EventProp.SpikesVector(new[] { new CW.EventProp.SpikesData(Times, Sources) });
    }

    [IterationCleanup]
    public void Cleanup()
    {
        ManagedLayer?.Dispose();
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
    public void Managed() => ManagedLayer!.Forward(ManagedInput);

    [Benchmark]
    public void Native() => NativeLayer!.Forward(NativeInput);
}

[MemoryDiagnoser]
public class Bench_LIF_Forward_Medium : BenchLayerBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    [Benchmark(Baseline = true)]
    public void Managed() => ManagedLayer!.Forward(ManagedInput);

    [Benchmark]
    public void Native() => NativeLayer!.Forward(NativeInput);
}

[MemoryDiagnoser]
public class Bench_LIF_Forward_Large : BenchLayerBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

    [Benchmark(Baseline = true)]
    public void Managed() => ManagedLayer!.Forward(ManagedInput);

    [Benchmark]
    public void Native() => NativeLayer!.Forward(NativeInput);
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
        ManagedLayer!.Forward(ManagedInput);
        var post = ManagedLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.ZeroGrad();
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        NativeLayer!.Forward(NativeInput);
        for (int i = 0; i < NativeLayer[0].NSpikes; i++)
            NativeLayer.SetError(0, i, 1.0);
        NativeLayer.ZeroGrad();
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
        ManagedLayer!.Forward(ManagedInput);
        var post = ManagedLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.ZeroGrad();
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        NativeLayer!.Forward(NativeInput);
        for (int i = 0; i < NativeLayer[0].NSpikes; i++)
            NativeLayer.SetError(0, i, 1.0);
        NativeLayer.ZeroGrad();
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
        ManagedLayer!.Forward(ManagedInput);
        var post = ManagedLayer.Output;
        for (int i = 0; i < post[0].NSpikes; i++)
            post[0].SetError(i, 1.0);
        ManagedLayer.ZeroGrad();
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        NativeLayer!.Forward(NativeInput);
        for (int i = 0; i < NativeLayer[0].NSpikes; i++)
            NativeLayer.SetError(0, i, 1.0);
        NativeLayer.ZeroGrad();
        NativeLayer.Backward();
    }
}

// ============================================================
// LI Layer — Forward & Backward Benchmarks (maxima-based)
// Uses LILayer: no spike threshold, computes membrane potential maxima.
// ============================================================

public abstract class BenchLIBase
{
    public int NPre { get; protected set; }
    public int NPost { get; protected set; }
    public int NSpikes { get; protected set; }

    protected const double TauMem = 0.02;
    protected const double TauSyn = 0.005;

    protected double[] W { get; set; } = null!;
    protected double[] Times { get; set; } = null!;
    protected int[] Sources { get; set; } = null!;

    protected SpikesVector ManagedInput { get; set; } = null!;
    protected CW.EventProp.SpikesVector NativeInput { get; set; } = null!;

    protected CW.EventProp.LILayer? ManagedLayer { get; set; }
    protected CApi.LILayer? NativeLayer { get; set; }

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

        // LI layers: no vTh, uses maxima instead of spikes
        var managedParams = new LILayerParameters(NPost, NPre, TauMem, TauSyn);
        var nativeParams = new CW.EventProp.LILayerParameters(NPost, NPre, TauMem, TauSyn);
        ManagedLayer = new CW.EventProp.LILayer(managedParams, W);
        NativeLayer = new CApi.LILayer(nativeParams, W);

        ManagedInput = new SpikesVector(new[] { new SpikesData(Times, Sources) });
        NativeInput = new CW.EventProp.SpikesVector(new[] { new CW.EventProp.SpikesData(Times, Sources) });
    }

    [IterationCleanup]
    public void Cleanup()
    {
        ManagedLayer?.Dispose();
        NativeLayer?.Dispose();
    }
}

// --- LI Forward ---

[MemoryDiagnoser]
public class Bench_LI_Forward_Small : BenchLIBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

    [Benchmark(Baseline = true)]
    public void Managed() => ManagedLayer!.Forward(ManagedInput);

    [Benchmark]
    public void Native() => NativeLayer!.Forward(NativeInput);
}

[MemoryDiagnoser]
public class Bench_LI_Forward_Medium : BenchLIBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    [Benchmark(Baseline = true)]
    public void Managed() => ManagedLayer!.Forward(ManagedInput);

    [Benchmark]
    public void Native() => NativeLayer!.Forward(NativeInput);
}

[MemoryDiagnoser]
public class Bench_LI_Forward_Large : BenchLIBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

    [Benchmark(Baseline = true)]
    public void Managed() => ManagedLayer!.Forward(ManagedInput);

    [Benchmark]
    public void Native() => NativeLayer!.Forward(NativeInput);
}

// --- LI Backward ---

[MemoryDiagnoser]
public class Bench_LI_Backward_Small : BenchLIBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        ManagedLayer!.Forward(ManagedInput);
        var maxima = ManagedLayer.MaximaBatch!;
        for (int i = 0; i < maxima[0].NNeurons; i++)
            maxima[0].SetError(i, 1.0);
        ManagedLayer.ZeroGrad();
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        NativeLayer!.Forward(NativeInput);
        var maxima = NativeLayer.MaximaBatch!;
        for (int i = 0; i < maxima[0].NNeurons; i++)
            maxima[0].SetError(i, 1.0);
        NativeLayer.ZeroGrad();
        NativeLayer.Backward();
    }
}

[MemoryDiagnoser]
public class Bench_LI_Backward_Medium : BenchLIBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        ManagedLayer!.Forward(ManagedInput);
        var maxima = ManagedLayer.MaximaBatch!;
        for (int i = 0; i < maxima[0].NNeurons; i++)
            maxima[0].SetError(i, 1.0);
        ManagedLayer.ZeroGrad();
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        NativeLayer!.Forward(NativeInput);
        var maxima = NativeLayer.MaximaBatch!;
        for (int i = 0; i < maxima[0].NNeurons; i++)
            maxima[0].SetError(i, 1.0);
        NativeLayer.ZeroGrad();
        NativeLayer.Backward();
    }
}

[MemoryDiagnoser]
public class Bench_LI_Backward_Large : BenchLIBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        ManagedLayer!.Forward(ManagedInput);
        var maxima = ManagedLayer.MaximaBatch!;
        for (int i = 0; i < maxima[0].NNeurons; i++)
            maxima[0].SetError(i, 1.0);
        ManagedLayer.ZeroGrad();
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        NativeLayer!.Forward(NativeInput);
        var maxima = NativeLayer.MaximaBatch!;
        for (int i = 0; i < maxima[0].NNeurons; i++)
            maxima[0].SetError(i, 1.0);
        NativeLayer.ZeroGrad();
        NativeLayer.Backward();
    }
}
