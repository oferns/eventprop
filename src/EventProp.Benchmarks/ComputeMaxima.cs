using BenchmarkDotNet.Attributes;
using CW.EventProp;

namespace EventProp.Benchmarks;

// ============================================================
// ComputeMaxima benchmarks — 5 sizes × 1 pattern (uniform) = 5 benchmarks
// Measures membrane potential maximum time and value per neuron.
// ============================================================

public abstract class Bench_ComputeMaxima_Base : BenchBase
{
    protected MaximaData? MaximaResult { get; set; }

    [GlobalSetup]
    public override void Setup()
    {
        // Don't call base.Setup() — we need custom spike patterns
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

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

        // Run forward pass once to get post-synaptic spikes
        MaximaResult = CW.EventProp.EventPropMath.ComputeMaxima(
            (ReadOnlySpan<double>)W, (ReadOnlySpan<double>)Times, (ReadOnlySpan<int>)Sources,
            NPost, TauMem, TauSyn, Buffers);
    }

    [Benchmark(Baseline = true)]
    public MaximaData Managed()
        => CW.EventProp.EventPropMath.ComputeMaxima(
            (ReadOnlySpan<double>)W, (ReadOnlySpan<double>)Times, (ReadOnlySpan<int>)Sources,
            NPost, TauMem, TauSyn, Buffers);

    [Benchmark]
    public void Native()
    {
        // Allocate output arrays for native compute_maxima
        double[] maximaTimesOut = new double[NPost];
        double[] maximaValuesOut = new double[NPost];

        CxxNative.ComputeMaxima(W, NPre, NPost, Times, Sources, TauMem, TauSyn,
            maximaTimesOut, maximaValuesOut);
    }
}

public class Bench_ComputeMaxima_Tiny : Bench_ComputeMaxima_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); }

public class Bench_ComputeMaxima_Small : Bench_ComputeMaxima_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); }

public class Bench_ComputeMaxima_Medium : Bench_ComputeMaxima_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); }

public class Bench_ComputeMaxima_Large : Bench_ComputeMaxima_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); }

public class Bench_ComputeMaxima_XL : Bench_ComputeMaxima_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (2000, 500, 5000); }
