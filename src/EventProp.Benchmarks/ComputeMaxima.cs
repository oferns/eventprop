using BenchmarkDotNet.Attributes;
using CW.EventProp;

namespace EventProp.Benchmarks;

// ============================================================
// ComputeMaxima benchmarks — 5 sizes × 3 patterns (uniform, poisson, burst)
// Measures membrane potential maximum time and value per neuron.
// ============================================================

public abstract class Bench_ComputeMaxima_Base : BenchBase
{
    protected MaximaData? MaximaResult { get; set; }

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
        
        // Generate uniform pattern by default (override in derived classes)
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
        using var output = Api.ComputeMaxima(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        var _ = output.Times[0]; // Access to prevent optimization
    }
}

// ============================================================
// Uniform pattern (existing)
// ============================================================
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

// ============================================================
// Poisson pattern (exponential inter-spike intervals)
// ============================================================
public abstract class Bench_ComputeMaxima_Poisson_Base : BenchBase
{
    protected MaximaData? MaximaResult { get; set; }

    public override void Setup()
    {
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        for (int idx = 0; idx < W.Length; idx++)
            W[idx] = 0.5 + rng.NextDouble() * 1.5;

        Times = new double[NSpikes];
        Sources = new int[NSpikes];

        // Poisson: exponential inter-spike intervals
        double rate = 200.0; // Hz per neuron
        double dt = 1.0 / (rate * NSpikes);

        for (int idx = 0; idx < NSpikes; idx++)
        {
            double u = rng.NextDouble();
            Times[idx] = -Math.Log(1.0 - u) / rate;
            Sources[idx] = rng.Next(NPre);

            // Ensure strictly increasing times
            if (idx > 0 && Times[idx] <= Times[idx - 1])
                Times[idx] = Times[idx - 1] + dt;
        }

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

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
        using var output = Api.ComputeMaxima(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        var _ = output.Times[0]; // Access to prevent optimization
    }
}

public class Bench_ComputeMaxima_Poisson_Tiny : Bench_ComputeMaxima_Poisson_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); }

public class Bench_ComputeMaxima_Poisson_Small : Bench_ComputeMaxima_Poisson_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); }

public class Bench_ComputeMaxima_Poisson_Medium : Bench_ComputeMaxima_Poisson_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); }

public class Bench_ComputeMaxima_Poisson_Large : Bench_ComputeMaxima_Poisson_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); }

public class Bench_ComputeMaxima_Poisson_XL : Bench_ComputeMaxima_Poisson_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (2000, 500, 5000); }

// ============================================================
// Burst pattern (clusters of 5 spikes with 10ms gaps)
// ============================================================
public abstract class Bench_ComputeMaxima_Burst_Base : BenchBase
{
    protected MaximaData? MaximaResult { get; set; }

    public override void Setup()
    {
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        for (int idx = 0; idx < W.Length; idx++)
            W[idx] = 0.5 + rng.NextDouble() * 1.5;

        Times = new double[NSpikes];
        Sources = new int[NSpikes];

        // Burst pattern: clusters of 5 spikes with 10ms gaps
        int spike_idx = 0;
        for (int nrn = 0; nrn < NPre && spike_idx < NSpikes; nrn++)
        {
            double burst_start = 0.05 + nrn * 0.1;
            for (int b = 0; burst_start < 1.0 && spike_idx < NSpikes; b++)
            {
                for (int s = 0; s < 5 && spike_idx < NSpikes; s++)
                {
                    Times[spike_idx] = burst_start + s * 0.002;
                    Sources[spike_idx] = nrn;
                    spike_idx++;
                }
                burst_start += 0.12;
            }
        }

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

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
        using var output = Api.ComputeMaxima(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        var _ = output.Times[0]; // Access to prevent optimization
    }
}

public class Bench_ComputeMaxima_Burst_Tiny : Bench_ComputeMaxima_Burst_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); }

public class Bench_ComputeMaxima_Burst_Small : Bench_ComputeMaxima_Burst_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); }

public class Bench_ComputeMaxima_Burst_Medium : Bench_ComputeMaxima_Burst_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); }

public class Bench_ComputeMaxima_Burst_Large : Bench_ComputeMaxima_Burst_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); }

public class Bench_ComputeMaxima_Burst_XL : Bench_ComputeMaxima_Burst_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (2000, 500, 5000); }
