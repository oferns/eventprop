using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

// ============================================================
// Maxima regression readout — Managed vs P/Invoke.
//
// Two families, because there are two different questions here and collapsing them
// produces a number that means the wrong thing:
//
//   MaximaRegression_<Size>      the per-sample kernel alone, every buffer reused.
//                                One sample, n_post maxima: the C loop's own cost plus
//                                one P/Invoke transition.
//
//   Batch_MaximaRegression_<n>   the layer-level step callers actually write
//                                (LILayer.Regression over a batch of 4/8/16), at the
//                                batch family's Small dimensions (50,20,100).
//
// Do NOT read either family as "the C readout is slow". The readout is O(batch × n_post)
// fused multiply-adds over a contiguous array and is effectively free at these shapes;
// what the batch rows measure is N marshalled calls, each doing its own malloc/free pair,
// because the layer cannot cache the native MaximaBatch (a ref struct) and the batch C
// entry point needs one. A large ratio there is a statement about crossing the boundary
// N times, not about the arithmetic — the same conflation that produced this suite's two
// false headline numbers (see the note in ComputeMaxima.cs).
//
// Only the .NET suite emits these families. There is no Eigen or reference-library
// counterpart to merge against, because the reference library has no regression loss; the
// merged table marks the cells `missing_counterpart` rather than inventing a peer impl.
// ============================================================

/// <summary>
/// Per-sample readout + loss with every buffer pre-allocated: the kernel on its own.
/// </summary>
public abstract class Bench_MaximaRegression_Base
{
    public int NPre { get; protected set; }
    public int NPost { get; protected set; }
    public int NSpikes { get; protected set; }

    protected const double TauMem = 0.02;
    protected const double TauSyn = 0.005;

    protected double[] Values = null!;      // maxima from a real forward pass
    protected double[] Errors = null!;
    protected double[] DecoderGrad = null!;
    protected double[] Decoder = null!;

    protected const double Bias = 0.15;
    protected const double Target = 0.7;

    [GlobalSetup]
    public void Setup()
    {
        Configure();
        var rng = new Random(42);

        var w = new double[NPre * NPost];
        for (int i = 0; i < w.Length; i++) w[i] = 0.5 + rng.NextDouble() * 1.5;

        var times = new double[NSpikes];
        var sources = new int[NSpikes];
        for (int i = 0; i < NSpikes; i++)
        {
            times[i] = (i + 1) * 1.0 / (NSpikes + 1);
            sources[i] = rng.Next(NPre);
        }

        // Maxima from a real forward pass, so the readout sees the magnitudes it will see in
        // training. Round numbers here would quietly change nothing about the timing, but they
        // would make the skip rule untestable from these rows, and a readout that skips
        // everything is doing less work than the row claims.
        using var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(NPost, NPre, NSpikes);
        using var maxima = EventPropMath.ComputeMaxima(w, times, sources, NPost, TauMem, TauSyn, buffers);

        Values = new double[NPost];
        for (int i = 0; i < NPost; i++) Values[i] = maxima.Values[i];
        if (Values.All(v => !double.IsFinite(v)))
            throw new InvalidOperationException("setup produced no usable maxima; the row would time an empty readout");

        Errors = new double[NPost];
        DecoderGrad = new double[NPost];
        Decoder = new double[NPost];
        for (int i = 0; i < NPost; i++) Decoder[i] = 0.4 - 0.5 * (i % 3) / 3.0;
    }

    protected abstract void Configure();

    [Benchmark(Baseline = true)]
    public double Managed()
    {
        var r = RegressionLoss.MaximaRegression(Values, Decoder, Bias, Target,
                                                double.PositiveInfinity, Errors, DecoderGrad);
        return r.Loss;
    }

    [Benchmark]
    public double Native()
    {
        var r = CApi.Loss.Regress(Values, Decoder, Bias, Target,
                                  double.PositiveInfinity, Errors, DecoderGrad);
        return r.Loss;
    }
}

[MemoryDiagnoser]
public class Bench_MaximaRegression_Medium : Bench_MaximaRegression_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); }

[MemoryDiagnoser]
public class Bench_MaximaRegression_Large : Bench_MaximaRegression_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); }

/// <summary>
/// The layer-level step over a batch, with the forward pass done once outside the timing.
/// </summary>
public abstract class Bench_Batch_MaximaRegression_Base
{
    protected int NPre { get; set; }
    protected int NPost { get; set; }
    protected int NSpikes { get; set; }
    protected int BatchSize { get; set; }

    protected const double TauMem = 0.02;
    protected const double TauSyn = 0.005;

    protected CW.EventProp.LILayer? ManagedLayer { get; set; }
    protected CApi.LILayer? NativeLayer { get; set; }

    protected double[] Decoder = null!;
    protected double[] Targets = null!;
    protected double[] DecGradManaged = null!;
    protected double[] DecGradNative = null!;
    protected double[] BiasGradManaged = null!;
    protected double[] BiasGradNative = null!;

    protected abstract void Configure();

    [GlobalSetup]
    public void Setup()
    {
        Configure();
        var rng = new Random(42);

        var w = new double[NPre * NPost];
        for (int i = 0; i < w.Length; i++) w[i] = 0.5 + rng.NextDouble() * 1.5;

        var managedInput = new SpikesVector();
        var nativeInput = new CW.EventProp.SpikesVector();
        for (int b = 0; b < BatchSize; b++)
        {
            var times = new double[NSpikes];
            var sources = new int[NSpikes];
            for (int i = 0; i < NSpikes; i++)
            {
                times[i] = (i + 1) * 1.0 / (NSpikes + 1);
                sources[i] = rng.Next(NPre);
            }
            managedInput.Add(new SpikesData(times, sources));
            nativeInput.Add(new CW.EventProp.SpikesData(times, sources));
        }

        var parameters = new LILayerParameters(NPost, NPre, TauMem, TauSyn);
        ManagedLayer = new CW.EventProp.LILayer(parameters, w);
        NativeLayer = new CApi.LILayer(parameters, w);

        // Forward once. Regression is a pure function of the stored maxima (it writes only the
        // error buffers), so repeating it without another forward is both valid and exactly the
        // steady state of a training loop that recomputes maxima outside the measured region.
        ManagedLayer.Forward(managedInput);
        NativeLayer.Forward(nativeInput);

        Decoder = new double[NPost];
        for (int i = 0; i < NPost; i++) Decoder[i] = 0.25 + 0.01 * i;   // never all-zero: see the parity suite
        Targets = new double[BatchSize];
        for (int b = 0; b < BatchSize; b++) Targets[b] = 0.5 + 0.1 * b;

        DecGradManaged = new double[NPost];
        DecGradNative = new double[NPost];
        BiasGradManaged = new double[1];
        BiasGradNative = new double[1];
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        ManagedLayer?.Dispose();
        NativeLayer?.Dispose();
    }

    // The accumulators are cleared inside the measured region on both sides: a training loop
    // zeroes them every step (zero_grad), and leaving them to grow would also make the two sides
    // diverge in float after enough iterations, which is not a property of either implementation.

    [Benchmark(Baseline = true)]
    public double Managed()
    {
        DecGradManaged.AsSpan().Clear();
        BiasGradManaged.AsSpan().Clear();
        return ManagedLayer!.Regression(Decoder, 0.15, Targets, double.PositiveInfinity,
                                        DecGradManaged, BiasGradManaged).Losses[0];
    }

    [Benchmark]
    public double Native()
    {
        DecGradNative.AsSpan().Clear();
        BiasGradNative.AsSpan().Clear();
        return NativeLayer!.Regression(Decoder, 0.15, Targets, double.PositiveInfinity,
                                       DecGradNative, BiasGradNative).Losses[0];
    }
}

[MemoryDiagnoser]
public class Bench_Batch_MaximaRegression_4 : Bench_Batch_MaximaRegression_Base
{ protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 4; } }

[MemoryDiagnoser]
public class Bench_Batch_MaximaRegression_8 : Bench_Batch_MaximaRegression_Base
{ protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 8; } }

[MemoryDiagnoser]
public class Bench_Batch_MaximaRegression_16 : Bench_Batch_MaximaRegression_Base
{ protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 16; } }
