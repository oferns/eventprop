using BenchmarkDotNet.Attributes;
using CW.EventProp;

namespace EventProp.Benchmarks;

// ============================================================
// Shared test data generation — matches C/C++ benchmark patterns exactly.
// Uses Random(42) seed, uniform_real_distribution equivalent for weights,
// and rate-based Poisson generation with strictly increasing times.
// ============================================================

public enum DataPattern { Uniform, Poisson, Burst }

static class SharedTestData
{
    public static void GenerateUniformData(int nPre, int nPost, int nSpikes,
                                            Random rng, double[] w, double[] times, int[] sources)
    {
        for (int i = 0; i < nPre * nPost; i++)
            w[i] = 0.5 + rng.NextDouble() * 1.5; // [0.5, 2.0)

        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = (i + 1) / (double)(nSpikes + 1);
            sources[i] = rng.Next(nPre);
        }
    }

    public static void GeneratePoissonData(int nPre, int nPost, int nSpikes,
                                            Random rng, double[] w, double[] times, int[] sources)
    {
        for (int i = 0; i < nPre * nPost; i++)
            w[i] = 0.5 + rng.NextDouble() * 1.5; // [0.5, 2.0)

        /* Poisson: exponential inter-spike intervals, ~200 Hz per neuron.
           Matches C/C++ bench_main.cpp generation exactly. */
        double rate = 200.0; // Hz
        double dt = 1.0 / (rate * nSpikes);

        for (int idx = 0; idx < nSpikes; idx++)
        {
            double u = rng.NextDouble();
            times[idx] = -Math.Log(1.0 - u) / rate;
            sources[idx] = rng.Next(nPre);

            /* Ensure strictly increasing times */
            if (idx > 0 && times[idx] <= times[idx - 1])
                times[idx] = times[idx - 1] + dt;
        }
    }

    public static void GenerateBurstData(int nPre, int nPost, int nSpikes,
                                          Random rng, double[] w, double[] times, int[] sources)
    {
        for (int i = 0; i < nPre * nPost; i++)
            w[i] = 0.5 + rng.NextDouble() * 1.5; // [0.5, 2.0)

        /* Burst: clusters of 5 spikes with 10ms gaps between bursts.
           Matches C/C++ bench_main.cpp generation exactly. */
        int spikeIdx = 0;
        for (int nrn = 0; nrn < nPre && spikeIdx < nSpikes; nrn++)
        {
            double burstStart = 0.05 + nrn * 0.1;
            for (int b = 0; burstStart < 1.0 && spikeIdx < nSpikes; b++)
            {
                for (int s = 0; s < 5 && spikeIdx < nSpikes; s++)
                {
                    times[spikeIdx] = burstStart + s * 0.002;
                    sources[spikeIdx] = nrn;
                    spikeIdx++;
                }
                burstStart += 0.12;
            }
        }

        /* Fill remaining spikes with uniform spacing */
        for (int idx = spikeIdx; idx < nSpikes; idx++)
        {
            times[idx] = (idx + 1) / (double)(nSpikes + 1);
            sources[idx] = rng.Next(nPre);
        }
    }

    public static void GenerateData(int nPre, int nPost, int nSpikes, DataPattern pattern,
                                     Random rng, double[] w, double[] times, int[] sources)
    {
        switch (pattern)
        {
            case DataPattern.Uniform: GenerateUniformData(nPre, nPost, nSpikes, rng, w, times, sources); break;
            case DataPattern.Poisson: GeneratePoissonData(nPre, nPost, nSpikes, rng, w, times, sources); break;
            case DataPattern.Burst:   GenerateBurstData(nPre, nPost, nSpikes, rng, w, times, sources); break;
        }
    }
}

// ============================================================
// Abstract base for 3-library comparison benchmarks.
// Targets: Managed (EventPropMath), Native (C via NativeApi), Cpp (Eigen via NativeCppApi)
// ============================================================

public abstract class CrossLangBase
{
    protected const double TauMem = 0.02;
    protected const double TauSyn = 0.005;
    protected const double VTh = 1.0;

    public int NPre { get; protected set; }
    public int NPost { get; protected set; }
    public int NSpikes { get; protected set; }

    protected double[] W { get; set; } = null!;
    protected double[] Times { get; set; } = null!;
    protected int[] Sources { get; set; } = null!;

    protected ScratchBuffers Buffers { get; set; } = null!;
    protected double[] GradManaged { get; set; } = null!;
    protected double[] GradNative { get; set; } = null!;
    protected double[] GradCpp { get; set; } = null!;

    protected double[] Sum0Native { get; set; } = null!;
    protected double[] Sum1Native { get; set; } = null!;
    protected double[] Sum0Cpp { get; set; } = null!;
    protected double[] Sum1Cpp { get; set; } = null!;

    protected double[] OutTimesNative { get; set; } = null!;
    protected int[] OutSourcesNative { get; set; } = null!;
    protected double[] OutCurrentsNative { get; set; } = null!;
    protected double[] OutTimesCpp { get; set; } = null!;
    protected int[] OutSourcesCpp { get; set; } = null!;
    protected double[] OutCurrentsCpp { get; set; } = null!;

    protected SpikesData PostSpikesManaged { get; set; } = null!;
    protected int NPostSpikes { get; set; }
    protected double[] PostTimesNative { get; set; } = null!;
    protected int[] PostSourcesNative { get; set; } = null!;
    protected double[] PostCurrentsNative { get; set; } = null!;
    protected double[] PostErrorsNative { get; set; } = null!;
    protected double[] SpikeErrorsNative { get; set; } = null!;
    protected double[] PostTimesCpp { get; set; } = null!;
    protected int[] PostSourcesCpp { get; set; } = null!;
    protected double[] PostCurrentsCpp { get; set; } = null!;
    protected double[] PostErrorsCpp { get; set; } = null!;
    protected double[] SpikeErrorsCpp { get; set; } = null!;

    protected abstract void Configure();
    protected abstract DataPattern Pattern { get; }

    [GlobalSetup]
    public virtual void Setup()
    {
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        Times = new double[NSpikes];
        Sources = new int[NSpikes];
        SharedTestData.GenerateData(NPre, NPost, NSpikes, Pattern, rng, W, Times, Sources);

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];
        GradCpp = new double[NPre * NPost];

        int sumRows = NSpikes + 1;
        Sum0Native = new double[sumRows * NPost];
        Sum1Native = new double[sumRows * NPost];
        Sum0Cpp = new double[sumRows * NPost];
        Sum1Cpp = new double[sumRows * NPost];

        int maxOutSpikes = NSpikes * NPost;
        OutTimesNative = new double[maxOutSpikes];
        OutSourcesNative = new int[maxOutSpikes];
        OutCurrentsNative = new double[maxOutSpikes];
        OutTimesCpp = new double[maxOutSpikes];
        OutSourcesCpp = new int[maxOutSpikes];
        OutCurrentsCpp = new double[maxOutSpikes];

        // Run forward pass once to get post-synaptic spikes for backward
        PostSpikesManaged = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        NPostSpikes = PostSpikesManaged.NSpikes;
        for (int i = 0; i < NPostSpikes; i++)
            PostSpikesManaged.SetError(i, 1.0);

        // Prepare native backward data
        PostTimesNative = new double[NPostSpikes];
        PostSourcesNative = new int[NPostSpikes];
        PostCurrentsNative = new double[NPostSpikes];
        PostErrorsNative = new double[NPostSpikes];
        SpikeErrorsNative = new double[NSpikes]; // separate output buffer for pre-synaptic spike errors
        for (int i = 0; i < NPostSpikes; i++)
        {
            PostTimesNative[i] = PostSpikesManaged.Times[i];
            PostSourcesNative[i] = PostSpikesManaged.Sources[i];
            PostCurrentsNative[i] = PostSpikesManaged.Currents[i];
            PostErrorsNative[i] = 1.0;
        }

        // Prepare Cpp backward data
        PostTimesCpp = new double[NPostSpikes];
        PostSourcesCpp = new int[NPostSpikes];
        PostCurrentsCpp = new double[NPostSpikes];
        PostErrorsCpp = new double[NPostSpikes];
        SpikeErrorsCpp = new double[NSpikes]; // separate output buffer for pre-synaptic spike errors
        for (int i = 0; i < NPostSpikes; i++)
        {
            PostTimesCpp[i] = PostSpikesManaged.Times[i];
            PostSourcesCpp[i] = PostSpikesManaged.Sources[i];
            PostCurrentsCpp[i] = PostSpikesManaged.Currents[i];
            PostErrorsCpp[i] = 1.0;
        }
    }
}

// ============================================================
// ComputeSums — Managed vs Native (C) vs Cpp (Eigen)
// Clear IS necessary: accumulation into pre-allocated buffers
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_ComputeSums_Base : CrossLangBase
{
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

    [Benchmark]
    public void Cpp()
    {
        Array.Clear(Sum0Cpp);
        Array.Clear(Sum1Cpp);
        NativeCppApi.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn,
            Sum0Cpp, Sum1Cpp);
    }
}

// ============================================================
// ComputeSpikes — Managed vs Native (C) vs Cpp (Eigen)
// NO clear needed: overwrites output arrays
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_ComputeSpikes_Base : CrossLangBase
{
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

    [Benchmark]
    public int Cpp()
    {
        Array.Clear(OutTimesCpp, 0, OutTimesCpp.Length);
        Array.Clear(OutSourcesCpp, 0, OutSourcesCpp.Length);
        Array.Clear(OutCurrentsCpp, 0, OutCurrentsCpp.Length);
        return NativeCppApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn,
            OutTimesCpp, OutSourcesCpp, OutCurrentsCpp);
    }
}

// ============================================================
// BackwardSpikes — Managed vs Native (C) vs Cpp (Eigen)
// Clear IS necessary: accumulation into gradient buffers
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_BackwardSpikes_Base : CrossLangBase
{
    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Array.Clear(GradManaged);
        EventPropMath.BackwardSpikes(W, Times, Sources, PostSpikesManaged, NPost,
            VTh, TauMem, TauSyn, GradManaged, Buffers);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(GradNative);
        Array.Clear(SpikeErrorsNative, 0, SpikeErrorsNative.Length);
        NativeApi.BackwardSpikes(Times, Sources,
            PostTimesNative, PostSourcesNative, PostCurrentsNative,
            PostErrorsNative, W, NPre, NPost,
            VTh, TauMem, TauSyn, GradNative, SpikeErrorsNative);
    }

    [Benchmark]
    public void Cpp()
    {
        Array.Clear(GradCpp);
        Array.Clear(SpikeErrorsCpp, 0, SpikeErrorsCpp.Length);
        NativeCppApi.BackwardSpikes(Times, Sources,
            PostTimesCpp, PostSourcesCpp, PostCurrentsCpp,
            PostErrorsCpp, W, NPre, NPost,
            VTh, TauMem, TauSyn, GradCpp, SpikeErrorsCpp);
    }
}

// ============================================================
// TrainingLoop (forward + backward) — Managed vs Native (C) vs Cpp (Eigen)
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_TrainingLoop_Base : CrossLangBase
{
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

        // Forward pass via native (allocation in C)
        int maxOut = NSpikes * NPost;
        Array.Clear(OutTimesNative, 0, OutTimesNative.Length);
        Array.Clear(OutSourcesNative, 0, OutSourcesNative.Length);
        Array.Clear(OutCurrentsNative, 0, OutCurrentsNative.Length);

        int nPost = NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn,
            OutTimesNative, OutSourcesNative, OutCurrentsNative);

        // Backward pass via native
        Array.Clear(SpikeErrorsNative, 0, SpikeErrorsNative.Length);
        NativeApi.BackwardSpikes(Times, Sources,
            OutTimesNative, OutSourcesNative, OutCurrentsNative,
            PostErrorsNative, W, NPre, NPost,
            VTh, TauMem, TauSyn, GradNative, SpikeErrorsNative);
    }

    [Benchmark]
    public void Cpp()
    {
        Array.Clear(GradCpp);

        // Forward pass via cpp (allocation in Eigen)
        int maxOut = NSpikes * NPost;
        Array.Clear(OutTimesCpp, 0, OutTimesCpp.Length);
        Array.Clear(OutSourcesCpp, 0, OutSourcesCpp.Length);
        Array.Clear(OutCurrentsCpp, 0, OutCurrentsCpp.Length);

        int nPost = NativeCppApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn,
            OutTimesCpp, OutSourcesCpp, OutCurrentsCpp);

        // Backward pass via cpp
        Array.Clear(SpikeErrorsCpp, 0, SpikeErrorsCpp.Length);
        NativeCppApi.BackwardSpikes(Times, Sources,
            OutTimesCpp, OutSourcesCpp, OutCurrentsCpp,
            PostErrorsCpp, W, NPre, NPost,
            VTh, TauMem, TauSyn, GradCpp, SpikeErrorsCpp);
    }
}

// ============================================================
// ComputeMaxima — Managed vs Native (C) vs Cpp (Eigen)
// Overwrites output: no clear needed
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_ComputeMaxima_Base : CrossLangBase
{
    protected double[] MaxTimesManaged { get; set; } = null!;
    protected double[] MaxValuesManaged { get; set; } = null!;
    protected double[] MaxTimesNative { get; set; } = null!;
    protected double[] MaxValuesNative { get; set; } = null!;
    protected double[] MaxTimesCpp { get; set; } = null!;
    protected double[] MaxValuesCpp { get; set; } = null!;

    public override void Setup()
    {
        base.Setup();
        MaxTimesManaged = new double[NPost];
        MaxValuesManaged = new double[NPost];
        MaxTimesNative = new double[NPost];
        MaxValuesNative = new double[NPost];
        MaxTimesCpp = new double[NPost];
        MaxValuesCpp = new double[NPost];
    }

    [Benchmark(Baseline = true)]
    public MaximaData Managed() =>
        EventPropMath.ComputeMaxima(W, Times, Sources, NPost, TauMem, TauSyn, Buffers);

    [Benchmark]
    public void Native() =>
        NativeApi.ComputeMaxima(W, NPre, NPost, Times, Sources, TauMem, TauSyn,
            MaxTimesNative, MaxValuesNative);

    [Benchmark]
    public void Cpp() =>
        NativeCppApi.ComputeMaxima(W, NPre, NPost, Times, Sources, TauMem, TauSyn,
            MaxTimesCpp, MaxValuesCpp);
}

// ============================================================
// Concrete benchmarks — Tiny size (10×5, 20 spikes)
// ============================================================

public class Cross_ComputeSums_Tiny_Uniform : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSums_Tiny_Poisson : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSums_Tiny_Burst : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeSpikes_Tiny_Uniform : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSpikes_Tiny_Poisson : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSpikes_Tiny_Burst : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_BackwardSpikes_Tiny_Uniform : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_BackwardSpikes_Tiny_Poisson : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_BackwardSpikes_Tiny_Burst : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_TrainingLoop_Tiny_Uniform : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_TrainingLoop_Tiny_Poisson : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_TrainingLoop_Tiny_Burst : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeMaxima_Tiny_Uniform : Cross_ComputeMaxima_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (10, 5, 20); protected override DataPattern Pattern => DataPattern.Uniform; }

// ============================================================
// Concrete benchmarks — Small size (50×20, 100 spikes)
// ============================================================

public class Cross_ComputeSums_Small_Uniform : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSums_Small_Poisson : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSums_Small_Burst : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeSpikes_Small_Uniform : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSpikes_Small_Poisson : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSpikes_Small_Burst : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_BackwardSpikes_Small_Uniform : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_BackwardSpikes_Small_Poisson : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_BackwardSpikes_Small_Burst : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_TrainingLoop_Small_Uniform : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_TrainingLoop_Small_Poisson : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_TrainingLoop_Small_Burst : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeMaxima_Small_Uniform : Cross_ComputeMaxima_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }

// ============================================================
// Concrete benchmarks — Medium size (200×80, 500 spikes)
// ============================================================

public class Cross_ComputeSums_Medium_Uniform : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSums_Medium_Poisson : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSums_Medium_Burst : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeSpikes_Medium_Uniform : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSpikes_Medium_Poisson : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSpikes_Medium_Burst : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_BackwardSpikes_Medium_Uniform : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_BackwardSpikes_Medium_Poisson : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_BackwardSpikes_Medium_Burst : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_TrainingLoop_Medium_Uniform : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_TrainingLoop_Medium_Poisson : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_TrainingLoop_Medium_Burst : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeMaxima_Medium_Uniform : Cross_ComputeMaxima_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }

// ============================================================
// Concrete benchmarks — Large size (500×200, 1000 spikes)
// ============================================================

public class Cross_ComputeSums_Large_Uniform : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSums_Large_Poisson : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSums_Large_Burst : Cross_ComputeSums_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeSpikes_Large_Uniform : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_ComputeSpikes_Large_Poisson : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_ComputeSpikes_Large_Burst : Cross_ComputeSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_BackwardSpikes_Large_Uniform : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_BackwardSpikes_Large_Poisson : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_BackwardSpikes_Large_Burst : Cross_BackwardSpikes_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_TrainingLoop_Large_Uniform : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }
public class Cross_TrainingLoop_Large_Poisson : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Poisson; }
public class Cross_TrainingLoop_Large_Burst : Cross_TrainingLoop_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Burst; }

public class Cross_ComputeMaxima_Large_Uniform : Cross_ComputeMaxima_Base { protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }
