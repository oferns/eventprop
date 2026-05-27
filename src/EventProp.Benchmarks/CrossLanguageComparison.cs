using BenchmarkDotNet.Attributes;
using CW.EventProp;

namespace EventProp.Benchmarks;

static class SharedTestData
{
    private static uint LcgNext(ref uint state)
    {
        state = state * 1103515245u + 12345u;
        return (state >> 16) & 0x7fff;
    }

    private static double Uniform(double lo, double hi, ref uint state) =>
        lo + (hi - lo) * LcgNext(ref state) / 32768.0;

    private static int IntRange(int lo, int hi, ref uint state) =>
        lo + (int)(LcgNext(ref state) % (uint)(hi - lo + 1));

    public static void GenerateUniformData(int nPre, int nPost, int nSpikes,
                                            double[] w, double[] times, int[] sources)
    {
        uint state = 12345u;
        for (int i = 0; i < nPre * nPost; i++)
            w[i] = Uniform(0.5, 2.0, ref state);
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = (i + 1) / (double)(nSpikes + 1);
            sources[i] = IntRange(0, nPre - 1, ref state);
        }
    }

    public static void GeneratePoissonData(int nPre, int nPost, int nSpikes,
                                            double[] w, double[] times, int[] sources)
    {
        uint state = 12345u;
        for (int i = 0; i < nPre * nPost; i++)
            w[i] = Uniform(0.5, 2.0, ref state);
        int spikeIdx = 0;
        for (int nrn = 0; nrn < nPre && spikeIdx < nSpikes; nrn++)
        {
            double t = 0.0;
            while (t < 1.0 && spikeIdx < nSpikes)
            {
                double u = Uniform(0.0001, 1.0, ref state);
                t -= 0.005 * (-Math.Log(u));
                if (t > 1.0) t = 0.999;
                times[spikeIdx] = t;
                sources[spikeIdx] = nrn;
                spikeIdx++;
            }
        }
    }
}

public enum DataPattern { Uniform, Poisson }

public abstract class CrossLanguageBase
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

    protected double[] Sum0Native { get; set; } = null!;
    protected double[] Sum1Native { get; set; } = null!;
    protected double[] OutTimesNative { get; set; } = null!;
    protected int[] OutSourcesNative { get; set; } = null!;
    protected double[] OutCurrentsNative { get; set; } = null!;
    protected double[] GradNative { get; set; } = null!;

    protected SpikesData PostSpikesManaged { get; set; } = null!;
    protected int NPostSpikes { get; set; }
    protected double[] PostTimesNative { get; set; } = null!;
    protected int[] PostSourcesNative { get; set; } = null!;
    protected double[] PostCurrentsNative { get; set; } = null!;
    protected double[] PostErrorsNative { get; set; } = null!;

    protected abstract void Configure();
    protected abstract DataPattern Pattern { get; }

    [GlobalSetup]
    public virtual void Setup()
    {
        Configure();
        W = new double[NPre * NPost];
        Times = new double[NSpikes];
        Sources = new int[NSpikes];

        if (Pattern == DataPattern.Uniform)
            SharedTestData.GenerateUniformData(NPre, NPost, NSpikes, W, Times, Sources);
        else
            SharedTestData.GeneratePoissonData(NPre, NPost, NSpikes, W, Times, Sources);

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);
        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

        int sumRows = NSpikes + 1;
        Sum0Native = new double[sumRows * NPost];
        Sum1Native = new double[sumRows * NPost];

        int maxOutSpikes = NSpikes * NPost;
        OutTimesNative = new double[maxOutSpikes];
        OutSourcesNative = new int[maxOutSpikes];
        OutCurrentsNative = new double[maxOutSpikes];

        PostSpikesManaged = EventPropMath.ComputeSpikes(
            W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        NPostSpikes = PostSpikesManaged.NSpikes;
        for (int i = 0; i < NPostSpikes; i++)
            PostSpikesManaged.SetError(i, 1.0);

        // Allocate to max capacity for TrainingLoop (re-runs forward pass)
        PostTimesNative = new double[maxOutSpikes];
        PostSourcesNative = new int[maxOutSpikes];
        PostCurrentsNative = new double[maxOutSpikes];
        PostErrorsNative = new double[maxOutSpikes];
        for (int i = 0; i < NPostSpikes; i++)
        {
            PostTimesNative[i] = PostSpikesManaged.Times[i];
            PostSourcesNative[i] = PostSpikesManaged.Sources[i];
            PostCurrentsNative[i] = PostSpikesManaged.Currents[i];
            PostErrorsNative[i] = 1.0;
        }
    }
}

// ============================================================
// ComputeSums - Managed vs Native (clear IS necessary: accumulation)
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_ComputeSums_Base : CrossLanguageBase
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
}

// ============================================================
// ComputeSpikes - Managed vs Native (NO clear: overwrites output)
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_ComputeSpikes_Base : CrossLanguageBase
{
    [Benchmark(Baseline = true)]
    public void Managed() =>
        EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public void Native() =>
        NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn,
            OutTimesNative, OutSourcesNative, OutCurrentsNative);
}

// ============================================================
// BackwardSpikes - Managed vs Native (clear IS necessary: accumulation)
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_BackwardSpikes_Base : CrossLanguageBase
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
        NativeApi.BackwardSpikes(Times, Sources, PostTimesNative, PostSourcesNative,
            PostCurrentsNative, PostErrorsNative, W, NPre, NPost, VTh, TauMem, TauSyn, GradNative);
    }
}

// ============================================================
// TrainingLoop (forward + backward) - Managed vs Native
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_TrainingLoop_Base : CrossLanguageBase
{
    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Array.Clear(GradManaged);
        var fwdResult = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        for (int i = 0; i < fwdResult.NSpikes; i++)
            fwdResult.SetError(i, 1.0);
        EventPropMath.BackwardSpikes(W, Times, Sources, fwdResult, NPost, VTh, TauMem, TauSyn, GradManaged, Buffers);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(GradNative);
        int n = NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn,
            OutTimesNative, OutSourcesNative, OutCurrentsNative);
        for (int i = 0; i < n; i++)
            PostErrorsNative[i] = 1.0;
        NativeApi.BackwardSpikes(Times, Sources, OutTimesNative, OutSourcesNative,
            OutCurrentsNative, PostErrorsNative, W, NPre, NPost, VTh, TauMem, TauSyn, GradNative);
    }
}

// ============================================================
// ComputeMaxima - Managed vs Native (overwrites output: no clear)
// ============================================================

[MemoryDiagnoser]
public abstract class Cross_ComputeMaxima_Base : CrossLanguageBase
{
    protected double[] MaxTimesManaged { get; set; } = null!;
    protected double[] MaxValuesManaged { get; set; } = null!;
    protected double[] MaxTimesNative { get; set; } = null!;
    protected double[] MaxValuesNative { get; set; } = null!;

    public override void Setup()
    {
        base.Setup();
        MaxTimesManaged = new double[NPost];
        MaxValuesManaged = new double[NPost];
        MaxTimesNative = new double[NPost];
        MaxValuesNative = new double[NPost];
    }

    [Benchmark(Baseline = true)]
    public void Managed() =>
        EventPropMath.ComputeMaxima(W, Times, Sources, NPost, TauMem, TauSyn, Buffers);

    [Benchmark]
    public void Native() =>
        NativeApi.ComputeMaxima(W, NPre, NPost, Times, Sources, TauMem, TauSyn,
            MaxTimesNative, MaxValuesNative);
}

// ============================================================
// Concrete benchmarks - Medium size (primary comparison target)
// ============================================================

public class Cross_ComputeSums_Medium_Uniform : Cross_ComputeSums_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_ComputeSums_Medium_Poisson : Cross_ComputeSums_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Poisson; }

public class Cross_ComputeSpikes_Medium_Uniform : Cross_ComputeSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_ComputeSpikes_Medium_Poisson : Cross_ComputeSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Poisson; }

public class Cross_BackwardSpikes_Medium_Uniform : Cross_BackwardSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_BackwardSpikes_Medium_Poisson : Cross_BackwardSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Poisson; }

public class Cross_TrainingLoop_Medium_Uniform : Cross_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_ComputeMaxima_Medium_Uniform : Cross_ComputeMaxima_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500); protected override DataPattern Pattern => DataPattern.Uniform; }

// ============================================================
// Small size benchmarks (for fast validation runs)
// ============================================================

public class Cross_ComputeSums_Small_Uniform : Cross_ComputeSums_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_ComputeSpikes_Small_Uniform : Cross_ComputeSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_BackwardSpikes_Small_Uniform : Cross_BackwardSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_TrainingLoop_Small_Uniform : Cross_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100); protected override DataPattern Pattern => DataPattern.Uniform; }

// ============================================================
// Large size benchmarks
// ============================================================

public class Cross_ComputeSums_Large_Uniform : Cross_ComputeSums_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_ComputeSpikes_Large_Uniform : Cross_ComputeSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_BackwardSpikes_Large_Uniform : Cross_BackwardSpikes_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }

public class Cross_TrainingLoop_Large_Uniform : Cross_TrainingLoop_Base
{ protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000); protected override DataPattern Pattern => DataPattern.Uniform; }
