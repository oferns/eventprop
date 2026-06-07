using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using CW.EventProp;

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);

public abstract class BenchBase
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
    protected ScratchBuffers Buffers { get; set; } = null!;

    protected double[] Sum0Native { get; set; } = null!;
    protected double[] Sum1Native { get; set; } = null!;
    protected double[] OutTimesNative { get; set; } = null!;
    protected int[] OutSourcesNative { get; set; } = null!;
    protected double[] OutCurrentsNative { get; set; } = null!;

    protected SpikesData PostSpikesManaged { get; set; } = null!;
    protected double[] PostTimesNative { get; set; } = null!;
    protected int[] PostSourcesNative { get; set; } = null!;
    protected double[] PostCurrentsNative { get; set; } = null!;
    protected double[] PostErrorsNative { get; set; } = null!;
    protected double[] SpikeErrorsNative { get; set; } = null!;

    protected double[] GradManaged { get; set; } = null!;
    protected double[] GradNative { get; set; } = null!;

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

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);

        int sumRows = NSpikes + 1;
        Sum0Native = new double[sumRows * NPost];
        Sum1Native = new double[sumRows * NPost];

        int maxOutSpikes = NSpikes * NPost;
        OutTimesNative = new double[maxOutSpikes];
        OutSourcesNative = new int[maxOutSpikes];
        OutCurrentsNative = new double[maxOutSpikes];

        var fwdResult = EventPropMath.ComputeSpikes(
            W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

        int nPost = fwdResult.NSpikes;
        PostSpikesManaged = fwdResult;
        for (int i = 0; i < nPost; i++)
            PostSpikesManaged.SetError(i, 1.0);

        PostTimesNative = new double[nPost];
        PostSourcesNative = new int[nPost];
        PostCurrentsNative = new double[nPost];
        PostErrorsNative = new double[nPost];
        SpikeErrorsNative = new double[NSpikes];
        for (int i = 0; i < nPost; i++)
        {
            PostTimesNative[i] = fwdResult.Times[i];
            PostSourcesNative[i] = fwdResult.Sources[i];
            PostCurrentsNative[i] = fwdResult.Currents[i];
            PostErrorsNative[i] = 1.0;
        }

        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];
    }
}

[MemoryDiagnoser]
public class Bench_ComputeSums_Small : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

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
        using var output = NativeApi.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        var _ = output.Sum0[0]; // Access to prevent optimization
    }
}

[MemoryDiagnoser]
public class Bench_ComputeSums_Medium : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

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
        using var output = NativeApi.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        var _ = output.Sum0[0]; // Access to prevent optimization
    }
}

[MemoryDiagnoser]
public class Bench_ComputeSums_Large : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

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
        using var output = NativeApi.ComputeSums(W, NPre, NPost, Times, Sources, TauMem, TauSyn);
        var _ = output.Sum0[0]; // Access to prevent optimization
    }
}

[MemoryDiagnoser]
public class Bench_ComputeSpikes_Small : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

    [Benchmark(Baseline = true)]
    public SpikesData Managed()
        => EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public int Native()
    {
        using var output = NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        return output.NSpikes;
    }
}

[MemoryDiagnoser]
public class Bench_ComputeSpikes_Medium : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

    [Benchmark(Baseline = true)]
    public SpikesData Managed()
        => EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public int Native()
    {
        using var output = NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        return output.NSpikes;
    }
}

[MemoryDiagnoser]
public class Bench_ComputeSpikes_Large : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

    [Benchmark(Baseline = true)]
    public SpikesData Managed()
        => EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);

    [Benchmark]
    public int Native()
    {
        using var output = NativeApi.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        return output.NSpikes;
    }
}

[MemoryDiagnoser]
public class Bench_BackwardSpikes_Small : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

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
        using var output = NativeApi.BackwardSpikes(Times, Sources,
            PostTimesNative, PostSourcesNative, PostCurrentsNative,
            PostErrorsNative, NSpikes, W, NPre, NPost,  // Use NSpikes as nPostSpikes
            VTh, TauMem, TauSyn);
        var _ = output.Grad[0]; // Access to prevent optimization
    }
}

[MemoryDiagnoser]
public class Bench_BackwardSpikes_Medium : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (200, 80, 500);

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
        using var output = NativeApi.BackwardSpikes(Times, Sources,
            PostTimesNative, PostSourcesNative, PostCurrentsNative,
            PostErrorsNative, NSpikes, W, NPre, NPost,  // Use NSpikes as nPostSpikes
            VTh, TauMem, TauSyn);
        var _ = output.Grad[0]; // Access to prevent optimization
    }
}

[MemoryDiagnoser]
public class Bench_BackwardSpikes_Large : BenchBase
{
    protected override void Configure() => (NPre, NPost, NSpikes) = (500, 200, 1000);

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
        using var output = NativeApi.BackwardSpikes(Times, Sources,
            PostTimesNative, PostSourcesNative, PostCurrentsNative,
            PostErrorsNative, NSpikes, W, NPre, NPost,  // Use NSpikes as nPostSpikes
            VTh, TauMem, TauSyn);
        var _ = output.Grad[0]; // Access to prevent optimization
    }
}
