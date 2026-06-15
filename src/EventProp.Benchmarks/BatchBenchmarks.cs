using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

// ============================================================
// Batch processing benchmarks — Managed (per-sample loop) vs Native (batch C API)
// Tests realistic batch sizes (4, 8, 16) matching Python benchmark coverage.
// ============================================================

public abstract class Bench_Batch_Base : BenchBase
{
    protected int BatchSize { get; set; }
    protected double[][] BatchTimes { get; set; } = null!;
    protected int[][] BatchSources { get; set; } = null!;

    // Layers
    protected CW.EventProp.LIFLayer? ManagedLayer { get; set; }
    protected CApi.LIFLayer? NativeLayer { get; set; }

    public override void Setup()
    {
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        for (int i = 0; i < W.Length; i++)
            W[i] = 0.5 + rng.NextDouble() * 1.5;

        // Generate per-sample spike data
        BatchTimes = new double[BatchSize][];
        BatchSources = new int[BatchSize][];
        for (int b = 0; b < BatchSize; b++)
        {
            BatchTimes[b] = new double[NSpikes];
            BatchSources[b] = new int[NSpikes];
            for (int i = 0; i < NSpikes; i++)
            {
                BatchTimes[b][i] = (i + 1) * 1.0 / (NSpikes + 1);
                BatchSources[b][i] = rng.Next(NPre);
            }
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
// Batch Forward — LIF layer forward pass with batch sizes 4, 8, 16
// ============================================================

[MemoryDiagnoser]
public class Bench_Batch_Forward_4 : Bench_Batch_Base
{
    protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 4; }

    [Benchmark(Baseline = true)]
    public void Managed_PerSampleLoop()
    {
        var batch = new SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new SpikesData(BatchTimes[b], BatchSources[b]));

        ManagedLayer!.Forward(batch);
    }

    [Benchmark]
    public void Native_BatchApi()
    {
        var batch = new CApi.SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new CApi.SpikesData(BatchTimes[b], BatchSources[b]));

        NativeLayer!.Forward(batch);
    }
}

[MemoryDiagnoser]
public class Bench_Batch_Forward_8 : Bench_Batch_Base
{
    protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 8; }

    [Benchmark(Baseline = true)]
    public void Managed_PerSampleLoop()
    {
        var batch = new SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new SpikesData(BatchTimes[b], BatchSources[b]));

        ManagedLayer!.Forward(batch);
    }

    [Benchmark]
    public void Native_BatchApi()
    {
        var batch = new CApi.SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new CApi.SpikesData(BatchTimes[b], BatchSources[b]));

        NativeLayer!.Forward(batch);
    }
}

[MemoryDiagnoser]
public class Bench_Batch_Forward_16 : Bench_Batch_Base
{
    protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 16; }

    [Benchmark(Baseline = true)]
    public void Managed_PerSampleLoop()
    {
        var batch = new SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new SpikesData(BatchTimes[b], BatchSources[b]));

        ManagedLayer!.Forward(batch);
    }

    [Benchmark]
    public void Native_BatchApi()
    {
        var batch = new CApi.SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new CApi.SpikesData(BatchTimes[b], BatchSources[b]));

        NativeLayer!.Forward(batch);
    }
}

// ============================================================
// Batch Forward + Backward — full training step with batch sizes 4, 8
// ============================================================

[MemoryDiagnoser]
public class Bench_Batch_TrainingStep_4 : Bench_Batch_Base
{
    protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 4; }

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var batch = new SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new SpikesData(BatchTimes[b], BatchSources[b]));

        ManagedLayer!.Forward(batch);
        var post = ManagedLayer.Output!;
        for (int b = 0; b < BatchSize; b++)
            for (int i = 0; i < post[b].NSpikes; i++)
                post[b].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var batch = new CApi.SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new CApi.SpikesData(BatchTimes[b], BatchSources[b]));

        NativeLayer!.Forward(batch);
        var post = NativeLayer.Output!;
        for (int b = 0; b < BatchSize; b++)
            for (int i = 0; i < post[b].NSpikes; i++)
                post[b].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}

[MemoryDiagnoser]
public class Bench_Batch_TrainingStep_8 : Bench_Batch_Base
{
    protected override void Configure() { (NPre, NPost, NSpikes) = (50, 20, 100); BatchSize = 8; }

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        var batch = new SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new SpikesData(BatchTimes[b], BatchSources[b]));

        ManagedLayer!.Forward(batch);
        var post = ManagedLayer.Output!;
        for (int b = 0; b < BatchSize; b++)
            for (int i = 0; i < post[b].NSpikes; i++)
                post[b].SetError(i, 1.0);
        ManagedLayer.Backward();
    }

    [Benchmark]
    public void Native()
    {
        var batch = new CApi.SpikesVector();
        for (int b = 0; b < BatchSize; b++)
            batch.Add(new CApi.SpikesData(BatchTimes[b], BatchSources[b]));

        NativeLayer!.Forward(batch);
        var post = NativeLayer.Output!;
        for (int b = 0; b < BatchSize; b++)
            for (int i = 0; i < post[b].NSpikes; i++)
                post[b].SetError(i, 1.0);
        NativeLayer.Backward();
    }
}

// ============================================================
// Raw batch API benchmarks — direct CApi.Batch.ComputeSpikesBatch call
// (bypasses layer overhead, measures pure C batch function)
// ============================================================

[MemoryDiagnoser]
public class Bench_Batch_RawApi_4 : BenchBase
{
    protected int BatchSize => 4;

    public override void Setup()
    {
        Configure();
        var rng = new Random(42);

        W = new double[NPre * NPost];
        for (int i = 0; i < W.Length; i++)
            W[i] = 0.5 + rng.NextDouble() * 1.5;

        // Flatten batch into concatenated arrays
        int totalSpikes = NSpikes * BatchSize;
        Times = new double[totalSpikes];
        Sources = new int[totalSpikes];
        var nSpikesPerSample = new int[BatchSize];

        for (int b = 0; b < BatchSize; b++)
        {
            nSpikesPerSample[b] = NSpikes;
            for (int i = 0; i < NSpikes; i++)
            {
                Times[b * NSpikes + i] = (i + 1) * 1.0 / (NSpikes + 1);
                Sources[b * NSpikes + i] = rng.Next(NPre);
            }
        }

        // Store for benchmarks
        _nSpikesPerSample = nSpikesPerSample;

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, NSpikes);
    }

    private int[] _nSpikesPerSample = null!;

    protected override void Configure() => (NPre, NPost, NSpikes) = (50, 20, 100);

    [Benchmark(Baseline = true)]
    public void Managed_PerSampleLoop()
    {
        for (int b = 0; b < BatchSize; b++)
        {
            int offset = b * NSpikes;
            var sampleTimes = new double[NSpikes];
            var sampleSources = new int[NSpikes];
            System.Array.Copy(Times, offset, sampleTimes, 0, NSpikes);
            System.Array.Copy(Sources, offset, sampleSources, 0, NSpikes);

            var result = EventPropMath.ComputeSpikes(W, sampleTimes, sampleSources, NPost, VTh, TauMem, TauSyn, Buffers);
        }
    }

    [Benchmark]
    public void Native_BatchApi()
    {
        using var output = CApi.Batch.ComputeSpikesBatch(
            W, NPre, NPost, Times, Sources, _nSpikesPerSample, VTh, TauMem, TauSyn);
        var _ = output[0].NSpikes; // prevent optimization
    }

    [Benchmark]
    public void Native_PerSampleLoop()
    {
        for (int b = 0; b < BatchSize; b++)
        {
            int offset = b * NSpikes;
            using var output = CApi.Api.ComputeSpikes(
                W, NPre, NPost,
                Times.AsSpan(offset, NSpikes), Sources.AsSpan(offset, NSpikes),
                VTh, TauMem, TauSyn);
            var _ = output.NSpikes; // prevent optimization
        }
    }
}
