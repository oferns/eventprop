using BenchmarkDotNet.Attributes;
using CW.EventProp;
using CApi = CW.EventProp.Native;

namespace EventProp.Benchmarks;

[MemoryDiagnoser]
public abstract class Bench_TrainingLoopSweep_Base : BenchBase
{
    public int SpikeCount { get; protected set; } = 0;
    protected SpikesData? PostSpikes { get; set; }

    public override void Setup()
    {
        Configure();
        var rng = new Random(42);
        W = new double[NPre * NPost];
        for (int i = 0; i < W.Length; i++) W[i] = 0.5 + rng.NextDouble() * 1.5;

        Times = new double[SpikeCount];
        Sources = new int[SpikeCount];
        for (int i = 0; i < SpikeCount; i++)
        {
            Times[i] = (i + 1) * 1.0 / (SpikeCount + 1);
            Sources[i] = rng.Next(NPre);
        }

        Buffers = new ScratchBuffers();
        Buffers.EnsureCapacity(NPost, NPre, SpikeCount);
        int sumRows = SpikeCount + 1;
        Sum0Native = new double[sumRows * NPost];
        Sum1Native = new double[sumRows * NPost];

        int maxOutSpikes = SpikeCount * NPost;
        OutTimesManaged = new double[maxOutSpikes];
        OutSourcesManaged = new int[maxOutSpikes];
        OutCurrentsManaged = new double[maxOutSpikes];
        OutErrorsManaged = new double[maxOutSpikes];
        GradManaged = new double[NPre * NPost];
        GradNative = new double[NPre * NPost];

        PostSpikes = EventPropMath.ComputeSpikes(W, Times, Sources, NPost, VTh, TauMem, TauSyn, Buffers);
        for (int i = 0; i < PostSpikes.NSpikes; i++) PostSpikes.SetError(i, 1.0);

        int nPost = PostSpikes.NSpikes;
        PostTimesNative = new double[nPost];
        PostSourcesNative = new int[nPost];
        PostErrorsNative = new double[nPost];
        for (int i = 0; i < nPost; i++)
        {
            PostTimesNative[i] = PostSpikes.Times[i];
            PostSourcesNative[i] = PostSpikes.Sources[i];
            PostErrorsNative[i] = 1.0;
        }
    }

    [Benchmark(Baseline = true)]
    public void Managed()
    {
        Array.Clear(GradManaged);
        int nOut = EventPropMath.ComputeSpikesInto(W, Times, Sources, NPost, VTh, TauMem, TauSyn,
            OutTimesManaged.AsSpan(), OutSourcesManaged.AsSpan(), OutCurrentsManaged.AsSpan(), Buffers);
        for (int i = 0; i < nOut; i++) OutErrorsManaged[i] = 1.0;
        EventPropMath.BackwardSpikesRaw(W, Times, Sources,
            OutTimesManaged.AsSpan(0, nOut), OutSourcesManaged.AsSpan(0, nOut),
            OutCurrentsManaged.AsSpan(0, nOut), OutErrorsManaged.AsSpan(0, nOut),
            NPost, VTh, TauMem, TauSyn, GradManaged.AsSpan(), Buffers);
    }

    [Benchmark]
    public void Native()
    {
        Array.Clear(GradNative);
        using var output = CApi.Api.ComputeSpikes(W, NPre, NPost, Times, Sources, VTh, TauMem, TauSyn);
        int nOut = output.NSpikes;
        using var backwardOutput = CApi.Api.BackwardSpikes(Times, Sources,
            output.Times, output.Sources, PostErrorsNative, nOut, W, NPre, NPost, VTh, TauMem, TauSyn);
    }
}

// Tiny (10->5) x spike counts
public class Bench_TrainingLoopSweep_Tiny_10 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(10,5,10); public override void Setup(){SpikeCount=10;base.Setup();} }
public class Bench_TrainingLoopSweep_Tiny_100 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(10,5,10); public override void Setup(){SpikeCount=100;base.Setup();} }
public class Bench_TrainingLoopSweep_Tiny_500 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(10,5,10); public override void Setup(){SpikeCount=500;base.Setup();} }
public class Bench_TrainingLoopSweep_Tiny_1000 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(10,5,10); public override void Setup(){SpikeCount=1000;base.Setup();} }

// Small (50->20) x spike counts
public class Bench_TrainingLoopSweep_Small_10 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(50,20,10); public override void Setup(){SpikeCount=10;base.Setup();} }
public class Bench_TrainingLoopSweep_Small_100 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(50,20,10); public override void Setup(){SpikeCount=100;base.Setup();} }
public class Bench_TrainingLoopSweep_Small_500 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(50,20,10); public override void Setup(){SpikeCount=500;base.Setup();} }
public class Bench_TrainingLoopSweep_Small_1000 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(50,20,10); public override void Setup(){SpikeCount=1000;base.Setup();} }

// Medium (200->80) x spike counts
public class Bench_TrainingLoopSweep_Medium_10 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(200,80,10); public override void Setup(){SpikeCount=10;base.Setup();} }
public class Bench_TrainingLoopSweep_Medium_100 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(200,80,10); public override void Setup(){SpikeCount=100;base.Setup();} }
public class Bench_TrainingLoopSweep_Medium_500 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(200,80,10); public override void Setup(){SpikeCount=500;base.Setup();} }
public class Bench_TrainingLoopSweep_Medium_1000 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(200,80,10); public override void Setup(){SpikeCount=1000;base.Setup();} }

// Large (500->200) x spike counts
public class Bench_TrainingLoopSweep_Large_10 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(500,200,10); public override void Setup(){SpikeCount=10;base.Setup();} }
public class Bench_TrainingLoopSweep_Large_100 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(500,200,10); public override void Setup(){SpikeCount=100;base.Setup();} }
public class Bench_TrainingLoopSweep_Large_500 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(500,200,10); public override void Setup(){SpikeCount=500;base.Setup();} }
public class Bench_TrainingLoopSweep_Large_1000 : Bench_TrainingLoopSweep_Base { protected override void Configure() => (NPre,NPost,NSpikes)=(500,200,10); public override void Setup(){SpikeCount=1000;base.Setup();} }
