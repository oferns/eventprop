using BenchmarkDotNet.Attributes;
using CW.EventProp;

namespace EventProp.Benchmarks;

// ============================================================
// Micro-benchmarks for hot-path operations
// These measure individual components rather than full algorithms.
// ============================================================

public abstract class MicroBase
{
    protected const double TauMem = 0.02;
    protected const double TauSyn = 0.005;

    // ExpPrecompute
    protected double[] SpikeTimes { get; set; } = null!;

    // BrentRootFinder
    protected double[] MembranePotentials { get; set; } = null!;

    // GradientAccumulation
    protected double[] W { get; set; } = null!;

    // SpikeMergeSort
    protected double[] InputTimes { get; set; } = null!;
    protected int[] InputSources { get; set; } = null!;
    protected double[] PostTimes { get; set; } = null!;
    protected int[] PostSources { get; set; } = null!;

    protected abstract void Configure();

    [GlobalSetup]
    public virtual void Setup()
    {
        Configure();
        var rng = new Random(42);

        // ExpPrecompute: spike times for exp precomputation
        SpikeTimes = new double[SpikeTimes.Length];
        for (int idx = 0; idx < SpikeTimes.Length; idx++)
            SpikeTimes[idx] = (idx + 1) * 0.01; // ~10ms spacing

        // BrentRootFinder: membrane potentials (simulated)
        MembranePotentials = new double[MembranePotentials.Length];
        for (int idx = 0; idx < MembranePotentials.Length; idx++)
            MembranePotentials[idx] = rng.NextDouble() * 2.0;

        // GradientAccumulation: setup
        int nPre = W.Length / 20; // assume post=20 for simplicity, adjust in Configure
        int nPost = W.Length / nPre;

        // SpikeMergeSort: input + post spike times
        int nInput = InputTimes.Length;
        int nPostSpikes = PostTimes.Length;
        InputTimes = new double[nInput];
        InputSources = new int[nInput];
        PostTimes = new double[nPostSpikes];
        PostSources = new int[nPostSpikes];
        for (int idx = 0; idx < nInput; idx++)
        {
            InputTimes[idx] = (idx + 1) * 0.01;
            InputSources[idx] = rng.Next(50);
        }
        for (int idx = 0; idx < nPostSpikes; idx++)
        {
            PostTimes[idx] = (idx + 1) * 0.02;
            PostSources[idx] = rng.Next(30);
        }
    }
}

// ============================================================
// ExpPrecompute: measure SpanAlgorithms.ExpNegDivTau throughput
// ============================================================
public class Micro_ExpPrecompute : MicroBase
{
    protected override void Configure() { }

    [Benchmark(Baseline = true)]
    public double[] Precompute_100()
    {
        SpikeTimes = new double[100];
        return SpanAlgorithms.ExpNegDivTau(SpikeTimes, TauMem);
    }

    [Benchmark]
    public double[] Precompute_500()
    {
        SpikeTimes = new double[500];
        return SpanAlgorithms.ExpNegDivTau(SpikeTimes, TauMem);
    }

    [Benchmark]
    public void Precompute_100_OutSpan()
    {
        SpikeTimes = new double[100];
        var output = new double[SpikeTimes.Length];
        SpanAlgorithms.ExpNegDivTau(SpikeTimes, TauMem, output);
    }

    [Benchmark]
    public void Precompute_500_OutSpan()
    {
        SpikeTimes = new double[500];
        var output = new double[SpikeTimes.Length];
        SpanAlgorithms.ExpNegDivTau(SpikeTimes, TauMem, output);
    }
}

// ============================================================
// BrentRootFinder: measure root-finding iteration cost
// ============================================================
public class Micro_BrentRootFinder : MicroBase
{
    protected override void Configure() { }

    [Benchmark(Baseline = true)]
    public int FindRoots_20()
    {
        MembranePotentials = new double[20];
        var rng = new Random(42);
        for (int idx = 0; idx < MembranePotentials.Length; idx++)
            MembranePotentials[idx] = rng.NextDouble() * 2.0;

        int iterations = 0;
        foreach (var v in MembranePotentials)
        {
            // Simulate a few Brent iterations (not the full algorithm, just measuring iteration cost)
            double a = 0.0, b = v;
            for (int iter = 0; iter < 10 && a < b; iter++)
            {
                double c = (a + b) / 2.0;
                iterations++;
                if (c * c < v) a = c; else b = c;
            }
        }
        return iterations;
    }

    [Benchmark]
    public int FindRoots_200()
    {
        MembranePotentials = new double[200];
        var rng = new Random(42);
        for (int idx = 0; idx < MembranePotentials.Length; idx++)
            MembranePotentials[idx] = rng.NextDouble() * 2.0;

        int iterations = 0;
        foreach (var v in MembranePotentials)
        {
            double a = 0.0, b = v;
            for (int iter = 0; iter < 10 && a < b; iter++)
            {
                double c = (a + b) / 2.0;
                iterations++;
                if (c * c < v) a = c; else b = c;
            }
        }
        return iterations;
    }
}

// ============================================================
// GradientAccumulation: measure backward inner loop throughput
// ============================================================
public class Micro_GradientAccumulation : MicroBase
{
    protected override void Configure() { }

    [Benchmark(Baseline = true)]
    public double Accumulate_50x20()
    {
        W = new double[50 * 20]; // pre=50, post=20
        int nPre = 50, nPost = 20;

        var rng = new Random(42);
        for (int idx = 0; idx < W.Length; idx++)
            W[idx] = 0.5 + rng.NextDouble() * 1.5;

        double[] lambdaV = new double[nPost];
        for (int idx = 0; idx < nPost; idx++) lambdaV[idx] = rng.NextDouble();

        double gradSum = 0;
        for (int pre = 0; pre < nPre; pre++)
            for (int post = 0; post < nPost; post++)
                gradSum += lambdaV[post] * W[pre * nPost + post];

        return gradSum;
    }

    [Benchmark]
    public double Accumulate_200x80()
    {
        W = new double[200 * 80]; // pre=200, post=80
        int nPre = 200, nPost = 80;

        var rng = new Random(42);
        for (int idx = 0; idx < W.Length; idx++)
            W[idx] = 0.5 + rng.NextDouble() * 1.5;

        double[] lambdaV = new double[nPost];
        for (int idx = 0; idx < nPost; idx++) lambdaV[idx] = rng.NextDouble();

        double gradSum = 0;
        for (int pre = 0; pre < nPre; pre++)
            for (int post = 0; post < nPost; post++)
                gradSum += lambdaV[post] * W[pre * nPost + post];

        return gradSum;
    }
}

// ============================================================
// SpikeMergeSort: measure merge + sort throughput
// ============================================================
public class Micro_SpikeMergeSort : MicroBase
{
    protected override void Configure() { }

    [Benchmark(Baseline = true)]
    public int MergeAndSort_100plus30()
    {
        InputTimes = new double[100];
        PostTimes = new double[30];

        for (int idx = 0; idx < InputTimes.Length; idx++)
            InputTimes[idx] = (idx + 1) * 0.01;
        for (int idx = 0; idx < PostTimes.Length; idx++)
            PostTimes[idx] = (idx + 1) * 0.02;

        // Merge two sorted arrays into one
        int n1 = InputTimes.Length, n2 = PostTimes.Length;
        double[] merged = new double[n1 + n2];
        int ii = 0, jj = 0, kk = 0;
        while (ii < n1 && jj < n2)
        {
            if (InputTimes[ii] <= PostTimes[jj]) merged[kk++] = InputTimes[ii++];
            else merged[kk++] = PostTimes[jj++];
        }
        while (ii < n1) merged[kk++] = InputTimes[ii++];
        while (jj < n2) merged[kk++] = PostTimes[jj++];

        // Sort descending
        Array.Sort(merged);
        Array.Reverse(merged);

        return merged.Length;
    }

    [Benchmark]
    public int MergeAndSort_500plus150()
    {
        InputTimes = new double[500];
        PostTimes = new double[150];

        for (int idx = 0; idx < InputTimes.Length; idx++)
            InputTimes[idx] = (idx + 1) * 0.01;
        for (int idx = 0; idx < PostTimes.Length; idx++)
            PostTimes[idx] = (idx + 1) * 0.02;

        int n1 = InputTimes.Length, n2 = PostTimes.Length;
        double[] merged = new double[n1 + n2];
        int iii = 0, jjj = 0, kkk = 0;
        while (iii < n1 && jjj < n2)
        {
            if (InputTimes[iii] <= PostTimes[jjj]) merged[kkk++] = InputTimes[iii++];
            else merged[kkk++] = PostTimes[jjj++];
        }
        while (iii < n1) merged[kkk++] = InputTimes[iii++];
        while (jjj < n2) merged[kkk++] = PostTimes[jjj++];

        Array.Sort(merged);
        Array.Reverse(merged);

        return merged.Length;
    }
}
