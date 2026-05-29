using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// C (NativeApi) vs .NET (EventPropMath) parity tests.
/// These verify that the managed C# implementation produces results identical to the native C library.
/// 
/// The C library is treated as a reference implementation — both should match the Eigen core (NativeCppApi)
/// within tolerance. If C and .NET disagree, investigate which one matches Eigen first.
/// </summary>
public class NativeVsManagedTests
{
    /// <summary>Matches C/C++ parity test tolerances: ABS_TOLERANCE=1e-10, REL_TOLERANCE=1e-9.
    /// Fails only if BOTH: diff > abs_tol AND diff > rel_tol * scale.</summary>
    private const double AbsTolerance = 1e-10;
    private const double RelTolerance = 1e-9;

    // ========================================================================
    // Shared Data Generation (seed 42, deterministic) — matches NativeLibraryParityTests
    // ========================================================================

    private static void GenerateUniformData(Random rng, int nPre, int nPost, int nSpikes,
        out double[] w, out double[] times, out int[] sources)
    {
        w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = rng.NextDouble() * 2.0 + 0.1;

        times = new double[nSpikes];
        sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = rng.NextDouble() * 1.0;
            sources[i] = rng.Next(nPre);
        }
    }

    private static void GeneratePoissonData(Random rng, int nPre, int nPost,
        out double[] w, out double[] times, out int[] sources)
    {
        w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = rng.NextDouble() * 2.0 + 0.1;

        var spikeList = new List<(double time, int source)>();
        double t = 0.0;
        while (t < 1.0)
        {
            double uP = rng.NextDouble();
            t += -Math.Log(1.0 - uP) / 200.0;
            if (t < 1.0)
                spikeList.Add((t, rng.Next(nPre)));
        }

        times = new double[spikeList.Count];
        sources = new int[spikeList.Count];
        for (int i = 0; i < spikeList.Count; i++)
        {
            times[i] = spikeList[i].time;
            sources[i] = spikeList[i].source;
        }
    }

    private static void GenerateBurstData(Random rng, int nPre, int nPost,
        out double[] w, out double[] times, out int[] sources)
    {
        w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = rng.NextDouble() * 2.0 + 0.1;

        var spikeList = new List<(double time, int source)>();
        double t = 0.0;
        while (t < 1.0)
        {
            int burstSize = rng.Next(3, 8);
            for (int b = 0; b < burstSize && t < 1.0; b++)
            {
                double uB = rng.NextDouble();
                t += -Math.Log(1.0 - uB) / 500.0;
                if (t < 1.0)
                    spikeList.Add((t, rng.Next(nPre)));
            }
            double uI = rng.NextDouble();
            t += -Math.Log(1.0 - uI) / 50.0;
        }

        times = new double[spikeList.Count];
        sources = new int[spikeList.Count];
        for (int i = 0; i < spikeList.Count; i++)
        {
            times[i] = spikeList[i].time;
            sources[i] = spikeList[i].source;
        }
    }

    private static void SortTimesSources(ref double[] times, ref int[] sources)
    {
        var pairTimes = new (double time, int source)[times.Length];
        for (int i = 0; i < times.Length; i++)
            pairTimes[i] = (times[i], sources[i]);
        Array.Sort(pairTimes, (a, b) => a.time.CompareTo(b.time));
        for (int i = 0; i < times.Length; i++)
        {
            times[i] = pairTimes[i].time;
            sources[i] = pairTimes[i].source;
        }
    }

    // ========================================================================
    // Scaled Tolerance Helper (matches C/C++ parity test logic)
    // ========================================================================

    private static void AssertScaled(double actual, double expected, string name, int index)
    {
        double diff = Math.Abs(actual - expected);
        double scale = Math.Max(Math.Abs(actual), Math.Abs(expected));
        if (diff > AbsTolerance && diff > RelTolerance * scale)
            throw new Exception($"Mismatch [{name}[{index}]]: {actual} vs {expected}, diff={diff}");
    }

    // ========================================================================
    // ComputeSums Tests — C vs .NET
    // ========================================================================

    [Test]
    public async Task ComputeSums_Uniform_Tiny() => await CompareNativeComputeSums("uniform", 10, 5, 20);

    [Test]
    public async Task ComputeSums_Uniform_Small() => await CompareNativeComputeSums("uniform", 50, 20, 100);

    [Test]
    public async Task ComputeSums_Uniform_Medium() => await CompareNativeComputeSums("uniform", 200, 80, 400);

    [Test]
    public async Task ComputeSums_Uniform_Large() => await CompareNativeComputeSums("uniform", 500, 200, 1000);

    [Test]
    public async Task ComputeSums_Poisson_Tiny() => await CompareNativeComputeSums("poisson", 10, 5, 20);

    [Test]
    public async Task ComputeSums_Poisson_Small() => await CompareNativeComputeSums("poisson", 50, 20, 100);

    [Test]
    public async Task ComputeSums_Poisson_Medium() => await CompareNativeComputeSums("poisson", 200, 80, 400);

    [Test]
    public async Task ComputeSums_Poisson_Large() => await CompareNativeComputeSums("poisson", 500, 200, 1000);

    [Test]
    public async Task ComputeSums_Burst_Tiny() => await CompareNativeComputeSums("burst", 10, 5, 20);

    [Test]
    public async Task ComputeSums_Burst_Small() => await CompareNativeComputeSums("burst", 50, 20, 100);

    [Test]
    public async Task ComputeSums_Burst_Medium() => await CompareNativeComputeSums("burst", 200, 80, 400);

    [Test]
    public async Task ComputeSums_Burst_Large() => await CompareNativeComputeSums("burst", 500, 200, 1000);

    private async Task CompareNativeComputeSums(string pattern, int nPre, int nPost, int maxSpikes)
    {
        var rng = new Random(42);

        double[] w, times; int[] sources;
        switch (pattern)
        {
            case "uniform": GenerateUniformData(rng, nPre, nPost, maxSpikes, out w, out times, out sources); break;
            case "poisson": GeneratePoissonData(rng, nPre, nPost, out w, out times, out sources); break;
            case "burst": GenerateBurstData(rng, nPre, nPost, out w, out times, out sources); break;
            default: throw new ArgumentException($"Unknown pattern: {pattern}");
        }

        SortTimesSources(ref times, ref sources);

        // .NET implementation
        var csSum0 = new double[(times.Length + 1) * nPost];
        var csSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, csSum0, csSum1, nPost,
            new double[times.Length], new double[times.Length]);

        // C library via P/Invoke
        var cSum0 = new double[(times.Length + 1) * nPost];
        var cSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, cSum0, cSum1);

        // Assert: .NET matches C (scaled tolerance)
        for (int i = 0; i < csSum0.Length; i++)
        {
            AssertScaled(csSum0[i], cSum0[i], "sum0", i);
            AssertScaled(csSum1[i], cSum1[i], "sum1", i);
        }
    }

    // ========================================================================
    // ComputeSpikes Tests — C vs .NET
    // ========================================================================

    [Test]
    public async Task ComputeSpikes_Uniform_Tiny() => await CompareNativeComputeSpikes("uniform", 10, 5, 20);

    [Test]
    public async Task ComputeSpikes_Uniform_Small() => await CompareNativeComputeSpikes("uniform", 50, 20, 100);

    [Test]
    public async Task ComputeSpikes_Uniform_Medium() => await CompareNativeComputeSpikes("uniform", 200, 80, 400);

    [Test]
    public async Task ComputeSpikes_Uniform_Large() => await CompareNativeComputeSpikes("uniform", 500, 200, 1000);

    [Test]
    public async Task ComputeSpikes_Poisson_Tiny() => await CompareNativeComputeSpikes("poisson", 10, 5, 20);

    [Test]
    public async Task ComputeSpikes_Poisson_Small() => await CompareNativeComputeSpikes("poisson", 50, 20, 100);

    [Test]
    public async Task ComputeSpikes_Poisson_Medium() => await CompareNativeComputeSpikes("poisson", 200, 80, 400);

    [Test]
    public async Task ComputeSpikes_Poisson_Large() => await CompareNativeComputeSpikes("poisson", 500, 200, 1000);

    [Test]
    public async Task ComputeSpikes_Burst_Tiny() => await CompareNativeComputeSpikes("burst", 10, 5, 20);

    [Test]
    public async Task ComputeSpikes_Burst_Small() => await CompareNativeComputeSpikes("burst", 50, 20, 100);

    [Test]
    public async Task ComputeSpikes_Burst_Medium() => await CompareNativeComputeSpikes("burst", 200, 80, 400);

    [Test]
    public async Task ComputeSpikes_Burst_Large() => await CompareNativeComputeSpikes("burst", 500, 200, 1000);

    private async Task CompareNativeComputeSpikes(string pattern, int nPre, int nPost, int maxSpikes)
    {
        var rng = new Random(42);

        double[] w, times; int[] sources;
        switch (pattern)
        {
            case "uniform": GenerateUniformData(rng, nPre, nPost, maxSpikes, out w, out times, out sources); break;
            case "poisson": GeneratePoissonData(rng, nPre, nPost, out w, out times, out sources); break;
            case "burst": GenerateBurstData(rng, nPre, nPost, out w, out times, out sources); break;
            default: throw new ArgumentException($"Unknown pattern: {pattern}");
        }

        SortTimesSources(ref times, ref sources);

        // .NET implementation
        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, times.Length);
        var csResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // C library via P/Invoke
        int maxOutSpikes = times.Length * nPost;
        var cTimes = new double[maxOutSpikes];
        var cSources = new int[maxOutSpikes];
        var cCurrents = new double[maxOutSpikes];
        int nCSpikes = NativeApi.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, cTimes, cSources, cCurrents);

        // Assert: spike counts match
        await Assert.That(csResult.NSpikes).IsEqualTo(nCSpikes);

        // Then element-wise comparison (scaled tolerance)
        for (int i = 0; i < csResult.NSpikes; i++)
        {
            AssertScaled(csResult.Times[i], cTimes[i], "times", i);
            await Assert.That(csResult.Sources[i]).IsEqualTo(cSources[i]);
            AssertScaled(csResult.Currents[i], cCurrents[i], "currents", i);
        }
    }

    // ========================================================================
    // ComputeMaxima Tests — C vs .NET
    // ========================================================================

    [Test]
    public async Task ComputeMaxima_Uniform_Tiny() => await CompareNativeComputeMaxima("uniform", 10, 5, 20);

    [Test]
    public async Task ComputeMaxima_Uniform_Small() => await CompareNativeComputeMaxima("uniform", 50, 20, 100);

    [Test]
    public async Task ComputeMaxima_Uniform_Medium() => await CompareNativeComputeMaxima("uniform", 200, 80, 400);

    [Test]
    public async Task ComputeMaxima_Uniform_Large() => await CompareNativeComputeMaxima("uniform", 500, 200, 1000);

    [Test]
    public async Task ComputeMaxima_Poisson_Tiny() => await CompareNativeComputeMaxima("poisson", 10, 5, 20);

    [Test]
    public async Task ComputeMaxima_Poisson_Small() => await CompareNativeComputeMaxima("poisson", 50, 20, 100);

    [Test]
    public async Task ComputeMaxima_Poisson_Medium() => await CompareNativeComputeMaxima("poisson", 200, 80, 400);

    [Test]
    public async Task ComputeMaxima_Poisson_Large() => await CompareNativeComputeMaxima("poisson", 500, 200, 1000);

    [Test]
    public async Task ComputeMaxima_Burst_Tiny() => await CompareNativeComputeMaxima("burst", 10, 5, 20);

    [Test]
    public async Task ComputeMaxima_Burst_Small() => await CompareNativeComputeMaxima("burst", 50, 20, 100);

    [Test]
    public async Task ComputeMaxima_Burst_Medium() => await CompareNativeComputeMaxima("burst", 200, 80, 400);

    [Test]
    public async Task ComputeMaxima_Burst_Large() => await CompareNativeComputeMaxima("burst", 500, 200, 1000);

    private async Task CompareNativeComputeMaxima(string pattern, int nPre, int nPost, int maxSpikes)
    {
        var rng = new Random(42);

        double[] w, times; int[] sources;
        switch (pattern)
        {
            case "uniform": GenerateUniformData(rng, nPre, nPost, maxSpikes, out w, out times, out sources); break;
            case "poisson": GeneratePoissonData(rng, nPre, nPost, out w, out times, out sources); break;
            case "burst": GenerateBurstData(rng, nPre, nPost, out w, out times, out sources); break;
            default: throw new ArgumentException($"Unknown pattern: {pattern}");
        }

        SortTimesSources(ref times, ref sources);

        // .NET implementation
        var csMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        // C library via P/Invoke
        var cTimes = new double[nPost];
        var cValues = new double[nPost];
        NativeApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cTimes, cValues);

        // Assert: C matches .NET (scaled tolerance)
        for (int i = 0; i < nPost; i++)
        {
            if (double.IsNaN(csMaxima.Times[i]))
            {
                await Assert.That(cTimes[i]).IsNaN();
            }
            else
            {
                AssertScaled(csMaxima.Times[i], cTimes[i], "max_times", i);
                AssertScaled(csMaxima.Values[i], cValues[i], "max_values", i);
            }
        }
    }

    // ========================================================================
    // BackwardSpikes Tests — C vs .NET
    // ========================================================================

    [Test]
    public async Task BackwardSpikes_Uniform_Tiny() => await CompareNativeBackwardSpikes("uniform", 10, 5, 20);

    [Test]
    public async Task BackwardSpikes_Uniform_Small() => await CompareNativeBackwardSpikes("uniform", 50, 20, 100);

    [Test]
    public async Task BackwardSpikes_Uniform_Medium() => await CompareNativeBackwardSpikes("uniform", 200, 80, 400);

    [Test]
    public async Task BackwardSpikes_Uniform_Large() => await CompareNativeBackwardSpikes("uniform", 500, 200, 1000);

    [Test]
    public async Task BackwardSpikes_Poisson_Tiny() => await CompareNativeBackwardSpikes("poisson", 10, 5, 20);

    [Test]
    public async Task BackwardSpikes_Poisson_Small() => await CompareNativeBackwardSpikes("poisson", 50, 20, 100);

    [Test]
    public async Task BackwardSpikes_Poisson_Medium() => await CompareNativeBackwardSpikes("poisson", 200, 80, 400);

    [Test]
    public async Task BackwardSpikes_Poisson_Large() => await CompareNativeBackwardSpikes("poisson", 500, 200, 1000);

    [Test]
    public async Task BackwardSpikes_Burst_Tiny() => await CompareNativeBackwardSpikes("burst", 10, 5, 20);

    [Test]
    public async Task BackwardSpikes_Burst_Small() => await CompareNativeBackwardSpikes("burst", 50, 20, 100);

    [Test]
    public async Task BackwardSpikes_Burst_Medium() => await CompareNativeBackwardSpikes("burst", 200, 80, 400);

    [Test]
    public async Task BackwardSpikes_Burst_Large() => await CompareNativeBackwardSpikes("burst", 500, 200, 1000);

    private async Task CompareNativeBackwardSpikes(string pattern, int nPre, int nPost, int maxSpikes)
    {
        var rng = new Random(42);

        double[] w, times; int[] sources;
        switch (pattern)
        {
            case "uniform": GenerateUniformData(rng, nPre, nPost, maxSpikes, out w, out times, out sources); break;
            case "poisson": GeneratePoissonData(rng, nPre, nPost, out w, out times, out sources); break;
            case "burst": GenerateBurstData(rng, nPre, nPost, out w, out times, out sources); break;
            default: throw new ArgumentException($"Unknown pattern: {pattern}");
        }

        SortTimesSources(ref times, ref sources);

        // Forward pass using .NET (we just need post-spike data; both C and .NET use the same input)
        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, times.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes == 0) return; // Skip — no post-spike activity

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        // .NET implementation
        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardSpikes(w, times, sources, postSpikes, nPost,
            1.0, 0.02, 0.005, csGradient, buffers);

        // C library via P/Invoke
        var cGradient = new double[nPre * nPost]; Array.Fill(cGradient, 0.0);
        NativeApi.BackwardSpikes(times, sources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cGradient, new double[1024]);

        // Assert: .NET matches C (scaled tolerance)
        for (int i = 0; i < csGradient.Length; i++)
        {
            AssertScaled(csGradient[i], cGradient[i], "gradient", i);
        }
    }

    // ========================================================================
    // BackwardMaxima Tests — C vs .NET
    // ========================================================================

    [Test]
    public async Task BackwardMaxima_Uniform_Tiny() => await CompareNativeBackwardMaxima("uniform", 10, 5, 20);

    [Test]
    public async Task BackwardMaxima_Uniform_Small() => await CompareNativeBackwardMaxima("uniform", 50, 20, 100);

    [Test]
    public async Task BackwardMaxima_Uniform_Medium() => await CompareNativeBackwardMaxima("uniform", 200, 80, 400);

    [Test]
    public async Task BackwardMaxima_Uniform_Large() => await CompareNativeBackwardMaxima("uniform", 500, 200, 1000);

    [Test]
    public async Task BackwardMaxima_Poisson_Tiny() => await CompareNativeBackwardMaxima("poisson", 10, 5, 20);

    [Test]
    public async Task BackwardMaxima_Poisson_Small() => await CompareNativeBackwardMaxima("poisson", 50, 20, 100);

    [Test]
    public async Task BackwardMaxima_Poisson_Medium() => await CompareNativeBackwardMaxima("poisson", 200, 80, 400);

    [Test]
    public async Task BackwardMaxima_Poisson_Large() => await CompareNativeBackwardMaxima("poisson", 500, 200, 1000);

    [Test]
    public async Task BackwardMaxima_Burst_Tiny() => await CompareNativeBackwardMaxima("burst", 10, 5, 20);

    [Test]
    public async Task BackwardMaxima_Burst_Small() => await CompareNativeBackwardMaxima("burst", 50, 20, 100);

    [Test]
    public async Task BackwardMaxima_Burst_Medium() => await CompareNativeBackwardMaxima("burst", 200, 80, 400);

    [Test]
    public async Task BackwardMaxima_Burst_Large() => await CompareNativeBackwardMaxima("burst", 500, 200, 1000);

    private async Task CompareNativeBackwardMaxima(string pattern, int nPre, int nPost, int maxSpikes)
    {
        var rng = new Random(42);

        double[] w, times; int[] sources;
        switch (pattern)
        {
            case "uniform": GenerateUniformData(rng, nPre, nPost, maxSpikes, out w, out times, out sources); break;
            case "poisson": GeneratePoissonData(rng, nPre, nPost, out w, out times, out sources); break;
            case "burst": GenerateBurstData(rng, nPre, nPost, out w, out times, out sources); break;
            default: throw new ArgumentException($"Unknown pattern: {pattern}");
        }

        SortTimesSources(ref times, ref sources);

        // Compute maxima using .NET (just need the data; both C and .NET backward passes use same input)
        var netMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        // Build backwards-compatible maxima arrays (n_maxima == n_post for C constraint)
        double[] maximaTimes = new double[nPost];
        double[] maximaValues = new double[nPost];
        double[] maximaErrors = new double[nPost];

        for (int i = 0; i < nPost; i++)
        {
            maximaTimes[i] = double.IsNaN(netMaxima.Times[i]) ? 0.5 : netMaxima.Times[i];
            maximaValues[i] = double.IsNaN(netMaxima.Values[i]) ? 1.0 : netMaxima.Values[i];
            maximaErrors[i] = (i % 2 == 0) ? 0.1 : -0.2;
        }

        // .NET implementation
        var csGradient = new double[nPre * nPost]; Array.Fill(csGradient, 0.0);
        EventPropMath.BackwardMaxima(times, sources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, csGradient);

        // C library via P/Invoke
        var cGradient = new double[nPre * nPost]; Array.Fill(cGradient, 0.0);
        NativeApi.BackwardMaxima(times, sources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cGradient, new double[1024]);

        // Assert: .NET matches C (scaled tolerance)
        for (int i = 0; i < csGradient.Length; i++)
        {
            AssertScaled(csGradient[i], cGradient[i], "gradient", i);
        }
    }
}
