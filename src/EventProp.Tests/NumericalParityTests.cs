using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// Three-way numerical parity tests: NativeApi (C) ≡ NativeCppApi (Eigen, source of truth) ≡ EventPropMath (.NET).
/// All three must produce identical results within 1e-10 tolerance.
/// 
/// These tests use shared RNG data generation (seed 42) to ensure all backends receive
/// identical inputs. They are designed to expose divergence points: tolerance thresholds,
/// bracketing guards, and t_before computation differences.
/// 
/// C++ (Eigen) is the source of truth — both .NET and C must match it.
/// </summary>
public class NumericalParityTests
{
    private const double Tolerance = 1e-10;

    // ========================================================================
    // Shared Data Generation (seed 42, deterministic)
    // ========================================================================

    private static void GenerateUniformData(Random rng, int nPre, int nPost, int nSpikes,
        out double[] w, out double[] times, out int[] sources)
    {
        w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = rng.NextDouble() * 2.0 + 0.1; // [0.1, 2.1] — realistic LIF weights

        times = new double[nSpikes];
        sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = rng.NextDouble() * 1.0; // [0, 1)
            sources[i] = rng.Next(nPre);
        }
    }

    private static void GeneratePoissonData(Random rng, int nPre, int nPost,
        out double[] w, out double[] times, out int[] sources)
    {
        // Match .NET benchmark pattern: rate=200Hz, strictly increasing times
        w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = rng.NextDouble() * 20.0 + 1.0;

        var spikeList = new List<(double time, int source)>();
        double t = 0.0;
        while (t < 1.0)
        {
            double uP = rng.NextDouble();
            t += -Math.Log(1.0 - uP) / 200.0; // rate=200Hz
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
        // Burst pattern: clusters of spikes from same source with inter-burst intervals
        w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = rng.NextDouble() * 20.0 + 1.0;

        var spikeList = new List<(double time, int source)>();
        double t = 0.0;
        while (t < 1.0)
        {
            int burstSize = rng.Next(3, 8); // 3-7 spikes per burst
            for (int b = 0; b < burstSize && t < 1.0; b++)
            {
                double uB = rng.NextDouble();
                t += -Math.Log(1.0 - uB) / 500.0; // intra-burst rate=500Hz
                if (t < 1.0)
                    spikeList.Add((t, rng.Next(nPre)));
            }
            double uI = rng.NextDouble();
            t += -Math.Log(1.0 - uI) / 50.0; // inter-burst rate=50Hz
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
    // ComputeSums Tests
    // ========================================================================

    [Test]
    public async Task ComputeSums_Uniform_Tiny() => await CompareComputeSums("uniform", 10, 5, 20);

    [Test]
    public async Task ComputeSums_Uniform_Small() => await CompareComputeSums("uniform", 50, 20, 100);

    [Test]
    public async Task ComputeSums_Uniform_Medium() => await CompareComputeSums("uniform", 200, 80, 400);

    [Test]
    public async Task ComputeSums_Uniform_Large() => await CompareComputeSums("uniform", 500, 200, 1000);

    [Test]
    public async Task ComputeSums_Poisson_Tiny() => await CompareComputeSums("poisson", 10, 5, 20);

    [Test]
    public async Task ComputeSums_Poisson_Small() => await CompareComputeSums("poisson", 50, 20, 100);

    [Test]
    public async Task ComputeSums_Poisson_Medium() => await CompareComputeSums("poisson", 200, 80, 400);

    [Test]
    public async Task ComputeSums_Poisson_Large() => await CompareComputeSums("poisson", 500, 200, 1000);

    [Test]
    public async Task ComputeSums_Burst_Tiny() => await CompareComputeSums("burst", 10, 5, 20);

    [Test]
    public async Task ComputeSums_Burst_Small() => await CompareComputeSums("burst", 50, 20, 100);

    [Test]
    public async Task ComputeSums_Burst_Medium() => await CompareComputeSums("burst", 200, 80, 400);

    [Test]
    public async Task ComputeSums_Burst_Large() => await CompareComputeSums("burst", 500, 200, 1000);

    private async Task CompareComputeSums(string pattern, int nPre, int nPost, int maxSpikes)
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
        var netSum0 = new double[(times.Length + 1) * nPost];
        var netSum1 = new double[(times.Length + 1) * nPost];
        EventPropMath.ComputeSums(w, times, sources, 0.02, 0.005, netSum0, netSum1, nPost,
            new double[times.Length], new double[times.Length]);

        // C library via P/Invoke
        var cSum0 = new double[(times.Length + 1) * nPost];
        var cSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, cSum0, cSum1);

        // C++ (Eigen) via P/Invoke
        var cppSum0 = new double[(times.Length + 1) * nPost];
        var cppSum1 = new double[(times.Length + 1) * nPost];
        NativeCppApi.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, cppSum0, cppSum1);

        // Assert: all three match
        for (int i = 0; i < netSum0.Length; i++)
        {
            await Assert.That(netSum0[i]).IsCloseTo(cppSum0[i], Tolerance);
            await Assert.That(netSum1[i]).IsCloseTo(cppSum1[i], Tolerance);
            await Assert.That(cSum0[i]).IsCloseTo(cppSum0[i], Tolerance);
            await Assert.That(cSum1[i]).IsCloseTo(cppSum1[i], Tolerance);
        }
    }

    // ========================================================================
    // ComputeSpikes Tests
    // ========================================================================

    [Test]
    public async Task ComputeSpikes_Uniform_Tiny() => await CompareComputeSpikes("uniform", 10, 5, 20);

    [Test]
    public async Task ComputeSpikes_Uniform_Small() => await CompareComputeSpikes("uniform", 50, 20, 100);

    [Test]
    public async Task ComputeSpikes_Uniform_Medium() => await CompareComputeSpikes("uniform", 200, 80, 400);

    [Test]
    public async Task ComputeSpikes_Uniform_Large() => await CompareComputeSpikes("uniform", 500, 200, 1000);

    [Test]
    public async Task ComputeSpikes_Poisson_Tiny() => await CompareComputeSpikes("poisson", 10, 5, 20);

    [Test]
    public async Task ComputeSpikes_Poisson_Small() => await CompareComputeSpikes("poisson", 50, 20, 100);

    [Test]
    public async Task ComputeSpikes_Poisson_Medium() => await CompareComputeSpikes("poisson", 200, 80, 400);

    [Test]
    public async Task ComputeSpikes_Poisson_Large() => await CompareComputeSpikes("poisson", 500, 200, 1000);

    [Test]
    public async Task ComputeSpikes_Burst_Tiny() => await CompareComputeSpikes("burst", 10, 5, 20);

    [Test]
    public async Task ComputeSpikes_Burst_Small() => await CompareComputeSpikes("burst", 50, 20, 100);

    [Test]
    public async Task ComputeSpikes_Burst_Medium() => await CompareComputeSpikes("burst", 200, 80, 400);

    [Test]
    public async Task ComputeSpikes_Burst_Large() => await CompareComputeSpikes("burst", 500, 200, 1000);

    private async Task CompareComputeSpikes(string pattern, int nPre, int nPost, int maxSpikes)
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
        var netResult = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);

        // C library via P/Invoke
        int maxOutSpikes = times.Length * nPost;
        var cTimes = new double[maxOutSpikes];
        var cSources = new int[maxOutSpikes];
        var cCurrents = new double[maxOutSpikes];
        int nCSpikes = NativeApi.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, cTimes, cSources, cCurrents);

        // C++ (Eigen) via P/Invoke
        var cppTimes = new double[maxOutSpikes];
        var cppSources = new int[maxOutSpikes];
        var cppCurrents = new double[maxOutSpikes];
        int nCppSpikes = NativeCppApi.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005, cppTimes, cppSources, cppCurrents);

        // Assert: spike counts match first
        await Assert.That(netResult.NSpikes).IsEqualTo(nCppSpikes);
        await Assert.That(nCSpikes).IsEqualTo(nCppSpikes);

        // Then element-wise comparison
        for (int i = 0; i < netResult.NSpikes; i++)
        {
            await Assert.That(netResult.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(netResult.Sources[i]).IsEqualTo(cppSources[i]);
            await Assert.That(netResult.Currents[i]).IsCloseTo(cppCurrents[i], Tolerance);
            await Assert.That(cTimes[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(cSources[i]).IsEqualTo(cppSources[i]);
            await Assert.That(cCurrents[i]).IsCloseTo(cppCurrents[i], Tolerance);
        }
    }

    // ========================================================================
    // ComputeMaxima Tests
    // ========================================================================

    [Test]
    public async Task ComputeMaxima_Uniform_Tiny() => await CompareComputeMaxima("uniform", 10, 5, 20);

    [Test]
    public async Task ComputeMaxima_Uniform_Small() => await CompareComputeMaxima("uniform", 50, 20, 100);

    [Test]
    public async Task ComputeMaxima_Uniform_Medium() => await CompareComputeMaxima("uniform", 200, 80, 400);

    [Test]
    public async Task ComputeMaxima_Uniform_Large() => await CompareComputeMaxima("uniform", 500, 200, 1000);

    [Test]
    public async Task ComputeMaxima_Poisson_Tiny() => await CompareComputeMaxima("poisson", 10, 5, 20);

    [Test]
    public async Task ComputeMaxima_Poisson_Small() => await CompareComputeMaxima("poisson", 50, 20, 100);

    [Test]
    public async Task ComputeMaxima_Poisson_Medium() => await CompareComputeMaxima("poisson", 200, 80, 400);

    [Test]
    public async Task ComputeMaxima_Poisson_Large() => await CompareComputeMaxima("poisson", 500, 200, 1000);

    [Test]
    public async Task ComputeMaxima_Burst_Tiny() => await CompareComputeMaxima("burst", 10, 5, 20);

    [Test]
    public async Task ComputeMaxima_Burst_Small() => await CompareComputeMaxima("burst", 50, 20, 100);

    [Test]
    public async Task ComputeMaxima_Burst_Medium() => await CompareComputeMaxima("burst", 200, 80, 400);

    [Test]
    public async Task ComputeMaxima_Burst_Large() => await CompareComputeMaxima("burst", 500, 200, 1000);

    private async Task CompareComputeMaxima(string pattern, int nPre, int nPost, int maxSpikes)
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
        var netMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        // C library via P/Invoke
        var cTimes = new double[nPost];
        var cValues = new double[nPost];
        NativeApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cTimes, cValues);

        // C++ (Eigen) via P/Invoke
        var cppTimes = new double[nPost];
        var cppValues = new double[nPost];
        NativeCppApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cppTimes, cppValues);

        // Assert: all three match
        for (int i = 0; i < nPost; i++)
        {
            if (double.IsNaN(netMaxima.Times[i]))
            {
                await Assert.That(cppTimes[i]).IsNaN();
                await Assert.That(cTimes[i]).IsNaN();
            }
            else
            {
                await Assert.That(netMaxima.Times[i]).IsCloseTo(cppTimes[i], Tolerance);
                await Assert.That(netMaxima.Values[i]).IsCloseTo(cppValues[i], Tolerance);
                await Assert.That(cTimes[i]).IsCloseTo(cppTimes[i], Tolerance);
                await Assert.That(cValues[i]).IsCloseTo(cppValues[i], Tolerance);
            }
        }
    }

    // ========================================================================
    // BackwardSpikes Tests
    // ========================================================================

    [Test]
    public async Task BackwardSpikes_Uniform_Tiny() => await CompareBackwardSpikes("uniform", 10, 5, 20);

    [Test]
    public async Task BackwardSpikes_Uniform_Small() => await CompareBackwardSpikes("uniform", 50, 20, 100);

    [Test]
    public async Task BackwardSpikes_Uniform_Medium() => await CompareBackwardSpikes("uniform", 200, 80, 400);

    [Test]
    public async Task BackwardSpikes_Uniform_Large() => await CompareBackwardSpikes("uniform", 500, 200, 1000);

    [Test]
    public async Task BackwardSpikes_Poisson_Tiny() => await CompareBackwardSpikes("poisson", 10, 5, 20);

    [Test]
    public async Task BackwardSpikes_Poisson_Small() => await CompareBackwardSpikes("poisson", 50, 20, 100);

    [Test]
    public async Task BackwardSpikes_Poisson_Medium() => await CompareBackwardSpikes("poisson", 200, 80, 400);

    [Test]
    public async Task BackwardSpikes_Poisson_Large() => await CompareBackwardSpikes("poisson", 500, 200, 1000);

    [Test]
    public async Task BackwardSpikes_Burst_Tiny() => await CompareBackwardSpikes("burst", 10, 5, 20);

    [Test]
    public async Task BackwardSpikes_Burst_Small() => await CompareBackwardSpikes("burst", 50, 20, 100);

    [Test]
    public async Task BackwardSpikes_Burst_Medium() => await CompareBackwardSpikes("burst", 200, 80, 400);

    [Test]
    public async Task BackwardSpikes_Burst_Large() => await CompareBackwardSpikes("burst", 500, 200, 1000);

    private async Task CompareBackwardSpikes(string pattern, int nPre, int nPost, int maxSpikes)
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

        // Forward pass (all three use EventPropMath for consistency)
        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, times.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes == 0) return; // Skip — no post-spike activity

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

        // .NET implementation
        var netGradient = new double[nPre * nPost]; Array.Fill(netGradient, 0.0);
        EventPropMath.BackwardSpikes(w, times, sources, postSpikes, nPost,
            1.0, 0.02, 0.005, netGradient, buffers);

        // C library via P/Invoke
        var cGradient = new double[nPre * nPost]; Array.Fill(cGradient, 0.0);
        NativeApi.BackwardSpikes(times, sources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cGradient);

        // C++ (Eigen) via P/Invoke
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);
        NativeCppApi.BackwardSpikes(times, sources,
            postSpikes.Times, postSpikes.Sources, postSpikes.Currents, postSpikes.Errors,
            w, nPre, nPost, 1.0, 0.02, 0.005, cppGradient);

        // Assert: all three match
        for (int i = 0; i < netGradient.Length; i++)
        {
            await Assert.That(netGradient[i]).IsCloseTo(cppGradient[i], Tolerance);
            await Assert.That(cGradient[i]).IsCloseTo(cppGradient[i], Tolerance);
        }
    }

    // ========================================================================
    // BackwardMaxima Tests
    // ========================================================================

    [Test]
    public async Task BackwardMaxima_Uniform_Tiny() => await CompareBackwardMaxima("uniform", 10, 5, 20);

    [Test]
    public async Task BackwardMaxima_Uniform_Small() => await CompareBackwardMaxima("uniform", 50, 20, 100);

    [Test]
    public async Task BackwardMaxima_Uniform_Medium() => await CompareBackwardMaxima("uniform", 200, 80, 400);

    [Test]
    public async Task BackwardMaxima_Uniform_Large() => await CompareBackwardMaxima("uniform", 500, 200, 1000);

    [Test]
    public async Task BackwardMaxima_Poisson_Tiny() => await CompareBackwardMaxima("poisson", 10, 5, 20);

    [Test]
    public async Task BackwardMaxima_Poisson_Small() => await CompareBackwardMaxima("poisson", 50, 20, 100);

    [Test]
    public async Task BackwardMaxima_Poisson_Medium() => await CompareBackwardMaxima("poisson", 200, 80, 400);

    [Test]
    public async Task BackwardMaxima_Poisson_Large() => await CompareBackwardMaxima("poisson", 500, 200, 1000);

    [Test]
    public async Task BackwardMaxima_Burst_Tiny() => await CompareBackwardMaxima("burst", 10, 5, 20);

    [Test]
    public async Task BackwardMaxima_Burst_Small() => await CompareBackwardMaxima("burst", 50, 20, 100);

    [Test]
    public async Task BackwardMaxima_Burst_Medium() => await CompareBackwardMaxima("burst", 200, 80, 400);

    [Test]
    public async Task BackwardMaxima_Burst_Large() => await CompareBackwardMaxima("burst", 500, 200, 1000);

    private async Task CompareBackwardMaxima(string pattern, int nPre, int nPost, int maxSpikes)
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

        // Compute maxima for all three (using EventPropMath as reference)
        var netMaxima = EventPropMath.ComputeMaxima(w, times, sources, nPost, 0.02, 0.005, new ScratchBuffers());

        // Build backwards-compatible maxima arrays (n_maxima == n_post for C++ constraint)
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
        var netGradient = new double[nPre * nPost]; Array.Fill(netGradient, 0.0);
        EventPropMath.BackwardMaxima(times, sources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPost, 0.02, 0.005, netGradient);

        // C library via P/Invoke
        var cGradient = new double[nPre * nPost]; Array.Fill(cGradient, 0.0);
        NativeApi.BackwardMaxima(times, sources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cGradient);

        // C++ (Eigen) via P/Invoke
        var cppGradient = new double[nPre * nPost]; Array.Fill(cppGradient, 0.0);
        NativeCppApi.BackwardMaxima(times, sources,
            maximaTimes, maximaValues, maximaErrors,
            w, nPre, nPost, 0.02, 0.005, cppGradient);

        // Assert: all three match
        for (int i = 0; i < netGradient.Length; i++)
        {
            await Assert.That(netGradient[i]).IsCloseTo(cppGradient[i], Tolerance);
            await Assert.That(cGradient[i]).IsCloseTo(cppGradient[i], Tolerance);
        }
    }
}
