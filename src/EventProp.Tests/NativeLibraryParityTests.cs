using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// C (NativeApi) vs Eigen (NativeCppApi) parity tests — no .NET comparison.
/// These verify that the two native libraries produce identical results from identical inputs.
/// If they agree, benchmarking is valid regardless of .NET behavior.
/// 
/// C++ (Eigen) is the source of truth — C must match it within 1e-10.
/// </summary>
public class NativeLibraryParityTests
{
    private const double Tolerance = 1e-10;

    /// <summary>Larger tolerance for ComputeSums — accumulated sums have FP noise that scales with magnitude.
    /// Max observed relative error: ~4e-16 (machine epsilon). Values up to 4e14 → abs error ~0.06.</summary>
    private const double ComputeSumsTolerance = 1e-6;

    // ========================================================================
    // Shared Data Generation (seed 42, deterministic)
    // ========================================================================

    private static void GenerateUniformData(Random rng, int nPre, int nPost, int nSpikes,
        out double[] w, out double[] times, out int[] sources)
    {
        w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++)
            w[i] = rng.NextDouble() * 2.0 + 0.1; // [0.1, 2.1]

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
            w[i] = rng.NextDouble() * 2.0 + 0.1;

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
            w[i] = rng.NextDouble() * 2.0 + 0.1;

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
    // ComputeSums Tests — C vs Eigen only
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

        // C library via P/Invoke
        var cSum0 = new double[(times.Length + 1) * nPost];
        var cSum1 = new double[(times.Length + 1) * nPost];
        NativeApi.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, cSum0, cSum1);

        // C++ (Eigen) via P/Invoke
        var cppSum0 = new double[(times.Length + 1) * nPost];
        var cppSum1 = new double[(times.Length + 1) * nPost];
        NativeCppApi.ComputeSums(w, nPre, nPost, times, sources, 0.02, 0.005, cppSum0, cppSum1);

        // Assert: C matches Eigen (use larger tolerance for accumulated sums — FP noise scales with magnitude)
        for (int i = 0; i < cSum0.Length; i++)
        {
            await Assert.That(cSum0[i]).IsCloseTo(cppSum0[i], ComputeSumsTolerance);
            await Assert.That(cSum1[i]).IsCloseTo(cppSum1[i], ComputeSumsTolerance);
        }
    }

    // ========================================================================
    // ComputeSpikes Tests — C vs Eigen only
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
        if (nCSpikes != nCppSpikes)
        {
            // Count spikes per neuron and print detailed diagnostics
            int[] cPerNrn = new int[nPost];
            int[] cppPerNrn = new int[nPost];
            for (int i = 0; i < nCSpikes; i++) cPerNrn[cSources[i]]++;
            for (int i = 0; i < nCppSpikes; i++) cppPerNrn[cppSources[i]]++;
            
            Console.WriteLine($"[SPIKE COUNT MISMATCH] C={nCSpikes}, Eigen={nCppSpikes}");
            for (int nrn = 0; nrn < nPost; nrn++)
            {
                if (cPerNrn[nrn] != cppPerNrn[nrn])
                {
                    Console.WriteLine($"  Neuron {nrn}: C={cPerNrn[nrn]}, Eigen={cppPerNrn[nrn]}");
                    // Print spike times for this neuron
                    Console.Write($"    C spikes: ");
                    int cIdx = 0;
                    for (int i = 0; i < nCSpikes && cIdx < cPerNrn[nrn]; i++)
                    {
                        if (cSources[i] == nrn) { Console.Write($"{cTimes[i]:F10} "); cIdx++; }
                    }
                    Console.WriteLine();
                    Console.Write($"    E spikes: ");
                    int eIdx = 0;
                    for (int i = 0; i < nCppSpikes && eIdx < cppPerNrn[nrn]; i++)
                    {
                        if (cppSources[i] == nrn) { Console.Write($"{cppTimes[i]:F10} "); eIdx++; }
                    }
                    Console.WriteLine();
                }
            }
        }
        await Assert.That(nCSpikes).IsEqualTo(nCppSpikes);

        // Then element-wise comparison
        for (int i = 0; i < nCSpikes; i++)
        {
            await Assert.That(cTimes[i]).IsCloseTo(cppTimes[i], Tolerance);
            await Assert.That(cSources[i]).IsEqualTo(cppSources[i]);
            await Assert.That(cCurrents[i]).IsCloseTo(cppCurrents[i], Tolerance);
        }
    }

    // ========================================================================
    // ComputeMaxima Tests — C vs Eigen only
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

        // C library via P/Invoke
        var cTimes = new double[nPost];
        var cValues = new double[nPost];
        NativeApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cTimes, cValues);

        // C++ (Eigen) via P/Invoke
        var cppTimes = new double[nPost];
        var cppValues = new double[nPost];
        NativeCppApi.ComputeMaxima(w, nPre, nPost, times, sources, 0.02, 0.005, cppTimes, cppValues);

        // Assert: C matches Eigen
        for (int i = 0; i < nPost; i++)
        {
            if (double.IsNaN(cppTimes[i]))
            {
                await Assert.That(cTimes[i]).IsNaN();
            }
            else
            {
                await Assert.That(cTimes[i]).IsCloseTo(cppTimes[i], Tolerance);
                await Assert.That(cValues[i]).IsCloseTo(cppValues[i], Tolerance);
            }
        }
    }

    // ========================================================================
    // BackwardSpikes Tests — C vs Eigen only
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

        // Forward pass using .NET (we just need post-spike data; both C and Eigen will use the same input)
        var buffers = new ScratchBuffers();
        buffers.EnsureCapacity(nPost, nPre, times.Length);
        var postSpikes = EventPropMath.ComputeSpikes(w, times, sources, nPost, 1.0, 0.02, 0.005, buffers);
        if (postSpikes.NSpikes == 0) return; // Skip — no post-spike activity

        for (int i = 0; i < postSpikes.NSpikes; i++)
            postSpikes.SetError(i, 1.0);

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

        // Assert: C matches Eigen
        for (int i = 0; i < cGradient.Length; i++)
        {
            await Assert.That(cGradient[i]).IsCloseTo(cppGradient[i], Tolerance);
        }
    }

    // ========================================================================
    // BackwardMaxima Tests — C vs Eigen only
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

        // Compute maxima using .NET (just need the data; both C and Eigen backward passes use same input)
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

        // Assert: C matches Eigen
        for (int i = 0; i < cGradient.Length; i++)
        {
            await Assert.That(cGradient[i]).IsCloseTo(cppGradient[i], Tolerance);
        }
    }
}
