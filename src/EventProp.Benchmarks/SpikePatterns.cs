namespace EventProp.Benchmarks;

// ============================================================
// Canonical spike patterns — shared by every benchmark suite.
//
// Mirrors GenerateBurstData in src/EventProp.Native.Benchmarks/bench_main.cpp
// and generate_burst_spikes in src/EventProp.Python.Benchmarks/conftest.py.
// Keep the three in sync; divergence makes cross-suite numbers incomparable.
//
// Contract every generator upholds, because the C library enforces it
// (EVENTPROP_UNSORTED_SPIKE_TIMES = 60) and the Eigen reference silently
// returns garbage without it:
//   * spike times are non-decreasing
//   * every spike lands inside [0, 1)
//   * exactly nSpikes spikes are produced, none left unwritten
//
// The last two matter for a reason: the previous .NET burst spread 100
// sequential bursts 10 ms apart, reaching 5.94 s. With tauSyn = 0.005 that is
// maxTime / tauSyn = 1188 > EVENTPROP_EXP_OVERFLOW_THRESHOLD (700), so the C
// library rejects the input outright (EVENTPROP_TIMES_TAU_MISMATCH) and the
// pattern could not be fed to the native side at all.
// ============================================================

public static class SpikePatterns
{
    public const int    SpikesPerBurst  = 5;
    public const int    BurstsPerNeuron = 5;
    public const double BurstGap        = 0.2;    /* > 5 * tauMem = 0.1 s */
    public const double SpikeStep       = 0.002;
    public const double Start           = 0.05;
    public const double WindowEnd       = 1.0;
    public const double EdgeEps         = 1e-6;   /* keep the last spike strictly inside [0, 1) */

    /// <summary>
    /// Each bursting neuron fires BurstsPerNeuron bursts of SpikesPerBurst spikes;
    /// spikes within a burst are SpikeStep apart and consecutive bursts are BurstGap
    /// apart, so the membrane decays between bursts. Neurons are staggered by a
    /// computed offset so exactly nSpikes spikes land inside [0, 1) at every size.
    ///
    /// The stagger is computed, not hardcoded: a fixed per-neuron offset saturates
    /// at a handful of bursting neurons, leaving most of the array unwritten —
    /// which produced thousands of spikes at t = 0 (zero-initialised memory) at
    /// Medium/Large/XL in the native suite.
    /// </summary>
    public static void Burst(int nPre, int nSpikes, double[] times, int[] sources, Random rng)
    {
        if (nSpikes <= 0) return;

        int perNeuron = SpikesPerBurst * BurstsPerNeuron;                     /* 25 */
        int nBursting = (nSpikes + perNeuron - 1) / perNeuron;                /* ceil */
        double burstSpan = (BurstsPerNeuron - 1) * BurstGap;                  /* 0.8  */
        double lastInBurst = (SpikesPerBurst - 1) * SpikeStep;                /* 0.008 */
        double room = WindowEnd - EdgeEps - Start - burstSpan - lastInBurst;
        double stagger = nBursting > 1 ? room / (nBursting - 1) : 0.0;

        int idx = 0;
        for (int j = 0; j < nBursting && idx < nSpikes; j++)
        {
            double burstBase = Start + j * stagger;
            int src = j % nPre;
            for (int b = 0; b < BurstsPerNeuron && idx < nSpikes; b++)
                for (int s = 0; s < SpikesPerBurst && idx < nSpikes; s++)
                {
                    times[idx] = burstBase + b * BurstGap + s * SpikeStep;
                    sources[idx] = src;
                    idx++;
                }
        }

        // Filled neuron by neuron, so the bursts interleave.
        SortTimesSources(times, sources);
    }

    /// <summary>
    /// "Poisson" pattern — the shared cross-suite key is `Poisson`, but this is NOT a
    /// homogeneous Poisson process over [0, 1). Each spike time is an independent
    /// exponential order statistic -log(1-u)/rate, so the train concentrates in the
    /// first few tens of ms: effective rate ~1.8 kHz (Tiny) to ~25 kHz (Large), not 200 Hz.
    /// Identical form in native and Python, so cross-suite ratios are valid. See AGENTS.md.
    /// </summary>
    public static void Poisson(int nPre, int nSpikes, double[] times, int[] sources, Random rng)
    {
        const double rate = 200.0; // nominal Hz; effective rate is ~10x higher (see above)
        double dt = nSpikes > 0 ? 1.0 / (rate * nSpikes) : 0.0;

        for (int idx = 0; idx < nSpikes; idx++)
        {
            double u = rng.NextDouble();
            times[idx] = -Math.Log(1.0 - u) / rate;
            sources[idx] = rng.Next(nPre);

            if (idx > 0 && times[idx] <= times[idx - 1])
                times[idx] = times[idx - 1] + dt;
        }

        SortTimesSources(times, sources);
    }

    /// <summary>Stable-sort spike times, carrying sources along (mirrors SortTimesSources in the native benchmarks).</summary>
    public static void SortTimesSources(double[] times, int[] sources)
    {
        if (times.Length <= 1) return;

        int n = times.Length;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;

        // Array.Sort on an index array with an explicit comparer is stable enough
        // for this use: equal times carry the same weight only through sources,
        // and sources are interchangeable within a simultaneous spike.
        Array.Sort(idx, (a, b) => times[a].CompareTo(times[b]));

        var sortedTimes = new double[n];
        var sortedSources = new int[n];
        for (int i = 0; i < n; i++)
        {
            sortedTimes[i] = times[idx[i]];
            sortedSources[i] = sources[idx[i]];
        }
        Array.Copy(sortedTimes, times, n);
        Array.Copy(sortedSources, sources, n);
    }
}
