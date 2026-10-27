using CW.EventProp;
using Native = CW.EventProp.Native;

namespace CW.EventProp.Tests;

/// <summary>
/// Guards against losing writes to managed memory that a concurrent GC relocates.
///
/// Two places take raw pointers from managed memory and hand them to code that
/// runs for a long time:
///   • SimdAlgorithms — Vector256 Load/Store through pointers derived from spans.
///   • the P/Invoke wrappers — EventPropInput carries raw pointers into
///     w/times/sources, and the native call consumes them after the struct is built.
/// In both cases the pin must span the whole operation. If it does not, a
/// concurrent GC relocating the array makes the writes land at the stale address:
/// the live array keeps its zero initialisation, and the stale region — already
/// recycled to another live object — is silently overwritten.
///
/// These are PROBABILISTIC guards, not proofs. They run a bounded window of
/// concurrent allocation to provoke collections and relocations. The original
/// defect needed ~30% of full-suite runs under TUnit's parallel scheduler to
/// show, and a 34-million-iteration single-test probe did NOT reproduce it, so
/// do not expect these to fail on their own if the pinning regresses — treat
/// them as cheap tripwires and keep the AGENTS.md post-mortem as the real record.
///
/// The window starts when the hammers start — there is no warm-up delay. Measured here
/// (8 cores, each variant in its own process): the 600 ms window already takes 347–472
/// gen0 collections with no warm-up, so a delay does not spin the hammers up; it only
/// ate ~15–20% of the iterations (2.5M against 3.1M), and iterations are the whole point.
/// </summary>
public class PinningStressTests
{
    /// <summary>
    /// Allocation hammers: force frequent GCs, including relocations.
    ///
    /// Takes a token, not a CancellationTokenSource: the caller owns and disposes the
    /// source, and a source handed to these tasks could be disposed while they still
    /// read it (the .Token getter throws ObjectDisposedException on a disposed source).
    /// </summary>
    private static Task[] StartPressure(int threads, CancellationToken stop)
        => Enumerable.Range(0, threads).Select(_ => Task.Run(() =>
        {
            var rng = new Random();
            long sink = 0;
            while (!stop.IsCancellationRequested)
                for (int k = 0; k < 32; k++)
                {
                    var a = new double[2000]; a[0] = rng.NextDouble(); sink += (long)a[0];
                    var b = new double[90000]; b[0] = rng.NextDouble(); sink += (long)b[0];
                }
            GC.KeepAlive(sink);
        })).ToArray();

    [Test]
    [Timeout(30_000)]
    public async Task FmaRow_WritesLandUnderConcurrentGcPressure(CancellationToken timeout)
    {
        // nPost = 5 >= Vector256<double>.Count (4), so FmaRow takes the SIMD path.
        const int nPost = 5;
        var src = new double[nPost];
        var w = new double[nPost];
        for (int j = 0; j < nPost; j++) { src[j] = 0.25 * j; w[j] = 0.5 + 0.1 * j; }
        const double scalar = 1.75;
        var expected = new double[nPost];
        for (int j = 0; j < nPost; j++) expected[j] = src[j] + w[j] * scalar;

        using var stop = new CancellationTokenSource(TimeSpan.FromMilliseconds(600));
        using var cancel = CancellationTokenSource.CreateLinkedTokenSource([timeout, stop.Token]);
        var pressure = StartPressure(4, cancel.Token);

        int lost = 0, iters = 0;
        while (!cancel.Token.IsCancellationRequested)
        {
            var dest = new double[nPost];          // fresh gen0 array, zero-initialised
            SimdAlgorithms.FmaRow(dest, src, w, scalar);
            Interlocked.Increment(ref iters);
            for (int j = 0; j < nPost; j++)
                if (dest[j] != expected[j]) { Interlocked.Increment(ref lost); break; }
            GC.KeepAlive(dest);
        }

        await Task.WhenAll(pressure);
        await Assert.That(lost).IsEqualTo(0);
    }

    [Test]
    [Timeout(60_000)]
    public async Task NativeForward_IsBitIdenticalUnderConcurrentGcPressure(CancellationToken timeout)
    {
        const int nPre = 50, nPost = 20, n = 200;
        var w = new double[nPre * nPost];
        for (int i = 0; i < w.Length; i++) w[i] = 0.5 + 0.001 * (i % 97);
        var times = new double[n];
        var sources = new int[n];
        for (int i = 0; i < n; i++) { times[i] = (i + 1) * 0.002; sources[i] = i % nPre; }

        (int nSpikes, double[] ts, int[] srcs) Reference()
        {
            using var o = Native.Api.ComputeSpikes(w, nPre, nPost, times, sources, 1.0, 0.02, 0.005);
            return (o.NSpikes, o.Times.ToArray(), o.Sources.ToArray());
        }

        var reference = Reference();

        using var stop = new CancellationTokenSource(TimeSpan.FromMilliseconds(600));
        using var cancel = CancellationTokenSource.CreateLinkedTokenSource([timeout, stop.Token]);
        // cancel.Token, not stop.Token: on a test timeout the hammers have to stop with
        // the test, or their gen0 churn leaks into whichever test runs next — the exact
        // interference these GC-sensitive guards cannot tolerate.
        var pressure = StartPressure(4, cancel.Token);

        int drift = 0, iters = 0;
        while (!cancel.Token.IsCancellationRequested)
        {
            // Fresh gen0 input arrays: the ones a relocating GC will move.
            var lw = (double[])w.Clone();
            var lt = (double[])times.Clone();
            var ls = (int[])sources.Clone();

            using var o = Native.Api.ComputeSpikes(lw, nPre, nPost, lt, ls, 1.0, 0.02, 0.005);
            Interlocked.Increment(ref iters);
            bool same = o.NSpikes == reference.nSpikes;
            var gotT = o.Times.ToArray();
            var gotS = o.Sources.ToArray();
            for (int i = 0; same && i < reference.ts.Length; i++) same = gotT[i] == reference.ts[i] && gotS[i] == reference.srcs[i];
            if (!same) Interlocked.Increment(ref drift);
            GC.KeepAlive(lw); GC.KeepAlive(lt); GC.KeepAlive(ls);
        }

        await Task.WhenAll(pressure);
        await Assert.That(drift).IsEqualTo(0);
    }
}
