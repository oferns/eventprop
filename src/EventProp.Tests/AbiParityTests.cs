using CW.EventProp;
using Native = CW.EventProp.Native;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

/// <summary>
/// The P/Invoke layer declares the C structs BY HAND (<c>[StructLayout]</c> in
/// <c>Api.cs</c> and <c>Batch.cs</c>), so a header change that moves a field offset is
/// invisible at compile time and invisible at the call site: the marshalling reads the
/// offsets this assembly compiled, against a library whose fields moved, and returns
/// plausible-looking garbage.
///
/// The Cython layer measured that failure (docs/POSTMORTEMS.md): a
/// <c>_native.cp311-win_amd64.pyd</c> built against July headers, loaded against a
/// library rebuilt from October headers, produced an access violation inside
/// <c>spikes.py __getitem__</c> 40 minutes into a benchmark run, in a frame with no hint
/// that a header/binary offset mismatch was the cause. Nothing in the build checked that
/// the extension was newer than the headers it was compiled from. The P/Invoke layer has
/// the same exposure, so it carries the same guard: <see cref="Native.Api.AbiCheck"/>
/// compares this assembly's compiled <c>sizeof</c> against the library's, and
/// <see cref="Native.Api.EnsureAbiMatches"/> runs it once per process from the layer
/// constructors and the friendly batch entry points.
///
/// These tests assert the guard passes, and that it COVERS every public struct — a
/// struct added to the headers without an entry is the drift that reintroduces the bug.
/// The same list is asserted in <c>eventprop_abi.c</c>'s C++ test and in
/// <c>EventProp.Native.Python/test/test_abi.py</c>, on purpose: adding a struct must
/// fail in every consumer.
/// </summary>
public class AbiParityTests
{
    /// <summary>The public structs a consumer embeds by value.</summary>
    private static readonly string[] PublicStructs =
    [
        "EventPropInput",
        "SpikeOutput",
        "SumsOutput",
        "MaximaOutput",
        "BackwardOutput",
        "SpikesBatch",
        "MaximaBatch",
        "BackwardsBatch",
        "GradientOutput",
        "AccumulatedBackwardResult",
        "DropoutOutput",
        "DropoutBatch",
        "VoltageTraceInput",
        "VoltageTraceOutput",
        "LambdaITraceInput",
        "LambdaITraceOutput",
    ];

    [Test]
    public void LibraryAbiVersion_IsPositiveAndMatchesTheCompiledConstant()
    {
        SyncAssert.That(Native.Api.LibraryAbiVersion()).IsGreaterThan(0);
        SyncAssert.That(Native.Api.LibraryAbiVersion())
                  .IsEqualTo(Native.Api.CompiledAbiVersion);
    }

    [Test]
    public void StructSizes_MatchTheLoadedLibrary()
    {
        // Two independent compilations of the same header: this assembly's sizeof, and
        // the library's. Any difference is an offset mismatch for a hand-declared field.
        foreach (var (name, compiledSize) in Native.Api.CompiledStructSizes())
        {
            int librarySize = Native.Api.LibraryStructSize(name);
            SyncAssert.That(librarySize).IsGreaterThan(0);
            SyncAssert.That(librarySize).IsEqualTo(compiledSize);
        }
    }

    [Test]
    public void CompiledStructSizes_CoversEveryPublicStruct()
    {
        var compiled = Native.Api.CompiledStructSizes();
        SyncAssert.That(compiled.Count).IsEqualTo(PublicStructs.Length);
        foreach (string name in PublicStructs)
            SyncAssert.That(compiled.ContainsKey(name)).IsTrue();
    }

    [Test]
    public void AbiCheck_ReportsNoMismatches()
    {
        var (ok, mismatches) = Native.Api.AbiCheck();
        SyncAssert.That(ok).IsTrue();
        SyncAssert.That(mismatches.Count).IsEqualTo(0);
    }

    [Test]
    public void EnsureAbiMatches_DoesNotThrow()
    {
        // The gate the layers call. If this throws, the loaded library and this
        // assembly disagree about a struct layout, and every number the layers produce
        // is read from the wrong offsets.
        Native.Api.EnsureAbiMatches();
        Native.Api.EnsureAbiMatches();   // idempotent: cached after the first pass
    }

    [Test]
    public void LibraryStructSize_UnknownNameIsZero()
    {
        // A consumer asking about a struct the library does not know must fail its own
        // check, not read 0 as a match.
        SyncAssert.That(Native.Api.LibraryStructSize("NotAStruct")).IsEqualTo(0);
        SyncAssert.That(Native.Api.LibraryStructSize("")).IsEqualTo(0);
    }

    [Test]
    public void NativeLayers_ConstructThroughTheAbiGate()
    {
        // The gate runs from the layer constructors, so it must not reject the normal
        // path: construct both native layers and run a forward pass.
        //
        // The input is deliberately a strong drive (40 spikes in 40 ms across 4
        // pre-synaptic neurons, every weight 1.5, v_th 1.0) because the firing
        // assertions below have to be non-vacuous. The parity suite's 20×8 / 60-spike /
        // U[0.5, 2.0) regime — the one NativeLayerParityTests uses — produces ZERO
        // output spikes for these parameters, which is why a parity test that only
        // compares counts can pass while exercising nothing.
        const int nPre = 4, nPost = 2, nSpikes = 40;
        const double tauMem = 0.02, tauSyn = 0.005, vTh = 1.0;

        var w = new double[nPre * nPost];
        Array.Fill(w, 1.5);
        var times = new double[nSpikes];
        var sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = (i + 1) * 0.001;
            sources[i] = i % nPre;
        }

        using var lif = new Native.LIFLayer(new LIFLayerParameters(nPost, nPre, tauMem, tauSyn, vTh), w);
        using var li = new Native.LILayer(new LILayerParameters(nPost, nPre, tauMem, tauSyn), w);

        var input = new SpikesVector([new SpikesData(times, sources)]);
        lif.Forward(input);
        SyncAssert.That(lif.Output[0].NSpikes).IsGreaterThan(0);

        var maximaInput = new SpikesVector([new SpikesData(times, sources)]);
        li.Forward(maximaInput);
        SyncAssert.That(li.MaximaBatch!.Count).IsGreaterThan(0);
    }

    [Test]
    public void NativeBatchEntryPoints_RunThroughTheAbiGate()
    {
        // ValidateFlat is the shared choke point for all six friendly batch calls, and
        // it is where the ABI check lives for direct (non-layer) callers. Strong drive,
        // so the spike-count assertion is non-vacuous.
        const int nPre = 4, nPost = 2, nSpikes = 40;
        var w = new double[nPre * nPost];
        Array.Fill(w, 1.5);
        var times = new double[nSpikes];
        var sources = new int[nSpikes];
        for (int i = 0; i < nSpikes; i++)
        {
            times[i] = (i + 1) * 0.001;
            sources[i] = i % nPre;
        }

        using var batch = Native.Batch.ComputeSpikesBatch(
            w, nPre, nPost, times, sources, [nSpikes], vTh: 1.0, tauMem: 0.02, tauSyn: 0.005);
        SyncAssert.That((int)batch.ErrorNo).IsEqualTo(0);
        SyncAssert.That(batch.BatchSize).IsEqualTo(1);
        SyncAssert.That(batch[0].NSpikes).IsGreaterThan(0);
    }
}
