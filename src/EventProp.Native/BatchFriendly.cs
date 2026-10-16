using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>
/// Managed-friendly batch entry points: flat arrays in, batch structs out.
///
/// The raw <see cref="Batch"/> P/Invokes take an <c>EventPropInput*</c> array, which
/// forces every caller to hand-build one struct per sample. These wrappers accept the
/// flat concatenated layout the FFI callers already use (times/sources concatenated
/// across samples, plus a per-sample spike count) and do the marshalling once.
///
/// Everything returned is borrowed from the native heap and freed by the returned
/// struct's Dispose; no managed buffers are allocated per call beyond the pinned
/// input array.
/// </summary>
public static unsafe partial class Batch
{
    /// <summary>Size of one <see cref="Api.EventPropInput"/>, for pointer arithmetic.</summary>
    private static int InputSize => Unsafe.SizeOf<Api.EventPropInput>();

    /// <summary>Size of one <see cref="Api.SpikeOutput"/>, for indexing a SpikesBatch.</summary>
    private static int SpikeOutputSize => Unsafe.SizeOf<Api.SpikeOutput>();

    /// <summary>Size of one <see cref="Api.BackwardOutput"/>, for indexing a BackwardsBatch.</summary>
    private static int BackwardOutputSize => Unsafe.SizeOf<Api.BackwardOutput>();

    // ── Per-sample indexing into a batch struct ──────────────────────

    /// <summary>Borrow sample <paramref name="index"/>'s spike output from a forward batch.</summary>
    /// <remarks>The result is owned by the batch — do NOT Dispose it.</remarks>
    public static Api.SpikeOutput SpikeAt(this SpikesBatch batch, int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, batch.BatchSize);

        var slot = (byte*)batch.Spikes + (long)index * SpikeOutputSize;
        return Unsafe.ReadUnaligned<Api.SpikeOutput>(slot);
    }

    /// <summary>Borrow sample <paramref name="index"/>'s backward output from a backwards batch.</summary>
    /// <remarks>The result is owned by the batch — do NOT Dispose it.</remarks>
    public static Api.BackwardOutput BackwardAt(this BackwardsBatch batch, int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, batch.BatchSize);

        var slot = (byte*)batch.Outputs + (long)index * BackwardOutputSize;
        return Unsafe.ReadUnaligned<Api.BackwardOutput>(slot);
    }

    /// <summary>Borrow sample <paramref name="index"/>'s maxima output from a maxima batch.</summary>
    public static MaximaEntry MaximaAt(this MaximaBatch batch, int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, batch.BatchSize);

        var slot = (byte*)batch.Outputs + (long)index * MaximaEntrySize;
        return Unsafe.ReadUnaligned<MaximaEntry>(slot);
    }

    private static int MaximaEntrySize => Unsafe.SizeOf<MaximaEntry>();

    /// <summary>
    /// One sample's maxima output, laid out as C's <c>MaximaOutput</c>:
    /// error code, n_post, then the times / values / errors pointers.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct MaximaEntry
    {
        public readonly Api.EventPropError ErrorNo;
        public readonly int NPost;
        public readonly IntPtr Times;
        public readonly IntPtr Values;
        public readonly IntPtr Errors;
    }

    // ── Flat-array forward ───────────────────────────────────────────

    /// <summary>
    /// Batch forward pass over concatenated samples.
    /// </summary>
    /// <param name="w">Row-major weight matrix (nPre × nPost), shared by every sample.</param>
    /// <param name="times">Input spike times, samples concatenated.</param>
    /// <param name="sources">Input spike sources, samples concatenated. Each sample's times must be non-decreasing.</param>
    /// <param name="nSpikesPerSample">Spike count for each sample; its length is the batch size.</param>
    public static SpikesBatch ComputeSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        double vTh, double tauMem, double tauSyn)
    {
        int batchSize = nSpikesPerSample.Length;
        ValidateFlat(batchSize, nSpikesPerSample, times.Length, sources.Length);

        var inputs = stackalloc Api.EventPropInput[batchSize];
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pn = nSpikesPerSample)
        {
            FillInputs(pw, nPre, nPost, pt, ps, pn, batchSize, vTh, tauMem, tauSyn, inputs);
            return eventprop_compute_spikes_batch((IntPtr)inputs, batchSize);
        }
    }

    /// <summary>Batch maxima (voltage) forward pass over concatenated samples.</summary>
    public static MaximaBatch ComputeMaximaBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        double tauMem, double tauSyn)
    {
        int batchSize = nSpikesPerSample.Length;
        ValidateFlat(batchSize, nSpikesPerSample, times.Length, sources.Length);

        var inputs = stackalloc Api.EventPropInput[batchSize];
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pn = nSpikesPerSample)
        {
            FillInputs(pw, nPre, nPost, pt, ps, pn, batchSize, 0.0, tauMem, tauSyn, inputs);
            // Maxima never reads v_th; the C validator has a matching no-v_th path.
            return eventprop_compute_maxima_batch((IntPtr)inputs, batchSize);
        }
    }

    // ── Flat-array backward ──────────────────────────────────────────

    /// <summary>
    /// Batch backward pass (spike loss). The caller supplies one derivative buffer per
    /// sample, sized to that sample's output spike count.
    /// </summary>
    public static BackwardsBatch BackwardSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        SpikesBatch spikes,
        ReadOnlySpan<IntPtr> postSpikeErrors,
        double vTh, double tauMem, double tauSyn)
    {
        int batchSize = nSpikesPerSample.Length;
        ValidateFlat(batchSize, nSpikesPerSample, times.Length, sources.Length);
        ValidateErrorPointers(batchSize, spikes, postSpikeErrors);

        var inputs = stackalloc Api.EventPropInput[batchSize];
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pn = nSpikesPerSample)
        {
            FillInputs(pw, nPre, nPost, pt, ps, pn, batchSize, vTh, tauMem, tauSyn, inputs);

            fixed (IntPtr* pErr = postSpikeErrors)
                return eventprop_backward_spikes_batch((IntPtr)inputs, spikes, (IntPtr)pErr);
        }
    }

    /// <summary>
    /// Batch backward pass (spike loss) with the gradients summed into one matrix.
    /// This is the primary FFI entry point: one call, no intermediate per-sample
    /// gradient copies on the managed side.
    /// </summary>
    public static AccumulatedBackwardResult BackwardSpikesBatchAccumulate(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        SpikesBatch spikes,
        ReadOnlySpan<IntPtr> postSpikeErrors,
        double vTh, double tauMem, double tauSyn)
    {
        int batchSize = nSpikesPerSample.Length;
        ValidateFlat(batchSize, nSpikesPerSample, times.Length, sources.Length);
        ValidateErrorPointers(batchSize, spikes, postSpikeErrors);

        var inputs = stackalloc Api.EventPropInput[batchSize];
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pn = nSpikesPerSample)
        {
            FillInputs(pw, nPre, nPost, pt, ps, pn, batchSize, vTh, tauMem, tauSyn, inputs);

            fixed (IntPtr* pErr = postSpikeErrors)
                return eventprop_backward_spikes_batch_accumulate((IntPtr)inputs, spikes, (IntPtr)pErr);
        }
    }

    /// <summary>Batch backward pass (maxima loss) over a maxima batch.</summary>
    public static BackwardsBatch BackwardMaximaBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        MaximaBatch maxima,
        double tauMem, double tauSyn)
    {
        int batchSize = nSpikesPerSample.Length;
        ValidateFlat(batchSize, nSpikesPerSample, times.Length, sources.Length);

        var inputs = stackalloc Api.EventPropInput[batchSize];
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pn = nSpikesPerSample)
        {
            FillInputs(pw, nPre, nPost, pt, ps, pn, batchSize, 0.0, tauMem, tauSyn, inputs);
            return eventprop_backward_maxima_batch((IntPtr)inputs, maxima);
        }
    }

    /// <summary>Batch backward pass (maxima loss) with gradients summed into one matrix.</summary>
    public static AccumulatedBackwardResult BackwardMaximaBatchAccumulate(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        MaximaBatch maxima,
        double tauMem, double tauSyn)
    {
        int batchSize = nSpikesPerSample.Length;
        ValidateFlat(batchSize, nSpikesPerSample, times.Length, sources.Length);

        var inputs = stackalloc Api.EventPropInput[batchSize];
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pn = nSpikesPerSample)
        {
            FillInputs(pw, nPre, nPost, pt, ps, pn, batchSize, 0.0, tauMem, tauSyn, inputs);
            return eventprop_backward_maxima_batch_accumulate((IntPtr)inputs, maxima);
        }
    }

    // ── Shared marshalling ───────────────────────────────────────────

    /// <summary>
    /// Fill a stack-allocated EventPropInput array from raw pointers.
    ///
    /// The CALLER owns the pins. EventPropInput stores raw pointers into
    /// w/times/sources/nSpikesPerSample, and `fixed` releases when its statement
    /// ends, so the pins must span the native call that consumes `inputs`. Taking
    /// them inside this helper would release them the moment it returns, leaving
    /// the native batch call reading arrays a concurrent GC is free to relocate —
    /// which shows up as intermittent EVENTPROP_UNSORTED_SPIKE_TIMES, invalid
    /// source indices, zeroed maxima and short output arrays, all of them
    /// intermittent and load-dependent.
    /// </summary>
    private static void FillInputs(
        double* pw, int nPre, int nPost, double* pt, int* ps, int* pn, int batchSize,
        double vTh, double tauMem, double tauSyn,
        Api.EventPropInput* inputs)
    {
        int offset = 0;
        for (int b = 0; b < batchSize; b++)
        {
            int n = pn[b];
            inputs[b] = Api.EventPropInput.Create(
                pw, nPre, nPost, pt + offset, ps + offset, n, vTh, tauMem, tauSyn);
            offset += n;
        }
    }

    private static void ValidateFlat(int batchSize, ReadOnlySpan<int> nSpikesPerSample,
        int timesLength, int sourcesLength)
    {
        if (batchSize <= 0)
            throw new ArgumentException("Batch size must be positive", nameof(nSpikesPerSample));
        if (timesLength != sourcesLength)
            throw new ArgumentException(
                $"times ({timesLength}) and sources ({sourcesLength}) must have the same length",
                nameof(timesLength));

        int total = 0;
        for (int b = 0; b < batchSize; b++)
        {
            int n = nSpikesPerSample[b];
            if (n < 0)
                throw new ArgumentException($"nSpikesPerSample[{b}] is negative ({n})", nameof(nSpikesPerSample));
            total += n;
        }
        if (total != timesLength)
            throw new ArgumentException(
                $"nSpikesPerSample sums to {total} but times has {timesLength} entries",
                nameof(nSpikesPerSample));
    }

    private static void ValidateErrorPointers(int batchSize, SpikesBatch spikes, ReadOnlySpan<IntPtr> postSpikeErrors)
    {
        if (postSpikeErrors.Length < batchSize)
            throw new ArgumentException(
                $"postSpikeErrors has {postSpikeErrors.Length} entries, batch size is {batchSize}",
                nameof(postSpikeErrors));
        if (spikes.BatchSize != batchSize)
            throw new ArgumentException(
                $"spikes batch has {spikes.BatchSize} samples, inputs have {batchSize}",
                nameof(spikes));
    }
}
