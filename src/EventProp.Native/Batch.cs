using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>P/Invoke bindings for eventprop_batch.h — struct-based batch operations only.</summary>
public static unsafe partial class Batch
{
    private const string Lib = "eventprop_c_api";

    // ── Native structs (match C layout exactly) ────────────────

    [StructLayout(LayoutKind.Sequential)]
    public struct BatchEventPropInput
    {
        public IntPtr Input;              // const EventPropInput*
        public IntPtr NSpikesPerSample;   // const int*
        public int BatchSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct DropoutOutput
    {
        public int NSpikes;
        public IntPtr Times;      // double* (malloc'd)
        public IntPtr Sources;    // int* (malloc'd)
    }

    // ── Native function declarations — struct-based only ───────

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_compute_spikes_batch(IntPtr input);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_backward_spikes_batch(IntPtr input, IntPtr fwdBatch, IntPtr postErrors);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_backward_spikes_batch_accumulate(IntPtr input, IntPtr fwdBatch, IntPtr postErrors);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_compute_maxima_batch(IntPtr input);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_backward_maxima_batch(IntPtr input, IntPtr fwdBatch, IntPtr maximaErrors);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_backward_maxima_batch_accumulate(IntPtr input, IntPtr fwdBatch, IntPtr maximaErrors);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_dropout_spikes_batch(IntPtr input, double pDrop, int seed);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_batch_spike_output(IntPtr output, int batchSize);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_backward_batch_output(IntPtr output, int batchSize);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_batch_maxima_output(IntPtr output, int batchSize);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_dropout_output(IntPtr output, int batchSize);

  // ── Flat-array batch (concatenated input) ────────────────────

    /// <summary>Batch forward pass (spikes) using concatenated flat arrays.</summary>
    public static OwnedBatchSpikeOutput ComputeSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        {
            Api.EventPropInput* epInput = stackalloc Api.EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = vTh;

            BatchEventPropInput* input = stackalloc BatchEventPropInput[1];
            input->Input = (IntPtr)epInput;
            input->NSpikesPerSample = (IntPtr)pnsps;
            input->BatchSize = nSpikesPerSample.Length;

            return new OwnedBatchSpikeOutput(eventprop_compute_spikes_batch((IntPtr)input), nSpikesPerSample.Length);
        }
    }

    /// <summary>Batch forward pass (maxima) using concatenated flat arrays.</summary>
    public static OwnedBatchMaximaOutput ComputeMaximaBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        {
            Api.EventPropInput* epInput = stackalloc Api.EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = 0;

            BatchEventPropInput* input = stackalloc BatchEventPropInput[1];
            input->Input = (IntPtr)epInput;
            input->NSpikesPerSample = (IntPtr)pnsps;
            input->BatchSize = nSpikesPerSample.Length;

            return new OwnedBatchMaximaOutput(eventprop_compute_maxima_batch((IntPtr)input), nSpikesPerSample.Length);
        }
    }

    /// <summary>Batch backward pass (spikes, per-sample gradients) using flat arrays.</summary>
    public static OwnedBatchBackwardOutput BackwardSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        IntPtr fwdBatchPtr,
        ReadOnlySpan<double> postErrors,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        fixed (double* pe = postErrors)
        {
            Api.EventPropInput* epInput = stackalloc Api.EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = vTh;

            BatchEventPropInput* input = stackalloc BatchEventPropInput[1];
            input->Input = (IntPtr)epInput;
            input->NSpikesPerSample = (IntPtr)pnsps;
            input->BatchSize = nSpikesPerSample.Length;

            return new OwnedBatchBackwardOutput(
                eventprop_backward_spikes_batch((IntPtr)input, fwdBatchPtr, (IntPtr)pe), nSpikesPerSample.Length);
        }
    }

    /// <summary>Batch backward pass (spikes, accumulated gradient) using flat arrays.</summary>
    public static Api.OwnedBackwardOutput BackwardSpikesBatchAccumulate(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        IntPtr fwdBatchPtr,
        ReadOnlySpan<double> postErrors,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        fixed (double* pe = postErrors)
        {
            Api.EventPropInput* epInput = stackalloc Api.EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = vTh;

            BatchEventPropInput* input = stackalloc BatchEventPropInput[1];
            input->Input = (IntPtr)epInput;
            input->NSpikesPerSample = (IntPtr)pnsps;
            input->BatchSize = nSpikesPerSample.Length;

            return new Api.OwnedBackwardOutput(
                eventprop_backward_spikes_batch_accumulate((IntPtr)input, fwdBatchPtr, (IntPtr)pe), times.Length);
        }
    }

    /// <summary>Batch backward pass (maxima, per-sample gradients) using flat arrays.</summary>
    public static OwnedBatchBackwardOutput BackwardMaximaBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        IntPtr fwdBatchPtr,
        ReadOnlySpan<double> maximaErrors,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        fixed (double* me = maximaErrors)
        {
            Api.EventPropInput* epInput = stackalloc Api.EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = 0;

            BatchEventPropInput* input = stackalloc BatchEventPropInput[1];
            input->Input = (IntPtr)epInput;
            input->NSpikesPerSample = (IntPtr)pnsps;
            input->BatchSize = nSpikesPerSample.Length;

            return new OwnedBatchBackwardOutput(
                eventprop_backward_maxima_batch((IntPtr)input, fwdBatchPtr, (IntPtr)me), nSpikesPerSample.Length);
        }
    }

    /// <summary>Batch backward pass (maxima, accumulated gradient) using flat arrays.</summary>
    public static Api.OwnedBackwardOutput BackwardMaximaBatchAccumulate(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        IntPtr fwdBatchPtr,
        ReadOnlySpan<double> maximaErrors,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        fixed (double* me = maximaErrors)
        {
            Api.EventPropInput* epInput = stackalloc Api.EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = 0;

            BatchEventPropInput* input = stackalloc BatchEventPropInput[1];
            input->Input = (IntPtr)epInput;
            input->NSpikesPerSample = (IntPtr)pnsps;
            input->BatchSize = nSpikesPerSample.Length;

            return new Api.OwnedBackwardOutput(
                eventprop_backward_maxima_batch_accumulate((IntPtr)input, fwdBatchPtr, (IntPtr)me), times.Length);
        }
    }

  // ── Batch dropout ────────────────────────────────────────────

    /// <summary>Randomly drop spikes from each sample in a batch.</summary>
    public static OwnedBatchDropoutOutput DropoutSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        double pDrop, int seed)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        {
            Api.EventPropInput* epInput = stackalloc Api.EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = 0;
            epInput->TauSyn = 0;
            epInput->VTh = 0;

            BatchEventPropInput* input = stackalloc BatchEventPropInput[1];
            input->Input = (IntPtr)epInput;
            input->NSpikesPerSample = (IntPtr)pnsps;
            input->BatchSize = nSpikesPerSample.Length;

            return new OwnedBatchDropoutOutput(eventprop_dropout_spikes_batch((IntPtr)input, pDrop, seed), nSpikesPerSample.Length);
        }
    }

  // ── IDisposable wrappers ────────────────────────────────────

    /// <summary>Owned batch spike output array — Dispose() frees all samples.</summary>
    public readonly struct OwnedBatchSpikeOutput : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _batchSize;

        public int BatchSize => _batchSize;

        /// <summary>Get per-sample spike output (borrowed — do not Dispose).</summary>
        public BorrowedSpikeOutput this[int index]
        {
            get
            {
                var basePtr = (Api.SpikeOutput*)_ptr.ToPointer();
                return new BorrowedSpikeOutput(basePtr[index]);
            }
        }

        internal OwnedBatchSpikeOutput(IntPtr ptr, int batchSize)
        {
            _ptr = ptr;
            _batchSize = batchSize;
        }

        /// <summary>Get the raw native handle (for passing to backward pass).</summary>
        public IntPtr GetHandle() => _ptr;

        public void Dispose() => eventprop_free_batch_spike_output(_ptr, _batchSize);
    }

    /// <summary>Owned batch maxima output array — Dispose() frees all samples.</summary>
    public readonly struct OwnedBatchMaximaOutput : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _batchSize;

        public int BatchSize => _batchSize;

        /// <summary>Get per-sample maxima output (borrowed — do not Dispose).</summary>
        public BorrowedMaximaOutput this[int index]
        {
            get
            {
                var basePtr = (Api.MaximaOutput*)_ptr.ToPointer();
                return new BorrowedMaximaOutput(basePtr[index]);
            }
        }

        internal OwnedBatchMaximaOutput(IntPtr ptr, int batchSize)
        {
            _ptr = ptr;
            _batchSize = batchSize;
        }

        /// <summary>Get the raw native handle (for passing to backward pass).</summary>
        public IntPtr GetHandle() => _ptr;

        public void Dispose() => eventprop_free_batch_maxima_output(_ptr, _batchSize);
    }

    /// <summary>Owned batch backward output array — Dispose() frees all samples.</summary>
    public readonly struct OwnedBatchBackwardOutput : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _batchSize;

        public int BatchSize => _batchSize;

        /// <summary>Get per-sample backward output (borrowed — do not Dispose).</summary>
        public BorrowedBackwardOutput this[int index]
        {
            get
            {
                var basePtr = (Api.BackwardOutput*)_ptr.ToPointer();
                return new BorrowedBackwardOutput(basePtr[index]);
            }
        }

        internal OwnedBatchBackwardOutput(IntPtr ptr, int batchSize)
        {
            _ptr = ptr;
            _batchSize = batchSize;
        }

        public void Dispose() => eventprop_free_backward_batch_output(_ptr, _batchSize);
    }

    /// <summary>Owned batch dropout output array — Dispose() frees all samples.</summary>
    public readonly struct OwnedBatchDropoutOutput : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _batchSize;

        public int BatchSize => _batchSize;

        /// <summary>Get per-sample dropout output (borrowed — do not Dispose).</summary>
        public BorrowedDropoutOutput this[int index]
        {
            get
            {
                var basePtr = (DropoutOutput*)_ptr.ToPointer();
                return new BorrowedDropoutOutput(basePtr[index]);
            }
        }

        internal OwnedBatchDropoutOutput(IntPtr ptr, int batchSize)
        {
            _ptr = ptr;
            _batchSize = batchSize;
        }

        public void Dispose() => eventprop_free_dropout_output(_ptr, _batchSize);
    }

  // ── Borrowed views (zero-copy, no Dispose needed) ────────────

    /// <summary>Borrowed view into a SpikeOutput — lifetime tied to owning batch.</summary>
    public readonly struct BorrowedSpikeOutput
    {
        private readonly Api.SpikeOutput _output;

        public int NSpikes => _output.NSpikes;
        public Span<double> Times => NS<double>(_output.Times, _output.NSpikes);
        public Span<int> Sources => NS<int>(_output.Sources, _output.NSpikes);
        public Span<double> Currents => NS<double>(_output.Currents, _output.NSpikes);

        internal BorrowedSpikeOutput(Api.SpikeOutput output) => _output = output;
    }

    /// <summary>Borrowed view into a MaximaOutput — lifetime tied to owning batch.</summary>
    public readonly struct BorrowedMaximaOutput
    {
        private readonly Api.MaximaOutput _output;

        public int NPost => _output.NPost;
        public Span<double> Times => NS<double>(_output.Times, _output.NPost);
        public Span<double> Values => NS<double>(_output.Values, _output.NPost);

        internal BorrowedMaximaOutput(Api.MaximaOutput output) => _output = output;
    }

    /// <summary>Borrowed view into a BackwardOutput — lifetime tied to owning batch.</summary>
    public readonly struct BorrowedBackwardOutput
    {
        private readonly Api.BackwardOutput _output;

        public int NPre => _output.NPre;
        public int NPost => _output.NPost;
        public Span<double> Gradient => NS<double>(_output.Grad, _output.NPre * _output.NPost);
        public Span<double> SpikeErrors => NS<double>(_output.SpikeErrors, _output.NPre);

        internal BorrowedBackwardOutput(Api.BackwardOutput output) => _output = output;
    }

    /// <summary>Borrowed view into a DropoutOutput — lifetime tied to owning batch.</summary>
    public readonly struct BorrowedDropoutOutput
    {
        private readonly DropoutOutput _output;

        public int NSpikes => _output.NSpikes;
        public Span<double> Times => NS<double>(_output.Times, _output.NSpikes);
        public Span<int> Sources => NS<int>(_output.Sources, _output.NSpikes);

        internal BorrowedDropoutOutput(DropoutOutput output) => _output = output;
    }

  // ── Helpers ─────────────────────────────────────────────────

    private static Span<T> NS<T>(IntPtr ptr, int len) where T : unmanaged
    {
        if (ptr == IntPtr.Zero || len == 0) return Span<T>.Empty;
        return new Span<T>((T*)ptr.ToPointer(), len);
    }
}
