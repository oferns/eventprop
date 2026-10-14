using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>P/Invoke bindings for eventprop_batch.h — batch operations.</summary>
public static unsafe partial class Batch
{
    private const string Lib = "eventprop_c_api";

    // ── Native structs (match C layout exactly) ───────────────────

    /// <summary>Owned spikes batch — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct SpikesBatch : IDisposable
    {
        internal readonly Api.EventPropError errorNo;
        internal readonly int batchSize;
        internal readonly int hasErrors;
        internal readonly IntPtr spikes;   // Api.SpikeOutput* (malloc'd, length = batch_size)

        public Api.EventPropError ErrorNo => errorNo;
        public int BatchSize => batchSize;
        public bool HasErrors => hasErrors != 0;
        public IntPtr Spikes => spikes;

        /// <summary>Borrow sample <paramref name="index"/>'s spike output (owned by the batch — do not Dispose).</summary>
        public Api.SpikeOutput this[int index] => SpikeAt(this, index);

        public void Dispose() => eventprop_free_spikes_batch(this);
    }

    /// <summary>Owned backwards batch — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct BackwardsBatch : IDisposable
    {
        internal readonly Api.EventPropError errorNo;
        internal readonly int batchSize;
        internal readonly int hasErrors;
        internal readonly IntPtr outputs;  // Api.BackwardOutput* (malloc'd, length = batch_size)

        public Api.EventPropError ErrorNo => errorNo;
        public int BatchSize => batchSize;
        public bool HasErrors => hasErrors != 0;
        public IntPtr Outputs => outputs;

        public void Dispose() => eventprop_free_backwards_batch(this);
    }

    /// <summary>Owned maxima batch — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct MaximaBatch : IDisposable
    {
        internal readonly Api.EventPropError errorNo;
        internal readonly int batchSize;
        internal readonly int hasErrors;
        internal readonly IntPtr outputs;  // Api.MaximaOutput* (malloc'd, length = batch_size)

        public Api.EventPropError ErrorNo => errorNo;
        public int BatchSize => batchSize;
        public bool HasErrors => hasErrors != 0;
        public IntPtr Outputs => outputs;

        public void Dispose() => eventprop_free_maxima_batch(this);
    }

    /// <summary>Owned gradient output — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct GradientOutput : IDisposable
    {
        internal readonly Api.EventPropError errorNo;
        internal readonly IntPtr data;     // double* (malloc'd, rows * cols)
        internal readonly int rows;
        internal readonly int cols;

        public Api.EventPropError ErrorNo => errorNo;
        public IntPtr Data => data;
        public int Rows => rows;
        public int Cols => cols;

        public void Dispose() => eventprop_free_gradient_output(this);
    }

    /// <summary>Owned accumulated backward result — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct AccumulatedBackwardResult : IDisposable
    {
        internal readonly Api.EventPropError errorNo;
        internal readonly IntPtr summedGrads;  // GradientOutput* (malloc'd)
        internal readonly int hasErrors;
        internal readonly IntPtr outputs;      // Api.BackwardOutput* (.grad freed, .spike_errors kept)
        internal readonly int batchSize;

        public Api.EventPropError ErrorNo => errorNo;

        /// <summary>Get the owned gradient output as a ref struct view.</summary>
        public GradientOutput GetSummedGrads()
        {
            if (summedGrads == IntPtr.Zero) return default;
            unsafe { return *(GradientOutput*)summedGrads.ToPointer(); }
        }

        public bool HasErrors => hasErrors != 0;
        public IntPtr Outputs => outputs;
        public int BatchSize => batchSize;

        public void Dispose() => eventprop_free_accumulated_backward_result(this);
    }

    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct DropoutOutput
    {
        internal readonly int nSpikes;
        internal readonly IntPtr times;      // double* (malloc'd)
        internal readonly IntPtr sources;    // int* (malloc'd)

        public int NSpikes => nSpikes;
        public IntPtr Times => times;
        public IntPtr Sources => sources;
    }

    /// <summary>Owned dropout batch — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct DropoutBatch : IDisposable
    {
        internal readonly int errorNo;     // EventPropError (0 = OK)
        internal readonly int batchSize;
        internal readonly int hasErrors;
        internal readonly IntPtr outputs;  // DropoutOutput* (malloc'd, length = batch_size)

        public int ErrorNo => errorNo;
        public int BatchSize => batchSize;
        public bool HasErrors => hasErrors != 0;
        public IntPtr Outputs => outputs;

        public void Dispose() => eventprop_free_dropout_batch(this);
    }

    // ── Native function declarations ───────────────────────────────

    [LibraryImport(Lib)]
    public static partial SpikesBatch eventprop_compute_spikes_batch(IntPtr inputs, int batchSize);

    [LibraryImport(Lib)]
    public static partial BackwardsBatch eventprop_backward_spikes_batch(IntPtr inputs, SpikesBatch spikesBatch, IntPtr postSpikeErrors);

    [LibraryImport(Lib)]
    public static partial AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(IntPtr inputs, SpikesBatch spikesBatch, IntPtr postSpikeErrors);

    [LibraryImport(Lib)]
    public static partial MaximaBatch eventprop_compute_maxima_batch(IntPtr inputs, int batchSize);

    [LibraryImport(Lib)]
    public static partial BackwardsBatch eventprop_backward_maxima_batch(IntPtr inputs, MaximaBatch maximaBatch);

    [LibraryImport(Lib)]
    public static partial AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(IntPtr inputs, MaximaBatch maximaBatch);

    [LibraryImport(Lib)]
    public static partial GradientOutput eventprop_accumulate_gradients(BackwardsBatch batch, int nPre, int nPost);

    [LibraryImport(Lib)]
    public static partial DropoutBatch eventprop_dropout_spikes_batch(IntPtr times, IntPtr sources, IntPtr nPerSample, int batchSize, double pDrop, int seed);

    // ── Free functions (public for manual lifetime management) ─────

    [LibraryImport(Lib)]
    public static partial void eventprop_free_spikes_batch(SpikesBatch batch);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_backwards_batch(BackwardsBatch batch);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_maxima_batch(MaximaBatch batch);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_gradient_output(GradientOutput output);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_accumulated_backward_result(AccumulatedBackwardResult result);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_dropout_batch(DropoutBatch batch);
}
