using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CW.EventProp;

/// <summary>
/// Zero-allocation P/Invoke bindings for libeventprop_c_api.dylib.
/// 
/// All functions use the C library's alloc/free API for automatic memory management.
/// Output provides zero-copy Span access via NativeMemory — no intermediate copies.
/// </summary>
public static unsafe partial class NativeApi
{
    private const string LibName = "eventprop_c_api";

    // ═══════════════════════════════════════════════════════════
    // Native struct declarations (match C layout exactly)
    // ═══════════════════════════════════════════════════════════

    [StructLayout(LayoutKind.Sequential)]
    public struct SpikeOutputNative
    {
        public int NSpikes;
        public IntPtr Times;      // double*
        public IntPtr Sources;    // int32_t*
        public IntPtr Currents;  // double*
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SumsOutputNative
    {
        public int NRows;         // n_spikes + 1
        public int NPost;
        public IntPtr Sum0;       // double*
        public IntPtr Sum1;       // double*
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MaximaOutputNative
    {
        public int NPost;
        public IntPtr Times;      // double*
        public IntPtr Values;     // double*
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BackwardSpikesOutputNative
    {
        public int NPre;
        public int NPost;
        public IntPtr Grad;       // double* (n_pre * n_post)
        public IntPtr SpikeErrors;// double* (n_spikes)
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BackwardMaximaOutputNative
    {
        public int NPre;
        public int NPost;
        public IntPtr Gradient;   // double* (n_pre * n_post)
        public IntPtr SpikeErrors;// double* (n_input_spikes)
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VoltageTraceOutputNative
    {
        public int NT;            // number of time points
        public IntPtr VTrace;     // double*
        public IntPtr Ts;         // double*
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LambdaITraceOutputNative
    {
        public int NT;            // number of time points
        public IntPtr LambdaTrace;// double*
        public IntPtr Ts;         // double*
    }

    // ═══════════════════════════════════════════════════════════
    // Native function declarations
    // ═══════════════════════════════════════════════════════════

    [LibraryImport(LibName)]
    private static partial SpikeOutputNative eventprop_compute_spikes(
        double* w, int nPre, int nPost,
        double* times, int* sources, int nSpikes,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_spike_output(ref SpikeOutputNative output);

    [LibraryImport(LibName)]
    private static partial SumsOutputNative eventprop_compute_sums(
        double* w, int nPre, int nPost,
        double* times, int* sources, int nSpikes,
        double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_sums_output(ref SumsOutputNative output);

    [LibraryImport(LibName)]
    private static partial MaximaOutputNative eventprop_compute_maxima(
        double* w, int nPre, int nPost,
        double* times, int* sources, int nSpikes,
        double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_maxima_output(ref MaximaOutputNative output);

    [LibraryImport(LibName)]
    private static partial BackwardSpikesOutputNative eventprop_backward_spikes(
        double* times, int* sources, int nSpikes,
        double* fwdTimes, int* fwdSources, double* fwdCurrents,
        double* postErrors, int nPostSpikes,
        double* w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_backward_spikes_output(ref BackwardSpikesOutputNative output);

    [LibraryImport(LibName)]
    private static partial BackwardMaximaOutputNative eventprop_backward_maxima(
        double* inputTimes, int* inputSources, int nInputSpikes,
        double* maximaTimes, double* maximaValues, double* maximaErrors, int nMaxima,
        double* w, int nPre, int nPost,
        double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_backward_maxima_output(ref BackwardMaximaOutputNative output);

    [LibraryImport(LibName)]
    private static partial VoltageTraceOutputNative eventprop_compute_voltage_trace(
        double tMax, double dt, int targetNrnIdx,
        double* w, int nPre, int nPost,
        double* times, int* sources, int nSpikes,
        double* fwdTimes, int* fwdSources, int nPostSpikes,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_voltage_trace_output(ref VoltageTraceOutputNative output);

    [LibraryImport(LibName)]
    private static partial LambdaITraceOutputNative eventprop_compute_lambda_i_trace(
        double tMax, double dt, int targetNrnIdx,
        double* fwdTimes, int nPostSpikes,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_lambda_i_trace_output(ref LambdaITraceOutputNative output);

    // Batch operations (return IntPtr to array of structs)
    [LibraryImport(LibName)]
    private static partial IntPtr eventprop_compute_spikes_batch(
        double* w, int nPre, int nPost,
        double* times, int* sources,
        int* nSpikesPerSample, int batchSize,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_batch_spike_output(IntPtr output, int batchSize);

    [LibraryImport(LibName)]
    private static partial IntPtr eventprop_backward_spikes_batch(
        double* times, int* sources,
        int* nSpikesPerSample,
        double* fwdTimes, int* fwdSources,
        double* fwdCurrents,
        double* postErrors,
        int* nFwdSpikesPerSample,
        int batchSize,
        double* w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    private static partial void eventprop_free_backward_spikes_batch_output(IntPtr output, int batchSize);

    // ═══════════════════════════════════════════════════════════
    // Public API — zero-allocation wrappers with Span access
    // ═══════════════════════════════════════════════════════════

    /// <summary>
    /// Forward pass — predicts post-synaptic spike times.
    /// Caller must call Dispose() to free native memory.
    /// </summary>
    public static SpikeOutput ComputeSpikes(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            var native = eventprop_compute_spikes(
                pw, nPre, nPost, pt, ps, times.Length, vTh, tauMem, tauSyn);
            return new SpikeOutput(native);
        }
    }

    /// <summary>Compute cumulative exponential-weighted sums.</summary>
    public static SumsOutput ComputeSums(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            var native = eventprop_compute_sums(
                pw, nPre, nPost, pt, ps, times.Length, tauMem, tauSyn);
            return new SumsOutput(native);
        }
    }

    /// <summary>Compute membrane potential maxima per neuron.</summary>
    public static MaximaOutput ComputeMaxima(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            var native = eventprop_compute_maxima(
                pw, nPre, nPost, pt, ps, times.Length, tauMem, tauSyn);
            return new MaximaOutput(native);
        }
    }

    /// <summary>Backward pass — spike-based gradient computation.</summary>
    public static BackwardSpikesOutput BackwardSpikes(
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<double> fwdTimes, ReadOnlySpan<int> fwdSources,
        ReadOnlySpan<double> fwdCurrents,
        ReadOnlySpan<double> postErrors, int nPostSpikes,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (double* pft = fwdTimes)
        fixed (int* pfs = fwdSources)
        fixed (double* pfc = fwdCurrents)
        fixed (double* pe = postErrors)
        fixed (double* pw = w)
        {
            var native = eventprop_backward_spikes(
                pt, ps, times.Length, pft, pfs, pfc, pe, nPostSpikes,
                pw, nPre, nPost, vTh, tauMem, tauSyn);
            return new BackwardSpikesOutput(native);
        }
    }

    /// <summary>Backward pass — maxima-based gradient computation.</summary>
    public static BackwardMaximaOutput BackwardMaxima(
        ReadOnlySpan<double> inputTimes, ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> maximaTimes, ReadOnlySpan<double> maximaValues,
        ReadOnlySpan<double> maximaErrors, int nMaxima,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double tauMem, double tauSyn)
    {
        fixed (double* pit = inputTimes)
        fixed (int* isrc = inputSources)
        fixed (double* mt = maximaTimes)
        fixed (double* mv = maximaValues)
        fixed (double* me = maximaErrors)
        fixed (double* pw = w)
        {
            var native = eventprop_backward_maxima(
                pit, isrc, inputTimes.Length, mt, mv, me, nMaxima,
                pw, nPre, nPost, tauMem, tauSyn);
            return new BackwardMaximaOutput(native);
        }
    }

    /// <summary>Voltage trace diagnostic.</summary>
    public static VoltageTraceOutput ComputeVoltageTrace(
        double tMax, double dt, int targetNrnIdx,
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<double> fwdTimes, ReadOnlySpan<int> fwdSources, int nPostSpikes,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (double* pft = fwdTimes)
        fixed (int* pfs = fwdSources)
        {
            var native = eventprop_compute_voltage_trace(
                tMax, dt, targetNrnIdx, pw, nPre, nPost,
                pt, ps, times.Length, pft, pfs, nPostSpikes, vTh, tauMem, tauSyn);
            return new VoltageTraceOutput(native);
        }
    }

    /// <summary>Lambda-I trace diagnostic.</summary>
    public static LambdaITraceOutput ComputeLambdaITrace(
        double tMax, double dt, int targetNrnIdx,
        ReadOnlySpan<double> fwdTimes, int nPostSpikes,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pft = fwdTimes)
        {
            var native = eventprop_compute_lambda_i_trace(
                tMax, dt, targetNrnIdx, pft, nPostSpikes, vTh, tauMem, tauSyn);
            return new LambdaITraceOutput(native);
        }
    }

    /// <summary>Batch forward pass — parallel processing via OpenMP.</summary>
    public static BatchSpikeOutput ComputeSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample, int batchSize,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        {
            var ptr = eventprop_compute_spikes_batch(
                pw, nPre, nPost, pt, ps, pnsps, batchSize, vTh, tauMem, tauSyn);
            return new BatchSpikeOutput(ptr, batchSize);
        }
    }

    /// <summary>Batch backward pass — parallel processing via OpenMP.</summary>
    public static BatchBackwardSpikesOutput BackwardSpikesBatch(
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<int> nSpikesPerSample,
        ReadOnlySpan<double> fwdTimes, ReadOnlySpan<int> fwdSources,
        ReadOnlySpan<double> fwdCurrents,
        ReadOnlySpan<double> postErrors,
        ReadOnlySpan<int> nFwdSpikesPerSample,
        int batchSize,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (int* pnsps = nSpikesPerSample)
        fixed (double* pft = fwdTimes)
        fixed (int* pfs = fwdSources)
        fixed (double* pfc = fwdCurrents)
        fixed (double* pe = postErrors)
        fixed (int* pnfsps = nFwdSpikesPerSample)
        fixed (double* pw = w)
        {
            var ptr = eventprop_backward_spikes_batch(
                pt, ps, pnsps, pft, pfs, pfc, pe, pnfsps,
                batchSize, pw, nPre, nPost, vTh, tauMem, tauSyn);
            return new BatchBackwardSpikesOutput(ptr, batchSize);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // IDisposable output wrappers with zero-copy Span access
    // ═══════════════════════════════════════════════════════════

    public class SpikeOutput : IDisposable
    {
        private SpikeOutputNative _native;

        public int NSpikes => _native.NSpikes;
        public Span<double> Times => NativeSpan<double>(_native.Times, _native.NSpikes);
        public Span<int> Sources => NativeSpan<int>(_native.Sources, _native.NSpikes);
        public Span<double> Currents => NativeSpan<double>(_native.Currents, _native.NSpikes);

        public SpikeOutput(SpikeOutputNative native) => _native = native;

        public void Dispose() => eventprop_free_spike_output(ref _native);
    }

    public class SumsOutput : IDisposable
    {
        private SumsOutputNative _native;

        public int NRows => _native.NRows;
        public int NPost => _native.NPost;
        public Span<double> Sum0 => NativeSpan<double>(_native.Sum0, _native.NRows * _native.NPost);
        public Span<double> Sum1 => NativeSpan<double>(_native.Sum1, _native.NRows * _native.NPost);

        public SumsOutput(SumsOutputNative native) => _native = native;

        public void Dispose() => eventprop_free_sums_output(ref _native);
    }

    public class MaximaOutput : IDisposable
    {
        private MaximaOutputNative _native;

        public int NPost => _native.NPost;
        public Span<double> Times => NativeSpan<double>(_native.Times, _native.NPost);
        public Span<double> Values => NativeSpan<double>(_native.Values, _native.NPost);

        public MaximaOutput(MaximaOutputNative native) => _native = native;

        public void Dispose() => eventprop_free_maxima_output(ref _native);
    }

    public class BackwardSpikesOutput : IDisposable
    {
        private BackwardSpikesOutputNative _native;

        public int NPre => _native.NPre;
        public int NPost => _native.NPost;
        public Span<double> Grad => NativeSpan<double>(_native.Grad, _native.NPre * _native.NPost);
        public Span<double> SpikeErrors => NativeSpan<double>(_native.SpikeErrors, _native.NPre);

        public BackwardSpikesOutput(BackwardSpikesOutputNative native) => _native = native;

        public void Dispose() => eventprop_free_backward_spikes_output(ref _native);
    }

    public class BackwardMaximaOutput : IDisposable
    {
        private BackwardMaximaOutputNative _native;

        public int NPre => _native.NPre;
        public int NPost => _native.NPost;
        public Span<double> Gradient => NativeSpan<double>(_native.Gradient, _native.NPre * _native.NPost);
        public Span<double> SpikeErrors => NativeSpan<double>(_native.SpikeErrors, _native.NPre);

        public BackwardMaximaOutput(BackwardMaximaOutputNative native) => _native = native;

        public void Dispose() => eventprop_free_backward_maxima_output(ref _native);
    }

    public class VoltageTraceOutput : IDisposable
    {
        private VoltageTraceOutputNative _native;

        public int NT => _native.NT;
        public Span<double> VTrace => NativeSpan<double>(_native.VTrace, _native.NT);
        public Span<double> Ts => NativeSpan<double>(_native.Ts, _native.NT);

        public VoltageTraceOutput(VoltageTraceOutputNative native) => _native = native;

        public void Dispose() => eventprop_free_voltage_trace_output(ref _native);
    }

    public class LambdaITraceOutput : IDisposable
    {
        private LambdaITraceOutputNative _native;

        public int NT => _native.NT;
        public Span<double> LambdaTrace => NativeSpan<double>(_native.LambdaTrace, _native.NT);
        public Span<double> Ts => NativeSpan<double>(_native.Ts, _native.NT);

        public LambdaITraceOutput(LambdaITraceOutputNative native) => _native = native;

        public void Dispose() => eventprop_free_lambda_i_trace_output(ref _native);
    }

    // Batch wrappers (own the array pointer)
    public class BatchSpikeOutput : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _batchSize;

        public int BatchSize => _batchSize;

        public BatchSpikeOutput(IntPtr ptr, int batchSize)
        {
            _ptr = ptr;
            _batchSize = batchSize;
        }

        public SpikeOutput Get(int index)
        {
            SpikeOutputNative* elements = (SpikeOutputNative*)_ptr.ToPointer();
            return new SpikeOutput(elements[index]);
        }

        public void Dispose() => eventprop_free_batch_spike_output(_ptr, _batchSize);
    }

    public class BatchBackwardSpikesOutput : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _batchSize;

        public int BatchSize => _batchSize;

        public BatchBackwardSpikesOutput(IntPtr ptr, int batchSize)
        {
            _ptr = ptr;
            _batchSize = batchSize;
        }

        public BackwardSpikesOutput Get(int index)
        {
            BackwardSpikesOutputNative* elements = (BackwardSpikesOutputNative*)_ptr.ToPointer();
            return new BackwardSpikesOutput(elements[index]);
        }

        public void Dispose() => eventprop_free_backward_spikes_batch_output(_ptr, _batchSize);
    }

    // ═══════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════

    private static Span<T> NativeSpan<T>(IntPtr ptr, int length) where T : unmanaged
    {
        if (ptr == IntPtr.Zero || length == 0) return Span<T>.Empty;
        return new Span<T>((T*)ptr.ToPointer(), length);
    }
}
