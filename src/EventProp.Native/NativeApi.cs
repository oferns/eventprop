using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CW.EventProp;

/// <summary>
/// P/Invoke bindings for the EventProp native C library (libeventprop_c_api).
/// Pure C implementation — no external dependencies, zero-allocation hot paths.
/// </summary>
public static partial class NativeApi
{
    private const string LibName = "eventprop_c_api";

    // ============================================================
    // Alloc output structs (match C struct layouts)
    // ============================================================

    [StructLayout(LayoutKind.Sequential)]
    public struct SpikeOutputNative
    {
        public int NSpikes;
        public IntPtr Times;      // double*
        public IntPtr Sources;    // int32_t*
        public IntPtr Currents;   // double*
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
        public IntPtr Grad;       // double*
        public IntPtr SpikeErrors;// double*
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BackwardMaximaOutputNative
    {
        public int NPre;
        public int NPost;
        public IntPtr Gradient;   // double*
        public IntPtr SpikeErrors;// double*
    }

    // ============================================================
    // Alloc functions (new API)
    // ============================================================

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial SpikeOutputNative eventprop_compute_spikes_alloc(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources, int nSpikes,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial void eventprop_free_spike_output(ref SpikeOutputNative output);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial SumsOutputNative eventprop_compute_sums_alloc(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources, int nSpikes,
        double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial void eventprop_free_sums_output(ref SumsOutputNative output);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial MaximaOutputNative eventprop_compute_maxima_alloc(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources, int nSpikes,
        double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial void eventprop_free_maxima_output(ref MaximaOutputNative output);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial BackwardSpikesOutputNative eventprop_backward_spikes_alloc(
        double[] times, int[] sources, int nSpikes,
        double[] fwdTimes, int[] fwdSources, double[] fwdCurrents,
        double[] postErrors, int nPostSpikes,
        double[] w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial void eventprop_free_backward_spikes_output(ref BackwardSpikesOutputNative output);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial BackwardMaximaOutputNative eventprop_backward_maxima_alloc(
        double[] inputTimes, int[] inputSources, int nInputSpikes,
        double[] maximaTimes, double[] maximaValues, double[] maximaErrors, int nMaxima,
        double[] w, int nPre, int nPost,
        double tauMem, double tauSyn);

    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    private static partial void eventprop_free_backward_maxima_output(ref BackwardMaximaOutputNative output);

    // ============================================================
    // Managed wrappers for alloc functions
    // ============================================================

    /// <summary>Call compute_spikes_alloc via native C library. Caller must free returned arrays.</summary>
    public static void ComputeSpikesAlloc(
        double[] w, int nPre, int _,
        double[] times, int[] sources,
        double vTh, double tauMem, double tauSyn,
        out int nSpikes, out double[] timesOut, out int[] sourcesOut, out double[] currentsOut)
    {
        var output = eventprop_compute_spikes_alloc(w, nPre, _, times, sources, times.Length,
            vTh, tauMem, tauSyn);
        nSpikes = output.NSpikes;

        timesOut = new double[nSpikes];
        sourcesOut = new int[nSpikes];
        currentsOut = new double[nSpikes];

        if (nSpikes > 0)
        {
            Marshal.Copy(output.Times, timesOut, 0, nSpikes);
            Marshal.Copy(output.Sources, sourcesOut, 0, nSpikes);
            Marshal.Copy(output.Currents, currentsOut, 0, nSpikes);
        }

        eventprop_free_spike_output(ref output);
    }

    /// <summary>Call compute_sums_alloc via native C library. Caller must free returned arrays.</summary>
    public static void ComputeSumsAlloc(
        double[] w, int nPre, int _,
        double[] times, int[] sources,
        double tauMem, double tauSyn,
        out int nRows, out int nPostOut, out double[] sum0Out, out double[] sum1Out)
    {
        var output = eventprop_compute_sums_alloc(w, nPre, _, times, sources, times.Length,
            tauMem, tauSyn);
        nRows = output.NRows;
        nPostOut = output.NPost;

        sum0Out = new double[nRows * nPostOut];
        sum1Out = new double[nRows * nPostOut];

        Marshal.Copy(output.Sum0, sum0Out, 0, nRows * nPostOut);
        Marshal.Copy(output.Sum1, sum1Out, 0, nRows * nPostOut);

        eventprop_free_sums_output(ref output);
    }

    /// <summary>Call compute_maxima_alloc via native C library. Caller must free returned arrays.</summary>
    public static void ComputeMaximaAlloc(
        double[] w, int nPre, int _,
        double[] times, int[] sources,
        double tauMem, double tauSyn,
        out int nPostOut, out double[] maximaTimesOut, out double[] maximaValuesOut)
    {
        var output = eventprop_compute_maxima_alloc(w, nPre, _, times, sources, times.Length,
            tauMem, tauSyn);
        nPostOut = output.NPost;

        maximaTimesOut = new double[nPostOut];
        maximaValuesOut = new double[nPostOut];

        Marshal.Copy(output.Times, maximaTimesOut, 0, nPostOut);
        Marshal.Copy(output.Values, maximaValuesOut, 0, nPostOut);

        eventprop_free_maxima_output(ref output);
    }

    /// <summary>Call backward_spikes_alloc via native C library. Caller must free returned arrays.</summary>
    public static void BackwardSpikesAlloc(
        double[] times, int[] sources,
        double[] fwdTimes, int[] fwdSources, double[] fwdCurrents,
        double[] postErrors,
        double[] w, int nPre, int _,
        double vTh, double tauMem, double tauSyn,
        out int nPreOut, out int nPostOut, out double[] gradOut, out double[] spikeErrorsOut)
    {
        var output = eventprop_backward_spikes_alloc(times, sources, times.Length,
            fwdTimes, fwdSources, fwdCurrents, postErrors, postErrors.Length,
            w, nPre, _, vTh, tauMem, tauSyn);
        nPreOut = output.NPre;
        nPostOut = output.NPost;

        gradOut = new double[nPreOut * nPostOut];
        spikeErrorsOut = new double[times.Length];

        Marshal.Copy(output.Grad, gradOut, 0, nPreOut * nPostOut);
        Marshal.Copy(output.SpikeErrors, spikeErrorsOut, 0, times.Length);

        eventprop_free_backward_spikes_output(ref output);
    }

    /// <summary>Call backward_maxima_alloc via native C library. Caller must free returned arrays.</summary>
    public static void BackwardMaximaAlloc(
        double[] inputTimes, int[] inputSources,
        double[] maximaTimes, double[] maximaValues, double[] maximaErrors,
        double[] w, int nPre, int _,
        double tauMem, double tauSyn,
        out int nPreOut, out int nPostOut, out double[] gradientOut, out double[] spikeErrorsOut)
    {
        var output = eventprop_backward_maxima_alloc(inputTimes, inputSources, inputTimes.Length,
            maximaTimes, maximaValues, maximaErrors, maximaTimes.Length,
            w, nPre, _, tauMem, tauSyn);
        nPreOut = output.NPre;
        nPostOut = output.NPost;

        gradientOut = new double[nPreOut * nPostOut];
        spikeErrorsOut = new double[inputTimes.Length];

        Marshal.Copy(output.Gradient, gradientOut, 0, nPreOut * nPostOut);
        Marshal.Copy(output.SpikeErrors, spikeErrorsOut, 0, inputTimes.Length);

        eventprop_free_backward_maxima_output(ref output);
    }

    // ============================================================
    // compute_sums
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv( CallConvs = [ typeof(CallConvCdecl) ])]
    private static partial void eventprop_compute_sums(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources, int nSpikes,
        double tauMem, double tauSyn,
        double[] sum0Out, double[] sum1Out);

    /// <summary>Call compute_sums via native C library.</summary>
    public static void ComputeSums(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources,
        double tauMem, double tauSyn,
        double[] sum0Out, double[] sum1Out)
    {
        eventprop_compute_sums(w, nPre, nPost, times, sources, times.Length, tauMem, tauSyn, sum0Out, sum1Out);
    }

    // ============================================================
    // compute_spikes
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [ typeof(CallConvCdecl) ])]
    private static partial int eventprop_compute_spikes(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources, int nSpikes,
        double vTh, double tauMem, double tauSyn,
        double[] outTimes, int[] outSources, double[] outCurrents);

    /// <summary>Call compute_spikes via native C library. Returns number of output spikes.</summary>
    public static int ComputeSpikes(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources,
        double vTh, double tauMem, double tauSyn,
        double[] outTimes, int[] outSources, double[] outCurrents)
    {
        return eventprop_compute_spikes(w, nPre, nPost, times, sources, times.Length, vTh, tauMem, tauSyn, outTimes, outSources, outCurrents);
    }

    // ============================================================
    // compute_maxima
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [ typeof(CallConvCdecl) ])]
    private static partial void eventprop_compute_maxima(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources, int nSpikes,
        double tauMem, double tauSyn,
        double[] maximaTimesOut, double[] maximaValuesOut);

    /// <summary>Call compute_maxima via native C library.</summary>
    public static void ComputeMaxima(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources,
        double tauMem, double tauSyn,
        double[] maximaTimesOut, double[] maximaValuesOut)
    {
        eventprop_compute_maxima(w, nPre, nPost, times, sources, times.Length, tauMem, tauSyn, maximaTimesOut, maximaValuesOut);
    }

    // ============================================================
    // backward_spikes — zero-copy output for gradient + spike errors
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [ typeof(CallConvCdecl) ])]
    private static partial void eventprop_backward_spikes(
        double[] inputTimes, int[] inputSources, int nInputSpikes,
        double[] postTimes, int[] postSources, double[] postCurrents,
        double[] postErrors, int nPostSpikes,
        double[] w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn,
        double[] gradient,      // zero-copy output: n_pre x n_post (row-major)
        double[] spikeErrors);  // zero-copy output: n_input_spikes (mutated)

    /// <summary>Call backward (Spikes loss) via native C library.
    /// gradient must be zero-initialized (n_pre * n_post). spikeErrors is mutated in-place.</summary>
    public static void BackwardSpikes(
        double[] inputTimes, int[] inputSources,
        double[] postTimes, int[] postSources, double[] postCurrents,
        double[] postErrors, double[] w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn,
        double[] gradient,      // zero-initialized output: n_pre * n_post
        double[] spikeErrors)   // mutated in-place: n_input_spikes
    {
        eventprop_backward_spikes(
            inputTimes, inputSources, inputTimes.Length,
            postTimes, postSources, postCurrents,
            postErrors, postErrors.Length,
            w, nPre, nPost,
            vTh, tauMem, tauSyn, gradient, spikeErrors);
    }

    // ============================================================
    // backward_maxima — zero-copy output for gradient + spike errors
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [ typeof(CallConvCdecl) ])]
    private static partial void eventprop_backward_maxima(
        double[] inputTimes, int[] inputSources, int nInputSpikes,
        double[] maximaTimes, double[] maximaValues, double[] maximaErrors, int nMaxima,
        double[] w, int nPre, int nPost,
        double tauMem, double tauSyn,
        double[] gradient,      // zero-copy output: n_pre x n_post (row-major)
        double[] spikeErrors);  // zero-copy output: n_input_spikes (mutated)

    /// <summary>Call backward (Maxima loss) via native C library.
    /// gradient must be zero-initialized (n_pre * n_post). spikeErrors is mutated in-place.</summary>
    public static void BackwardMaxima(
        double[] inputTimes, int[] inputSources,
        double[] maximaTimes, double[] maximaValues, double[] maximaErrors,
        double[] w, int nPre, int nPost,
        double tauMem, double tauSyn,
        double[] gradient,      // zero-initialized output: n_pre * n_post
        double[] spikeErrors)   // mutated in-place: n_input_spikes
    {
        eventprop_backward_maxima(inputTimes, inputSources, inputTimes.Length,
            maximaTimes, maximaValues, maximaErrors, maximaTimes.Length,
            w, nPre, nPost, tauMem, tauSyn, gradient, spikeErrors);
    }
}
