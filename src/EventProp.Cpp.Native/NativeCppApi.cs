using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CW.EventProp;

/// <summary>
/// P/Invoke bindings for the EventProp native C++ wrapper (libeventprop_cpp_wrapper).
/// Eigen/Boost-based reference implementation with parallel execution support.
/// </summary>
public static partial class NativeCppApi
{
    private const string LibName = "libeventprop_cpp_wrapper";

    // ============================================================
    // compute_sums
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [ typeof(CallConvCdecl) ])]
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
    // backward_spikes
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [ typeof(CallConvCdecl) ])]
    private static partial void eventprop_backward_spikes(
        double[] inputTimes, int[] inputSources, int nInputSpikes,
        double[] postTimes, int[] postSources, double[] postCurrents,
        double[] postErrors, int nPostSpikes,
        double[] w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn,
        double[] gradient);

    /// <summary>Call backward (Spikes loss) via native C library.</summary>
    public static void BackwardSpikes(
        double[] inputTimes, int[] inputSources,
        double[] postTimes, int[] postSources, double[] postCurrents,
        double[] postErrors, double[] w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn,
        double[] gradient)
    {
        eventprop_backward_spikes(
            inputTimes, inputSources, inputTimes.Length,
            postTimes, postSources, postCurrents,
            postErrors, postErrors.Length,
            w, nPre, nPost,
            vTh, tauMem, tauSyn, gradient);
    }

    // ============================================================
    // backward_maxima
    // ============================================================
    [LibraryImport(LibName)]
    [UnmanagedCallConv(CallConvs = [ typeof(CallConvCdecl) ])]
    private static partial void eventprop_backward_maxima(
        double[] inputTimes, int[] inputSources, int nInputSpikes,
        double[] maximaTimes, double[] maximaValues, double[] maximaErrors, int nMaxima,
        double[] w, int nPre, int nPost,
        double tauMem, double tauSyn,
        double[] gradient);

    /// <summary>Call backward (Maxima loss) via native C library.</summary>
    public static void BackwardMaxima(
        double[] inputTimes, int[] inputSources,
        double[] maximaTimes, double[] maximaValues, double[] maximaErrors,
        double[] w, int nPre, int nPost,
        double tauMem, double tauSyn,
        double[] gradient)
    {
        eventprop_backward_maxima(inputTimes, inputSources, inputTimes.Length,
            maximaTimes, maximaValues, maximaErrors, maximaTimes.Length,
            w, nPre, nPost, tauMem, tauSyn, gradient);
    }
}
