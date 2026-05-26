using System.Runtime.InteropServices;

namespace CW.EventProp;

/// <summary>
/// P/Invoke wrappers for the native C++ EventProp library.
/// Provides ground-truth reference implementations for comparison testing.
/// The native library (libeventprop_c_api.dylib) must be in the same directory as this assembly.
/// </summary>
public static class CxxNative
{
    private const string LibName = "libeventprop_c_api";

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void eventprop_compute_sums(
        double[] w, int n_pre, int n_post,
        double[] times, int[] sources, int n_spikes,
        double tau_mem, double tau_syn,
        double[] sum0_out, double[] sum1_out);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern int eventprop_compute_spikes(
        double[] w, int n_pre, int n_post,
        double[] times, int[] sources, int n_spikes,
        double v_th, double tau_mem, double tau_syn,
        double[] out_times, int[] out_sources, double[] out_currents);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void eventprop_compute_maxima(
        double[] w, int n_pre, int n_post,
        double[] times, int[] sources, int n_spikes,
        double tau_mem, double tau_syn,
        double[] maxima_times, double[] maxima_values);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void eventprop_backward_spikes(
        double[] input_times, int[] input_sources, int n_input_spikes,
        double[] post_times, int[] post_sources, double[] post_currents,
        double[] post_errors, int n_post_spikes,
        double[] w, int n_pre, int n_post,
        double v_th, double tau_mem, double tau_syn,
        double[] gradient);

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern void eventprop_backward_maxima(
        double[] input_times, int[] input_sources, int n_input_spikes,
        double[] maxima_times, double[] maxima_values, double[] maxima_errors, int n_maxima,
        double[] w, int n_pre, int n_post,
        double tau_mem, double tau_syn,
        double[] gradient);

    // --- Public API wrappers (array-based for P/Invoke compatibility) ---

    /// <summary>Call compute_sums via native C++ library.</summary>
    public static void ComputeSums(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources,
        double tauMem, double tauSyn,
        double[] sum0_out, double[] sum1_out)
    {
        eventprop_compute_sums(w, nPre, nPost, times, sources, times.Length, tauMem, tauSyn, sum0_out, sum1_out);
    }

    /// <summary>Call compute_spikes via native C++ library. Returns number of output spikes.</summary>
    public static int ComputeSpikes(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources,
        double vTh, double tauMem, double tauSyn,
        double[] outTimes, int[] outSources, double[] outCurrents)
    {
        return eventprop_compute_spikes(w, nPre, nPost, times, sources, times.Length, vTh, tauMem, tauSyn, outTimes, outSources, outCurrents);
    }

    /// <summary>Call compute_maxima via native C++ library.</summary>
    public static void ComputeMaxima(
        double[] w, int nPre, int nPost,
        double[] times, int[] sources,
        double tauMem, double tauSyn,
        double[] maximaTimes, double[] maximaValues)
    {
        eventprop_compute_maxima(w, nPre, nPost, times, sources, times.Length, tauMem, tauSyn, maximaTimes, maximaValues);
    }

    /// <summary>Call backward (Spikes loss) via native C++ library.</summary>
    public static void BackwardSpikes(
        double[] inputTimes, int[] inputSources,
        double[] postTimes, int[] postSources, double[] postCurrents,
        double[] postErrors, double[] w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn,
        double[] gradient)
    {
        eventprop_backward_spikes(inputTimes, inputSources, inputTimes.Length,
            postTimes, postSources, postCurrents, postErrors, postTimes.Length,
            w, nPre, nPost, vTh, tauMem, tauSyn, gradient);
    }

    /// <summary>Call backward (Maxima loss) via native C++ library.</summary>
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
