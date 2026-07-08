using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>P/Invoke bindings for eventprop.h — core single-sample operations.</summary>
public static unsafe partial class Api
{
    private const string Lib = "eventprop_c_api";

    // ── Error codes (match C enum exactly) ────────────────────────

    public enum EventPropError : int
    {
        Ok                       = 0,
        InvalidDimension         = 1,
        NegativeSpikes           = 2,
        NullPointer              = 3,
        InvalidSource            = 4,
        InvalidTau               = 5,
        AllocationFailed         = 6,
        InvalidCurrent           = 7,
        NegativeSpikeTime        = 8,
        ExcessiveSpikeTime       = 9,
        TauTooSmall              = 10,
        TauTooLarge              = 11,
        InvalidVTh               = 12,
        DimensionTooLarge        = 13,
        TimesTauMismatch         = 14,
        TauTooClose              = 15,
    }

    // ── Native structs (match C layout exactly) ───────────────────

    /// <summary>Non-owning view into weight matrix + spike input.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct EventPropInput
    {
        public IntPtr W;           // const double*
        public int NPre;
        public int NPost;
        public IntPtr Times;       // const double*
        public IntPtr Sources;     // const int*
        public int NSpikes;
        public double TauMem;
        public double TauSyn;
        public double VTh;

        /// <summary>Build from raw pointers.</summary>
        public static EventPropInput Create(double* w, int nPre, int nPost,
            double* times, int* sources, int nSpikes,
            double vTh, double tauMem, double tauSyn)
        {
            return new EventPropInput
            {
                W = (IntPtr)w,
                NPre = nPre,
                NPost = nPost,
                Times = (IntPtr)times,
                Sources = (IntPtr)sources,
                NSpikes = nSpikes,
                TauMem = tauMem,
                TauSyn = tauSyn,
                VTh = vTh,
            };
        }

        /// <summary>Build from managed arrays (pins via fixed).</summary>
        public static EventPropInput Create(ReadOnlySpan<double> w, int nPre, int nPost,
            ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
            double vTh, double tauMem, double tauSyn)
        {
            fixed (double* pw = w)
            fixed (double* pt = times)
            fixed (int* ps = sources)
                return Create(pw, nPre, nPost, pt, ps, times.Length, vTh, tauMem, tauSyn);
        }
    }

    /// <summary>Owned spike output — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct SpikeOutput : IDisposable
    {
        internal readonly EventPropError errorNo;
        internal readonly int nSpikes;
        internal readonly IntPtr times;       // double* (malloc'd)
        internal readonly IntPtr sources;     // int* (malloc'd)
        internal readonly IntPtr currents;    // double* (malloc'd)
        internal readonly IntPtr errors;      // double* (malloc'd, written by caller before backward)

        public EventPropError ErrorNo => errorNo;
        public int NSpikes => nSpikes;

        /// <summary>Zero-copy span over spike times.</summary>
        public Span<double> Times => NS<double>(times, nSpikes);
        /// <summary>Zero-copy span over spike sources.</summary>
        public Span<int> Sources => NS<int>(sources, nSpikes);
        /// <summary>Zero-copy span over synaptic currents at spike times.</summary>
        public Span<double> Currents => NS<double>(currents, nSpikes);
        /// <summary>Zero-copy mutable span for writing error signals before backward pass.</summary>
        public Span<double> Errors => NS<double>(errors, nSpikes);

        public void Dispose() => eventprop_free_spike_output(this);
    }

    /// <summary>Owned sums output — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct SumsOutput : IDisposable
    {
        internal readonly EventPropError errorNo;
        internal readonly int nRows;
        internal readonly int nPost;
        internal readonly IntPtr sum0;        // double* (malloc'd, nRows * nPost)
        internal readonly IntPtr sum1;        // double* (malloc'd, nRows * nPost);

        public EventPropError ErrorNo => errorNo;
        /// <summary>Zero-copy span over sum0.</summary>
        public Span<double> Sum0 => NS<double>(sum0, nRows * nPost);
        /// <summary>Zero-copy span over sum1.</summary>
        public Span<double> Sum1 => NS<double>(sum1, nRows * nPost);

        public void Dispose() => eventprop_free_sums_output(this);
    }

    /// <summary>Owned maxima output — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct MaximaOutput : IDisposable
    {
        internal readonly EventPropError errorNo;
        internal readonly int nPost;
        internal readonly IntPtr times;       // double* (malloc'd, nPost)
        internal readonly IntPtr values;      // double* (malloc'd, nPost)
        internal readonly IntPtr errors;      // double* (malloc'd, nPost, written by caller before backward);

        public EventPropError ErrorNo => errorNo;
        /// <summary>Zero-copy span over maxima times.</summary>
        public Span<double> Times => NS<double>(times, nPost);
        /// <summary>Zero-copy span over maxima values.</summary>
        public Span<double> Values => NS<double>(values, nPost);

        public void Dispose() => eventprop_free_maxima_output(this);
    }

    /// <summary>Owned backward output — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct BackwardOutput : IDisposable
    {
        internal readonly EventPropError errorNo;
        internal readonly int nPre;
        internal readonly int nPost;
        internal readonly int nSpikes;
        internal readonly IntPtr grad;        // double* (malloc'd, nPre * nPost)
        internal readonly IntPtr spikeErrors; // double* (malloc'd, nSpikes)

        public EventPropError ErrorNo => errorNo;
        /// <summary>Zero-copy span over weight gradient.</summary>
        public Span<double> Gradient => NS<double>(grad, nPre * nPost);
        /// <summary>Zero-copy span over input spike errors.</summary>
        public Span<double> SpikeErrors => NS<double>(spikeErrors, nSpikes);

        public void Dispose() => eventprop_free_backward_output(this);
    }

    // ── Native function declarations ───────────────────────────────

    [LibraryImport(Lib)]
    public static partial SpikeOutput eventprop_compute_spikes(EventPropInput input);

    [LibraryImport(Lib)]
    public static partial SumsOutput eventprop_compute_sums(EventPropInput input);

    [LibraryImport(Lib)]
    public static partial MaximaOutput eventprop_compute_maxima(EventPropInput input);

    [LibraryImport(Lib)]
    public static partial BackwardOutput eventprop_backward_spikes(EventPropInput input, SpikeOutput spike);

    [LibraryImport(Lib)]
    public static partial BackwardOutput eventprop_backward_maxima(EventPropInput input, IntPtr maximaTimes, IntPtr maximaErrors);

    // ── Free functions (public for manual lifetime management) ─────

    [LibraryImport(Lib)]
    public static partial void eventprop_free_spike_output(SpikeOutput output);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_sums_output(SumsOutput output);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_maxima_output(MaximaOutput output);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_backward_output(BackwardOutput output);

    [LibraryImport(Lib)]
    public static partial EventPropError eventprop_validate_input(EventPropInput input);

    // ── Helpers ───────────────────────────────────────────────────

    private static Span<T> NS<T>(IntPtr ptr, int len) where T : unmanaged
    {
        if (ptr == IntPtr.Zero || len <= 0) return Span<T>.Empty;
        return new Span<T>((T*)ptr.ToPointer(), len);
    }
}
