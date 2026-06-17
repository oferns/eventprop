using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>P/Invoke bindings for eventprop.h — core single-sample operations.</summary>
public static unsafe partial class Api
{
    private const string Lib = "eventprop_c_api";

    // ── Native structs (match C layout exactly) ────────────────

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
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SpikeOutput
    {
        public int NSpikes;
        public IntPtr Times;       // double* (malloc'd)
        public IntPtr Sources;     // int* (malloc'd)
        public IntPtr Currents;    // double* (malloc'd)
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SumsOutput
    {
        public int NRows;          // n_spikes + 1
        public int NPost;
        public IntPtr Sum0;        // double* (malloc'd, n_rows * n_post)
        public IntPtr Sum1;        // double* (malloc'd, n_rows * n_post)
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MaximaOutput
    {
        public int NPost;
        public IntPtr Times;       // double* (malloc'd, n_post)
        public IntPtr Values;      // double* (malloc'd, n_post)
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BackwardSpikesInput
    {
        public IntPtr Input;       // const EventPropInput*
        public IntPtr FwdTimes;    // const double*
        public IntPtr FwdSources;  // const int*
        public IntPtr FwdCurrents;// const double*
        public IntPtr PostErrors;  // const double*
        public int NPostSpikes;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BackwardMaximaInput
    {
        public IntPtr Input;       // const EventPropInput*
        public IntPtr MaximaTimes;// const double*
        public IntPtr MaximaValues;// const double*
        public IntPtr MaximaErrors;// const double*
        public int NMaxima;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct BackwardOutput
    {
        public int NPre;
        public int NPost;
        public IntPtr Grad;        // double* (malloc'd, n_pre * n_post)
        public IntPtr SpikeErrors;// double* (malloc'd, n_spikes)
    }

    // ── Native function declarations ───────────────────────────

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_compute_spikes(IntPtr input);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_compute_sums(IntPtr input);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_compute_maxima(IntPtr input);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_backward_spikes(IntPtr input);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_backward_maxima(IntPtr input);

    [LibraryImport(Lib)]
    private static partial void eventprop_free_spike_output(IntPtr output);

    [LibraryImport(Lib)]
    private static partial void eventprop_free_sums_output(IntPtr output);

    [LibraryImport(Lib)]
    private static partial void eventprop_free_maxima_output(IntPtr output);

    [LibraryImport(Lib)]
    private static partial void eventprop_free_backward_output(IntPtr output);

  // ── Managed wrappers (single-sample) ────────────────────────

    /// <summary>Forward pass — compute post-synaptic spikes.</summary>
    public static OwnedSpikeOutput ComputeSpikes(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            EventPropInput* input = stackalloc EventPropInput[1];
            input->W = (IntPtr)pw;
            input->NPre = nPre;
            input->NPost = nPost;
            input->Times = (IntPtr)pt;
            input->Sources = (IntPtr)ps;
            input->NSpikes = times.Length;
            input->TauMem = tauMem;
            input->TauSyn = tauSyn;
            input->VTh = vTh;

            return new OwnedSpikeOutput(eventprop_compute_spikes((IntPtr)input));
        }
    }

    /// <summary>Compute cumulative exponential-weighted sums.</summary>
    public static OwnedSumsOutput ComputeSums(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            EventPropInput* input = stackalloc EventPropInput[1];
            input->W = (IntPtr)pw;
            input->NPre = nPre;
            input->NPost = nPost;
            input->Times = (IntPtr)pt;
            input->Sources = (IntPtr)ps;
            input->NSpikes = times.Length;
            input->TauMem = tauMem;
            input->TauSyn = tauSyn;
            input->VTh = 0;

            return new OwnedSumsOutput(eventprop_compute_sums((IntPtr)input));
        }
    }

    /// <summary>Compute membrane potential maxima per neuron.</summary>
    public static OwnedMaximaOutput ComputeMaxima(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            EventPropInput* input = stackalloc EventPropInput[1];
            input->W = (IntPtr)pw;
            input->NPre = nPre;
            input->NPost = nPost;
            input->Times = (IntPtr)pt;
            input->Sources = (IntPtr)ps;
            input->NSpikes = times.Length;
            input->TauMem = tauMem;
            input->TauSyn = tauSyn;
            input->VTh = 0;

            return new OwnedMaximaOutput(eventprop_compute_maxima((IntPtr)input));
        }
    }

    /// <summary>Backward pass — compute weight gradients from spike errors.</summary>
    public static OwnedBackwardOutput BackwardSpikes(
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<double> fwdTimes, ReadOnlySpan<int> fwdSources,
        ReadOnlySpan<double> fwdCurrents,
        ReadOnlySpan<double> postErrors, int nPostSpikes,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (double* pft = fwdTimes)
        fixed (int* pfs = fwdSources)
        fixed (double* pfc = fwdCurrents)
        fixed (double* pe = postErrors)
        {
            EventPropInput* epInput = stackalloc EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = vTh;

            BackwardSpikesInput* input = stackalloc BackwardSpikesInput[1];
            input->Input = (IntPtr)epInput;
            input->FwdTimes = (IntPtr)pft;
            input->FwdSources = (IntPtr)pfs;
            input->FwdCurrents = (IntPtr)pfc;
            input->PostErrors = (IntPtr)pe;
            input->NPostSpikes = nPostSpikes;

            return new OwnedBackwardOutput(
                eventprop_backward_spikes((IntPtr)input), times.Length);
        }
    }

    /// <summary>Backward pass — compute weight gradients from maxima errors.</summary>
    public static OwnedBackwardOutput BackwardMaxima(
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<double> maximaTimes, ReadOnlySpan<double> maximaValues,
        ReadOnlySpan<double> maximaErrors, int nMaxima,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (double* mt = maximaTimes)
        fixed (double* mv = maximaValues)
        fixed (double* me = maximaErrors)
        {
            EventPropInput* epInput = stackalloc EventPropInput[1];
            epInput->W = (IntPtr)pw;
            epInput->NPre = nPre;
            epInput->NPost = nPost;
            epInput->Times = (IntPtr)pt;
            epInput->Sources = (IntPtr)ps;
            epInput->NSpikes = times.Length;
            epInput->TauMem = tauMem;
            epInput->TauSyn = tauSyn;
            epInput->VTh = 0;

            BackwardMaximaInput* input = stackalloc BackwardMaximaInput[1];
            input->Input = (IntPtr)epInput;
            input->MaximaTimes = (IntPtr)mt;
            input->MaximaValues = (IntPtr)mv;
            input->MaximaErrors = (IntPtr)me;
            input->NMaxima = nMaxima;

            return new OwnedBackwardOutput(
                eventprop_backward_maxima((IntPtr)input), times.Length);
        }
    }

  // ── IDisposable wrappers ────────────────────────────────────

    /// <summary>Owned spike output — Dispose() frees native memory.</summary>
    public readonly struct OwnedSpikeOutput : IDisposable
    {
        private readonly IntPtr _ptr;

        public int NSpikes => Unsafe.Read<int>(_ptr.ToPointer());
        public Span<double> Times
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 8);
                return NS<double>(ptr, NSpikes);
            }
        }
        public Span<int> Sources
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 16);
                return NS<int>(ptr, NSpikes);
            }
        }
        public Span<double> Currents
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 24);
                return NS<double>(ptr, NSpikes);
            }
        }

        internal OwnedSpikeOutput(IntPtr ptr) => _ptr = ptr;
        public void Dispose() => eventprop_free_spike_output(_ptr);
    }

    /// <summary>Owned sums output — Dispose() frees native memory.</summary>
    public readonly struct OwnedSumsOutput : IDisposable
    {
        private readonly IntPtr _ptr;

        public int NRows => Unsafe.Read<int>(_ptr.ToPointer());
        public int NPost => Unsafe.Read<int>((void*)((byte*)_ptr.ToPointer() + 4));
        public Span<double> Sum0
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 8);
                return NS<double>(ptr, NRows * NPost);
            }
        }
        public Span<double> Sum1
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 16);
                return NS<double>(ptr, NRows * NPost);
            }
        }

        internal OwnedSumsOutput(IntPtr ptr) => _ptr = ptr;
        public void Dispose() => eventprop_free_sums_output(_ptr);
    }

    /// <summary>Owned maxima output — Dispose() frees native memory.</summary>
    public readonly struct OwnedMaximaOutput : IDisposable
    {
        private readonly IntPtr _ptr;

        public int NPost => Unsafe.Read<int>(_ptr.ToPointer());
        public Span<double> Times
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 8);
                return NS<double>(ptr, NPost);
            }
        }
        public Span<double> Values
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 16);
                return NS<double>(ptr, NPost);
            }
        }

        internal OwnedMaximaOutput(IntPtr ptr) => _ptr = ptr;
        public void Dispose() => eventprop_free_maxima_output(_ptr);
    }

    /// <summary>Owned backward output — Dispose() frees native memory.</summary>
    public readonly struct OwnedBackwardOutput : IDisposable
    {
        private readonly IntPtr _ptr;
        private readonly int _nSpikes;

        public int NPre => Unsafe.Read<int>(_ptr.ToPointer());
        public int NPost => Unsafe.Read<int>((void*)((byte*)_ptr.ToPointer() + 4));
        public Span<double> Gradient
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 8);
                return NS<double>(ptr, NPre * NPost);
            }
        }
        public Span<double> SpikeErrors
        {
            get
            {
                IntPtr ptr = *(IntPtr*)((byte*)_ptr.ToPointer() + 16);
                return NS<double>(ptr, _nSpikes);
            }
        }

        internal OwnedBackwardOutput(IntPtr ptr, int nSpikes)
        {
            _ptr = ptr;
            _nSpikes = nSpikes;
        }
        public void Dispose() => eventprop_free_backward_output(_ptr);
    }

  // ── Helpers ─────────────────────────────────────────────────

    private static Span<T> NS<T>(IntPtr ptr, int len) where T : unmanaged
    {
        if (ptr == IntPtr.Zero || len == 0) return Span<T>.Empty;
        return new Span<T>((T*)ptr.ToPointer(), len);
    }
}
