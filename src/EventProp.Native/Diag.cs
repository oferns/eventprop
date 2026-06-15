using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>P/Invoke bindings for eventprop_diag.h — diagnostic helpers.</summary>
public static unsafe partial class Diag
{
    private const string Lib = "eventprop_c_api";

    [StructLayout(LayoutKind.Sequential)]
    public struct VoltageTraceInput
    {
        public IntPtr Input;       // const EventPropInput*
        public int TargetNrnIdx;
        public IntPtr FwdTimes;    // const double*
        public IntPtr FwdSources;  // const int*
        public int NPostSpikes;
        public double TMax;
        public double Dt;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct VoltageTraceOutput
    {
        public int NT;
        public IntPtr VTrace;      // double* (malloc'd)
        public IntPtr Ts;          // double* (malloc'd)
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LambdaITraceInput
    {
        public IntPtr FwdTimes;    // const double*
        public int NPostSpikes;
        public int TargetNrnIdx;
        public double VTh;
        public double TauMem;
        public double TauSyn;
        public double TMax;
        public double Dt;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LambdaITraceOutput
    {
        public int NT;
        public IntPtr LambdaTrace;// double* (malloc'd)
        public IntPtr Ts;          // double* (malloc'd)
    }

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_compute_voltage_trace(IntPtr input);

    [LibraryImport(Lib)]
    private static partial IntPtr eventprop_compute_lambda_i_trace(IntPtr input);

    [LibraryImport(Lib)]
    private static partial void eventprop_free_voltage_trace_output(IntPtr output);

    [LibraryImport(Lib)]
    private static partial void eventprop_free_lambda_i_trace_output(IntPtr output);

    /// <summary>Compute membrane potential trace for a single post-synaptic neuron.</summary>
    public static OwnedVoltageTraceOutput ComputeVoltageTrace(
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        ReadOnlySpan<double> fwdTimes, ReadOnlySpan<int> fwdSources,
        int targetNrnIdx, double tMax, double dt,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        fixed (double* pft = fwdTimes)
        fixed (int* pfs = fwdSources)
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

            VoltageTraceInput* input = stackalloc VoltageTraceInput[1];
            input->Input = (IntPtr)epInput;
            input->TargetNrnIdx = targetNrnIdx;
            input->FwdTimes = (IntPtr)pft;
            input->FwdSources = (IntPtr)pfs;
            input->NPostSpikes = fwdTimes.Length;
            input->TMax = tMax;
            input->Dt = dt;

            return new OwnedVoltageTraceOutput(eventprop_compute_voltage_trace((IntPtr)input));
        }
    }

    /// <summary>Compute synaptic adjoint (lambda_i) trace for a single neuron.</summary>
    public static OwnedLambdaITraceOutput ComputeLambdaITrace(
        ReadOnlySpan<double> fwdTimes, int targetNrnIdx,
        double vTh, double tauMem, double tauSyn,
        double tMax, double dt)
    {
        fixed (double* pft = fwdTimes)
        {
            LambdaITraceInput* input = stackalloc LambdaITraceInput[1];
            input->FwdTimes = (IntPtr)pft;
            input->NPostSpikes = fwdTimes.Length;
            input->TargetNrnIdx = targetNrnIdx;
            input->VTh = vTh;
            input->TauMem = tauMem;
            input->TauSyn = tauSyn;
            input->TMax = tMax;
            input->Dt = dt;

            return new OwnedLambdaITraceOutput(eventprop_compute_lambda_i_trace((IntPtr)input));
        }
    }

  // ── IDisposable wrappers ────────────────────────────────────

    public readonly ref struct OwnedVoltageTraceOutput : IDisposable
    {
        private readonly IntPtr _ptr;

        public int NT => Read().NT;
        public Span<double> VTrace
        {
            get { var o = Read(); return NS<double>(o.VTrace, o.NT); }
        }
        public Span<double> Ts
        {
            get { var o = Read(); return NS<double>(o.Ts, o.NT); }
        }

        internal OwnedVoltageTraceOutput(IntPtr ptr) => _ptr = ptr;
        public void Dispose() => eventprop_free_voltage_trace_output(_ptr);
        private VoltageTraceOutput Read() => Marshal.PtrToStructure<VoltageTraceOutput>(_ptr)!;
    }

    public readonly ref struct OwnedLambdaITraceOutput : IDisposable
    {
        private readonly IntPtr _ptr;

        public int NT => Read().NT;
        public Span<double> LambdaTrace
        {
            get { var o = Read(); return NS<double>(o.LambdaTrace, o.NT); }
        }
        public Span<double> Ts
        {
            get { var o = Read(); return NS<double>(o.Ts, o.NT); }
        }

        internal OwnedLambdaITraceOutput(IntPtr ptr) => _ptr = ptr;
        public void Dispose() => eventprop_free_lambda_i_trace_output(_ptr);
        private LambdaITraceOutput Read() => Marshal.PtrToStructure<LambdaITraceOutput>(_ptr)!;
    }

  // ── Helpers ─────────────────────────────────────────────────

    private static Span<T> NS<T>(IntPtr ptr, int len) where T : unmanaged
    {
        if (ptr == IntPtr.Zero || len == 0) return Span<T>.Empty;
        return new Span<T>((T*)ptr.ToPointer(), len);
    }
}
