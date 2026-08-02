using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>P/Invoke bindings for eventprop_diag.h — diagnostic helpers.</summary>
public static unsafe partial class Diag
{
    private const string Lib = "eventprop_c_api";

    // ── Native structs (match C layout exactly) ───────────────────

    /// <summary>Voltage trace input — plain data, no ownership.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct VoltageTraceInput
    {
        public Api.EventPropInput Input;  // embedded EventPropInput (value)
        public int TargetNrnIdx;
        public IntPtr FwdTimes;           // const double*
        public IntPtr FwdSources;         // const int*
        public int NPostSpikes;
        public double TMax;
        public double Dt;

        /// <summary>Build from raw pointers.</summary>
        public static VoltageTraceInput Create(Api.EventPropInput input,
            int targetNrnIdx, double* fwdTimes, int* fwdSources, int nPostSpikes,
            double tMax, double dt)
        {
            return new VoltageTraceInput
            {
                Input = input,
                TargetNrnIdx = targetNrnIdx,
                FwdTimes = (IntPtr)fwdTimes,
                FwdSources = (IntPtr)fwdSources,
                NPostSpikes = nPostSpikes,
                TMax = tMax,
                Dt = dt,
            };
        }

        /// <summary>Build from managed arrays (pins via fixed).</summary>
        public static VoltageTraceInput Create(ReadOnlySpan<double> w, int nPre, int nPost,
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
                var epInput = Api.EventPropInput.Create(pw, nPre, nPost, pt, ps, times.Length, vTh, tauMem, tauSyn);

                return new VoltageTraceInput
                {
                    Input = epInput,
                    TargetNrnIdx = targetNrnIdx,
                    FwdTimes = (IntPtr)pft,
                    FwdSources = (IntPtr)pfs,
                    NPostSpikes = fwdTimes.Length,
                    TMax = tMax,
                    Dt = dt,
                };
            }
        }
    }

    /// <summary>Owned voltage trace output — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct VoltageTraceOutput : IDisposable
    {
        internal readonly int nT;
        internal readonly IntPtr vTrace;      // double* (malloc'd)
        internal readonly IntPtr ts;          // double* (malloc'd)
        internal readonly Api.EventPropError errorNo;

        public int NT => nT;
        public IntPtr VTrace => vTrace;
        public IntPtr Ts => ts;
        public Api.EventPropError ErrorNo => errorNo;

        public void Dispose() => eventprop_free_voltage_trace_output(this);
    }

    /// <summary>lambda-I trace input — plain data, no ownership.</summary>
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

        /// <summary>Build from raw pointers.</summary>
        public static LambdaITraceInput Create(double* fwdTimes, int nPostSpikes, int targetNrnIdx,
            double vTh, double tauMem, double tauSyn, double tMax, double dt)
        {
            return new LambdaITraceInput
            {
                FwdTimes = (IntPtr)fwdTimes,
                NPostSpikes = nPostSpikes,
                TargetNrnIdx = targetNrnIdx,
                VTh = vTh,
                TauMem = tauMem,
                TauSyn = tauSyn,
                TMax = tMax,
                Dt = dt,
            };
        }

        /// <summary>Build from managed arrays (pins via fixed).</summary>
        public static LambdaITraceInput Create(ReadOnlySpan<double> fwdTimes, int targetNrnIdx,
            double vTh, double tauMem, double tauSyn, double tMax, double dt)
        {
            fixed (double* pft = fwdTimes)
                return Create(pft, fwdTimes.Length, targetNrnIdx, vTh, tauMem, tauSyn, tMax, dt);
        }
    }

    /// <summary>Owned lambda-I trace output — freed via Dispose (<c>using</c>). Zero boxing: compiler calls Dispose directly.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly ref struct LambdaITraceOutput : IDisposable
    {
        internal readonly int nT;
        internal readonly IntPtr lambdaTrace;// double* (malloc'd)
        internal readonly IntPtr ts;          // double* (malloc'd)
        internal readonly Api.EventPropError errorNo;

        public int NT => nT;
        public IntPtr LambdaTrace => lambdaTrace;
        public IntPtr Ts => ts;
        public Api.EventPropError ErrorNo => errorNo;

        public void Dispose() => eventprop_free_lambda_i_trace_output(this);
    }

    // ── Native function declarations ───────────────────────────────

    [LibraryImport(Lib)]
    public static partial VoltageTraceOutput eventprop_compute_voltage_trace(VoltageTraceInput input);

    [LibraryImport(Lib)]
    public static partial LambdaITraceOutput eventprop_compute_lambda_i_trace(LambdaITraceInput input);

    // ── Free functions (public for manual lifetime management) ─────

    [LibraryImport(Lib)]
    public static partial void eventprop_free_voltage_trace_output(VoltageTraceOutput output);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_lambda_i_trace_output(LambdaITraceOutput output);
}
