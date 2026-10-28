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

    /// <summary>lambda-I trace input — plain data, no ownership.
    /// Mirrors <see cref="VoltageTraceInput"/> plus the error signal the adjoint is driven by.
    /// fwdTimes/fwdSources must hold the spikes of EVERY post-synaptic neuron, not only the
    /// traced one: the adjoint decays lambda_v for all neurons at every spike.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct LambdaITraceInput
    {
        public Api.EventPropInput Input;
        public int TargetNrnIdx;
        public IntPtr FwdTimes;             // const double*
        public IntPtr FwdSources;           // const int*
        public IntPtr PostErrors;           // const double*
        public int NPostSpikes;
        public double TMax;
        public double Dt;

        /// <summary>Build from raw pointers.</summary>
        public static LambdaITraceInput Create(Api.EventPropInput input,
            int targetNrnIdx, double* fwdTimes, int* fwdSources, double* postErrors,
            int nPostSpikes, double tMax, double dt)
        {
            return new LambdaITraceInput
            {
                Input = input,
                TargetNrnIdx = targetNrnIdx,
                FwdTimes = (IntPtr)fwdTimes,
                FwdSources = (IntPtr)fwdSources,
                PostErrors = (IntPtr)postErrors,
                NPostSpikes = nPostSpikes,
                TMax = tMax,
                Dt = dt,
            };
        }

        /// <summary>Build from managed arrays (pins via fixed).</summary>
        /// <remarks>The pointers this hands back are UNPINNED by contract, as for the
        /// <c>EventPropInput</c> span overloads: the caller owns the pin and must keep the
        /// managed arrays fixed for the whole native call.</remarks>
        public static LambdaITraceInput Create(ReadOnlySpan<double> w, int nPre, int nPost,
            ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
            ReadOnlySpan<double> fwdTimes, ReadOnlySpan<int> fwdSources,
            ReadOnlySpan<double> postErrors,
            int targetNrnIdx, double tMax, double dt,
            double vTh, double tauMem, double tauSyn)
        {
            fixed (double* pw = w)
            fixed (double* pt = times)
            fixed (int* ps = sources)
            fixed (double* pft = fwdTimes)
            fixed (int* pfs = fwdSources)
            fixed (double* pe = postErrors)
            {
                var epInput = Api.EventPropInput.Create(pw, nPre, nPost, pt, ps, times.Length, vTh, tauMem, tauSyn);

                return Create(epInput, targetNrnIdx, pft, pfs, pe, fwdTimes.Length, tMax, dt);
            }
        }
    }

    /// <summary>Pointwise lambda-I output — holds no heap memory, so there is nothing to dispose.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct LambdaIAtTimeOutput
    {
        public double LambdaI;
        public Api.EventPropError ErrorNo;
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

    [LibraryImport(Lib)]
    public static partial LambdaIAtTimeOutput eventprop_compute_lambda_i_at_time(LambdaITraceInput input, double t);

    // ── Free functions (public for manual lifetime management) ─────

    [LibraryImport(Lib)]
    public static partial void eventprop_free_voltage_trace_output(VoltageTraceOutput output);

    [LibraryImport(Lib)]
    public static partial void eventprop_free_lambda_i_trace_output(LambdaITraceOutput output);
}
