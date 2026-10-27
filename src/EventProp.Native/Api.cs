using System.Runtime.InteropServices;
using System.Text;

namespace CW.EventProp.Native;

/// <summary>P/Invoke bindings for eventprop.h — core single-sample operations.</summary>
public static unsafe partial class Api
{
    private const string Lib = "eventprop_c_api";

    // ── Error codes (match C enum exactly) ────────────────────────

    public enum EventPropError : int
    {
        Ok                       = 0,
        InvalidDimension         = 1,   // n_pre <= 0 or n_post <= 0
        NegativeSpikes           = 2,   // n_spikes < 0
        NullPointer              = 3,   // w/times/sources is NULL when required
        InvalidSource            = 4,   // sources[i] out of [0, n_pre) range
        InvalidTau               = 5,   // tau_mem <= 0, tau_syn <= 0, or tau_mem == tau_syn
        InvalidCurrent           = 6,   // spike.currents[i] == v_th (division by zero)
        NegativeSpikeTime        = 7,   // times[i] < 0
        ExcessiveSpikeTime       = 8,   // max(times) > MAX_SPIKE_TIME
        TauTooSmall              = 9,   // tau_mem or tau_syn < MIN_TAU
        TauTooLarge              = 10,  // tau_mem or tau_syn > MAX_TAU
        InvalidVTh               = 11,  // v_th <= 0
        DimensionTooLarge        = 12,  // n_pre or n_post > MAX_DIM
        TimesTauMismatch         = 13,  // max(times)/min(tau) >= EXP_OVERFLOW_THRESHOLD
        TauTooClose              = 14,  // |tau_mem - tau_syn| / min(tau) < TAU_REL_DIFF_MIN
        BatchSizeZeroOrNeg       = 15,  // bad batch size
        AllocSum0                = 16,  // result.sum0
        AllocSum1                = 17,  // result.sum1
        AllocExpMem              = 18,  // exp_mem lookup table
        AllocExpSyn              = 19,  // exp_syn lookup table
        AllocTimesArr            = 20,  // post_times_arr
        AllocCurrentsArr         = 21,  // post_currents_arr
        AllocSpikeCaps           = 22,  // spike_caps
        AllocSpikeCounts         = 23,  // spike_counts
        AllocNeuronBuf           = 24,  // per-neuron times/currents buffer
        AllocEntries             = 25,  // entries (sorted collection)
        AllocResultTimes         = 26,  // result.times
        AllocResultSources       = 27,  // result.sources
        AllocResultCurrents      = 28,  // result.currents
        AllocResultErrors        = 29,  // result.errors
        AllocMaximaTimes         = 30,  // maxima result.times
        AllocMaximaValues        = 31,  // maxima result.values
        AllocMaximaErrors        = 32,  // maxima result.errors
        AllocBwdGrad             = 33,  // backward result.grad
        AllocBwdSpikeErrors      = 34,  // backward result.spike_errors
        AllocBwdMerged           = 35,  // backward merged spike array
        AllocBwdAllErrors        = 36,  // backward all_errors buffer
        AllocBwdLambdaV          = 37,  // backward lambda_v
        AllocBwdLambdaI          = 38,  // backward lambda_i
        AllocBwdJumpTime         = 39,  // backward jump_time array
        AllocBwdJumpMag          = 40,  // backward jump_mag array
        AllocBwdJumpCount        = 41,  // backward jump_count array
        AllocBwdJumpCap          = 42,  // backward jump_cap array
        AllocBwdPerNeuron        = 43,  // per-neuron jump_time/mag buffer
        AllocBatchOutputs        = 44,  // batch outputs array
        AllocBatchGrad           = 45,  // accumulate grad pointer
        AllocDropoutSample       = 46,  // dropout per-sample buffer
        AllocDiagVTrace          = 47,  // voltage trace v_trace array
        AllocDiagTs              = 48,  // trace ts (time) array
        AllocDiagLambda          = 49,  // lambda-I trace array
        BatchEmpty               = 50,  // batch input was null
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

        /// <summary>
        /// Build from managed arrays. The pins taken here are released when this method
        /// returns, so the returned struct's pointers are UNPINNED afterwards. Only safe
        /// when the native call happens inside the same `fixed` scope — which this
        /// overload cannot provide, because it returns the struct. Prefer the
        /// pointer-based overload with an explicit `fixed` block that spans the native
        /// call (see the convenience wrappers below).
        /// </summary>
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

        /// <summary>Construct from borrowed pointers (for wrapper methods). Caller must keep arrays alive during the call.</summary>
        internal SpikeOutput(EventPropError errorNo, int nSpikes, IntPtr times, IntPtr sources)
        {
            this.errorNo = errorNo;
            this.nSpikes = nSpikes;
            this.times = times;
            this.sources = sources;
        }

        public EventPropError ErrorNo => errorNo;
        public int NSpikes => nSpikes;

        /// <summary>Zero-copy span over spike times.</summary>
        public Span<double> Times => NS<double>(times, nSpikes);
        /// <summary>Zero-copy span over spike sources.</summary>
        public Span<int> Sources => NS<int>(sources, nSpikes);

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
    public static partial BackwardOutput eventprop_backward_spikes(EventPropInput input, SpikeOutput spike, IntPtr postSpikeErrors);

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

    /// <summary>Get human-readable error string for an error code.</summary>
    [LibraryImport(Lib, EntryPoint = "eventprop_error_string")]
    private static unsafe partial IntPtr NativeErrorString(int errorNo);

    // ── ABI self-check ────────────────────────────────────────────
    //
    // This assembly declares the C structs BY HAND — [StructLayout] here and in
    // Batch.cs — so a header change that moves a field offset is invisible at compile
    // time and invisible at the call site: the marshalling reads the offsets this
    // assembly compiled, against a library whose fields moved, and returns
    // plausible-looking garbage. The Cython layer measured exactly that failure
    // (docs/POSTMORTEMS.md: a .pyd built against July headers,
    // loaded against an October library, faulted 40 minutes later inside
    // spikes.py __getitem__). The P/Invoke layer has the same exposure, so it gets
    // the same guard.
    //
    // CompiledAbiVersion is the version these declarations were written against; keep
    // it in step with EVENTPROP_ABI_VERSION in eventprop.h. The struct-size check
    // needs no such discipline, which is why both halves exist.

    /// <summary>The ABI version these struct declarations were written against. Keep in step with EVENTPROP_ABI_VERSION in eventprop.h.</summary>
    public const int CompiledAbiVersion = 1;

    [LibraryImport(Lib)]
    private static partial int eventprop_abi_version();

    [LibraryImport(Lib)]
    private static partial nint eventprop_abi_struct_size(byte* structNameUtf8);

    /// <summary>The ABI version the loaded library was compiled with.</summary>
    public static int LibraryAbiVersion() => eventprop_abi_version();

    /// <summary>sizeof of a public struct as compiled inside the library. 0 for a name the library does not know.</summary>
    public static int LibraryStructSize(string name)
    {
        int byteCount = Encoding.UTF8.GetByteCount(name);
        Span<byte> buffer = stackalloc byte[byteCount + 1];   // +1 for the NUL
        Encoding.UTF8.GetBytes(name, buffer);
        buffer[byteCount] = 0;
        fixed (byte* p = buffer)
        {
            return (int)eventprop_abi_struct_size(p);
        }
    }

    /// <summary>sizeof of every C struct this assembly declares, as compiled here.</summary>
    /// <remarks>
    /// The key set is the public surface a consumer embeds by value, and it is
    /// deliberately duplicated in eventprop_abi.c, _native.pyx and the test suites:
    /// adding a struct to the headers must fail a test in every consumer, which is what
    /// forces this check to be extended. sizeof works on the Batch.cs ref structs
    /// because this is an unsafe context and they are unmanaged.
    /// </remarks>
    public static IReadOnlyDictionary<string, int> CompiledStructSizes() => new Dictionary<string, int>
    {
        ["EventPropInput"] = sizeof(EventPropInput),
        ["SpikeOutput"] = sizeof(SpikeOutput),
        ["SumsOutput"] = sizeof(SumsOutput),
        ["MaximaOutput"] = sizeof(MaximaOutput),
        ["BackwardOutput"] = sizeof(BackwardOutput),
        ["SpikesBatch"] = sizeof(Batch.SpikesBatch),
        ["MaximaBatch"] = sizeof(Batch.MaximaBatch),
        ["BackwardsBatch"] = sizeof(Batch.BackwardsBatch),
        ["GradientOutput"] = sizeof(Batch.GradientOutput),
        ["AccumulatedBackwardResult"] = sizeof(Batch.AccumulatedBackwardResult),
        ["DropoutOutput"] = sizeof(Batch.DropoutOutput),
        ["DropoutBatch"] = sizeof(Batch.DropoutBatch),
    };

    /// <summary>Compare this assembly's compiled layouts against the loaded library. Never throws.</summary>
    public static (bool Ok, IReadOnlyList<string> Mismatches) AbiCheck()
    {
        var mismatches = new List<string>();
        int libraryVersion = eventprop_abi_version();
        if (libraryVersion != CompiledAbiVersion)
            mismatches.Add($"abi_version: assembly compiled={CompiledAbiVersion}, library={libraryVersion}");

        foreach (var (name, compiledSize) in CompiledStructSizes())
        {
            int librarySize = LibraryStructSize(name);
            if (librarySize != compiledSize)
                mismatches.Add($"{name}: assembly compiled sizeof={compiledSize}, library sizeof={librarySize}");
        }
        return (mismatches.Count == 0, mismatches);
    }

    private static bool _abiChecked;
    private static readonly object _abiLock = new();

    /// <summary>
    /// Run <see cref="AbiCheck"/> once per process and throw on a mismatch. Called from
    /// the layer constructors, so a mismatch surfaces at construction naming both
    /// values — not as a wrong number, and not as an access violation minutes later.
    /// </summary>
    public static void EnsureAbiMatches()
    {
        if (_abiChecked) return;
        lock (_abiLock)
        {
            if (_abiChecked) return;
            var (ok, mismatches) = AbiCheck();
            if (!ok)
            {
                throw new InvalidOperationException(
                    "EventProp C ABI mismatch: this assembly's P/Invoke struct declarations do not " +
                    "match the loaded eventprop_c_api library, so every marshalled value would be read " +
                    "from the wrong offsets. Rebuild EventProp.Native against the current headers, and " +
                    "check that the loaded library is a build output of this repo. Mismatches: " +
                    string.Join("; ", mismatches));
            }
            _abiChecked = true;
        }
    }

    public static string GetErrorString(EventPropError error)
    {
        var ptr = NativeErrorString((int)error);
        if (ptr == IntPtr.Zero) return $"Unknown error ({(int)error})";
        return Marshal.PtrToStringUTF8(ptr) ?? $"Unknown error ({(int)error})";
    }

    // ── Convenience wrappers (managed arrays → P/Invoke) ─────────

    /// <summary>Compute sums from managed arrays.</summary>
    public static SumsOutput ComputeSums(ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            var input = EventPropInput.Create(pw, nPre, nPost, pt, ps, times.Length, 0.0, tauMem, tauSyn);
            return eventprop_compute_sums(input);
        }
    }

    /// <summary>Compute spikes from managed arrays.</summary>
    public static SpikeOutput ComputeSpikes(ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            var input = EventPropInput.Create(pw, nPre, nPost, pt, ps, times.Length, vTh, tauMem, tauSyn);
            return eventprop_compute_spikes(input);
        }
    }

    /// <summary>Compute maxima from managed arrays.</summary>
    public static MaximaOutput ComputeMaxima(ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = times)
        fixed (int* ps = sources)
        {
            var input = EventPropInput.Create(pw, nPre, nPost, pt, ps, times.Length, 0.0, tauMem, tauSyn);
            return eventprop_compute_maxima(input);
        }
    }

    /// <summary>Backward pass (spike loss) from managed arrays.</summary>
    public static BackwardOutput BackwardSpikes(
        ReadOnlySpan<double> inputTimes, ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> postTimes, ReadOnlySpan<int> postSources,
        ReadOnlySpan<double> postErrors,
        int nPostSpikes,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double vTh, double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = inputTimes)
        fixed (int* ps = inputSources)
        {
            var input = EventPropInput.Create(pw, nPre, nPost, pt, ps, inputTimes.Length, vTh, tauMem, tauSyn);

            // Pin post-spike arrays and construct SpikeOutput with borrowed pointers
            fixed (double* ptPost = postTimes)
            fixed (int* psPost = postSources)
            fixed (double* pe = postErrors)
            {
                var spike = new SpikeOutput(EventPropError.Ok, nPostSpikes,
                    (IntPtr)ptPost, (IntPtr)psPost);
                return eventprop_backward_spikes(input, spike, (IntPtr)pe);
            }
        }
    }

    /// <summary>Backward pass (maxima loss) from managed arrays.</summary>
    public static BackwardOutput BackwardMaxima(
        ReadOnlySpan<double> inputTimes, ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> maximaTimes, ReadOnlySpan<double> maximaValues,
        ReadOnlySpan<double> maximaErrors, int nMaxima,
        ReadOnlySpan<double> w, int nPre, int nPost,
        double tauMem, double tauSyn)
    {
        fixed (double* pw = w)
        fixed (double* pt = inputTimes)
        fixed (int* ps = inputSources)
        {
            var input = EventPropInput.Create(pw, nPre, nPost, pt, ps, inputTimes.Length, 0.0, tauMem, tauSyn);

            fixed (double* mt = maximaTimes)
            fixed (double* me = maximaErrors)
            {
                return eventprop_backward_maxima(input, (IntPtr)mt, (IntPtr)me);
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────

    private static Span<T> NS<T>(IntPtr ptr, int len) where T : unmanaged
    {
        if (ptr == IntPtr.Zero || len <= 0) return Span<T>.Empty;
        return new Span<T>((T*)ptr.ToPointer(), len);
    }
}
