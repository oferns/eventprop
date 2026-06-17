using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>
/// Leaky Integrate-and-Fire (LIF) layer backed by the C library via P/Invoke.
/// 
/// Zero-allocation hot path: Forward uses _ptrs batch API (no flattening),
/// errors are written to native memory, Backward passes native pointers directly.
/// 
/// Usage:
///   layer.Forward(inputBatch);
///   for (int i = 0; i < layer.PostSpikes[i].NSpikes; i++)
///       layer.SetError(i, spikeIdx, 1.0);  // writes to native buffer
///   layer.Backward();
/// </summary>
public sealed unsafe class LIFLayer : Layer
{
    private readonly LIFLayerParameters _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    private double[] _weights;

    /// <summary>Raw native pointer to forward batch output (for backward pass).</summary>
    private IntPtr _fwdBatchPtr = IntPtr.Zero;

    /// <summary>Zero-copy view over native forward output.</summary>
    private NativeSpikesVector? _outputNative;

    /// <summary>Native errors buffer (malloc'd, one per sample via pointer array).</summary>
    private IntPtr[] _errorBuffers = Array.Empty<IntPtr>();

    /// <summary>Spike count per sample (for error buffer offsets).</summary>
    private int[] _postSpikesPerSample = Array.Empty<int>();

    /// <summary>
    /// Create a LIF layer with the given parameters.
    /// Weights are initialized using the initializer from parameters (default: Gaussian(0, 1)).
    /// </summary>
    public LIFLayer(LIFLayerParameters parameters)
        : this(parameters, null) { }

    /// <summary>
    /// Create a LIF layer with pre-initialized weights.
    /// </summary>
    public LIFLayer(LIFLayerParameters parameters, double[]? weights)
    {
        _params = parameters;

        if (weights is not null)
        {
            var expectedLen = parameters.NIn * parameters.N;
            if (weights.Length != expectedLen)
                throw new ArgumentException(
                    $"Weight array length ({weights.Length}) must be {expectedLen} (nIn={parameters.NIn} × nOut={parameters.N})",
                    nameof(weights));
            _weights = weights;
        }
        else
        {
            var init = parameters.Weights ?? new GaussianInitializer();
            _weights = init.GetWeights(parameters.NIn, parameters.N);
        }

        Gradient = new double[_weights.Length];
    }

    public LIFLayerParameters Parameters => _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    public double[] Weights => _weights;

    /// <summary>Number of input neurons.</summary>
    public override int NIn => _params.NIn;

    /// <summary>Number of output neurons.</summary>
    public override int NOut => _params.N;

    /// <summary>Get the weight matrix as a Span (row-major, nIn × nOut).</summary>
    public override ReadOnlySpan<double> GetWeights() => _weights.AsSpan();

    /// <summary>Set the weight matrix from a Span (row-major, nIn × nOut).</summary>
    public override void SetWeights(ReadOnlySpan<double> weights)
    {
        if (weights.Length != _weights.Length)
            throw new ArgumentException($"Weight span length ({weights.Length}) must match (_weights.Length={_weights.Length})", nameof(weights));
        weights.CopyTo(_weights);
    }

    /// <summary>Number of samples in the last forward pass.</summary>
    public int BatchSize => _outputNative?.Count ?? 0;

    /// <summary>
    /// Get zero-copy view of post-synaptic spikes for a sample.
    /// Use this to read spike times/sources and determine error count.
    /// </summary>
    public NativeSpikesSample this[int batchIdx]
    {
        get
        {
            if (_outputNative is null)
                throw new InvalidOperationException("Run Forward() first.");
            return _outputNative[batchIdx];
        }
    }

    /// <summary>
    /// Set error signal for a specific spike in a sample.
    /// Writes directly to native memory — zero allocation.
    /// </summary>
    public void SetError(int batchIdx, int spikeIdx, double error)
    {
        if (_outputNative is null)
            throw new InvalidOperationException("Run Forward() first.");
        if (batchIdx < 0 || batchIdx >= _outputNative.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var errors = GetErrorSpan(batchIdx);
        if (spikeIdx < 0 || spikeIdx >= errors.Length)
            throw new ArgumentOutOfRangeException(nameof(spikeIdx));
        errors[spikeIdx] = error;
    }

    /// <summary>
    /// Get mutable span over error buffer for a sample.
    /// Write errors here before calling Backward().
    /// </summary>
    public Span<double> GetErrorSpan(int batchIdx)
    {
        if (_outputNative is null)
            throw new InvalidOperationException("Run Forward() first.");
        if (batchIdx < 0 || batchIdx >= _outputNative.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        int nSpikes = _postSpikesPerSample[batchIdx];
        return new Span<double>((double*)_errorBuffers[batchIdx].ToPointer(), nSpikes);
    }

    /// <summary>
    /// Run forward pass: compute post-synaptic spike times given input spikes.
    /// Zero-allocation: _ptrs batch API, no flattening, no managed output copy.
    /// </summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;

        // Free previous forward output and error buffers
        if (_fwdBatchPtr != IntPtr.Zero)
        {
            Batch.eventprop_free_batch_spike_output(_fwdBatchPtr, _outputNative?.Count ?? 1);
            _fwdBatchPtr = IntPtr.Zero;
        }
        FreeErrorBuffers();

        int batchSize = inputBatch.Count;
        var timesPtrs = new IntPtr[batchSize];
        var sourcesPtrs = new IntPtr[batchSize];
        var nSpikes = new int[batchSize];

        // Pin all sample arrays for the duration of the C call
        var handles = new GCHandle[batchSize * 2];
        try
        {
            for (int i = 0; i < batchSize; i++)
            {
                var sample = inputBatch[i];
                handles[i * 2] = GCHandle.Alloc(sample.Times, GCHandleType.Pinned);
                handles[i * 2 + 1] = GCHandle.Alloc(sample.Sources, GCHandleType.Pinned);
                timesPtrs[i] = handles[i * 2].AddrOfPinnedObject();
                sourcesPtrs[i] = handles[i * 2 + 1].AddrOfPinnedObject();
                nSpikes[i] = sample.NSpikes;
            }

            var nativeOutput = Batch.ComputeSpikesBatchPtrs(
                _weights, _params.NIn, _params.N,
                timesPtrs, sourcesPtrs, nSpikes,
                _params.VTh, _params.TauMem, _params.TauSyn);

            _fwdBatchPtr = nativeOutput.GetHandle();
        }
        finally
        {
            for (int i = 0; i < handles.Length; i++)
                if (handles[i].IsAllocated) handles[i].Free();
        }

        // Zero-copy output view over native memory
        _outputNative = new NativeSpikesVector(_fwdBatchPtr, batchSize);

        // Allocate native error buffers (one per sample)
        _errorBuffers = new IntPtr[batchSize];
        _postSpikesPerSample = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            int nPostSpikes = _outputNative[i].NSpikes;
            _postSpikesPerSample[i] = nPostSpikes;
            _errorBuffers[i] = nPostSpikes > 0
                ? Marshal.AllocHGlobal(nPostSpikes * sizeof(double))
                : IntPtr.Zero;
        }

        _output = null; // No managed output — use this[batchIdx] and SetError()
        _ranForward = true;
    }

    /// <summary>
    /// Run backward pass: compute weight gradients via the adjoint method.
    /// Zero-allocation: _ptrs batch API for input, native error pointers for output.
    /// </summary>
    public override void Backward()
    {
        if (!_ranForward)
            throw new InvalidOperationException("Run Forward() before Backward().");

        // Zero gradient before accumulation
        Gradient.AsSpan().Clear();

        if (_inputBatch is null || _outputNative is null)
            throw new InvalidOperationException("Forward has not been run.");

        int batchSize = _inputBatch.Count;
        var timesPtrs = new IntPtr[batchSize];
        var sourcesPtrs = new IntPtr[batchSize];
        var nSpikes = new int[batchSize];

        // Pin all input arrays for the duration of the C call
        var handles = new GCHandle[batchSize * 2];
        try
        {
            for (int i = 0; i < batchSize; i++)
            {
                var inputSample = _inputBatch[i];
                handles[i * 2] = GCHandle.Alloc(inputSample.Times, GCHandleType.Pinned);
                handles[i * 2 + 1] = GCHandle.Alloc(inputSample.Sources, GCHandleType.Pinned);
                timesPtrs[i] = handles[i * 2].AddrOfPinnedObject();
                sourcesPtrs[i] = handles[i * 2 + 1].AddrOfPinnedObject();
                nSpikes[i] = inputSample.NSpikes;
            }

            // Error pointers are native — no pinning needed
            var nErrors = new int[batchSize];
            for (int i = 0; i < batchSize; i++)
                nErrors[i] = _postSpikesPerSample[i];

            using var result = Batch.BackwardSpikesBatchAccumulatePtrs(
                _weights, _params.NIn, _params.N,
                timesPtrs, sourcesPtrs, nSpikes,
                _fwdBatchPtr,
                _errorBuffers, nErrors,
                _params.VTh, _params.TauMem, _params.TauSyn);

            result.Gradient.CopyTo(Gradient.AsSpan());
        }
        finally
        {
            for (int i = 0; i < handles.Length; i++)
                if (handles[i].IsAllocated) handles[i].Free();
        }

        _ranBackward = true;
    }

    /// <summary>Zero the weight gradient.</summary>
    public override void ZeroGrad() => Gradient.AsSpan().Clear();

    /// <summary>
    /// Compute the voltage trace for a single neuron during forward pass.
    /// </summary>
    public override (double[] times, double[] voltages) GetVoltageTrace(
        int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
    {
        if (!_ranForward || _inputBatch is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _inputBatch.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var inputSpikes = _inputBatch[batchIdx];
        var postSample = _outputNative![batchIdx];

        using var result = Diag.ComputeVoltageTrace(
            _weights, _params.NIn, _params.N,
            inputSpikes.Times, inputSpikes.Sources,
            postSample.Times.ToArray(),
            postSample.Sources.ToArray(),
            targetNrnIdx, tMax, dt,
            _params.VTh, _params.TauMem, _params.TauSyn);

        var times = new double[result.NT];
        var voltages = new double[result.NT];
        result.Ts.CopyTo(times);
        result.VTrace.CopyTo(voltages);

        return (times, voltages);
    }

    /// <summary>
    /// Compute λ_i at a specific time for a single neuron.
    /// </summary>
    public override double GetLambdaI(int batchIdx, int targetNrnIdx, double t)
    {
        if (!_ranForward || _outputNative is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _outputNative.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var postSample = _outputNative[batchIdx];

        double lambdaI = 0;
        double expFactor = 1.0 / _params.TauMem;
        for (int s = 0; s < postSample.NSpikes; s++)
        {
            if (postSample.Times[s] > t)
            {
                lambdaI += _params.VTh * Math.Exp((t - postSample.Times[s]) / _params.TauMem) * expFactor;
            }
        }
        return lambdaI;
    }

    /// <summary>
    /// Compute the λ_i trace over time for a single neuron.
    /// </summary>
    public override (double[] times, double[] lambdaI) GetLambdaITrace(
        int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
    {
        if (!_ranForward || _outputNative is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _outputNative.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var postSample = _outputNative[batchIdx];

        using var result = Diag.ComputeLambdaITrace(
            postSample.Times.ToArray(), targetNrnIdx,
            _params.VTh, _params.TauMem, _params.TauSyn,
            tMax, dt);

        var times = new double[result.NT];
        var lambdaTrace = new double[result.NT];
        result.Ts.CopyTo(times);
        result.LambdaTrace.CopyTo(lambdaTrace);

        return (times, lambdaTrace);
    }

    /// <summary>Dispose native resources.</summary>
    public void Dispose()
    {
        if (_fwdBatchPtr != IntPtr.Zero)
        {
            Batch.eventprop_free_batch_spike_output(_fwdBatchPtr, _outputNative?.Count ?? 1);
            _fwdBatchPtr = IntPtr.Zero;
        }
        FreeErrorBuffers();
    }

    private void FreeErrorBuffers()
    {
        foreach (var ptr in _errorBuffers)
            if (ptr != IntPtr.Zero)
                Marshal.FreeHGlobal(ptr);
        _errorBuffers = Array.Empty<IntPtr>();
        _postSpikesPerSample = Array.Empty<int>();
    }
}
