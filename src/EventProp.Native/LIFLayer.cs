namespace CW.EventProp.Native;

/// <summary>
/// Leaky Integrate-and-Fire (LIF) layer backed by the C library via P/Invoke.
/// 
/// Forward pass: computes post-synaptic spike times using batch operations
/// (single C call for the entire batch via OpenMP parallelization).
/// 
/// Backward pass: computes weight gradients via the adjoint method,
/// accumulating across all samples in a single C call.
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

    /// <summary>
    /// Run forward pass: compute post-synaptic spike times given input spikes.
    /// Uses batch C API — single call for entire batch. Zero-copy: no flattening, no output copy.
    /// </summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;

        // Free previous forward output
        if (_fwdBatchPtr != IntPtr.Zero)
        {
            Batch.eventprop_free_batch_spike_output(_fwdBatchPtr, inputBatch.Count);
            _fwdBatchPtr = IntPtr.Zero;
        }

        int batchSize = inputBatch.Count;
        var timesPtrs = new IntPtr[batchSize];
        var sourcesPtrs = new IntPtr[batchSize];
        var nSpikes = new int[batchSize];

        // Pin all sample arrays for the duration of the C call
        var handles = new System.Runtime.InteropServices.GCHandle[batchSize * 2];
        try
        {
            for (int i = 0; i < batchSize; i++)
            {
                var sample = inputBatch[i];
                handles[i * 2] = System.Runtime.InteropServices.GCHandle.Alloc(sample.Times, System.Runtime.InteropServices.GCHandleType.Pinned);
                handles[i * 2 + 1] = System.Runtime.InteropServices.GCHandle.Alloc(sample.Sources, System.Runtime.InteropServices.GCHandleType.Pinned);
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

        // Managed output view for backward pass (errors set by user)
        _output = new SpikesVector(batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            var sample = _outputNative[i];
            var times = new double[sample.NSpikes];
            var sources = new int[sample.NSpikes];
            sample.Times.CopyTo(times);
            sample.Sources.CopyTo(sources);
            _output.Add(new SpikesData(times, sources));
        }

        _ranForward = true;
    }

    /// <summary>
    /// Run backward pass: compute weight gradients via the adjoint method.
    /// Uses batch C API — single call for entire batch (accumulated gradient).
    /// </summary>
    public override void Backward()
    {
        if (!_ranForward)
            throw new InvalidOperationException("Run Forward() before Backward().");

        // Zero gradient before accumulation
        Gradient.AsSpan().Clear();

        if (_inputBatch is null || _outputNative is null)
            throw new InvalidOperationException("Forward has not been run.");

        // Flatten input batch (temporary — _ptrs backward crashes at Large scale)
        var (flatTimes, flatSources, nSpikesPerSample) = FlattenSpikes(_inputBatch);

        // Copy errors from managed output into flat array
        int totalPostSpikes = 0;
        for (int i = 0; i < _outputNative.Count; i++)
            totalPostSpikes += _outputNative[i].NSpikes;
        var flatErrors = new double[totalPostSpikes];
        int offset = 0;
        for (int i = 0; i < _output!.Count; i++)
        {
            System.Array.Copy(_output[i].Errors, 0, flatErrors, offset, _output[i].NSpikes);
            offset += _output[i].NSpikes;
        }

        // Batch backward pass — flat array variant (stable)
        using var result = Batch.BackwardSpikesBatchAccumulate(
            _weights, _params.NIn, _params.N,
            flatTimes, flatSources, nSpikesPerSample,
            _fwdBatchPtr,
            flatErrors,
            _params.VTh, _params.TauMem, _params.TauSyn);

        result.Gradient.CopyTo(Gradient.AsSpan());
        _ranBackward = true;
    }

    /// <summary>Flatten a SpikesVector into concatenated arrays + per-sample counts.</summary>
    private static (double[] times, int[] sources, int[] nSpikesPerSample) FlattenSpikes(SpikesVector batch)
    {
        int totalSpikes = 0;
        var nSpikesPerSample = new int[batch.Count];
        for (int i = 0; i < batch.Count; i++)
        {
            nSpikesPerSample[i] = batch[i].NSpikes;
            totalSpikes += batch[i].NSpikes;
        }

        var times = new double[totalSpikes];
        var sources = new int[totalSpikes];
        int offset = 0;
        for (int i = 0; i < batch.Count; i++)
        {
            var sample = batch[i];
            System.Array.Copy(sample.Times, 0, times, offset, sample.NSpikes);
            System.Array.Copy(sample.Sources, 0, sources, offset, sample.NSpikes);
            offset += sample.NSpikes;
        }

        return (times, sources, nSpikesPerSample);
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

        // λ_i(t) = Σ_{spike times > t} vTh * exp((t - t_spike) / tauMem) / tauMem
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
            int batchSize = _inputBatch?.Count ?? 1;
            Batch.eventprop_free_batch_spike_output(_fwdBatchPtr, batchSize);
            _fwdBatchPtr = IntPtr.Zero;
        }
    }

    // ── Helpers ────────────────────────────────────────────────
}
