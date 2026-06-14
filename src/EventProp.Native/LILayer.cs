namespace CW.EventProp;

/// <summary>
/// Leaky Integrate (LI) layer backed by the C library via P/Invoke.
/// 
/// Forward pass: computes membrane potential maxima per neuron using batch operations
/// (single C call for the entire batch via OpenMP parallelization).
/// 
/// Backward pass: computes weight gradients via the adjoint method,
/// accumulating across all samples in a single C call.
/// </summary>
public sealed class LILayer : Layer
{
    private readonly LILayerParameters _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    private double[] _weights;

    /// <summary>Raw native pointer to forward batch output (for backward pass).</summary>
    private IntPtr _fwdBatchPtr = IntPtr.Zero;

    /// <summary>Membrane potential maxima from the forward pass.</summary>
    private MaximaVector? _maximaBatch;

    /// <summary>
    /// Create an LI layer with the given parameters.
    /// Weights are initialized using the initializer from parameters (default: Gaussian(0, 1)).
    /// </summary>
    public LILayer(LILayerParameters parameters)
        : this(parameters, null) { }

    /// <summary>
    /// Create an LI layer with pre-initialized weights.
    /// </summary>
    public LILayer(LILayerParameters parameters, double[]? weights)
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

    public LILayerParameters Parameters => _params;

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

    /// <summary>Membrane potential maxima from the forward pass.</summary>
    public MaximaVector MaximaBatch => _maximaBatch ?? throw new InvalidOperationException("Forward has not been run.");

    /// <summary>
    /// Run forward pass: compute membrane potential maxima per neuron given input spikes.
    /// Uses batch C API — single call for entire batch.
    /// </summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;

        // Free previous forward output
        if (_fwdBatchPtr != IntPtr.Zero)
        {
            Batch.eventprop_free_batch_maxima_output(_fwdBatchPtr, inputBatch.Count);
            _fwdBatchPtr = IntPtr.Zero;
        }

        // Flatten batch into concatenated arrays + per-sample spike counts
        var (flatTimes, flatSources, nSpikesPerSample) = FlattenSpikes(inputBatch);

        // Batch forward pass — single C call
        using var nativeOutput = Batch.ComputeMaximaBatch(
            _weights, _params.NIn, _params.N,
            flatTimes, flatSources, nSpikesPerSample,
            _params.TauMem, _params.TauSyn);

        // Steal the native pointer for backward pass
        _fwdBatchPtr = nativeOutput.GetHandle();

        // Reconstruct managed MaximaData from native output
        var maximaBatch = new MaximaVector(inputBatch.Count);
        for (int i = 0; i < inputBatch.Count; i++)
        {
            var sample = nativeOutput[i];
            var times = new double[sample.NPost];
            var values = new double[sample.NPost];
            sample.Times.CopyTo(times);
            sample.Values.CopyTo(values);
            maximaBatch.Add(new MaximaData(times, values));
        }

        _maximaBatch = maximaBatch;
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

        if (_inputBatch is null || _maximaBatch is null)
            throw new InvalidOperationException("Forward has not been run.");

        // Flatten input batch
        var (flatTimes, flatSources, nSpikesPerSample) = FlattenSpikes(_inputBatch);

        // Flatten maxima errors
        var flatErrors = FlattenMaximaErrors(_maximaBatch);

        // Batch backward pass — single C call (accumulated gradient)
        using var result = Batch.BackwardMaximaBatchAccumulate(
            _weights, _params.NIn, _params.N,
            flatTimes, flatSources, nSpikesPerSample,
            _fwdBatchPtr,
            flatErrors,
            _params.TauMem, _params.TauSyn);

        // Copy gradient from native output
        result.Gradient.CopyTo(Gradient.AsSpan());

        _ranBackward = true;
    }

    /// <summary>Zero the weight gradient.</summary>
    public override void ZeroGrad() => Gradient.AsSpan().Clear();

    /// <summary>
    /// Compute the voltage trace for a single neuron during forward pass.
    /// Uses vTh = double.PositiveInfinity to simulate an LI neuron (no threshold).
    /// </summary>
    public override (double[] times, double[] voltages) GetVoltageTrace(
        int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
    {
        if (!_ranForward || _inputBatch is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _inputBatch.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var inputSpikes = _inputBatch[batchIdx];

        // LI layer has no spikes — pass empty forward output
        using var result = Diag.ComputeVoltageTrace(
            _weights, _params.NIn, _params.N,
            inputSpikes.Times, inputSpikes.Sources,
            Array.Empty<double>(),
            Array.Empty<int>(),
            targetNrnIdx, tMax, dt,
            double.PositiveInfinity, _params.TauMem, _params.TauSyn);

        var times = new double[result.NT];
        var voltages = new double[result.NT];
        result.Ts.CopyTo(times);
        result.VTrace.CopyTo(voltages);

        return (times, voltages);
    }

    /// <summary>Dispose native resources.</summary>
    public void Dispose()
    {
        if (_fwdBatchPtr != IntPtr.Zero)
        {
            int batchSize = _inputBatch?.Count ?? 1;
            Batch.eventprop_free_batch_maxima_output(_fwdBatchPtr, batchSize);
            _fwdBatchPtr = IntPtr.Zero;
        }
    }

    // ── Helpers ────────────────────────────────────────────────

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

    /// <summary>Flatten maxima errors from batch into a single array.</summary>
    private static double[] FlattenMaximaErrors(MaximaVector batch)
    {
        int total = 0;
        for (int i = 0; i < batch.Count; i++)
            total += batch[i].NNeurons;

        var errors = new double[total];
        int offset = 0;
        for (int i = 0; i < batch.Count; i++)
        {
            System.Array.Copy(batch[i].Errors, 0, errors, offset, batch[i].NNeurons);
            offset += batch[i].NNeurons;
        }

        return errors;
    }
}
