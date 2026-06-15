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
public sealed class LIFLayer : Layer
{
    private readonly LIFLayerParameters _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    private double[] _weights;

    /// <summary>Raw native pointer to forward batch output (for backward pass).</summary>
    private IntPtr _fwdBatchPtr = IntPtr.Zero;

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
    /// Uses batch C API — single call for entire batch.
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

        // Flatten batch into concatenated arrays + per-sample spike counts
        var (flatTimes, flatSources, nSpikesPerSample) = FlattenSpikes(inputBatch);

        // Batch forward pass — single C call
        using var nativeOutput = Batch.ComputeSpikesBatch(
            _weights, _params.NIn, _params.N,
            flatTimes, flatSources, nSpikesPerSample,
            _params.VTh, _params.TauMem, _params.TauSyn);

        // Keep the native pointer alive for backward pass
        // We need to "steal" it from the owned wrapper so we can free it later
        _fwdBatchPtr = GetNativePointer(nativeOutput);

        // Reconstruct managed SpikesData from native output
        var postBatch = new SpikesVector(inputBatch.Count);
        for (int i = 0; i < inputBatch.Count; i++)
        {
            var sample = GetSampleOutput(nativeOutput, i);
            postBatch.Add(new SpikesData(sample.times, sample.sources, sample.currents));
        }

        _output = postBatch;
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

        if (_inputBatch is null || _output is null)
            throw new InvalidOperationException("Forward has not been run.");

        // Flatten input batch
        var (flatTimes, flatSources, nSpikesPerSample) = FlattenSpikes(_inputBatch);

        // Flatten post-errors from output batch
        var flatErrors = FlattenPostErrors(_output);

        // Batch backward pass — single C call (accumulated gradient)
        using var result = Batch.BackwardSpikesBatchAccumulate(
            _weights, _params.NIn, _params.N,
            flatTimes, flatSources, nSpikesPerSample,
            _fwdBatchPtr,
            flatErrors,
            _params.VTh, _params.TauMem, _params.TauSyn);

        // Copy gradient from native output
        result.Gradient.CopyTo(Gradient.AsSpan());

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
        var postSpikes = _output?[batchIdx];

        using var result = Diag.ComputeVoltageTrace(
            _weights, _params.NIn, _params.N,
            inputSpikes.Times, inputSpikes.Sources,
            postSpikes?.Times ?? Array.Empty<double>(),
            postSpikes?.Sources ?? Array.Empty<int>(),
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
        if (!_ranForward || _output is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _output.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var postSpikes = _output[batchIdx];

        // λ_i(t) = Σ_{spike times > t} vTh * exp((t - t_spike) / tauMem) / tauMem
        double lambdaI = 0;
        double expFactor = 1.0 / _params.TauMem;
        for (int s = 0; s < postSpikes.NSpikes; s++)
        {
            if (postSpikes.Times[s] > t)
            {
                lambdaI += _params.VTh * Math.Exp((t - postSpikes.Times[s]) / _params.TauMem) * expFactor;
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
        if (!_ranForward || _output is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _output.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var postSpikes = _output[batchIdx];

        using var result = Diag.ComputeLambdaITrace(
            postSpikes.Times, targetNrnIdx,
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

    /// <summary>Flatten post-errors from output batch into a single array.</summary>
    private static double[] FlattenPostErrors(SpikesVector batch)
    {
        int totalSpikes = 0;
        for (int i = 0; i < batch.Count; i++)
            totalSpikes += batch[i].NSpikes;

        var errors = new double[totalSpikes];
        int offset = 0;
        for (int i = 0; i < batch.Count; i++)
        {
            System.Array.Copy(batch[i].Errors, 0, errors, offset, batch[i].NSpikes);
            offset += batch[i].NSpikes;
        }

        return errors;
    }

    /// <summary>Extract the native pointer from an OwnedBatchSpikeOutput (for stealing).</summary>
    private static IntPtr GetNativePointer(Batch.OwnedBatchSpikeOutput output)
    {
        // Use reflection-free approach: the OwnedBatchSpikeOutput wraps an IntPtr internally.
        // We access it through a helper method on Batch class.
        return output.GetHandle();
    }

    /// <summary>Extract per-sample data from batch native output.</summary>
    private static (double[] times, int[] sources, double[] currents) GetSampleOutput(
        Batch.OwnedBatchSpikeOutput output, int index)
    {
        var sample = output[index];
        var times = new double[sample.NSpikes];
        var sources = new int[sample.NSpikes];
        var currents = new double[sample.NSpikes];
        sample.Times.CopyTo(times);
        sample.Sources.CopyTo(sources);
        sample.Currents.CopyTo(currents);
        return (times, sources, currents);
    }
}
