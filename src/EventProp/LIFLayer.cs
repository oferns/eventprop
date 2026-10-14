namespace CW.EventProp;

/// <summary>
/// Leaky Integrate-and-Fire (LIF) layer with spike-time-based loss.
/// 
/// Forward pass: computes post-synaptic spike times using the EventProp algorithm
/// with α-function voltage model and Toms748 root-finding for threshold crossings.
/// 
/// Backward pass: computes weight gradients via the adjoint method, propagating
/// error signals from post-synaptic spike times back through the merged input/post
/// spike timeline.
/// 
/// Zero-allocation hot paths via pre-allocated ScratchBuffers.
/// </summary>
public sealed class LIFLayer : Layer, IDisposable
{
    private readonly LIFLayerParameters _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    private double[] _weights;

    /// <summary>Post-synaptic spike data from the forward pass.</summary>
    private SpikesVector? _postBatch;

    /// <summary>Reusable scratch buffers for zero-allocation computation.</summary>
    private readonly ScratchBuffers _buffers = new();

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
    /// Processes samples in parallel when batch size >= 2.
    /// </summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;

        // Ensure scratch buffers are sized for the largest sample (used by sequential path)
        int maxSpikes = 0;
        foreach (var spikes in inputBatch)
            maxSpikes = Math.Max(maxSpikes, spikes.NSpikes);

        var postBatch = new SpikesVector(inputBatch.Count);

        if (inputBatch.Count >= 2)
        {
            // Parallel path: each thread gets its own ScratchBuffers
            double[] wArr = _weights;
            var results = new SpikesData[inputBatch.Count];
            Parallel.For(0, inputBatch.Count, () =>
            {
                var buffers = new ScratchBuffers();
                buffers.EnsureCapacity(_params.N, _params.NIn, maxSpikes);
                return buffers;
            },
            (idx, _, buffers) =>
            {
                var inputSpikes = inputBatch[idx];
                buffers.Clear();

                var post = EventPropMath.ComputeSpikes(
                    wArr,
                    inputSpikes.Times,
                    inputSpikes.Sources,
                    _params.N,
                    _params.VTh,
                    _params.TauMem,
                    _params.TauSyn,
                    buffers);

                results[idx] = post;
                return buffers;
            },
            buffers => buffers.Dispose());

            postBatch = new SpikesVector(results);
        }
        else
        {
            // Sequential path: reuse single buffer
            _buffers.EnsureCapacity(_params.N, _params.NIn, maxSpikes);

            foreach (var inputSpikes in inputBatch)
            {
                _buffers.Clear();

                var post = EventPropMath.ComputeSpikes(
                    _weights.AsSpan(),
                    inputSpikes.Times,
                    inputSpikes.Sources,
                    _params.N,
                    _params.VTh,
                    _params.TauMem,
                    _params.TauSyn,
                    _buffers);

                postBatch.Add(post);
            }
        }

        _postBatch = postBatch;
        _output = postBatch;
        _ranForward = true;
    }

    /// <summary>
    /// Run backward pass: compute weight gradients via the adjoint method.
    /// Gradients are **accumulated** into <see cref="Layer.Gradient"/> (not zeroed) —
    /// matching the Python reference, whose training loop calls zero_grad() separately.
    /// Call <see cref="ZeroGrad"/> before Backward for a single-step gradient.
    /// Processes samples in parallel when batch size >= 2.
    /// </summary>
    public override void Backward()
    {
        if (!_ranForward)
            throw new InvalidOperationException("Run Forward() before Backward().");

        int batchCount = _inputBatch!.Count;
        int totalSpikes = 0;
        foreach (var s in _inputBatch) totalSpikes += s.NSpikes;
        if (totalSpikes == 0)
        {
            _ranBackward = true;
            return;
        }

        if (batchCount >= 2)
        {
            // Parallel path: each thread computes local gradient, then accumulate with lock
            double[] wArr = _weights;
            int gradSize = Gradient.Length;
            Parallel.For(0, batchCount, () =>
            {
                var buffers = new ScratchBuffers();
                var localGrad = new double[gradSize];
                return (buffers, localGrad);
            },
            (idx, _, local) =>
            {
                var buffers = local.buffers;
                int maxNeurons = Math.Max(_params.NIn, _params.N);
                buffers.EnsureCapacity(
                    nNeurons: maxNeurons,
                    nPre: _params.NIn,
                    nSpikes: _inputBatch[idx].NSpikes + _postBatch![idx].NSpikes);

                EventPropMath.BackwardSpikes(
                    wArr,
                    _inputBatch[idx].Times,
                    _inputBatch[idx].Sources,
                    _postBatch[idx],
                    _params.N,
                    _params.VTh,
                    _params.TauMem,
                    _params.TauSyn,
                    local.localGrad,
                    buffers);

                return local;
            },
            local =>
            {
                lock (Gradient)
                {
                    for (int i = 0; i < gradSize; i++)
                        Gradient[i] += local.localGrad[i];
                }
                local.buffers.Dispose();
            });
        }
        else
        {
            // Sequential path: accumulate directly into Gradient
            int maxNeurons = Math.Max(_params.NIn, _params.N);
            foreach (var inputSpikes in _inputBatch)
                foreach (var postSpikes in _postBatch!)
                {
                    _buffers.EnsureCapacity(
                        nNeurons: maxNeurons,
                        nPre: _params.NIn,
                        nSpikes: inputSpikes.NSpikes + postSpikes.NSpikes);

                    EventPropMath.BackwardSpikes(
                        _weights.AsSpan(),
                        inputSpikes.Times,
                        inputSpikes.Sources,
                        postSpikes,
                        _params.N,
                        _params.VTh,
                        _params.TauMem,
                        _params.TauSyn,
                        Gradient.AsSpan(),
                        _buffers);
                }
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
        if (!_ranForward || _postBatch is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _inputBatch!.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var inputSpikes = _inputBatch[batchIdx];
        return EventPropMath.ComputeVoltageTrace(
            tMax, dt, targetNrnIdx,
            _weights.AsSpan(), _params.NIn, _params.N,
            inputSpikes.Times, inputSpikes.Sources,
            _params.VTh, _params.TauMem, _params.TauSyn,
            _buffers);
    }

    /// <summary>
    /// Compute λ_i at a specific time for a single neuron.
    /// </summary>
    public override double GetLambdaI(int batchIdx, int targetNrnIdx, double t)
    {
        if (!_ranForward || _postBatch is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _postBatch.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var postSpikes = _postBatch[batchIdx];
        return EventPropMath.ComputeLambdaI(
            t, targetNrnIdx,
            postSpikes.Times, postSpikes.Sources,
            _params.VTh, _params.TauMem, _params.TauSyn);
    }

    /// <summary>
    /// Compute the λ_i trace over time for a single neuron.
    /// </summary>
    public override (double[] times, double[] lambdaI) GetLambdaITrace(
        int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
    {
        if (!_ranForward || _postBatch is null)
            throw new InvalidOperationException("Run Forward() first.");

        if (batchIdx < 0 || batchIdx >= _postBatch.Count)
            throw new ArgumentOutOfRangeException(nameof(batchIdx));

        var postSpikes = _postBatch[batchIdx];
        return EventPropMath.ComputeLambdaITrace(
            tMax, dt, targetNrnIdx,
            postSpikes.Times, postSpikes.Sources,
            _params.VTh, _params.TauMem, _params.TauSyn);
    }

    /// <summary>
    /// Zero-allocation training step: forward + backward for a single sample.
    /// Writes output spike times/sources directly into the provided buffers.
    /// Returns the number of output spikes written.
    /// </summary>
    public override int TrainingStep(
        ReadOnlySpan<double> inputTimes,
        ReadOnlySpan<int> inputSources,
        Span<double> outputTimes,
        Span<int> outputSources,
        Span<double> outputCurrents)
    {
        _buffers.EnsureCapacity(_params.N, _params.NIn, inputTimes.Length);
        _buffers.Clear();

        return EventPropMath.ComputeSpikesInto(
            _weights,
            inputTimes, inputSources,
            _params.N,
            _params.VTh,
            _params.TauMem,
            _params.TauSyn,
            outputTimes, outputSources, outputCurrents,
            _buffers);
    }

    /// <summary>
    /// Zero-allocation backward pass after a TrainingStep forward.
    /// Computes weight gradients into the existing Gradient array (zeroed first).
    /// </summary>
    public override void TrainingStepBackward(
        ReadOnlySpan<double> inputTimes,
        ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> outputTimes,
        ReadOnlySpan<int> outputSources,
        ReadOnlySpan<double> outputCurrents,
        int nOutputSpikes,
        ReadOnlySpan<double> outputErrors)
    {
        Gradient.AsSpan().Clear();

        if (nOutputSpikes == 0)
            return;

        // Trim spans to actual spike count (no copy — just slice)
        EventPropMath.BackwardSpikesRaw(
            _weights,
            inputTimes, inputSources,
            outputTimes.Slice(0, nOutputSpikes),
            outputSources.Slice(0, nOutputSpikes),
            outputCurrents.Slice(0, nOutputSpikes),
            outputErrors.Slice(0, nOutputSpikes),
            _params.N,
            _params.VTh,
            _params.TauMem,
            _params.TauSyn,
            Gradient.AsSpan(),
            _buffers);
    }

    /// <summary>Dispose scratch buffers.</summary>
    public override void Dispose() => _buffers.Dispose();
}
