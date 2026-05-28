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
public sealed class LIFLayer : Layer
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
    /// </summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;

        // Ensure scratch buffers are sized for the largest sample
        int maxSpikes = 0;
        foreach (var spikes in inputBatch)
            maxSpikes = Math.Max(maxSpikes, spikes.NSpikes);

        _buffers.EnsureCapacity(_params.N, _params.NIn, maxSpikes);
        _buffers.Clear();

        var postBatch = new SpikesVector(inputBatch.Count);
        foreach (var inputSpikes in inputBatch)
        {
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

        _postBatch = postBatch;
        _output = postBatch;
        _ranForward = true;
    }

    /// <summary>
    /// Run backward pass: compute weight gradients via the adjoint method.
    /// Gradients are accumulated into self.Gradient (not zeroed — call ZeroGrad() first if needed).
    /// </summary>
    public override void Backward()
    {
        if (!_ranForward)
            throw new InvalidOperationException("Run Forward() before Backward().");

        // Zero gradient before accumulation
        Gradient.AsSpan().Clear();

        foreach (var inputSpikes in _inputBatch!)
            foreach (var postSpikes in _postBatch!)
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

    /// <summary>Dispose scratch buffers.</summary>
    public void Dispose() => _buffers.Dispose();
}
