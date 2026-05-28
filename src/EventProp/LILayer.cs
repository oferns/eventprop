namespace CW.EventProp;

/// <summary>
/// Leaky Integrate (LI) layer with maxima-based loss.
/// 
/// Forward pass: computes membrane potential maxima per neuron (no spike threshold).
/// The maxima represent the peak voltage reached by each neuron given the input spikes.
/// 
/// Backward pass: computes weight gradients via the adjoint method, propagating
/// error signals from membrane potential maxima back through the input spike timeline.
/// 
/// Zero-allocation hot paths via pre-allocated ScratchBuffers.
/// </summary>
public sealed class LILayer : Layer
{
    private readonly LILayerParameters _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    private double[] _weights;

    /// <summary>Membrane potential maxima from the forward pass.</summary>
    private MaximaVector? _maximaBatch;

    /// <summary>Reusable scratch buffers for zero-allocation computation.</summary>
    private readonly ScratchBuffers _buffers = new();

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

    /// <summary>
    /// Run forward pass: compute membrane potential maxima per neuron given input spikes.
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

        var maximaBatch = new MaximaVector(inputBatch.Count);
        foreach (var inputSpikes in inputBatch)
        {
            var maxima = EventPropMath.ComputeMaxima(
                _weights.AsSpan(),
                inputSpikes.Times,
                inputSpikes.Sources,
                _params.N,
                _params.TauMem,
                _params.TauSyn,
                _buffers);

            maximaBatch.Add(maxima);
        }

        _maximaBatch = maximaBatch;
        // LI layer doesn't produce spikes — Output throws if accessed
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
            foreach (var maxima in _maximaBatch!)
                EventPropMath.BackwardMaxima(
                    inputSpikes.Times,
                    inputSpikes.Sources,
                    maxima.Times,
                    maxima.Values,
                    maxima.Errors,
                    _weights.AsSpan(),
                    _params.N,
                    _params.TauMem,
                    _params.TauSyn,
                    Gradient.AsSpan());

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
        // Use Infinity as threshold to simulate LI (no spike reset)
        return EventPropMath.ComputeVoltageTrace(
            tMax, dt, targetNrnIdx,
            _weights.AsSpan(), _params.NIn, _params.N,
            inputSpikes.Times, inputSpikes.Sources,
            double.PositiveInfinity, _params.TauMem, _params.TauSyn,
            _buffers);
    }

    /// <summary>Dispose scratch buffers.</summary>
    public void Dispose() => _buffers.Dispose();
}
