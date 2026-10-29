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
public sealed class LILayer : Layer, IDisposable
{
    private readonly LILayerParameters _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    private double[] _weights;

    /// <summary>Membrane potential maxima from the forward pass.</summary>
    private MaximaVector? _maximaBatch;

    /// <summary>Reusable scratch buffers for zero-allocation computation.</summary>
    private readonly ScratchBuffers _buffers = new();

    /// <summary>Per-sample dL/da scratch for <see cref="Regression"/>, which sums samples itself.</summary>
    private double[] _regressionScratch = [];

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

    /// <summary>Membrane potential maxima from the forward pass.</summary>
    public MaximaVector? MaximaBatch => _maximaBatch;

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
    /// Processes samples in parallel when batch size >= 2.
    /// </summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;

        // Determine max spikes for buffer sizing
        int maxSpikes = 0;
        foreach (var spikes in inputBatch)
            maxSpikes = Math.Max(maxSpikes, spikes.NSpikes);

        var maximaBatch = new MaximaVector(inputBatch.Count);

        if (inputBatch.Count >= 2)
        {
            // Parallel path: each thread gets its own ScratchBuffers
            double[] wArr = _weights;
            var results = new MaximaData[inputBatch.Count];
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

                var maxima = EventPropMath.ComputeMaxima(
                    wArr,
                    inputSpikes.Times,
                    inputSpikes.Sources,
                    _params.N,
                    _params.TauMem,
                    _params.TauSyn,
                    buffers);

                results[idx] = maxima;
                return buffers;
            },
            buffers => buffers.Dispose());

            maximaBatch = new MaximaVector(results);
        }
        else
        {
            // Sequential path: reuse single buffer
            _buffers.EnsureCapacity(_params.N, _params.NIn, maxSpikes);

            foreach (var inputSpikes in inputBatch)
            {
                _buffers.Clear();

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
        }

        _maximaBatch = maximaBatch;
        // LI layer doesn't produce spikes — Output throws if accessed
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

        if (batchCount >= 2)
        {
            // Parallel path: each thread computes local gradient, then accumulate with lock
            double[] wArr = _weights;
            int gradSize = Gradient.Length;
            Parallel.For(0, batchCount, () => new double[gradSize],
                (idx, _, localGrad) =>
                {
                    EventPropMath.BackwardMaxima(
                        _inputBatch[idx].Times,
                        _inputBatch[idx].Sources,
                        _maximaBatch![idx].Times,
                        _maximaBatch[idx].Values,
                        _maximaBatch[idx].Errors,
                        wArr,
                        _params.N,
                        _params.TauMem,
                        _params.TauSyn,
                        localGrad);

                    return localGrad;
                },
                localGrad =>
                {
                    lock (Gradient)
                    {
                        for (int i = 0; i < gradSize; i++)
                            Gradient[i] += localGrad[i];
                    }
                });
        }
        else
        {
            // Sequential path: accumulate directly into Gradient
            foreach (var inputSpikes in _inputBatch)
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
        }

        _ranBackward = true;
    }

    /// <summary>Zero the weight gradient.</summary>
    public override void ZeroGrad() => Gradient.AsSpan().Clear();

    /// <summary>
    /// Continuous readout over the maxima from the last <see cref="Forward"/>, plus its loss, with
    /// dL/dmaxima written into each sample's maxima error buffers — so the step is
    /// <c>Forward(); Regression(...); Backward()</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Managed mirror of <c>CW.EventProp.Native.LILayer.Regression</c>, down to the refusal when
    /// <see cref="Forward"/> has not run and the accumulation rules, and the two are held to each
    /// other by <c>RegressionLayerParityTests</c> — including the weight gradient each produces
    /// after <see cref="Backward"/>, which is the only assertion that covers the whole chain.
    /// </para>
    /// <para>
    /// Gradients are <b>sums</b> over the batch and <b>accumulate</b> into
    /// <paramref name="decoderGrad"/>/<paramref name="biasGrad"/> when supplied. Non-finite maxima
    /// are neurons that never peaked: excluded from the readout and counted in
    /// <see cref="MaximaRegressionStep.Skipped"/>; a neuron at rest has a finite maximum of 0.0 and
    /// is not skipped.
    /// </para>
    /// </remarks>
    /// <param name="decoder">Readout weights a[n], length N; every entry must be finite.</param>
    /// <param name="bias">Readout bias b.</param>
    /// <param name="targets">One target per sample.</param>
    /// <param name="huberDelta">Non-finite (the default) for plain squared error; finite positive
    /// for the smooth Huber loss.</param>
    /// <param name="decoderGrad">Optional caller-owned dL/da accumulator.</param>
    /// <param name="biasGrad">Optional caller-owned length-1 dL/db accumulator.</param>
    public MaximaRegressionStep Regression(ReadOnlySpan<double> decoder, double bias,
                                           ReadOnlySpan<double> targets,
                                           double huberDelta = double.PositiveInfinity,
                                           double[]? decoderGrad = null,
                                           double[]? biasGrad = null)
    {
        if (!_ranForward || _maximaBatch is null)
            throw new InvalidOperationException(
                "Run Forward() before Regression(): dL/dmaxima is written into the maxima error buffers that Forward allocated.");

        int batch = _inputBatch!.Count;
        if (decoder.Length != _params.N)
            throw new ArgumentException(
                $"decoder has {decoder.Length} entries, the layer has {_params.N} neurons", nameof(decoder));
        if (targets.Length < batch)
            throw new ArgumentException(
                $"targets has {targets.Length} entries, the batch has {batch}", nameof(targets));

        double[] dGrad = decoderGrad ?? new double[_params.N];
        double[] bGrad = biasGrad ?? new double[1];
        if (dGrad.Length < _params.N)
            throw new ArgumentException(
                $"decoderGrad must hold {_params.N} entries, got {dGrad.Length}", nameof(decoderGrad));
        if (bGrad.Length < 1)
            throw new ArgumentException("biasGrad must hold at least 1 entry", nameof(biasGrad));

        var predictions = new double[batch];
        var losses = new double[batch];
        var skipped = new int[batch];
        if (_regressionScratch.Length < _params.N) _regressionScratch = new double[_params.N];

        for (int b = 0; b < batch; b++)
        {
            var maxima = _maximaBatch[b];
            int n = maxima.NNeurons;
            if (n == 0) continue;   // nothing to read out for this sample

            var result = RegressionLoss.MaximaRegression(
                maxima.Values.AsSpan(0, n), decoder, bias, targets[b], huberDelta,
                maxima.Errors.AsSpan(0, n), _regressionScratch.AsSpan(0, n));
            for (int i = 0; i < n; i++) dGrad[i] += _regressionScratch[i];

            bGrad[0] += result.BiasGrad;
            predictions[b] = result.Prediction;
            losses[b] = result.Loss;
            skipped[b] = result.NSkipped;
        }

        return new MaximaRegressionStep(predictions, losses, skipped, dGrad, bGrad);
    }

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
    public override void Dispose() => _buffers.Dispose();
}
