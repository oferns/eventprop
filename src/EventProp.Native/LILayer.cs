using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>
/// Leaky Integrate (LI) layer backed by the C library through P/Invoke.
///
/// Mirrors <see cref="CW.EventProp.LILayer"/>: no spike threshold, forward computes
/// membrane maxima, backward uses the maxima-based loss. Same <see cref="Layer"/>
/// contract as the managed layer so the two are comparable in benchmarks.
///
/// The forward maxima are exposed through <see cref="MaximaBatch"/>; the caller writes
/// loss derivatives with <c>MaximaBatch[b].SetError(i, …)</c> and Backward reads them.
/// </summary>
public sealed unsafe class LILayer : Layer, IDisposable
{
    private readonly LILayerParameters _params;
    private readonly double[] _weights;

    /// <summary>Per-sample maxima from the last forward pass (managed view of native output).</summary>
    private MaximaVector? _maximaBatch;

    /// <summary>Pinned pointer array: one native double* per sample, into MaximaData.Errors.</summary>
    private IntPtr[] _errorPtrs = [];
    private GCHandle[] _errorPins = [];
    private double[] _regressionScratch = [];

    public LILayer(LILayerParameters parameters)
        : this(parameters, null) { }

    public LILayer(LILayerParameters parameters, double[]? weights)
    {
        // The P/Invoke structs are hand-declared, so a header/library offset mismatch
        // is invisible here and shows up as wrong numbers. Check once per process.
        Api.EnsureAbiMatches();

        _params = parameters;

        if (weights is not null)
        {
            int expectedLen = parameters.NIn * parameters.N;
            if (weights.Length != expectedLen)
                throw new ArgumentException(
                    $"Weight array length ({weights.Length}) must be {expectedLen} (nIn={parameters.NIn} × nOut={parameters.N})",
                    nameof(weights));
            _weights = weights;
        }
        else
        {
            WeightInitializer init = parameters.Weights ?? new GaussianInitializer();
            _weights = init.GetWeights(parameters.NIn, parameters.N);
        }

        Gradient = new double[_weights.Length];
    }

    public LILayerParameters Parameters => _params;

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    public double[] Weights => _weights;

    /// <summary>Per-sample maxima from the last forward pass (the LI layer's loss signal).</summary>
    public MaximaVector? MaximaBatch => _maximaBatch;

    public override int NIn => _params.NIn;
    public override int NOut => _params.N;

    public override ReadOnlySpan<double> GetWeights() => _weights.AsSpan();

    public override void SetWeights(ReadOnlySpan<double> weights)
    {
        if (weights.Length != _weights.Length)
            throw new ArgumentException(
                $"Weight span length ({weights.Length}) must match ({_weights.Length})",
                nameof(weights));
        weights.CopyTo(_weights);
    }

    /// <summary>Forward pass: one batched native maxima call, exposed as managed MaximaData.</summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;
        int batchSize = inputBatch.Count;

        Flatten(inputBatch, out var flatTimes, out var flatSources, out var nPerSample);

        using var maxima = Batch.ComputeMaximaBatch(
            _weights, _params.NIn, _params.N, flatTimes, flatSources, nPerSample,
            _params.TauMem, _params.TauSyn);

        if (maxima.ErrorNo != Api.EventPropError.Ok)
            throw new InvalidOperationException(
                $"Forward pass failed: {Api.GetErrorString(maxima.ErrorNo)}");

        var maximaBatch = new MaximaVector(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            var entry = maxima.MaximaAt(b);
            if (entry.ErrorNo != Api.EventPropError.Ok)
                throw new InvalidOperationException(
                    $"Forward pass failed for sample {b}: {Api.GetErrorString(entry.ErrorNo)}");

            int n = entry.NPost;
            var times = new double[n];
            var values = new double[n];
            new ReadOnlySpan<double>(entry.Times.ToPointer(), n).CopyTo(times);
            new ReadOnlySpan<double>(entry.Values.ToPointer(), n).CopyTo(values);
            maximaBatch.Add(new MaximaData(times, values));
        }

        BuildErrorPointers(maximaBatch);

        _maximaBatch = maximaBatch;
        _ranForward = true;
        _ranBackward = false;
    }

    /// <summary>
    /// Backward pass: gradients come straight out of the native maxima accumulate call.
    /// Reads loss derivatives from <see cref="MaximaBatch"/>'s per-neuron error buffers.
    /// **Accumulates** into <see cref="Layer.Gradient"/> (not zeroed) — matching the
    /// managed layer and the Python reference. Call <see cref="ZeroGrad"/> first for a
    /// single-step gradient.
    /// </summary>
    public override void Backward()
    {
        if (!_ranForward)
            throw new InvalidOperationException("Run Forward() before Backward().");

        var inputBatch = _inputBatch!;
        int batchSize = inputBatch.Count;

        int totalSpikes = 0;
        for (int b = 0; b < batchSize; b++) totalSpikes += inputBatch[b].NSpikes;
        if (totalSpikes == 0)
        {
            _ranBackward = true;
            return;
        }

        Flatten(inputBatch, out var flatTimes, out var flatSources, out var nPerSample);

        // MaximaBatch is a ref struct, so the forward maxima cannot be stored on the layer.
        // Recompute it (deterministic for the same weights + input) and feed the backward
        // accumulate call in the same scope.
        using var maxima = Batch.ComputeMaximaBatch(
            _weights, _params.NIn, _params.N, flatTimes, flatSources, nPerSample,
            _params.TauMem, _params.TauSyn);

        // The C backward reads the loss derivative from each sample's native errors array.
        // Copy the caller's managed MaximaData.Errors into it.
        for (int b = 0; b < batchSize; b++)
        {
            var entry = maxima.MaximaAt(b);
            int n = entry.NPost;
            if (n == 0) continue;
            _maximaBatch![b].Errors.AsSpan(0, n).CopyTo(new Span<double>(entry.Errors.ToPointer(), n));
        }

        using var accumulated = Batch.BackwardMaximaBatchAccumulate(
            _weights, _params.NIn, _params.N, flatTimes, flatSources, nPerSample,
            maxima, _params.TauMem, _params.TauSyn);

        if (accumulated.ErrorNo != Api.EventPropError.Ok)
            throw new InvalidOperationException(
                $"Backward pass failed: {Api.GetErrorString(accumulated.ErrorNo)}");

        var summed = accumulated.GetSummedGrads();
        if (summed.Rows != _params.NIn || summed.Cols != _params.N)
            throw new InvalidOperationException(
                $"Gradient shape mismatch: got {summed.Rows}x{summed.Cols}, expected {_params.NIn}x{_params.N}");

        int gradLen = Math.Min(summed.Rows * summed.Cols, Gradient.Length);
        var native = new ReadOnlySpan<double>(summed.Data.ToPointer(), gradLen);
        for (int i = 0; i < gradLen; i++) Gradient[i] += native[i];
        _ranBackward = true;
    }

    public override void ZeroGrad() => Gradient.AsSpan().Clear();

    /// <summary>
    /// Continuous readout over the maxima from the last <see cref="Forward"/>, plus its loss, with
    /// dL/dmaxima written into each sample's maxima error buffers — so the step is
    /// <c>Forward(); Regression(...); Backward()</c> and no loss arithmetic happens on the
    /// managed side.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the managed mirror of <c>CW.EventProp.LILayer.Regression</c> and of
    /// <c>eventprop_c.LILayer.regression</c> in the Cython binding: same formulas, same refusal if
    /// <see cref="Forward"/> has not run, same accumulation rules. The native side reaches the C
    /// library through <see cref="Loss.Regress"/> per sample rather than
    /// <see cref="Loss.RegressBatch"/>, because the batch entry point needs a native
    /// <see cref="Batch.MaximaBatch"/>, which is a ref struct the layer cannot cache — recomputing
    /// the maxima to build one would cost a full forward pass to save N cheap calls.
    /// </para>
    /// <para>
    /// Gradients are <b>sums</b> over the batch and <b>accumulate</b> into
    /// <paramref name="decoderGrad"/>/<paramref name="biasGrad"/> when supplied. A non-finite
    /// maximum is a neuron that never peaked: it is excluded and counted in
    /// <see cref="MaximaRegressionStep.Skipped"/>. A neuron at rest contributes a finite maximum of
    /// 0.0, which is a value and not a skip.
    /// </para>
    /// </remarks>
    /// <param name="decoder">Readout weights a[n], length N; every entry must be finite.</param>
    /// <param name="bias">Readout bias b.</param>
    /// <param name="targets">One target per sample.</param>
    /// <param name="huberDelta"><see cref="Loss.LeastSquares"/> (or any non-finite value) for
    /// plain squared error; a finite positive value for the smooth Huber loss.</param>
    /// <param name="decoderGrad">Optional caller-owned dL/da accumulator.</param>
    /// <param name="biasGrad">Optional caller-owned length-1 dL/db accumulator.</param>
    public MaximaRegressionStep Regression(ReadOnlySpan<double> decoder, double bias,
                                           ReadOnlySpan<double> targets,
                                           double huberDelta = Loss.LeastSquares,
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

            // Loss.Regress overwrites the decoder-gradient span it is given (it is the per-sample
            // primitive); the batch sum is accumulated here.
            var result = Loss.Regress(maxima.Values.AsSpan(0, n), decoder, bias, targets[b],
                                      huberDelta, maxima.Errors.AsSpan(0, n),
                                      _regressionScratch.AsSpan(0, n));
            for (int i = 0; i < n; i++) dGrad[i] += _regressionScratch[i];

            bGrad[0] += result.BiasGrad;
            predictions[b] = result.Prediction;
            losses[b] = result.Loss;
            skipped[b] = result.NSkipped;
        }

        return new MaximaRegressionStep(predictions, losses, skipped, dGrad, bGrad);
    }

    public override void Dispose() => FreeErrorPointers();

    // ── Helpers ──────────────────────────────────────────────────────

    private void Flatten(SpikesVector batch, out double[] times, out int[] sources, out int[] nPerSample)
    {
        int total = 0;
        for (int b = 0; b < batch.Count; b++) total += batch[b].NSpikes;

        times = new double[total];
        sources = new int[total];
        nPerSample = new int[batch.Count];

        int offset = 0;
        for (int b = 0; b < batch.Count; b++)
        {
            int n = batch[b].NSpikes;
            nPerSample[b] = n;
            batch[b].Times.AsSpan(0, n).CopyTo(times.AsSpan(offset));
            batch[b].Sources.AsSpan(0, n).CopyTo(sources.AsSpan(offset));
            offset += n;
        }
    }

    private void BuildErrorPointers(MaximaVector maximaBatch)
    {
        FreeErrorPointers();

        int batchSize = maximaBatch.Count;
        _errorPtrs = new IntPtr[batchSize];
        _errorPins = new GCHandle[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            int n = maximaBatch[b].NNeurons;
            if (n == 0)
            {
                _errorPtrs[b] = IntPtr.Zero;
                continue;
            }
            var handle = GCHandle.Alloc(maximaBatch[b].Errors, GCHandleType.Pinned);
            _errorPins[b] = handle;
            _errorPtrs[b] = handle.AddrOfPinnedObject();
        }
    }

    private void FreeErrorPointers()
    {
        for (int b = 0; b < _errorPins.Length; b++)
        {
            if (_errorPins[b].IsAllocated) _errorPins[b].Free();
            _errorPins[b] = default;
        }
        _errorPins = [];
        _errorPtrs = [];
    }
}
