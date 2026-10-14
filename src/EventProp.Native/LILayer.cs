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

    public LILayer(LILayerParameters parameters)
        : this(parameters, null) { }

    public LILayer(LILayerParameters parameters, double[]? weights)
    {
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
