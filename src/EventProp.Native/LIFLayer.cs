using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace CW.EventProp.Native;

/// <summary>
/// Leaky Integrate-and-Fire layer backed by the C library through P/Invoke.
///
/// Mirrors <see cref="CW.EventProp.LIFLayer"/> — same <see cref="Layer"/> contract, same
/// constructor validation, same Forward/Backward/ZeroGrad semantics — so the two are
/// interchangeable at call sites and comparable in benchmarks.
///
/// Forward caches the native spike batch (freed on the next Forward, Dispose, or
/// replacement) and exposes managed <see cref="SpikesData"/> through
/// <see cref="Layer.Output"/>. The caller writes loss derivatives into
/// <c>Output[b].SetError(i, …)</c>; Backward reads them straight from those buffers.
/// </summary>
public sealed unsafe class LIFLayer : Layer, IDisposable
{
    private readonly LIFLayerParameters _params;
    private readonly double[] _weights;

    /// <summary>Post-synaptic spikes from the last forward pass (managed view of native output).</summary>
    private SpikesVector? _postBatch;

    /// <summary>Pinned pointer array: one native double* per sample, into SpikesData.Errors.</summary>
    private IntPtr[] _errorPtrs = [];

    /// <summary>Pinned handles for each sample's SpikesData.Errors array (freed on next Forward/Dispose).</summary>
    private GCHandle[] _errorPins = [];

    public LIFLayer(LIFLayerParameters parameters)
        : this(parameters, null) { }

    public LIFLayer(LIFLayerParameters parameters, double[]? weights)
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

    public LIFLayerParameters Parameters => _params;

    /// <summary>Per-sample forward output (same view as <see cref="Layer.Output"/>).</summary>
    public SpikesData this[int batchIdx] => (_postBatch ?? throw new InvalidOperationException("Run Forward() first."))[batchIdx];

    /// <summary>Set the loss derivative for one output spike; Backward reads these.</summary>
    public void SetError(int batchIdx, int spikeIdx, double value)
    {
        var post = _postBatch ?? throw new InvalidOperationException("Run Forward() before SetError().");
        post[batchIdx].SetError(spikeIdx, value);
    }

    /// <summary>Weight matrix (row-major, nIn × nOut).</summary>
    public double[] Weights => _weights;

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

    /// <summary>
    /// Forward pass: one batched native call over all samples. The native batch is cached
    /// for Backward; a managed <see cref="SpikesData"/> view is exposed through Output.
    /// </summary>
    public override void Forward(SpikesVector inputBatch)
    {
        _inputBatch = inputBatch;
        int batchSize = inputBatch.Count;

        int totalSpikes = Flatten(inputBatch, out var flatTimes, out var flatSources, out var nPerSample);

        using var spikes = Batch.ComputeSpikesBatch(
            _weights, _params.NIn, _params.N, flatTimes, flatSources, nPerSample,
            _params.VTh, _params.TauMem, _params.TauSyn);

        if (spikes.ErrorNo != Api.EventPropError.Ok)
            throw new InvalidOperationException(
                $"Forward pass failed: {Api.GetErrorString(spikes.ErrorNo)}");

        var postBatch = new SpikesVector(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            var sample = spikes.SpikeAt(b);
            if (sample.ErrorNo != Api.EventPropError.Ok)
                throw new InvalidOperationException(
                    $"Forward pass failed for sample {b}: {Api.GetErrorString(sample.ErrorNo)}");

            int n = sample.NSpikes;
            var times = new double[n];
            var sources = new int[n];
            sample.Times.CopyTo(times);
            sample.Sources.CopyTo(sources);
            postBatch.Add(new SpikesData(times, sources));
        }

        BuildErrorPointers(postBatch);

        _postBatch = postBatch;
        _output = postBatch;
        _ranForward = true;
        _ranBackward = false;
    }

    /// <summary>
    /// Backward pass: gradients come straight out of the native accumulate call.
    /// Reads loss derivatives from <see cref="Layer.Output"/>'s per-spike error buffers.
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

        // SpikesBatch is a ref struct, so the forward batch cannot be stored on the layer.
        // Recompute it here (deterministic for the same weights + input) and feed the
        // backward accumulate call in the same scope.
        using var spikes = Batch.ComputeSpikesBatch(
            _weights, _params.NIn, _params.N, flatTimes, flatSources, nPerSample,
            _params.VTh, _params.TauMem, _params.TauSyn);

        using var accumulated = Batch.BackwardSpikesBatchAccumulate(
            _weights, _params.NIn, _params.N, flatTimes, flatSources, nPerSample,
            spikes, _errorPtrs,
            _params.VTh, _params.TauMem, _params.TauSyn);

        if (accumulated.ErrorNo != Api.EventPropError.Ok)
            throw new InvalidOperationException(
                $"Backward pass failed: {Api.GetErrorString(accumulated.ErrorNo)}");

        var summed = accumulated.GetSummedGrads();
        if (summed.Rows != _params.NIn || summed.Cols != _params.N)
            throw new InvalidOperationException(
                $"Gradient shape mismatch: got {summed.Rows}x{summed.Cols}, expected {_params.NIn}x{_params.N}");

        // Add the summed gradient (rows*cols doubles) out of the native buffer.
        int gradLen = summed.Rows * summed.Cols;
        if (gradLen > Gradient.Length)
            gradLen = Gradient.Length;
        var native = new ReadOnlySpan<double>(summed.Data.ToPointer(), gradLen);
        for (int i = 0; i < gradLen; i++) Gradient[i] += native[i];
        _ranBackward = true;
    }

    public override void ZeroGrad() => Gradient.AsSpan().Clear();

    /// <summary>Free the pinned error-pointer handles.</summary>
    public override void Dispose()
    {
        FreeErrorPointers();
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private int Flatten(SpikesVector batch, out double[] times, out int[] sources, out int[] nPerSample)
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
        return total;
    }

    /// <summary>
    /// Build the native post-spike-error pointer array from the output's per-spike error
    /// buffers. Each sample's Errors array is pinned so its address survives GC compaction
    /// until the next Forward (or Dispose) frees the handles.
    /// </summary>
    private void BuildErrorPointers(SpikesVector postBatch)
    {
        FreeErrorPointers();

        int batchSize = postBatch.Count;
        _errorPtrs = new IntPtr[batchSize];
        _errorPins = new GCHandle[batchSize];
        for (int b = 0; b < batchSize; b++)
        {
            int n = postBatch[b].NSpikes;
            if (n == 0)
            {
                _errorPtrs[b] = IntPtr.Zero;
                continue;
            }
            var handle = GCHandle.Alloc(postBatch[b].Errors, GCHandleType.Pinned);
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
