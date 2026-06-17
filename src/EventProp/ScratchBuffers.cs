using System.Buffers;

namespace CW.EventProp;

/// <summary>
/// Reusable scratch buffers backed by ArrayPool — zero-allocation after initial rent.
/// Rent once per thread/work-item and reuse across calls. Call Dispose() to return arrays to the pool.
/// </summary>
public sealed class ScratchBuffers : IDisposable
{
    static readonly ArrayPool<double> D = ArrayPool<double>.Shared;
    static readonly ArrayPool<int> I = ArrayPool<int>.Shared;

    double[] _sum0 = [];
    double[] _sum1 = [];
    double[] _lambdaV = [];
    double[] _lambdaI = [];
    double[] _transposeW = [];
    double[] _voltageTrace = [];
    double[] _lambdaITrace = [];
    double[] _spikeErrors = [];
    double[] _lambdaIJumpsMagnitude = [];
    double[] _lambdaIJumpsTimeOffset = [];
    int[] _lambdaIJumpsStart = [];
    int[] _lambdaIJumpsEnd = [];
    int[] _lambdaIJumpsNeuronCap = [];
    MergedSpike[] _mergedSpikes = [];
    SpikeEntry[] _spikeEntries = [];
    double[] _outputSpikeTimes = [];
    int[] _outputSpikeSources = [];
    double[] _outputSpikeCurrents = [];
    double[] _expInputMem = [];
    double[] _expInputSyn = [];
    double[] _postSpikeTimes = [];
    double[] _postSpikeCurrents = [];
    double[] _postSpikeExpSt = []; // Precomputed exp(spikeTime * tauMemInv) for factorized ComputeV
    int _maxSpikesPerNeuron;
    int[] _spikeCountPerNeuron = [];
    int[] _spikeOffsetPerNeuron = [];
    bool _disposed;

    public double[] Sum0 => _sum0;
    public double[] Sum1 => _sum1;
    public double[] LambdaV => _lambdaV;
    public double[] LambdaI => _lambdaI;
    public double[] TransposeW => _transposeW;
    public double[] VoltageTrace => _voltageTrace;
    public double[] LambdaITrace => _lambdaITrace;
    public double[] SpikeErrors => _spikeErrors;
    public double[] LambdaIJumpsMagnitude => _lambdaIJumpsMagnitude;
    public double[] LambdaIJumpsTimeOffset => _lambdaIJumpsTimeOffset;
    public int[] LambdaIJumpsStart => _lambdaIJumpsStart;
    public int[] LambdaIJumpsEnd => _lambdaIJumpsEnd;
    public int[] LambdaIJumpsNeuronCap => _lambdaIJumpsNeuronCap;
    public MergedSpike[] MergedSpikes => _mergedSpikes;
    public SpikeEntry[] SpikeEntries => _spikeEntries;
    public double[] OutputSpikeTimes => _outputSpikeTimes;
    public int[] OutputSpikeSources => _outputSpikeSources;
    public double[] OutputSpikeCurrents => _outputSpikeCurrents;
    public double[] ExpInputMem => _expInputMem;
    public double[] ExpInputSyn => _expInputSyn;
    public double[] PostSpikeTimes => _postSpikeTimes;
    public double[] PostSpikeCurrents => _postSpikeCurrents;
    public double[] PostSpikeExpSt => _postSpikeExpSt;
    public int MaxSpikesPerNeuron => _maxSpikesPerNeuron;
    public int[] SpikeCountPerNeuron => _spikeCountPerNeuron;
    public int[] SpikeOffsetPerNeuron => _spikeOffsetPerNeuron;

    // ── ArrayPool helpers (operate on backing fields) ───────────

    void ReplaceDouble(ref double[] field, int minLength)
    {
        var old = field;
        field = D.Rent(minLength);
        if (old.Length > 0) D.Return(old);
    }

    void ReplaceInt(ref int[] field, int minLength)
    {
        var old = field;
        field = I.Rent(minLength);
        if (old.Length > 0) I.Return(old);
    }

    static void ReturnDouble(ref double[] arr)
    {
        if (arr.Length > 0) D.Return(arr);
        arr = [];
    }

    static void ReturnInt(ref int[] arr)
    {
        if (arr.Length > 0) I.Return(arr);
        arr = [];
    }

    // ── Capacity management ─────────────────────────────────────

    public void EnsureCapacity(int nNeurons, int nPre, int nSpikes)
    {
        int sumsSize = (nSpikes + 1) * nNeurons;
        if (_sum0.Length < sumsSize) ReplaceDouble(ref _sum0, sumsSize);
        if (_sum1.Length < sumsSize) ReplaceDouble(ref _sum1, sumsSize);

        if (_lambdaV.Length < nNeurons) ReplaceDouble(ref _lambdaV, nNeurons);
        if (_lambdaI.Length < nNeurons) ReplaceDouble(ref _lambdaI, nNeurons);

        if (_transposeW.Length < nPre * nNeurons) ReplaceDouble(ref _transposeW, nPre * nNeurons);

        if (_spikeErrors.Length < nSpikes) ReplaceDouble(ref _spikeErrors, nSpikes);

        int nPostSpikes = Math.Max(nSpikes, 1);
        if (_lambdaIJumpsMagnitude.Length < nPostSpikes)
        {
            ReplaceDouble(ref _lambdaIJumpsMagnitude, nPostSpikes);
            ReplaceDouble(ref _lambdaIJumpsTimeOffset, nPostSpikes);
        }
        if (_lambdaIJumpsStart.Length < nNeurons + 1) ReplaceInt(ref _lambdaIJumpsStart, nNeurons + 1);
        if (_lambdaIJumpsEnd.Length < nNeurons + 1) ReplaceInt(ref _lambdaIJumpsEnd, nNeurons + 1);
        if (_lambdaIJumpsNeuronCap.Length < nNeurons) ReplaceInt(ref _lambdaIJumpsNeuronCap, nNeurons);

        if (_mergedSpikes.Length < nSpikes * 2)
            _mergedSpikes = new MergedSpike[nSpikes * 2];

        if (_spikeEntries.Length < nSpikes)
            _spikeEntries = new SpikeEntry[nSpikes];

        if (_outputSpikeTimes.Length < nSpikes) ReplaceDouble(ref _outputSpikeTimes, nSpikes);
        if (_outputSpikeSources.Length < nSpikes) ReplaceInt(ref _outputSpikeSources, nSpikes);
        if (_outputSpikeCurrents.Length < nSpikes) ReplaceDouble(ref _outputSpikeCurrents, nSpikes);

        if (_expInputMem.Length < nSpikes) ReplaceDouble(ref _expInputMem, nSpikes);
        if (_expInputSyn.Length < nSpikes) ReplaceDouble(ref _expInputSyn, nSpikes);

        int estimatedPerNeuron = Math.Max(nSpikes / nNeurons + 1, 4);
        GrowPostSpikeBuffers(nNeurons, estimatedPerNeuron);
    }

    public void InitPostSpikeBuffers(int nNeurons, int maxSpikesPerNeuron)
        => GrowPostSpikeBuffers(nNeurons, maxSpikesPerNeuron);

    internal void GrowPostSpikeBuffers(int nNeurons, int estimatedPerNeuron)
    {
        int size = estimatedPerNeuron * nNeurons;
        if (_postSpikeTimes.Length < size) ReplaceDouble(ref _postSpikeTimes, size);
        if (_postSpikeCurrents.Length < size) ReplaceDouble(ref _postSpikeCurrents, size);
        if (_postSpikeExpSt.Length < size) ReplaceDouble(ref _postSpikeExpSt, size);
        if (_spikeCountPerNeuron.Length < nNeurons) ReplaceInt(ref _spikeCountPerNeuron, nNeurons);
        if (_spikeOffsetPerNeuron.Length < nNeurons + 1) ReplaceInt(ref _spikeOffsetPerNeuron, nNeurons + 1);
        _maxSpikesPerNeuron = estimatedPerNeuron;
    }

    /// <summary>
    /// Ensure SpikeEntries and output spike buffers can hold at least <paramref name="capacity"/> entries.
    /// Called after forward pass when total output spike count is known.
    /// </summary>
    internal void EnsureOutputCapacity(int capacity)
    {
        if (_spikeEntries.Length < capacity)
            _spikeEntries = new SpikeEntry[capacity];
        if (_outputSpikeTimes.Length < capacity) ReplaceDouble(ref _outputSpikeTimes, capacity);
        if (_outputSpikeSources.Length < capacity) ReplaceInt(ref _outputSpikeSources, capacity);
        if (_outputSpikeCurrents.Length < capacity) ReplaceDouble(ref _outputSpikeCurrents, capacity);
    }

    // ── Lifecycle ───────────────────────────────────────────────

    public void Clear()
    {
        _sum0.AsSpan().Clear();
        _sum1.AsSpan().Clear();
        _lambdaV.AsSpan().Clear();
        _lambdaI.AsSpan().Clear();
        _spikeErrors.AsSpan().Clear();
        _postSpikeTimes.AsSpan().Clear();
        _postSpikeCurrents.AsSpan().Clear();
        _voltageTrace.AsSpan().Clear();
        _lambdaITrace.AsSpan().Clear();
        _lambdaIJumpsMagnitude.AsSpan().Clear();
        _lambdaIJumpsTimeOffset.AsSpan().Clear();
        Array.Clear(_lambdaIJumpsNeuronCap, 0, _lambdaIJumpsNeuronCap.Length);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        ReturnDouble(ref _sum0);
        ReturnDouble(ref _sum1);
        ReturnDouble(ref _lambdaV);
        ReturnDouble(ref _lambdaI);
        ReturnDouble(ref _transposeW);
        ReturnDouble(ref _voltageTrace);
        ReturnDouble(ref _lambdaITrace);
        ReturnDouble(ref _spikeErrors);
        ReturnDouble(ref _expInputMem);
        ReturnDouble(ref _expInputSyn);
        ReturnDouble(ref _postSpikeTimes);
        ReturnDouble(ref _postSpikeCurrents);
        ReturnDouble(ref _postSpikeExpSt);
        ReturnDouble(ref _lambdaIJumpsMagnitude);
        ReturnDouble(ref _lambdaIJumpsTimeOffset);
        ReturnDouble(ref _outputSpikeTimes);
        ReturnDouble(ref _outputSpikeCurrents);

        ReturnInt(ref _spikeCountPerNeuron);
        ReturnInt(ref _spikeOffsetPerNeuron);
        ReturnInt(ref _lambdaIJumpsStart);
        ReturnInt(ref _lambdaIJumpsEnd);
        ReturnInt(ref _lambdaIJumpsNeuronCap);
        ReturnInt(ref _outputSpikeSources);

        _mergedSpikes = [];
        _spikeEntries = [];
    }
}
