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
    // One growable row per post-synaptic neuron. A neuron can emit any number of
    // spikes — there is no bound expressible in terms of the input spike count —
    // so rows grow on demand instead of living at a fixed stride in one slab.
    PostSpikeRow[] _postSpikeRows = [];
    int _maxSpikesPerNeuron;
    int[] _spikeCountPerNeuron = [];
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
    /// <summary>Scratch row holding one post-synaptic neuron's spikes (times, currents, exp(st/tau_mem)).</summary>
    public PostSpikeRow PostSpikeRowAt(int targetNrn) => _postSpikeRows[targetNrn];
    /// <summary>Initial per-row capacity hint. Rows grow past this on demand; it is not a bound.</summary>
    public int MaxSpikesPerNeuron => _maxSpikesPerNeuron;
    public int[] SpikeCountPerNeuron => _spikeCountPerNeuron;

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
        int initialCap = Math.Max(estimatedPerNeuron, 1);

        if (_postSpikeRows.Length < nNeurons)
        {
            var rows = new PostSpikeRow[nNeurons];
            for (int i = 0; i < _postSpikeRows.Length; i++) rows[i] = _postSpikeRows[i];
            for (int i = _postSpikeRows.Length; i < nNeurons; i++) rows[i] = new PostSpikeRow(initialCap);
            _postSpikeRows = rows;
        }
        else
        {
            for (int i = 0; i < nNeurons; i++) _postSpikeRows[i].Grow(0, initialCap);
        }

        if (_spikeCountPerNeuron.Length < nNeurons) ReplaceInt(ref _spikeCountPerNeuron, nNeurons);
        _maxSpikesPerNeuron = initialCap;
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
        ReturnDouble(ref _lambdaIJumpsMagnitude);
        ReturnDouble(ref _lambdaIJumpsTimeOffset);
        ReturnDouble(ref _outputSpikeTimes);
        ReturnDouble(ref _outputSpikeCurrents);

        ReturnInt(ref _spikeCountPerNeuron);
        ReturnInt(ref _lambdaIJumpsStart);
        ReturnInt(ref _lambdaIJumpsEnd);
        ReturnInt(ref _lambdaIJumpsNeuronCap);
        ReturnInt(ref _outputSpikeSources);

        _mergedSpikes = [];
        _spikeEntries = [];

        foreach (var row in _postSpikeRows) row.Dispose();
        _postSpikeRows = [];
    }
}

/// <summary>
/// Growable scratch row holding one post-synaptic neuron's output spikes.
/// Backs ScratchBuffers.PostSpikeRowAt. The forward pass writes one spike per
/// threshold crossing, and a neuron may cross the threshold any number of times,
/// so the row grows (x3) instead of overflowing a fixed-size region.
/// </summary>
public sealed class PostSpikeRow : IDisposable
{
    static readonly ArrayPool<double> D = ArrayPool<double>.Shared;

    double[] _times = [];
    double[] _currents = [];
    double[] _expSt = [];
    bool _disposed;

    public double[] Times => _times;
    public double[] Currents => _currents;
    /// <summary>exp(times[s] / tau_mem), recomputed per interval for the factorized voltage model.</summary>
    public double[] ExpSt => _expSt;
    public int Capacity => _times.Length;

    public PostSpikeRow(int initialCapacity)
    {
        int cap = Math.Max(initialCapacity, 1);
        _times = D.Rent(cap);
        _currents = D.Rent(cap);
        _expSt = D.Rent(cap);
    }

    /// <summary>
    /// Grow every array so the row holds at least <paramref name="minCapacity"/> spikes,
    /// carrying over the <paramref name="liveCount"/> spikes already written.
    /// The superseded arrays are deliberately not returned to the pool: the sweep caches
    /// them in locals until it re-reads the row, so handing them over here could give the
    /// same memory to another thread. Growth is logarithmic, so the abandoned arrays are
    /// a handful per row and the garbage collector reclaims them.
    /// </summary>
    public void Grow(int liveCount, int minCapacity)
    {
        if (_times.Length >= minCapacity) return;

        int cap = Math.Max(_times.Length * 3, minCapacity);

        var newTimes = D.Rent(cap);
        var newCurrents = D.Rent(cap);
        var newExpSt = D.Rent(cap);

        for (int i = 0; i < liveCount; i++)
        {
            newTimes[i] = _times[i];
            newCurrents[i] = _currents[i];
            newExpSt[i] = _expSt[i];
        }

        _times = newTimes;
        _currents = newCurrents;
        _expSt = newExpSt;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_times.Length > 0) D.Return(_times);
        if (_currents.Length > 0) D.Return(_currents);
        if (_expSt.Length > 0) D.Return(_expSt);
        _times = [];
        _currents = [];
        _expSt = [];
    }
}
