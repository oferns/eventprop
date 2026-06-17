using System.Runtime.CompilerServices;

namespace CW.EventProp.Native;

/// <summary>
/// Zero-copy view over a single sample in native batch spike output.
/// </summary>
public unsafe struct NativeSpikesSample
{
    private readonly byte* _batchPtr;
    private readonly int _index;

    internal NativeSpikesSample(byte* batchPtr, int index)
    {
        _batchPtr = batchPtr;
        _index = index;
    }

    /// <summary>Number of spikes in this sample.</summary>
    public int NSpikes => *(int*)(_batchPtr + _index * 32);

    /// <summary>Zero-copy span over spike times.</summary>
    public ReadOnlySpan<double> Times => new((double*)(_batchPtr + _index * 32 + 8), NSpikes);

    /// <summary>Zero-copy span over spike sources.</summary>
    public ReadOnlySpan<int> Sources => new((int*)(_batchPtr + _index * 32 + 16), NSpikes);

    /// <summary>Zero-copy mutable span over synaptic currents (used for error signals in backward pass).</summary>
    public Span<double> Currents => new((double*)(_batchPtr + _index * 32 + 24), NSpikes);
}

/// <summary>
/// Zero-copy view over a native batch spike output pointer.
/// Exposes per-sample data via NativeSpikesSample without allocating managed arrays.
/// </summary>
public sealed unsafe class NativeSpikesVector : IDisposable
{
    private byte* _batchPtr;
    private readonly int _batchSize;

    internal NativeSpikesVector(IntPtr batchPtr, int batchSize)
    {
        _batchPtr = (byte*)batchPtr.ToPointer();
        _batchSize = batchSize;
    }

    /// <summary>Number of samples in the batch.</summary>
    public int Count => _batchSize;

    /// <summary>Get zero-copy view of a sample.</summary>
    public NativeSpikesSample this[int index] => new(_batchPtr, index);

    /// <summary>Free native batch output.</summary>
    public void Dispose()
    {
        if (_batchPtr != null)
        {
            Batch.eventprop_free_batch_spike_output((IntPtr)_batchPtr, _batchSize);
            _batchPtr = null;
        }
    }
}
