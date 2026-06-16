namespace CW.EventProp;

/// <summary>
/// Merged spike entry for sorting input + post-spike spikes by time.
/// Pre-computed Time avoids capturing ReadOnlySpan in the comparer.
/// </summary>
public struct MergedSpike : IComparable<MergedSpike>
{
    public readonly int Idx;
    public readonly bool IsInput;
    public readonly double Time;

    public MergedSpike(int idx, bool isInput, double time)
    {
        Idx = idx;
        IsInput = isInput;
        Time = time;
    }

    public int CompareTo(MergedSpike other)
        => other.Time.CompareTo(Time); // descending by time
}

/// <summary>
/// Spike entry for sorting output spikes by time.
/// </summary>
public struct SpikeEntry : IComparable<SpikeEntry>
{
    public readonly double Time;
    public readonly int Source;
    public readonly double Current;

    public SpikeEntry(double time, int source, double current)
    {
        Time = time;
        Source = source;
        Current = current;
    }

    public int CompareTo(SpikeEntry other)
    {
        int cmp = Time.CompareTo(other.Time);
        return cmp != 0 ? cmp : Source.CompareTo(other.Source);
    }
}
