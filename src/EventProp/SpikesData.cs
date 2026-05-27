namespace CW.EventProp;

/// <summary>
/// Holds spike event data for a single sample. Times and sources are sorted by time.
/// Errors and Currents arrays must always match Times.Length.
/// </summary>
public sealed class SpikesData
{
    public double[] Times { get; }
    public int[] Sources { get; }

    /// <summary>Error signals for each spike. Must always match Times.Length.</summary>
    public double[] Errors { get; set; }

    /// <summary>Synaptic currents at each spike time. Must always match Times.Length.</summary>
    public double[] Currents { get; set; }

    /// <summary>Number of spikes.</summary>
    public int NSpikes => Times.Length;

    public SpikesData(double[] times, int[] sources)
    {
        Times = times;
        Sources = sources;
        Errors = new double[times.Length];
        Currents = new double[times.Length];
    }

    public SpikesData(double[] times, int[] sources, double[] currents)
    {
        Times = times;
        Sources = sources;
        Errors = new double[times.Length];
        Currents = currents;
    }

    /// <summary>Set error signal for a specific spike.</summary>
    public void SetError(int spikeIdx, double error)
    {
        if (spikeIdx < 0 || spikeIdx >= Errors.Length)
            throw new ArgumentOutOfRangeException(nameof(spikeIdx),
                $"Spike index {spikeIdx} out of range [0, {Errors.Length - 1}]");
        Errors[spikeIdx] = error;
    }

    /// <summary>Set spike time for a specific spike.</summary>
    public void SetTime(int spikeIdx, double time)
    {
        if (spikeIdx < 0 || spikeIdx >= Times.Length)
            throw new ArgumentOutOfRangeException(nameof(spikeIdx),
                $"Spike index {spikeIdx} out of range [0, {Times.Length - 1}]");
        Times[spikeIdx] = time;
    }

    /// <summary>
    /// Replace the Errors array. Validates that the new array matches Times.Length.
    /// </summary>
    public void SetErrors(double[] errors)
    {
        if (errors.Length != Times.Length)
            throw new ArgumentException(
                $"Errors array length ({errors.Length}) must match Times length ({Times.Length})", nameof(errors));
        Errors = errors;
    }

    /// <summary>
    /// Replace the Currents array. Validates that the new array matches Times.Length.
    /// </summary>
    public void SetCurrents(double[] currents)
    {
        if (currents.Length != Times.Length)
            throw new ArgumentException(
                $"Currents array length ({currents.Length}) must match Times length ({Times.Length})", nameof(currents));
        Currents = currents;
    }
}
