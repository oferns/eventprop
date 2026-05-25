namespace CW.EventProp;

/// <summary>
/// Holds spike event data for a single sample. Times and sources are sorted by time.
/// </summary>
public sealed class SpikesData
{
    public double[] Times { get; }
    public int[] Sources { get; }
    public double[] Errors { get; set; }
    public double[] Currents { get; set; }
    public int NSpikes => Times.Length;

    public SpikesData(double[] times, int[] sources)
    {
        Times = times;
        Sources = sources;
        Errors = new double[times.Length];
        Currents = Array.Empty<double>();
    }

    public void SetError(int spikeIdx, double error) => Errors[spikeIdx] = error;
    public void SetTime(int spikeIdx, double time) => Times[spikeIdx] = time;
}
