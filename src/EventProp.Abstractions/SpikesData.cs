using System.Buffers;

namespace CW.EventProp;

/// <summary>
/// Holds spike event data for a single sample. Times and sources are sorted by time.
/// Errors and Currents arrays must always match Times.Length.
/// </summary>
public sealed class SpikesData : IDisposable
{
    static readonly ArrayPool<double> D = ArrayPool<double>.Shared;
    static readonly ArrayPool<int> I = ArrayPool<int>.Shared;

    double[] _times;
    int[] _sources;
    double[] _errors;
    double[] _currents;
    bool _rented;

    public double[] Times => _times;
    public int[] Sources => _sources;
    public double[] Errors { get; set; }
    public double[] Currents { get; set; }
    public int NSpikes => _times.Length;

    /// <summary>
    /// Rent a SpikesData from ArrayPool. Call Return() or Dispose() when done.
    /// </summary>
    public static SpikesData Rent(int capacity)
    {
        return new SpikesData(D.Rent(capacity), I.Rent(capacity), D.Rent(capacity), D.Rent(capacity), rented: true);
    }

    /// <summary>Return a rented SpikesData to ArrayPool.</summary>
    public void Return()
    {
        if (!_rented) return;
        _rented = false;

        D.Return(_times, clearArray: false);
        I.Return(_sources, clearArray: false);
        D.Return(_errors, clearArray: false);
        D.Return(_currents, clearArray: false);

        _times = [];
        _sources = [];
        _errors = [];
        Errors = [];
        _currents = [];
        Currents = [];
    }

    public SpikesData(double[] times, int[] sources)
    {
        _times = times;
        _sources = sources;
        _errors = new double[times.Length];
        Errors = _errors;
        _currents = new double[times.Length];
        Currents = _currents;
    }

    public SpikesData(double[] times, int[] sources, double[] currents)
    {
        _times = times;
        _sources = sources;
        _errors = new double[times.Length];
        Errors = _errors;
        _currents = currents;
        Currents = _currents;
    }

    SpikesData(double[] times, int[] sources, double[] errors, double[] currents, bool rented)
    {
        _times = times;
        _sources = sources;
        _errors = errors;
        Errors = errors;
        _currents = currents;
        Currents = currents;
        _rented = rented;
    }

    public void SetError(int spikeIdx, double error)
    {
        if (spikeIdx < 0 || spikeIdx >= _errors.Length)
            throw new ArgumentOutOfRangeException(nameof(spikeIdx),
                $"Spike index {spikeIdx} out of range [0, {_errors.Length - 1}]");
        _errors[spikeIdx] = error;
    }

    public void SetTime(int spikeIdx, double time)
    {
        if (spikeIdx < 0 || spikeIdx >= _times.Length)
            throw new ArgumentOutOfRangeException(nameof(spikeIdx),
                $"Spike index {spikeIdx} out of range [0, {_times.Length - 1}]");
        _times[spikeIdx] = time;
    }

    public void SetErrors(double[] errors)
    {
        if (errors.Length != _times.Length)
            throw new ArgumentException(
                $"Errors array length ({errors.Length}) must match Times length ({_times.Length})", nameof(errors));
        _errors = errors;
        Errors = errors;
    }

    public void SetCurrents(double[] currents)
    {
        if (currents.Length != _times.Length)
            throw new ArgumentException(
                $"Currents array length ({currents.Length}) must match Times length ({_times.Length})", nameof(currents));
        _currents = currents;
        Currents = currents;
    }

    public void Dispose() => Return();
}
