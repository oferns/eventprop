using System.Buffers;

namespace CW.EventProp;

/// <summary>
/// Holds membrane potential maxima for a single sample, one entry per neuron.
/// </summary>
public sealed class MaximaData : IDisposable
{
    static readonly ArrayPool<double> D = ArrayPool<double>.Shared;

    double[] _times;
    double[] _values;
    bool _rented;

    public double[] Times => _times;
    public double[] Values => _values;
    public double[] Errors { get; set; }
    public int NNeurons => _times.Length;

    /// <summary>
    /// Rent a MaximaData from ArrayPool. Call Return() or Dispose() when done.
    /// </summary>
    public static MaximaData Rent(int capacity)
    {
        return new MaximaData(D.Rent(capacity), D.Rent(capacity), D.Rent(capacity), rented: true);
    }

    /// <summary>Return a rented MaximaData to ArrayPool.</summary>
    public void Return()
    {
        if (!_rented) return;
        _rented = false;

        D.Return(_times, clearArray: false);
        D.Return(_values, clearArray: false);
        D.Return(Errors!, clearArray: false);

        _times = [];
        _values = [];
        Errors = [];
    }

    public MaximaData(double[] times, double[] values)
        : this(times, values, new double[times.Length]) { }

    public MaximaData(double[] times, double[] values, double[] errors)
    {
        _times = times;
        _values = values;
        Errors = errors;
    }

    MaximaData(double[] times, double[] values, double[] errors, bool rented)
    {
        _times = times;
        _values = values;
        Errors = errors;
        _rented = rented;
    }

    public void SetError(int neuronIdx, double error) => Errors![neuronIdx] = error;
    public void SetValue(int neuronIdx, double value) => _values[neuronIdx] = value;

    public void Dispose() => Return();
}
