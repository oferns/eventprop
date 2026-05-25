namespace CW.EventProp;

/// <summary>
/// Holds membrane potential maxima for a single sample, one entry per neuron.
/// </summary>
public sealed class MaximaData
{
    public double[] Times { get; }
    public double[] Values { get; }
    public double[] Errors { get; set; }

    public int NNeurons => Times.Length;

    public MaximaData(double[] times, double[] values)
        : this(times, values, new double[times.Length]) { }

    public MaximaData(double[] times, double[] values, double[] errors)
    {
        Times = times;
        Values = values;
        Errors = errors;
    }

    public void SetError(int neuronIdx, double error) => Errors[neuronIdx] = error;
    public void SetValue(int neuronIdx, double value) => Values[neuronIdx] = value;
}
