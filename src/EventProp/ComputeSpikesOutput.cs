namespace CW.EventProp;

/// <summary>Output from ComputeSpikes forward pass. Owned by caller.</summary>
public sealed class ComputeSpikesOutput : IDisposable
{
    public int NSpikes { get; }

    /// <summary>Spike times (sorted).</summary>
    public double[] Times { get; }

    /// <summary>Source neurons for each spike.</summary>
    public int[] Sources { get; }

    /// <summary>Synaptic currents at each spike time.</summary>
    public double[] Currents { get; }

    internal ComputeSpikesOutput(double[] times, int[] sources, double[] currents)
    {
        NSpikes = times.Length;
        Times = times;
        Sources = sources;
        Currents = currents;
    }

    public void Dispose() { /* GC will handle cleanup */ }
}

/// <summary>Output from ComputeSums.</summary>
public sealed class ComputeSumsOutput : IDisposable
{
    public int NRows { get; }  // n_spikes + 1
    public int NPost { get; }

    /// <summary>Cumulative sum0 matrix (n_rows × n_post, row-major).</summary>
    public double[] Sum0 { get; }

    /// <summary>Cumulative sum1 matrix (n_rows × n_post, row-major).</summary>
    public double[] Sum1 { get; }

    internal ComputeSumsOutput(int nRows, int nPost, double[] sum0, double[] sum1)
    {
        NRows = nRows;
        NPost = nPost;
        Sum0 = sum0;
        Sum1 = sum1;
    }

    public void Dispose() { /* GC will handle cleanup */ }
}

/// <summary>Output from ComputeMaxima.</summary>
public sealed class ComputeMaximaOutput : IDisposable
{
    public int NPost { get; }

    /// <summary>Times of membrane potential maxima.</summary>
    public double[] Times { get; }

    /// <summary>Voltage values at maxima.</summary>
    public double[] Values { get; }

    internal ComputeMaximaOutput(int nPost, double[] times, double[] values)
    {
        NPost = nPost;
        Times = times;
        Values = values;
    }

    public void Dispose() { /* GC will handle cleanup */ }
}

/// <summary>Output from BackwardSpikes.</summary>
public sealed class BackwardSpikesOutput : IDisposable
{
    public int NPre { get; }
    public int NPost { get; }

    /// <summary>Weight gradient (n_pre × n_post, row-major).</summary>
    public double[] Gradient { get; }

    /// <summary>Spike errors for input spikes.</summary>
    public double[] SpikeErrors { get; }

    internal BackwardSpikesOutput(int nPre, int nPost, double[] gradient, double[] spikeErrors)
    {
        NPre = nPre;
        NPost = nPost;
        Gradient = gradient;
        SpikeErrors = spikeErrors;
    }

    public void Dispose() { /* GC will handle cleanup */ }
}

/// <summary>Output from BackwardMaxima.</summary>
public sealed class BackwardMaximaOutput : IDisposable
{
    public int NPre { get; }
    public int NPost { get; }

    /// <summary>Weight gradient (n_pre × n_post, row-major).</summary>
    public double[] Gradient { get; }

    /// <summary>Spike errors for input spikes.</summary>
    public double[] SpikeErrors { get; }

    internal BackwardMaximaOutput(int nPre, int nPost, double[] gradient, double[] spikeErrors)
    {
        NPre = nPre;
        NPost = nPost;
        Gradient = gradient;
        SpikeErrors = spikeErrors;
    }

    public void Dispose() { /* GC will handle cleanup */ }
}
