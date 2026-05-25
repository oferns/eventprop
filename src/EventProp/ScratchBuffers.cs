namespace CW.EventProp;

/// <summary>
/// Reusable scratch buffers for allocation-free computation.
/// Allocate once per thread/work-item and reuse across calls.
/// </summary>
public sealed class ScratchBuffers : IDisposable
{
    /// <summary>Pre-allocated buffer for compute_sums sum0 matrix (n_spikes+1 rows x n_cols cols).</summary>
    public double[] Sum0 { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for compute_sums sum1 matrix.</summary>
    public double[] Sum1 { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for lambda_v vector (n_neurons elements).</summary>
    public double[] LambdaV { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for lambda_i vector (n_neurons elements, reused in backward).</summary>
    public double[] LambdaI { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for transposed weight matrix.</summary>
    public double[] TransposeW { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for voltage trace.</summary>
    public double[] VoltageTrace { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for lambda_i trace.</summary>
    public double[] LambdaITrace { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for spike errors in backward pass.</summary>
    public double[] SpikeErrors { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for post-spike times (max per neuron).</summary>
    public double[] PostSpikeTimes { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for post-spike currents (max per neuron).</summary>
    public double[] PostSpikeCurrents { get; private set; } = Array.Empty<double>();

    /// <summary>Number of spikes allocated per neuron in PostSpike buffers.</summary>
    public int MaxSpikesPerNeuron { get; private set; }

    /// <summary>Count of spikes per neuron (output from forward pass).</summary>
    public int[] SpikeCountPerNeuron { get; private set; } = Array.Empty<int>();

    /// <summary>Offset into flat post-spike arrays per neuron.</summary>
    public int[] SpikeOffsetPerNeuron { get; private set; } = Array.Empty<int>();

    /// <summary>
    /// Initialize or grow buffers to accommodate the given dimensions.
    /// </summary>
    public void EnsureCapacity(int nNeurons, int nPre, int nSpikes)
    {
        if (Sum0.Length < (nSpikes + 1) * nNeurons)
            Sum0 = new double[(nSpikes + 1) * nNeurons];

        if (Sum1.Length < (nSpikes + 1) * nNeurons)
            Sum1 = new double[(nSpikes + 1) * nNeurons];

        if (LambdaV.Length < nNeurons)
            LambdaV = new double[nNeurons];

        if (LambdaI.Length < nNeurons)
            LambdaI = new double[nNeurons];

        if (TransposeW.Length < nPre * nNeurons)
            TransposeW = new double[nPre * nNeurons];

        // Voltage trace: we don't know t_max ahead of time, allocate a reasonable default
        if (VoltageTrace.Length < 10000)
            VoltageTrace = new double[10000];

        if (LambdaITrace.Length < 10000)
            LambdaITrace = new double[10000];

        if (SpikeErrors.Length < nSpikes)
            SpikeErrors = new double[nSpikes];

        // Post-spike buffers: estimate max spikes per neuron
        int estimatedPerNeuron = Math.Max(nSpikes / nNeurons + 1, 4);
        if (PostSpikeTimes.Length < estimatedPerNeuron * nNeurons)
        {
            MaxSpikesPerNeuron = estimatedPerNeuron;
            PostSpikeTimes = new double[estimatedPerNeuron * nNeurons];
            PostSpikeCurrents = new double[estimatedPerNeuron * nNeurons];
            SpikeCountPerNeuron = new int[nNeurons];
            SpikeOffsetPerNeuron = new int[nNeurons + 1];
        }
    }

    public void Clear()
    {
        Sum0.AsSpan().Clear();
        Sum1.AsSpan().Clear();
        LambdaV.AsSpan().Clear();
        LambdaI.AsSpan().Clear();
        SpikeErrors.AsSpan().Clear();
    }

    public void Dispose()
    {
        Sum0 = Array.Empty<double>();
        Sum1 = Array.Empty<double>();
        LambdaV = Array.Empty<double>();
        LambdaI = Array.Empty<double>();
        TransposeW = Array.Empty<double>();
        VoltageTrace = Array.Empty<double>();
        LambdaITrace = Array.Empty<double>();
        SpikeErrors = Array.Empty<double>();
        PostSpikeTimes = Array.Empty<double>();
        PostSpikeCurrents = Array.Empty<double>();
        SpikeCountPerNeuron = Array.Empty<int>();
        SpikeOffsetPerNeuron = Array.Empty<int>();
    }
}
