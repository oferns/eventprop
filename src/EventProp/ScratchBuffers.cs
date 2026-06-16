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

    /// <summary>Pre-allocated flat buffer for lambda_i jump magnitudes (one per post-spike).</summary>
    public double[] LambdaIJumpsMagnitude { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated flat buffer for lambda_i jump time offsets (one per post-spike).</summary>
    public double[] LambdaIJumpsTimeOffset { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated per-neuron start index into lambda_i jumps flat arrays.</summary>
    public int[] LambdaIJumpsStart { get; private set; } = Array.Empty<int>();

    /// <summary>Pre-allocated per-neuron end index into lambda_i jumps flat arrays.</summary>
    public int[] LambdaIJumpsEnd { get; private set; } = Array.Empty<int>();

    /// <summary>Per-neuron jump capacity tracker (for dynamic growth of flat jump buffers).</summary>
    public int[] LambdaIJumpsNeuronCap { get; private set; } = Array.Empty<int>();

    /// <summary>Current total jump count across all neurons (into flat LambdaIJumps* arrays).</summary>
    public int LambdaIJumpsTotalCount { get; internal set; }

    /// <summary>Pre-allocated buffer for merged spike timeline in backward pass.</summary>
    public MergedSpike[] MergedSpikes { get; private set; } = Array.Empty<MergedSpike>();

    /// <summary>Pre-allocated buffer for spike entries (used by ComputeSpikes output sorting).</summary>
    public SpikeEntry[] SpikeEntries { get; private set; } = Array.Empty<SpikeEntry>();

    /// <summary>Pre-allocated output buffer for post-spike times.</summary>
    public double[] OutputSpikeTimes { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated output buffer for post-spike sources.</summary>
    public int[] OutputSpikeSources { get; private set; } = Array.Empty<int>();

    /// <summary>Pre-allocated output buffer for post-spike currents.</summary>
    public double[] OutputSpikeCurrents { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for exp(-t/tau_mem) values (one per input spike).</summary>
    public double[] ExpInputMem { get; private set; } = Array.Empty<double>();

    /// <summary>Pre-allocated buffer for exp(-t/tau_syn) values (one per input spike).</summary>
    public double[] ExpInputSyn { get; private set; } = Array.Empty<double>();

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

        // VoltageTrace and LambdaITrace are reserved for future algorithms.
        // They are not used by the current EventProp forward/backward passes.

        if (SpikeErrors.Length < nSpikes)
            SpikeErrors = new double[nSpikes];

        // Lambda_i jumps: one per post-spike, plus per-neuron start/end indices
        int nPostSpikes = Math.Max(nSpikes, 1);
        if (LambdaIJumpsMagnitude.Length < nPostSpikes)
        {
            LambdaIJumpsMagnitude = new double[nPostSpikes];
            LambdaIJumpsTimeOffset = new double[nPostSpikes];
        }
        if (LambdaIJumpsStart.Length < nNeurons + 1)
        {
            LambdaIJumpsStart = new int[nNeurons + 1];
            LambdaIJumpsEnd = new int[nNeurons + 1];
        }
        if (LambdaIJumpsNeuronCap.Length < nNeurons)
            LambdaIJumpsNeuronCap = new int[nNeurons];

        // Merged spikes for backward pass (input + post)
        if (MergedSpikes.Length < nSpikes * 2)
            MergedSpikes = new MergedSpike[nSpikes * 2];

        // Spike entries for output sorting
        if (SpikeEntries.Length < nSpikes)
            SpikeEntries = new SpikeEntry[nSpikes];

        // Output spike arrays
        if (OutputSpikeTimes.Length < nSpikes)
        {
            OutputSpikeTimes = new double[nSpikes];
            OutputSpikeSources = new int[nSpikes];
            OutputSpikeCurrents = new double[nSpikes];
        }

        // Exp buffers: one value per input spike
        if (ExpInputMem.Length < nSpikes)
        {
            ExpInputMem = new double[nSpikes];
            ExpInputSyn = new double[nSpikes];
        }

        // Post-spike buffers: estimate max spikes per neuron
        int estimatedPerNeuron = Math.Max(nSpikes / nNeurons + 1, 4);
        GrowPostSpikeBuffers(nNeurons, estimatedPerNeuron);
    }

    /// <summary>
    /// Initialize post-spike buffers for a specific configuration.
    /// Used when caller knows exact dimensions (e.g. tests).
    /// </summary>
    public void InitPostSpikeBuffers(int nNeurons, int maxSpikesPerNeuron)
    {
        GrowPostSpikeBuffers(nNeurons, maxSpikesPerNeuron);
    }

    internal void GrowPostSpikeBuffers(int nNeurons, int estimatedPerNeuron)
    {
        if (PostSpikeTimes.Length < estimatedPerNeuron * nNeurons)
        {
            MaxSpikesPerNeuron = estimatedPerNeuron;
            PostSpikeTimes = new double[estimatedPerNeuron * nNeurons];
            PostSpikeCurrents = new double[estimatedPerNeuron * nNeurons];
        }
        if (SpikeCountPerNeuron.Length < nNeurons)
        {
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
        PostSpikeTimes.AsSpan().Clear();
        PostSpikeCurrents.AsSpan().Clear();
        VoltageTrace.AsSpan().Clear();
        LambdaITrace.AsSpan().Clear();
        LambdaIJumpsMagnitude.AsSpan().Clear();
        LambdaIJumpsTimeOffset.AsSpan().Clear();
        Array.Clear(LambdaIJumpsNeuronCap, 0, LambdaIJumpsNeuronCap.Length);
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
        ExpInputMem = Array.Empty<double>();
        ExpInputSyn = Array.Empty<double>();
        PostSpikeTimes = Array.Empty<double>();
        PostSpikeCurrents = Array.Empty<double>();
        SpikeCountPerNeuron = Array.Empty<int>();
        SpikeOffsetPerNeuron = Array.Empty<int>();
        LambdaIJumpsNeuronCap = Array.Empty<int>();
        MergedSpikes = Array.Empty<MergedSpike>();
        SpikeEntries = Array.Empty<SpikeEntry>();
        OutputSpikeTimes = Array.Empty<double>();
        OutputSpikeSources = Array.Empty<int>();
        OutputSpikeCurrents = Array.Empty<double>();
    }
}
