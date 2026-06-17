using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace CW.EventProp;

/// <summary>
/// Result of a backward pass for a single sample.
/// </summary>
public sealed class BackwardResult
{
    /// <summary>Weight gradient (n_pre x n_post), row-major.</summary>
    public double[] Gradient { get; }

    /// <summary>Error signal for each input spike.</summary>
    public double[] SpikeErrors { get; }

    public BackwardResult(double[] gradient, double[] spikeErrors)
    {
        Gradient = gradient;
        SpikeErrors = spikeErrors;
    }
}

/// <summary>
/// Batch operations for EventProp — parallel processing of multiple samples.
/// Mirrors eventprop_batch.h (struct-based functions only).
/// </summary>
public static class EventPropBatch
{
    private const int ParallelThreshold = 2;

  // ── Batch forward pass (spikes) ──────────────────────────────

    /// <summary>
    /// Forward pass for a batch of samples. Each sample is processed independently.
    /// </summary>
    public static SpikesData[] ComputeSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs,
        double vTh, double tauMem, double tauSyn)
    {
        var results = new SpikesData[inputs.Length];

        if (inputs.Length >= ParallelThreshold)
        {
            double[] wArr = w.ToArray();
            Parallel.For(0, inputs.Length, idx => ProcessSpikesSample(idx, wArr, nPre, nPost, inputs, vTh, tauMem, tauSyn, results));
        }
        else
        {
            var buffers = new ScratchBuffers();
            for (int i = 0; i < inputs.Length; i++)
                ProcessSpikesSample(i, w, nPre, nPost, inputs, vTh, tauMem, tauSyn, results);
        }

        return results;
    }

  // ── Batch forward pass (maxima) ──────────────────────────────

    /// <summary>
    /// Compute membrane potential maxima for a batch of samples.
    /// </summary>
    public static MaximaData[] ComputeMaximaBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs,
        double tauMem, double tauSyn)
    {
        var results = new MaximaData[inputs.Length];

        if (inputs.Length >= ParallelThreshold)
        {
            double[] wArr = w.ToArray();
            Parallel.For(0, inputs.Length, idx => ProcessMaximaSample(idx, wArr, nPre, nPost, inputs, tauMem, tauSyn, results));
        }
        else
        {
            var buffers = new ScratchBuffers();
            for (int i = 0; i < inputs.Length; i++)
                ProcessMaximaSample(i, w, nPre, nPost, inputs, tauMem, tauSyn, results);
        }

        return results;
    }

  // ── Batch backward pass (spikes) — per-sample gradients ──────

    /// <summary>
    /// Backward pass for a batch of samples. Returns per-sample gradients (not accumulated).
    /// </summary>
    public static BackwardResult[] BackwardSpikesBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, SpikesData[] fwdOutputs,
        double vTh, double tauMem, double tauSyn)
    {
        var results = new BackwardResult[inputs.Length];

        if (inputs.Length >= ParallelThreshold)
        {
            double[] wArr = w.ToArray();
            Parallel.For(0, inputs.Length, idx => ProcessBackwardSpikesSample(idx, wArr, nPre, nPost, inputs, fwdOutputs, vTh, tauMem, tauSyn, results));
        }
        else
        {
            var buffers = new ScratchBuffers();
            for (int i = 0; i < inputs.Length; i++)
                ProcessBackwardSpikesSample(i, w, nPre, nPost, inputs, fwdOutputs, vTh, tauMem, tauSyn, results);
        }

        return results;
    }

  // ── Batch backward pass (spikes) — accumulated gradient ──────

    /// <summary>
    /// Backward pass for a batch of samples with gradient accumulation.
    /// Equivalent to summing per-sample gradients into a single result.
    /// </summary>
    public static BackwardResult BackwardSpikesBatchAccumulate(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, SpikesData[] fwdOutputs,
        double vTh, double tauMem, double tauSyn)
    {
        int gradSize = nPre * nPost;
        int totalInputSpikes = 0;
        for (int i = 0; i < inputs.Length; i++) totalInputSpikes += inputs[i].NSpikes;

        double[] gradAcc = new double[gradSize];
        double[] spikeErrorsFlat = new double[totalInputSpikes];

        int[] inputOffsets = new int[inputs.Length];
        inputOffsets[0] = 0;
        for (int i = 1; i < inputs.Length; i++)
            inputOffsets[i] = inputOffsets[i - 1] + inputs[i - 1].NSpikes;

        if (inputs.Length >= ParallelThreshold)
        {
            double[] wArr = w.ToArray();
            AccumulateBackwardSpikesParallel(wArr, nPre, nPost, inputs, fwdOutputs, inputOffsets,
                vTh, tauMem, tauSyn, gradAcc, spikeErrorsFlat);
        }
        else
        {
            AccumulateBackwardSpikesSequential(w, nPre, nPost, inputs, fwdOutputs, inputOffsets,
                vTh, tauMem, tauSyn, gradAcc, spikeErrorsFlat);
        }

        return new BackwardResult(gradAcc, spikeErrorsFlat);
    }

  // ── Batch backward pass (maxima) — per-sample gradients ──────

    /// <summary>
    /// Backward pass for a batch of samples using maxima. Returns per-sample gradients.
    /// </summary>
    public static BackwardResult[] BackwardMaximaBatch(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, MaximaData[] fwdOutputs,
        double tauMem, double tauSyn)
    {
        var results = new BackwardResult[inputs.Length];

        if (inputs.Length >= ParallelThreshold)
        {
            double[] wArr = w.ToArray();
            Parallel.For(0, inputs.Length, idx => ProcessBackwardMaximaSample(idx, wArr, nPre, nPost, inputs, fwdOutputs, tauMem, tauSyn, results));
        }
        else
        {
            var buffers = new ScratchBuffers();
            for (int i = 0; i < inputs.Length; i++)
                ProcessBackwardMaximaSample(i, w, nPre, nPost, inputs, fwdOutputs, tauMem, tauSyn, results);
        }

        return results;
    }

  // ── Batch backward pass (maxima) — accumulated gradient ──────

    /// <summary>
    /// Backward pass for a batch of samples using maxima with gradient accumulation.
    /// </summary>
    public static BackwardResult BackwardMaximaBatchAccumulate(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, MaximaData[] fwdOutputs,
        double tauMem, double tauSyn)
    {
        int gradSize = nPre * nPost;
        int totalInputSpikes = 0;
        for (int i = 0; i < inputs.Length; i++) totalInputSpikes += inputs[i].NSpikes;

        double[] gradAcc = new double[gradSize];
        double[] spikeErrorsFlat = new double[totalInputSpikes];

        if (inputs.Length >= ParallelThreshold)
        {
            double[] wArr = w.ToArray();
            AccumulateBackwardMaximaParallel(wArr, nPre, nPost, inputs, fwdOutputs, tauMem, tauSyn, gradAcc);
        }
        else
        {
            AccumulateBackwardMaximaSequential(w, nPre, nPost, inputs, fwdOutputs, tauMem, tauSyn, gradAcc);
        }

        return new BackwardResult(gradAcc, spikeErrorsFlat);
    }

  // ── Batch dropout ────────────────────────────────────────────

    /// <summary>
    /// Randomly drop spikes from each sample in a batch.
    /// Uses LCG PRNG seeded per-sample for reproducibility (matches C library).
    /// </summary>
    /// <param name="inputs">Input spikes per sample.</param>
    /// <param name="pDrop">Dropout probability (0 = no dropout, 1.0 = drop all).</param>
    /// <param name="seed">Random seed (sample i uses seed + i).</param>
    public static SpikesData[] DropoutSpikesBatch(
        SpikesData[] inputs, double pDrop, int seed)
    {
        var results = new SpikesData[inputs.Length];

        for (int b = 0; b < inputs.Length; b++)
            results[b] = DropoutSpikesSample(inputs[b], pDrop, seed + b);

        return results;
    }

  // ── Per-sample helpers (aggressively inlined for sequential path) ─

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessSpikesSample(
        int idx, ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, double vTh, double tauMem, double tauSyn,
        SpikesData[] results)
    {
        var buffers = new ScratchBuffers();
        buffers.InitPostSpikeBuffers(nPost, inputs[idx].NSpikes * 4);
        results[idx] = EventPropMath.ComputeSpikes(
            w, inputs[idx].Times, inputs[idx].Sources, nPost, vTh, tauMem, tauSyn, buffers);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessMaximaSample(
        int idx, ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, double tauMem, double tauSyn,
        MaximaData[] results)
    {
        var buffers = new ScratchBuffers();
        results[idx] = EventPropMath.ComputeMaxima(
            w, inputs[idx].Times, inputs[idx].Sources, nPost, tauMem, tauSyn, buffers);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessBackwardSpikesSample(
        int idx, ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, SpikesData[] fwdOutputs,
        double vTh, double tauMem, double tauSyn,
        BackwardResult[] results)
    {
        var buffers = new ScratchBuffers();
        int maxNeurons = Math.Max(nPre, nPost);
        buffers.EnsureCapacity(nNeurons: maxNeurons, nPre: nPre,
            nSpikes: inputs[idx].NSpikes + fwdOutputs[idx].NSpikes);

        var gradient = new double[nPre * nPost];
        EventPropMath.BackwardSpikes(
            w, inputs[idx].Times, inputs[idx].Sources,
            fwdOutputs[idx], nPost, vTh, tauMem, tauSyn, gradient, buffers);

        results[idx] = new BackwardResult(gradient, (double[])fwdOutputs[idx].Errors.Clone());
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ProcessBackwardMaximaSample(
        int idx, ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, MaximaData[] fwdOutputs,
        double tauMem, double tauSyn,
        BackwardResult[] results)
    {
        var gradient = new double[nPre * nPost];
        EventPropMath.BackwardMaxima(
            inputs[idx].Times, inputs[idx].Sources,
            fwdOutputs[idx].Times, fwdOutputs[idx].Values, fwdOutputs[idx].Errors,
            w, nPost, tauMem, tauSyn, gradient);

        results[idx] = new BackwardResult(gradient, new double[inputs[idx].NSpikes]);
    }

  // ── Accumulation helpers ──────────────────────────────────────

    private static void AccumulateBackwardSpikesSequential(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, SpikesData[] fwdOutputs, int[] inputOffsets,
        double vTh, double tauMem, double tauSyn,
        double[] gradAcc, double[] spikeErrorsFlat)
    {
        var buffers = new ScratchBuffers();
        int gradSize = nPre * nPost;

        for (int i = 0; i < inputs.Length; i++)
        {
            int maxNeurons = Math.Max(nPre, nPost);
            buffers.EnsureCapacity(nNeurons: maxNeurons, nPre: nPre,
                nSpikes: inputs[i].NSpikes + fwdOutputs[i].NSpikes);

            var localGrad = new double[gradSize];
            EventPropMath.BackwardSpikes(
                w, inputs[i].Times, inputs[i].Sources,
                fwdOutputs[i], nPost, vTh, tauMem, tauSyn, localGrad, buffers);

            for (int g = 0; g < gradSize; g++)
                gradAcc[g] += localGrad[g];
        }
    }

    private static void AccumulateBackwardSpikesParallel(
        double[] w, int nPre, int nPost,
        SpikesData[] inputs, SpikesData[] fwdOutputs, int[] inputOffsets,
        double vTh, double tauMem, double tauSyn,
        double[] gradAcc, double[] spikeErrorsFlat)
    {
        int gradSize = nPre * nPost;
        var localGrads = new ConcurrentBag<double[]>();

        Parallel.For(0, inputs.Length, () =>
            (buffers: new ScratchBuffers(), grad: new double[gradSize]),
        (idx, _, local) =>
        {
            var b = local.buffers;
            int maxNeurons = Math.Max(nPre, nPost);
            b.EnsureCapacity(nNeurons: maxNeurons, nPre: nPre,
                nSpikes: inputs[idx].NSpikes + fwdOutputs[idx].NSpikes);

            EventPropMath.BackwardSpikes(
                w, inputs[idx].Times, inputs[idx].Sources,
                fwdOutputs[idx], nPost, vTh, tauMem, tauSyn, local.grad, b);

            localGrads.Add(local.grad);
            return local;
        }, _ => { });

        foreach (var lg in localGrads)
            for (int i = 0; i < gradSize; i++)
                gradAcc[i] += lg[i];
    }

    private static void AccumulateBackwardMaximaSequential(
        ReadOnlySpan<double> w, int nPre, int nPost,
        SpikesData[] inputs, MaximaData[] fwdOutputs,
        double tauMem, double tauSyn,
        double[] gradAcc)
    {
        int gradSize = nPre * nPost;

        for (int i = 0; i < inputs.Length; i++)
        {
            var localGrad = new double[gradSize];
            EventPropMath.BackwardMaxima(
                inputs[i].Times, inputs[i].Sources,
                fwdOutputs[i].Times, fwdOutputs[i].Values, fwdOutputs[i].Errors,
                w, nPost, tauMem, tauSyn, localGrad);

            for (int g = 0; g < gradSize; g++)
                gradAcc[g] += localGrad[g];
        }
    }

    private static void AccumulateBackwardMaximaParallel(
        double[] w, int nPre, int nPost,
        SpikesData[] inputs, MaximaData[] fwdOutputs,
        double tauMem, double tauSyn,
        double[] gradAcc)
    {
        int gradSize = nPre * nPost;
        var localGrads = new ConcurrentBag<double[]>();

        Parallel.For(0, inputs.Length, () => new double[gradSize],
            (idx, _, localGrad) =>
            {
                EventPropMath.BackwardMaxima(
                    inputs[idx].Times, inputs[idx].Sources,
                    fwdOutputs[idx].Times, fwdOutputs[idx].Values, fwdOutputs[idx].Errors,
                    w, nPost, tauMem, tauSyn, localGrad);

                localGrads.Add(localGrad);
                return localGrad;
            }, _ => { });

        foreach (var lg in localGrads)
            for (int i = 0; i < gradSize; i++)
                gradAcc[i] += lg[i];
    }

  // ── Dropout helper ───────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static SpikesData DropoutSpikesSample(SpikesData input, double pDrop, int seed)
    {
        int nSpikes = input.NSpikes;

        if (nSpikes == 0 || pDrop <= 0.0)
            return new SpikesData(
                (double[])input.Times.Clone(),
                (int[])input.Sources.Clone());

        // LCG PRNG (glibc parameters, matches C library exactly)
        ulong state = (ulong)seed + 1;

        var times = new double[nSpikes];
        var sources = new int[nSpikes];
        int idx = 0;

        for (int i = 0; i < nSpikes; i++)
        {
            state = state * 6364136223846793005UL + 1442695040888963407UL;
            double rand = ((double)(state >> 11)) / 9007199254740992.0;

            if (rand > pDrop)
            {
                times[idx] = input.Times[i];
                sources[idx] = input.Sources[i];
                idx++;
            }
        }

        Array.Resize(ref times, idx);
        Array.Resize(ref sources, idx);
        return new SpikesData(times, sources);
    }
}
