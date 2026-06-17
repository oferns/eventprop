namespace CW.EventProp;

/// <summary>
/// Core EventProp algorithm computations. All methods are allocation-free when scratch buffers are provided.
/// </summary>
public static class EventPropMath
{
    /// <summary>
    /// Compute cumulative sums used for membrane potential evaluation.
    /// sum0: accumulated synaptic currents (weighted by tau_syn decay)
    /// sum1: accumulated membrane potentials (weighted by tau_mem decay)
    /// </summary>
    /// <param name="w">Weight matrix (n_pre x n_post), row-major.</param>
    /// <param name="times">Spike times, sorted ascending.</param>
    /// <param name="sources">Source neuron index for each spike.</param>
    /// <param name="tauMem">Membrane time constant.</param>
    /// <param name="tauSyn">Synaptic time constant.</param>
    /// <param name="sum0">Output buffer for sum0 matrix ((n_spikes+1) x n_post).</param>
    /// <param name="sum1">Output buffer for sum1 matrix ((n_spikes+1) x n_post).</param>
    /// <param name="nPost">Number of post-synaptic neurons.</param>
    /// <param name="expInputMem">Scratch buffer for exp(t/tau_mem), length >= times.Length.</param>
    /// <param name="expInputSyn">Scratch buffer for exp(t/tau_syn), length >= times.Length.</param>
    public static void ComputeSums(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        double tauMem,
        double tauSyn,
        Span<double> sum0,
        Span<double> sum1,
        int nPost,
        Span<double> expInputMem,
        Span<double> expInputSyn)
    {
        SpanAlgorithms.ExpDivScalar(times, tauMem, expInputMem);
        SpanAlgorithms.ExpDivScalar(times, tauSyn, expInputSyn);

        int nSpikes = times.Length;

        // Row 0 is all zeros (already zeroed by caller)
        for (int i = 0; i < nSpikes; i++)
        {
            int src = sources[i];
            double expSyn = expInputSyn[i];
            double expMem = expInputMem[i];

            int rowOut = i + 1;
            int rowIn = i;
            int srcOffset = src * nPost;

            // SIMD-accelerated FMA row update: dest[j] = prev[j] + scalar * wRow[j]
            SimdAlgorithms.FmaRow(
                sum0.Slice(rowOut * nPost, nPost),
                sum0.Slice(rowIn * nPost, nPost),
                w.Slice(srcOffset, nPost),
                expSyn);

            SimdAlgorithms.FmaRow(
                sum1.Slice(rowOut * nPost, nPost),
                sum1.Slice(rowIn * nPost, nPost),
                w.Slice(srcOffset, nPost),
                expMem);
        }
    }

    /// <summary>
    /// Compute the membrane potential maximum time and value for each neuron.
    /// Zero-allocation when scratch buffers are provided and sized appropriately.
    /// </summary>
    public static MaximaData ComputeMaxima(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        int nPost,
        double tauMem,
        double tauSyn,
        ScratchBuffers buffers)
    {
        int nSpikes = times.Length;

        // Ensure scratch buffers are sized for this computation
        int nPre = w.Length / nPost;
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: nSpikes);
        buffers.Clear();

        // Use scratch buffers for sum matrices (already sized by EnsureCapacity)
        int nSums = (nSpikes + 1) * nPost;

        ComputeSums(w, times, sources, tauMem, tauSyn,
            buffers.Sum0.AsSpan(0, nSums),
            buffers.Sum1.AsSpan(0, nSums),
            nPost,
            buffers.ExpInputMem.AsSpan(0, nSpikes),
            buffers.ExpInputSyn.AsSpan(0, nSpikes));

        // Allocate result arrays (unavoidable — caller needs to own them)
        var maximaValues = new double[nPost];
        var maximaTimes = new double[nPost];
        for (int i = 0; i < nPost; i++) maximaTimes[i] = double.NaN;

        // Parallel across post-synaptic neurons (each neuron is independent)
        ComputeMaximaParallel(times, nPost, tauMem, tauSyn,
            buffers.Sum0, buffers.Sum1, nSums / nPost,
            maximaTimes, maximaValues);

        return new MaximaData(maximaTimes, maximaValues);
    }

    /// <summary>
    /// Zero-allocation variant of ComputeMaxima that writes into caller-provided spans.
    /// </summary>
    public static void ComputeMaximaInto(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        int nPost,
        double tauMem,
        double tauSyn,
        Span<double> outMaximaTimes,
        Span<double> outMaximaValues,
        ScratchBuffers buffers)
    {
        int nSpikes = times.Length;

        int nPre = w.Length / nPost;
        buffers.EnsureCapacity(nNeurons: nPost, nPre: nPre, nSpikes: nSpikes);
        buffers.Clear();

        int nSums = (nSpikes + 1) * nPost;

        ComputeSums(w, times, sources, tauMem, tauSyn,
            buffers.Sum0.AsSpan(0, nSums),
            buffers.Sum1.AsSpan(0, nSums),
            nPost,
            buffers.ExpInputMem.AsSpan(0, nSpikes),
            buffers.ExpInputSyn.AsSpan(0, nSpikes));

        outMaximaTimes.Fill(double.NaN);

        // Use scratch buffers for parallel write, then copy back to output spans
        double[] mtScratch = buffers.LambdaV;
        double[] mvScratch = buffers.LambdaI;

        ComputeMaximaParallel(times, nPost, tauMem, tauSyn,
            buffers.Sum0, buffers.Sum1, nSums / nPost,
            mtScratch, mvScratch);

        // Copy results back to output spans (sequential, no allocation)
        for (int i = 0; i < nPost; i++)
        {
            outMaximaTimes[i] = mtScratch[i];
            outMaximaValues[i] = mvScratch[i];
        }
    }

    /// <summary>
    /// Parallel per-neuron maxima computation for ComputeMaxima.
    /// Each neuron is independent, so we parallelize across post-synaptic neurons.
    /// </summary>
    static void ComputeMaximaParallel(
        ReadOnlySpan<double> times, int nPost,
        double tauMem, double tauSyn,
        double[] sum0, double[] sum1, int nRows,
        double[] maximaTimes, double[] maximaValues)
    {
        // Convert span to array for Parallel.For lambda capture (small allocation, ~8KB at Large)
        double[] timesArr = times.ToArray();
        int nSpikes = timesArr.Length;

        double kPrefactor = tauSyn / (tauMem - tauSyn);
        double tmaxPrefactor = 1.0 / (1.0 / tauMem - 1.0 / tauSyn);
        double tmaxSummand = Math.Log(tauSyn / tauMem);

        Parallel.For(0, nPost, targetNrn =>
        {
            double bestValue = 0.0;
            double bestTime = double.NaN;

            for (int i = 0; i <= nSpikes; i++)
            {
                double tInput = (i == nSpikes) ? double.PositiveInfinity : timesArr[i];
                double tBefore = (i > 0) ? timesArr[i - 1] : 0.0;

                double sum0Elem = sum0[i * nPost + targetNrn];
                double sum1Elem = sum1[i * nPost + targetNrn];

                double tmax = tmaxPrefactor * (tmaxSummand + Math.Log(sum1Elem / sum0Elem));

                if (tBefore <= tmax && tmax <= tInput)
                {
                    double vmax = kPrefactor * (sum1Elem * Math.Exp(-tmax / tauMem) - sum0Elem * Math.Exp(-tmax / tauSyn));
                    if (vmax > bestValue)
                    {
                        bestValue = vmax;
                        bestTime = tmax;
                    }
                }

                // Also check at the input spike time itself
                double vAtInput = kPrefactor * (sum1Elem * Math.Exp(-tInput / tauMem) - sum0Elem * Math.Exp(-tInput / tauSyn));
                if (vAtInput > bestValue)
                {
                    bestValue = vAtInput;
                    bestTime = tInput == double.PositiveInfinity ? tBefore : tInput;
                }
            }

            maximaValues[targetNrn] = bestValue;
            maximaTimes[targetNrn] = bestTime;
        });
    }

    /// <summary>
    /// Forward pass: predict post-synaptic spike times given input spikes and weights.
    /// Uses the α-function voltage model with Toms748 root-finding for threshold crossings.
    /// Zero-internal-allocation when scratch buffers are provided (return arrays unavoidable).
    /// </summary>
    public static SpikesData ComputeSpikes(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        int nPost,
        double vTh,
        double tauMem,
        double tauSyn,
        ScratchBuffers buffers)
    {
        int nSpikes = times.Length;

        // Use scratch buffers for sum matrices (already sized by EnsureCapacity)
        int nSums = (nSpikes + 1) * nPost;
        buffers.EnsureCapacity(nNeurons: nPost, nPre: w.Length / nPost, nSpikes: nSpikes);
        buffers.Clear();

        ComputeSums(w, times, sources, tauMem, tauSyn,
            buffers.Sum0.AsSpan(0, nSums),
            buffers.Sum1.AsSpan(0, nSums),
            nPost,
            buffers.ExpInputMem.AsSpan(0, nSpikes),
            buffers.ExpInputSyn.AsSpan(0, nSpikes));

        // Constants for α-function
        double kPrefactor = tauSyn / (tauMem - tauSyn);
        double tmaxPrefactor = 1.0 / (1.0 / tauMem - 1.0 / tauSyn);
        double tmaxSummand = Math.Log(tauSyn / tauMem);

        // Use scratch buffers for per-neuron spike tracking
        int maxSpikesPerNeuron = Math.Max(nSpikes + 1, buffers.MaxSpikesPerNeuron);
        if (maxSpikesPerNeuron * nPost > buffers.PostSpikeTimes.Length)
        {
            buffers.GrowPostSpikeBuffers(nPost, maxSpikesPerNeuron);
        }

        // For each post-synaptic neuron, find threshold crossings (parallelized)
        ComputeSpikesParallel(times, sources, nPost, vTh, tauMem, tauSyn,
            kPrefactor, tmaxPrefactor, tmaxSummand, maxSpikesPerNeuron,
            buffers);

        // Count total spikes and build output from scratch buffers (single allocation)
        int totalSpikes = 0;
        for (int n = 0; n < nPost; n++)
            totalSpikes += buffers.SpikeCountPerNeuron[n];

        if (totalSpikes == 0)
            return new SpikesData([], []);

        // Use pre-allocated SpikeEntries buffer for sorting (no intermediate allocation)
        buffers.EnsureOutputCapacity(totalSpikes);
        Span<SpikeEntry> entries = buffers.SpikeEntries.AsSpan(0, totalSpikes);
        int idx = 0;
        for (int n = 0; n < nPost; n++)
        {
            int count = buffers.SpikeCountPerNeuron[n];
            int offset = n * maxSpikesPerNeuron;
            for (int k = 0; k < count; k++)
            {
                entries[idx++] = new SpikeEntry(
                    buffers.PostSpikeTimes[offset + k], n,
                    buffers.PostSpikeCurrents[offset + k]);
            }
        }

        Array.Sort(buffers.SpikeEntries, 0, totalSpikes);

        var sortedTimes = new double[totalSpikes];
        var sortedSources = new int[totalSpikes];
        var sortedCurrents = new double[totalSpikes];
        for (int i = 0; i < totalSpikes; i++)
        {
            sortedTimes[i] = entries[i].Time;
            sortedSources[i] = entries[i].Source;
            sortedCurrents[i] = entries[i].Current;
        }

        return new SpikesData(sortedTimes, sortedSources, sortedCurrents);
    }

    /// <summary>
    /// Zero-allocation forward pass: writes output spike times/sources/currents
    /// directly into caller-provided spans. Returns the number of spikes written.
    /// </summary>
    public static int ComputeSpikesInto(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        int nPost,
        double vTh,
        double tauMem,
        double tauSyn,
        Span<double> outputTimes,
        Span<int> outputSources,
        Span<double> outputCurrents,
        ScratchBuffers buffers)
    {
        int nSpikes = times.Length;

        int nSums = (nSpikes + 1) * nPost;
        buffers.EnsureCapacity(nNeurons: nPost, nPre: w.Length / nPost, nSpikes: nSpikes);
        buffers.Clear();

        ComputeSums(w, times, sources, tauMem, tauSyn,
            buffers.Sum0.AsSpan(0, nSums),
            buffers.Sum1.AsSpan(0, nSums),
            nPost,
            buffers.ExpInputMem.AsSpan(0, nSpikes),
            buffers.ExpInputSyn.AsSpan(0, nSpikes));

        double kPrefactor = tauSyn / (tauMem - tauSyn);
        double tmaxPrefactor = 1.0 / (1.0 / tauMem - 1.0 / tauSyn);
        double tmaxSummand = Math.Log(tauSyn / tauMem);

        int maxSpikesPerNeuron = Math.Max(nSpikes + 1, buffers.MaxSpikesPerNeuron);
        if (maxSpikesPerNeuron * nPost > buffers.PostSpikeTimes.Length)
            buffers.GrowPostSpikeBuffers(nPost, maxSpikesPerNeuron);

        ComputeSpikesParallel(times, sources, nPost, vTh, tauMem, tauSyn,
            kPrefactor, tmaxPrefactor, tmaxSummand, maxSpikesPerNeuron, buffers);

        int totalSpikes = 0;
        for (int n = 0; n < nPost; n++)
            totalSpikes += buffers.SpikeCountPerNeuron[n];

        if (totalSpikes == 0)
            return 0;

        // Use pre-allocated entry buffer for sorting
        Span<SpikeEntry> entries = buffers.SpikeEntries.AsSpan(0, totalSpikes);
        int idx = 0;
        for (int n = 0; n < nPost; n++)
        {
            int count = buffers.SpikeCountPerNeuron[n];
            int offset = n * maxSpikesPerNeuron;
            for (int k = 0; k < count; k++)
            {
                entries[idx++] = new SpikeEntry(
                    buffers.PostSpikeTimes[offset + k], n,
                    buffers.PostSpikeCurrents[offset + k]);
            }
        }

        Array.Sort(buffers.SpikeEntries, 0, totalSpikes);

        for (int i = 0; i < totalSpikes; i++)
        {
            outputTimes[i] = entries[i].Time;
            outputSources[i] = entries[i].Source;
            outputCurrents[i] = entries[i].Current;
        }

        return totalSpikes;
    }

    /// <summary>
    /// Parallel per-neuron spike computation for ComputeSpikes.
    /// Each neuron is independent, so we parallelize across post-synaptic neurons.
    /// </summary>
    static void ComputeSpikesParallel(
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        int nPost, double vTh, double tauMem, double tauSyn,
        double kPrefactor, double tmaxPrefactor, double tmaxSummand,
        int maxSpikesPerNeuron, ScratchBuffers buffers)
    {
        // Convert span to array for Parallel.For lambda capture (small allocation, ~8KB at Large)
        double[] timesArr = times.ToArray();
        int nSpikes = timesArr.Length;

        Parallel.For(0, nPost, targetNrn =>
        {
            int neuronOffset = targetNrn * maxSpikesPerNeuron;
            int neuronSpikeCount = 0;

            bool finished = false;
            int processedUpTo = 0;

            // Precomputed exp(st * tauMemInv) for factorized ComputeV
            double[] postExpSt = buffers.PostSpikeExpSt;

            while (!finished)
            {
                bool foundSpikeInThisPass = false;
                for (int i = processedUpTo; i <= nSpikes; i++)
                {
                    if (i == nSpikes)
                    {
                        double sum0Val = buffers.Sum0[nSpikes * nPost + targetNrn];
                        double sum1Val = buffers.Sum1[nSpikes * nPost + targetNrn];
                        for (int s = 0; s < neuronSpikeCount; s++)
                            sum1Val += -vTh * Math.Exp(buffers.PostSpikeTimes[neuronOffset + s] / tauMem);
                        double tmax = tmaxPrefactor * (tmaxSummand + Math.Log(sum1Val / sum0Val));
                        if (!double.IsInfinity(tmax) && !double.IsNaN(tmax))
                        {
                            // Precompute exp(st * tauMemInv) for factorized ComputeV

                            for (int s = 0; s < neuronSpikeCount; s++)
                                postExpSt[neuronOffset + s] = Math.Exp(buffers.PostSpikeTimes[neuronOffset + s] / tauMem);

                            double vmax = ComputeVFactored(tmax, i, neuronSpikeCount,
                                buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                targetNrn, nPost, kPrefactor, 1.0 / tauMem, 1.0 / tauSyn, vTh, postExpSt);
                            if (vmax > vTh + 1e-14)
                            {
                                double tLower = GetTLower(neuronSpikeCount, buffers.PostSpikeTimes, neuronOffset, tmax);
                                int solveSpikeCount = neuronSpikeCount;
                                double tPost = Toms748.Solve(
                                    t => ComputeVFactored(t, i, solveSpikeCount,
                                        buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                        targetNrn, nPost, kPrefactor, 1.0 / tauMem, 1.0 / tauSyn, vTh, postExpSt) - vTh,
                                    tLower, tmax);
                                if (tPost > tLower && tPost < tmax)
                                {
                                    double current = ComputeI(tPost, nSpikes, timesArr,
                                        buffers.Sum0, targetNrn, nPost, tauSyn);
                                    if (neuronOffset + neuronSpikeCount < buffers.PostSpikeTimes.Length)
                                    {
                                        buffers.PostSpikeTimes[neuronOffset + neuronSpikeCount] = tPost;
                                        buffers.PostSpikeCurrents[neuronOffset + neuronSpikeCount] = current;
                                        neuronSpikeCount++;
                                        foundSpikeInThisPass = true;
                                    }
                                }
                            }
                        }
                        if (!foundSpikeInThisPass) finished = true;
                        else break;
                    }
                    else
                    {
                        double tInput = timesArr[i];
                        double tmax = ComputeTmax(i, nSpikes, timesArr, neuronSpikeCount,
                            buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                            targetNrn, nPost, tmaxPrefactor, tmaxSummand, tauMem, vTh);

                        if (tmax != double.PositiveInfinity)
                        {
                            // Precompute exp(st * tauMemInv) for factorized ComputeV

                            for (int s = 0; s < neuronSpikeCount; s++)
                                postExpSt[neuronOffset + s] = Math.Exp(buffers.PostSpikeTimes[neuronOffset + s] / tauMem);

                            double vmax = ComputeVFactored(tmax, i, neuronSpikeCount,
                                buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                targetNrn, nPost, kPrefactor, 1.0 / tauMem, 1.0 / tauSyn, vTh, postExpSt);
                            if (vmax > vTh + 1e-14)
                            {
                                double tBefore = (i > 0) ? timesArr[i - 1] : 0.0;
                                for (int s = 0; s < neuronSpikeCount; s++)
                                {
                                    double spikeTime = buffers.PostSpikeTimes[neuronOffset + s];
                                    if (spikeTime > tInput) break;
                                    if (tBefore < spikeTime)
                                        tBefore = spikeTime;
                                }

                                int solveSpikeCount = neuronSpikeCount;
                                double tPost = Toms748.Solve(
                                    t => ComputeVFactored(t, i, solveSpikeCount,
                                        buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                        targetNrn, nPost, kPrefactor, 1.0 / tauMem, 1.0 / tauSyn, vTh, postExpSt) - vTh,
                                    tBefore, tmax);

                                if (tPost > tBefore && tPost < tmax)
                                {
                                    double current = ComputeI(tPost, nSpikes, timesArr,
                                        buffers.Sum0, targetNrn, nPost, tauSyn);
                                    if (neuronOffset + neuronSpikeCount < buffers.PostSpikeTimes.Length)
                                    {
                                        buffers.PostSpikeTimes[neuronOffset + neuronSpikeCount] = tPost;
                                        buffers.PostSpikeCurrents[neuronOffset + neuronSpikeCount] = current;
                                        neuronSpikeCount++;
                                        foundSpikeInThisPass = true;
                                    }
                                }
                                processedUpTo = i;
                                break;
                            }
                        }

                        // Precompute exp(st * tauMemInv) for factorized ComputeV

                        for (int s = 0; s < neuronSpikeCount; s++)
                            postExpSt[neuronOffset + s] = Math.Exp(buffers.PostSpikeTimes[neuronOffset + s] / tauMem);

                        double vAtInput = ComputeVFactored(tInput, i, neuronSpikeCount,
                            buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                            targetNrn, nPost, kPrefactor, 1.0 / tauMem, 1.0 / tauSyn, vTh, postExpSt);
                        if (vAtInput > vTh + 1e-14)
                        {
                            double tBefore = (i > 0) ? timesArr[i - 1] : 0.0;
                            for (int s = 0; s < neuronSpikeCount; s++)
                            {
                                double spikeTime = buffers.PostSpikeTimes[neuronOffset + s];
                                if (spikeTime > tInput) break;
                                if (tBefore < spikeTime)
                                    tBefore = spikeTime;
                            }

                            int solveSpikeCount = neuronSpikeCount;
                            double tPost = Toms748.Solve(
                                t => ComputeVFactored(t, i, solveSpikeCount,
                                    buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                    targetNrn, nPost, kPrefactor, 1.0 / tauMem, 1.0 / tauSyn, vTh, postExpSt) - vTh,
                                tBefore, tInput);

                            if (tPost > tBefore && tPost < tInput)
                            {
                                double current = ComputeI(tPost, nSpikes, timesArr,
                                    buffers.Sum0, targetNrn, nPost, tauSyn);
                                if (neuronOffset + neuronSpikeCount < buffers.PostSpikeTimes.Length)
                                {
                                    buffers.PostSpikeTimes[neuronOffset + neuronSpikeCount] = tPost;
                                    buffers.PostSpikeCurrents[neuronOffset + neuronSpikeCount] = current;
                                    neuronSpikeCount++;
                                    foundSpikeInThisPass = true;
                                }
                            }
                            processedUpTo = i;
                            break;
                        }
                    }
                }

                if (!foundSpikeInThisPass)
                    processedUpTo = Math.Min(processedUpTo + 1, nSpikes);
            }

            buffers.SpikeCountPerNeuron[targetNrn] = neuronSpikeCount;
        });
    }

    /// <summary>
    /// Adjoint method backward pass for spike-time-based loss.
    /// Computes weight gradients and input spike errors by propagating λ_v (membrane)
    /// and λ_i (synaptic) adjoints backward through the merged input/post-spike timeline.
    /// Zero-internal-allocation when scratch buffers are provided (MergedSpike[] and error
    /// arrays use pre-allocated ScratchBuffers; return values are in-place via gradient span).
    /// </summary>
    public static void BackwardSpikes(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> inputTimes,
        ReadOnlySpan<int> inputSources,
        SpikesData postSpikes,
        int nPost,
        double vTh,
        double tauMem,
        double tauSyn,
        Span<double> gradient,
        ScratchBuffers buffers)
    {
        int nPre = w.Length / nPost;
        int nInputSpikes = inputTimes.Length;
        int nPostSpikes = postSpikes.NSpikes;

        // No post-spike activity — nothing to compute
        if (nPostSpikes == 0)
            return;

        // Ensure scratch buffers are sized
        int totalSpikes = nInputSpikes + nPostSpikes;
        // Size for max(nPre, nPost) since spikeSource can be either pre- or post-synaptic index
        int maxNeurons = nPre > nPost ? nPre : nPost;
        buffers.EnsureCapacity(nNeurons: maxNeurons, nPre: nPre, nSpikes: totalSpikes);
        // NOTE: no buffers.Clear() — BackwardSpikes initializes all buffers it uses explicitly.

        // Merge input + post-spike indices and sort by time descending (use pre-allocated buffer)
        Span<MergedSpike> merged = buffers.MergedSpikes.AsSpan(0, totalSpikes);
        for (int i = 0; i < nInputSpikes; i++)
            merged[i] = new MergedSpike(i, true, inputTimes[i]);
        for (int i = 0; i < nPostSpikes; i++)
            merged[nInputSpikes + i] = new MergedSpike(i, false, postSpikes.Times[i]);

        Array.Sort(buffers.MergedSpikes, 0, totalSpikes);

        // Combined error array: input spike errors [0..nInputSpikes-1] + post-spike errors [nInputSpikes..totalSpikes-1].
        double[] allErrors = buffers.SpikeErrors;
        Array.Fill(allErrors, 0.0);
        for (int i = 0; i < nPostSpikes; i++)
            allErrors[nInputSpikes + i] = postSpikes.Errors[i];

        // Initialize λ_v = 0, previous_t = largest_time
        double[] lambdaV = buffers.LambdaV;
        Array.Fill(lambdaV, 0.0);

        double largestTime = merged[0].Time;
        double previousT = largestTime;

        // Backward prefactor (inlined, no lambda)
        double kBwdPrefactor = tauMem / (tauMem - tauSyn);

        // Pre-count post-spike jumps per neuron → compute flat buffer layout.
        // Each post-synaptic spike generates exactly one jump for its source neuron.
        int[] jumpCount = buffers.LambdaIJumpsNeuronCap; // reuse as per-neuron count
        Array.Fill(jumpCount, 0);
        for (int i = 0; i < nPostSpikes; i++)
            jumpCount[postSpikes.Sources[i]]++;

        // Compute start indices (prefix sum) into flat jump buffers
        int[] jumpStart = buffers.LambdaIJumpsStart;
        int[] jumpCursor = buffers.LambdaIJumpsEnd; // reuse as write cursor
        jumpStart[0] = 0;
        for (int nrn = 0; nrn < nPost; nrn++)
        {
            jumpStart[nrn + 1] = jumpStart[nrn] + jumpCount[nrn];
            jumpCursor[nrn] = jumpStart[nrn]; // cursor starts at beginning
        }

        // Flat jump buffers (pre-allocated in ScratchBuffers, sized to nPostSpikes)
        double[] flatJumpTime = buffers.LambdaIJumpsTimeOffset;
        double[] flatJumpMag = buffers.LambdaIJumpsMagnitude;

        // Main backward sweep through merged timeline (single pass, matching C++ structure)
        for (int s = 0; s < totalSpikes; s++)
        {
            int spikeIdx = merged[s].Idx;
            bool isInput = merged[s].IsInput;
            double spikeTime = merged[s].Time;
            int spikeSource = isInput ? inputSources[spikeIdx] : postSpikes.Sources[spikeIdx];

            // Time elapsed since largest time (increases as we go backward)
            double tBwd = largestTime - spikeTime;

            // Propagate λ_v backward through membrane decay
            double dt = previousT - tBwd;
            SimdAlgorithms.ScaleInPlace(lambdaV.AsSpan(0, nPost), Math.Exp(dt / tauMem));

            if (isInput)
            {
                // Input spike: compute gradient and outbound signal
                double[] lambdaI = buffers.LambdaI;
                Array.Fill(lambdaI, 0.0);

                // Compute λ_i from accumulated jumps for EACH post-synaptic neuron.
                // Jumps are stored in flat buffers with per-neuron start/cursor indices.
                for (int nrn = 0; nrn < nPost; nrn++)
                {
                    int start = jumpStart[nrn];
                    int end = jumpCursor[nrn];
                    for (int j = start; j < end; j++)
                    {
                        lambdaI[nrn] += flatJumpMag[j] * kBwdPrefactor *
                            (Math.Exp(-(tBwd - flatJumpTime[j]) / tauMem)
                             - Math.Exp(-(tBwd - flatJumpTime[j]) / tauSyn));
                    }
                }

                // SIMD: gradient[source, nrn] += -tau_syn * lambda_i[nrn]
                SimdAlgorithms.AddScaled(
                    gradient.Slice(spikeSource * nPost, nPost),
                    lambdaI.AsSpan(0, nPost),
                    -tauSyn);

                // SIMD: outbound = w[source] · (lambda_v - lambda_i)
                // Reuse lambdaI as diff buffer after gradient is accumulated
                SimdAlgorithms.Subtract(
                    lambdaI.AsSpan(0, nPost),
                    lambdaV.AsSpan(0, nPost),
                    lambdaI.AsSpan(0, nPost));
                double outboundSignal = SimdAlgorithms.Dot(
                    w.Slice(spikeSource * nPost, nPost),
                    lambdaI.AsSpan(0, nPost));

                // Add outbound signal to input spike error
                allErrors[spikeIdx] += outboundSignal;
            }
            else
            {
                // Post-synaptic spike: record λ_i jump and update λ_v
                double current = postSpikes.Currents[spikeIdx];
                double invICurrent = 1.0 / (current - vTh);

                // Record λ_i jump for this neuron in flat buffer.
                int jIdx = jumpCursor[spikeSource]++;
                double postError = allErrors[nInputSpikes + spikeIdx]; // post-spikes stored at offset nInputSpikes
                flatJumpMag[jIdx] =
                    invICurrent * (vTh * lambdaV[spikeSource] + postError);
                flatJumpTime[jIdx] = tBwd;

                // Update λ_v for this neuron
                lambdaV[spikeSource] =
                    (current / (current - vTh)) * lambdaV[spikeSource] +
                    invICurrent * postError;
            }

            previousT = tBwd;
        }
    }

    /// <summary>
    /// Zero-allocation backward pass: accepts raw spans instead of SpikesData.
    /// </summary>
    public static void BackwardSpikesRaw(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> inputTimes,
        ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> postTimes,
        ReadOnlySpan<int> postSources,
        ReadOnlySpan<double> postCurrents,
        ReadOnlySpan<double> postErrors,
        int nPost,
        double vTh,
        double tauMem,
        double tauSyn,
        Span<double> gradient,
        ScratchBuffers buffers)
    {
        int nPre = w.Length / nPost;
        int nInputSpikes = inputTimes.Length;
        int nPostSpikes = postTimes.Length;

        if (nPostSpikes == 0)
            return;

        int totalSpikes = nInputSpikes + nPostSpikes;
        int maxNeurons = nPre > nPost ? nPre : nPost;
        buffers.EnsureCapacity(nNeurons: maxNeurons, nPre: nPre, nSpikes: totalSpikes);

        // Merge input + post-spike indices and sort by time descending
        Span<MergedSpike> merged = buffers.MergedSpikes.AsSpan(0, totalSpikes);
        for (int i = 0; i < nInputSpikes; i++)
            merged[i] = new MergedSpike(i, true, inputTimes[i]);
        for (int i = 0; i < nPostSpikes; i++)
            merged[nInputSpikes + i] = new MergedSpike(i, false, postTimes[i]);

        Array.Sort(buffers.MergedSpikes, 0, totalSpikes);

        // Combined error array
        double[] allErrors = buffers.SpikeErrors;
        Array.Fill(allErrors, 0.0);
        for (int i = 0; i < nPostSpikes; i++)
            allErrors[nInputSpikes + i] = postErrors[i];

        // Initialize λ_v = 0
        double[] lambdaV = buffers.LambdaV;
        Array.Fill(lambdaV, 0.0);

        double largestTime = merged[0].Time;
        double previousT = largestTime;
        double kBwdPrefactor = tauMem / (tauMem - tauSyn);

        // Pre-count post-spike jumps per neuron → compute flat buffer layout
        int[] jumpCount = buffers.LambdaIJumpsNeuronCap;
        Array.Fill(jumpCount, 0);
        for (int i = 0; i < nPostSpikes; i++)
            jumpCount[postSources[i]]++;

        int[] jumpStart = buffers.LambdaIJumpsStart;
        int[] jumpCursor = buffers.LambdaIJumpsEnd;
        jumpStart[0] = 0;
        for (int nrn = 0; nrn < nPost; nrn++)
        {
            jumpStart[nrn + 1] = jumpStart[nrn] + jumpCount[nrn];
            jumpCursor[nrn] = jumpStart[nrn];
        }

        double[] flatJumpTime = buffers.LambdaIJumpsTimeOffset;
        double[] flatJumpMag = buffers.LambdaIJumpsMagnitude;

        for (int s = 0; s < totalSpikes; s++)
        {
            int spikeIdx = merged[s].Idx;
            bool isInput = merged[s].IsInput;
            double spikeTime = merged[s].Time;
            int spikeSource = isInput ? inputSources[spikeIdx] : postSources[spikeIdx];

            double tBwd = largestTime - spikeTime;
            double dt = previousT - tBwd;
            SimdAlgorithms.ScaleInPlace(lambdaV.AsSpan(0, nPost), Math.Exp(dt / tauMem));

            if (isInput)
            {
                double[] lambdaI = buffers.LambdaI;
                Array.Fill(lambdaI, 0.0);

                for (int nrn = 0; nrn < nPost; nrn++)
                {
                    int start = jumpStart[nrn];
                    int end = jumpCursor[nrn];
                    for (int j = start; j < end; j++)
                    {
                        lambdaI[nrn] += flatJumpMag[j] * kBwdPrefactor *
                            (Math.Exp(-(tBwd - flatJumpTime[j]) / tauMem)
                             - Math.Exp(-(tBwd - flatJumpTime[j]) / tauSyn));
                    }
                }

                // SIMD: gradient[source, nrn] += -tau_syn * lambda_i[nrn]
                SimdAlgorithms.AddScaled(
                    gradient.Slice(spikeSource * nPost, nPost),
                    lambdaI.AsSpan(0, nPost),
                    -tauSyn);

                // SIMD: outbound = w[source] · (lambda_v - lambda_i)
                SimdAlgorithms.Subtract(
                    lambdaI.AsSpan(0, nPost),
                    lambdaV.AsSpan(0, nPost),
                    lambdaI.AsSpan(0, nPost));
                double outboundSignal = SimdAlgorithms.Dot(
                    w.Slice(spikeSource * nPost, nPost),
                    lambdaI.AsSpan(0, nPost));

                allErrors[spikeIdx] += outboundSignal;
            }
            else
            {
                double current = postCurrents[spikeIdx];
                double invICurrent = 1.0 / (current - vTh);

                int jIdx = jumpCursor[spikeSource]++;
                double postError = allErrors[nInputSpikes + spikeIdx];
                flatJumpMag[jIdx] = invICurrent * (vTh * lambdaV[spikeSource] + postError);
                flatJumpTime[jIdx] = tBwd;

                lambdaV[spikeSource] =
                    (current / (current - vTh)) * lambdaV[spikeSource] +
                    invICurrent * postError;
            }

            previousT = tBwd;
        }
    }

    /// <summary>
    /// Computes weight gradients using adjoint method with maxima-based loss.
    /// Propagates error signals from membrane potential maxima backward through the input spike timeline.
    /// Zero-allocation implementation using Span&lt;double&gt; throughout.
    /// </summary>
    /// <param name="inputTimes">Input spike times (ascending order)</param>
    /// <param name="inputSources">Source neuron index for each input spike</param>
    /// <param name="maximaTimes">Membrane potential maxima times per post-synaptic neuron (n_maxima == n_post)</param>
    /// <param name="maximaValues">Membrane potential values at maxima (not used in gradient computation)</param>
    /// <param name="maximaErrors">Error signals at each maxima (dL/dV_max)</param>
    /// <param name="w">Weight matrix (row-major: w[source * nPost + nrn])</param>
    /// <param name="nPost">Number of post-synaptic neurons</param>
    /// <param name="tauMem">Membrane time constant (seconds)</param>
    /// <param name="tauSyn">Synaptic time constant (seconds)</param>
    /// <param name="gradient">Output gradient array (row-major, caller zero-initialized)</param>
    public static void BackwardMaxima(
        ReadOnlySpan<double> inputTimes,
        ReadOnlySpan<int> inputSources,
        ReadOnlySpan<double> maximaTimes,
        ReadOnlySpan<double> maximaValues, // Not used — gradient depends on maximaTimes and errors only
        ReadOnlySpan<double> maximaErrors,
        ReadOnlySpan<double> w,
        int nPost,
        double tauMem,
        double tauSyn,
        Span<double> gradient)
    {
        int nInputSpikes = inputTimes.Length;
        int nMaxima = maximaTimes.Length;

        double largestTime = inputTimes[nInputSpikes - 1];
        double kBwdPrefactor = tauMem / (tauMem - tauSyn);

        // Sweep backward through input spikes
        for (int spikeIdx = nInputSpikes - 1; spikeIdx >= 0; spikeIdx--)
        {
            double tBwd = largestTime - inputTimes[spikeIdx];

            for (int maxIdx = 0; maxIdx < nMaxima; maxIdx++)
            {
                if (double.IsNaN(maximaTimes[maxIdx])) continue;

                double tVmaxBwd = largestTime - maximaTimes[maxIdx];
                if (tVmaxBwd > tBwd) continue;

                double lambdaV = -Math.Exp(-(tBwd - tVmaxBwd) / tauMem) * maximaErrors[maxIdx] / tauMem;
                double kBwdVal = kBwdPrefactor * (Math.Exp(-(tBwd - tVmaxBwd) / tauMem) - Math.Exp(-(tBwd - tVmaxBwd) / tauSyn));
                double lambdaI = -kBwdVal * maximaErrors[maxIdx] / tauMem;

                int sourceNrn = inputSources[spikeIdx];
                gradient[sourceNrn * nPost + maxIdx] -= tauSyn * lambdaI;
            }
        }
    }

    // --- Static helpers (no lambda capture of ReadOnlySpan) ---

    /// <summary>
    /// Compute membrane potential at time t for interval i.
    /// </summary>
    static double ComputeV(double t, int i, int spikeCount,
        double[] postSpikeTimes, int neuronOffset,
        double[] sum0, double[] sum1,
        int targetNrn, int nPost,
        double kPrefactor, double tauMemInv, double tauSynInv, double vTh)
    {
        double mem = kPrefactor * (sum1[i * nPost + targetNrn] * Math.Exp(-t * tauMemInv)
                        - sum0[i * nPost + targetNrn] * Math.Exp(-t * tauSynInv));
        for (int s = 0; s < spikeCount; s++)
        {
            double spikeTime = postSpikeTimes[neuronOffset + s];
            if (spikeTime <= t)  // Include all spikes where spikeTime <= t, don't break early
            {
                mem -= vTh * Math.Exp(-(t - spikeTime) * tauMemInv);
            }
        }
        return mem;
    }

    /// <summary>
    /// Factorized version of ComputeV.
    /// Precomputed postSpikeExpSt[s] = exp(postSpikeTimes[neuronOffset+s] * tauMemInv).
    /// Factorization: exp(-(t-st)/tau) = exp(-t/tau) * exp(st/tau)
    /// Eliminates N exp() calls per iteration to just 1 (shared with base terms).
    /// </summary>
    static double ComputeVFactored(double t, int i, int spikeCount,
        double[] postSpikeTimes, int neuronOffset,
        double[] sum0, double[] sum1,
        int targetNrn, int nPost,
        double kPrefactor, double tauMemInv, double tauSynInv, double vTh,
        double[] postSpikeExpSt)
    {
        double eT = Math.Exp(-t * tauMemInv);
        double mem = kPrefactor * (sum1[i * nPost + targetNrn] * eT
                        - sum0[i * nPost + targetNrn] * Math.Exp(-t * tauSynInv));
        if (postSpikeExpSt != null)
        {
            for (int s = 0; s < spikeCount; s++)
            {
                if (postSpikeTimes[neuronOffset + s] <= t)
                    mem -= vTh * eT * postSpikeExpSt[neuronOffset + s];
            }
        }
        else
        {
            for (int s = 0; s < spikeCount; s++)
            {
                double spikeTime = postSpikeTimes[neuronOffset + s];
                if (spikeTime <= t)
                    mem -= vTh * Math.Exp(-(t - spikeTime) * tauMemInv);
            }
        }
        return mem;
    }

    /// <summary>
    /// Find time of maximum voltage in interval i.
    /// </summary>
    static double ComputeTmax(int i, int nSpikes, ReadOnlySpan<double> times,
        int spikeCount, double[] postSpikeTimes, int neuronOffset,
        double[] sum0, double[] sum1,
        int targetNrn, int nPost,
        double tmaxPrefactor, double tmaxSummand, double tauMem, double vTh)
    {
        if (i == 0) return 0.0;

        double tInput = (i == nSpikes) ? double.PositiveInfinity : times[i];
        double tBefore = (i > 0) ? times[i - 1] : 0.0;

        // Update tBefore to include previous output spikes before this interval
        for (int s = 0; s < spikeCount; s++)
        {
            double spikeTime = postSpikeTimes[neuronOffset + s];
            if (spikeTime > tInput) break;
            if (spikeTime > tBefore)
                tBefore = spikeTime;
        }

        double sum0Elem = sum0[i * nPost + targetNrn];
        double sum1Elem = sum1[i * nPost + targetNrn];

        // Skip if either sum is zero or negative (no valid tmax)
        if (sum0Elem <= 0 || sum1Elem <= 0) return double.PositiveInfinity;

        // Adjust sum1 for previous output spikes (they reduce the membrane potential)
        double sum1Adj = sum1Elem;
        for (int s = 0; s < spikeCount; s++)
        {
            double spikeTime = postSpikeTimes[neuronOffset + s];
            if (spikeTime > tInput) break;
            sum1Adj += -vTh * Math.Exp(spikeTime / tauMem);
        }

        double tmax = tmaxPrefactor * (tmaxSummand + Math.Log(sum1Adj / sum0Elem));

        if (tBefore <= tmax && (i == nSpikes || tmax <= tInput))
            return tmax;
        return double.PositiveInfinity;
    }

    /// <summary>
    /// Compute synaptic current at time t.
    /// </summary>
    static double ComputeI(double t, int nSpikes, ReadOnlySpan<double> times,
        double[] sum0, int targetNrn, int nPost, double tauSyn)
    {
        int idx = 0;
        for (; idx < nSpikes; idx++)
        {
            if (times[idx] > t) break;
        }
        return sum0[idx * nPost + targetNrn] * Math.Exp(-t / tauSyn);
    }

    /// <summary>
    /// Get lower bracket time for root-finding.
    /// </summary>
    static double GetTLower(int spikeCount, double[] postSpikeTimes, int neuronOffset, double tUpper)
    {
        if (spikeCount > 0)
        {
            return postSpikeTimes[neuronOffset + spikeCount - 1];
        }
        return 0.0;
    }

    // ── Diagnostic methods (zero-allocation hot path) ───────────────────

    /// <summary>
    /// Compute the membrane potential trace for a single neuron over time.
    /// Zero-allocation version — uses caller-provided ScratchBuffers for all intermediate storage.
    /// Output arrays (times, voltages) are still allocated since this method returns data.
    /// </summary>
    public static (double[] times, double[] voltages) ComputeVoltageTrace(
        double tMax, double dt,
        int targetNrnIdx,
        ReadOnlySpan<double> w, int nPre, int nPost,
        ReadOnlySpan<double> times, ReadOnlySpan<int> sources,
        double vTh, double tauMem, double tauSyn,
        ScratchBuffers buffers)
    {
        int nSpikes = times.Length;

        // Use caller-provided buffers — no allocations.
        // Sum0/Sum1 are the same buffers used by ComputeSums.
        int nSums = (nSpikes + 1) * nPost;
        buffers.EnsureCapacity(nPost, nPre, nSpikes);

        Span<double> sum0 = buffers.Sum0.AsSpan(0, nSums);
        Span<double> sum1 = buffers.Sum1.AsSpan(0, nSums);

        // Clear sums before accumulation (caller may reuse buffers)
        sum0.Clear();
        sum1.Clear();

        // Compute exp values in-place using pre-allocated buffers.
        Span<double> expMem = buffers.ExpInputMem.AsSpan(0, nSpikes);
        for (int i = 0; i < nSpikes; i++)
            expMem[i] = Math.Exp(times[i] / tauMem);

        Span<double> expSyn = buffers.ExpInputSyn.AsSpan(0, nSpikes);
        for (int i = 0; i < nSpikes; i++)
            expSyn[i] = Math.Exp(times[i] / tauSyn);

        // Accumulate sums (same logic as ComputeSums)
        for (int i = 0; i < nSpikes; i++)
        {
            int src = sources[i];
            double expSynI = expSyn[i], expMemI = expMem[i];
            int rowOut = i + 1, rowIn = i;
            int srcOffset = src * nPost;

            for (int j = 0; j < nPost; j++)
            {
                sum0[rowOut * nPost + j] = sum0[rowIn * nPost + j] + w[srcOffset + j] * expSynI;
                sum1[rowOut * nPost + j] = sum1[rowIn * nPost + j] + w[srcOffset + j] * expMemI;
            }
        }

        double kPrefactor = tauSyn / (tauMem - tauSyn);
        int nT = (int)(tMax / dt) + 1;

        var ts = new double[nT];
        var vTrace = new double[nT];

        for (int ti = 0; ti < nT; ti++)
        {
            double t = dt * ti;

            // Find the number of spikes with times[i] <= t (binary search).
            int tPreIdx;
            if (nSpikes == 0)
                tPreIdx = 0;
            else
            {
                int lo = 0, hi = nSpikes - 1;
                while (lo < hi)
                {
                    int mid = (lo + hi + 1) / 2;
                    if (times[mid] <= t) lo = mid; else hi = mid - 1;
                }
                // lo is the largest index with times[lo] <= t, or 0 if none qualify
                // Check: if times[lo] > t, then no spike is before/at t
                tPreIdx = (times[lo] <= t) ? lo + 1 : 0;
            }

            double mem = kPrefactor * (
                sum1[tPreIdx * nPost + targetNrnIdx] * Math.Exp(-t / tauMem) -
                sum0[tPreIdx * nPost + targetNrnIdx] * Math.Exp(-t / tauSyn)
            );

            vTrace[ti] = mem;
            ts[ti] = t;
        }

        return (ts, vTrace);
    }

    /// <summary>
    /// Compute λ_i (synaptic adjoint) at a specific time t for a single neuron.
    /// Zero-allocation — single pass over post-spike timeline.
    /// </summary>
    public static double ComputeLambdaI(
        double t, int targetNrnIdx,
        ReadOnlySpan<double> postTimes, ReadOnlySpan<int> postSources,
        double vTh, double tauMem, double tauSyn)
    {
        int nSpikes = postTimes.Length;

        if (nSpikes == 0) return 0.0;

        double largestTime = postTimes[nSpikes - 1];
        double tBwdTotal = largestTime - t;

        // Single pass: accumulate jump contributions directly.
        // Jump magnitude = 1 / (vTh + epsilon) for each post-spike after time t.
        double kBwdPrefactor = tauMem / (tauMem - tauSyn);
        double lambdaI = 0.0;

        for (int i = nSpikes - 1; i >= 0; i--)
        {
            if (postTimes[i] < t) break;
            double kb = kBwdPrefactor * (Math.Exp(-tBwdTotal / tauMem) - Math.Exp(-tBwdTotal / tauSyn));
            lambdaI += 1.0 / (vTh + 0.001) * kb;
        }

        return lambdaI;
    }

    /// <summary>
    /// Compute the λ_i trace over time for a single neuron.
    /// </summary>
    public static (double[] times, double[] lambdaI) ComputeLambdaITrace(
        double tMax, double dt,
        int targetNrnIdx,
        ReadOnlySpan<double> postTimes, ReadOnlySpan<int> postSources,
        double vTh, double tauMem, double tauSyn)
    {
        int nT = (int)(tMax / dt) + 1;
        var ts = new double[nT];
        var lambdaTrace = new double[nT];

        for (int ti = 0; ti < nT; ti++)
        {
            lambdaTrace[ti] = ComputeLambdaI(
                dt * ti, targetNrnIdx,
                postTimes, postSources,
                vTh, tauMem, tauSyn);
            ts[ti] = dt * ti;
        }

        return (ts, lambdaTrace);
    }
}
