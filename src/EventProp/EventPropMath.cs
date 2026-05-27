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

        double kPrefactor = tauSyn / (tauMem - tauSyn);
        double tmaxPrefactor = 1.0 / (1.0 / tauMem - 1.0 / tauSyn);
        double tmaxSummand = Math.Log(tauSyn / tauMem);

        for (int targetNrn = 0; targetNrn < nPost; targetNrn++)
        {
            for (int i = 0; i <= nSpikes; i++)
            {
                double tInput = (i == nSpikes) ? double.PositiveInfinity : times[i];
                double tBefore = (i > 0) ? times[i - 1] : 0.0;

                double sum0Elem = buffers.Sum0[i * nPost + targetNrn];
                double sum1Elem = buffers.Sum1[i * nPost + targetNrn];

                double tmax = tmaxPrefactor * (tmaxSummand + Math.Log(sum1Elem / sum0Elem));

                if (tBefore <= tmax && tmax <= tInput)
                {
                    double vmax = kPrefactor * (sum1Elem * Math.Exp(-tmax / tauMem) - sum0Elem * Math.Exp(-tmax / tauSyn));
                    if (vmax > maximaValues[targetNrn])
                    {
                        maximaValues[targetNrn] = vmax;
                        maximaTimes[targetNrn] = tmax;
                    }
                }

                // Also check at the input spike time itself
                double vAtInput = kPrefactor * (sum1Elem * Math.Exp(-tInput / tauMem) - sum0Elem * Math.Exp(-tInput / tauSyn));
                if (vAtInput > maximaValues[targetNrn])
                {
                    maximaValues[targetNrn] = vAtInput;
                    maximaTimes[targetNrn] = tInput == double.PositiveInfinity ? tBefore : tInput;
                }
            }
        }

        return new MaximaData(maximaTimes, maximaValues);
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

        // For each post-synaptic neuron, find threshold crossings
        for (int targetNrn = 0; targetNrn < nPost; targetNrn++)
        {
            int neuronOffset = targetNrn * maxSpikesPerNeuron;
            int neuronSpikeCount = 0;

            // Process intervals — while loop re-processes from processedUpTo when a spike is found
            // (matching the C++ algorithm's behavior of finding multiple spikes per interval)
            bool finished = false;
            int processedUpTo = 0;

            while (!finished)
            {
                bool foundSpikeInThisPass = false;
                for (int i = processedUpTo; i <= nSpikes; i++)
                {
                    if (i == nSpikes)
                    {
                        // Last interval — check after last input spike
                        double sum0Val = buffers.Sum0[targetNrn * (nSpikes + 1) + nSpikes];
                        double sum1Val = buffers.Sum1[targetNrn * (nSpikes + 1) + nSpikes];
                        for (int s = 0; s < neuronSpikeCount; s++)
                            sum1Val += -vTh * Math.Exp(buffers.PostSpikeTimes[neuronOffset + s] / tauMem);
                        double tmax = tmaxPrefactor * (tmaxSummand + Math.Log(sum1Val / sum0Val));
                        if (!double.IsInfinity(tmax))
                        {
                            double vmax = ComputeV(tmax, i, neuronSpikeCount,
                                buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                targetNrn, nPost, kPrefactor, tauMem, tauSyn, vTh);
                            if (vmax > vTh + 1e-14)
                            {
                                double tLower = GetTLower(neuronSpikeCount, buffers.PostSpikeTimes, neuronOffset, tmax);
                                int solveSpikeCount = neuronSpikeCount;
                                double tPost = Toms748.Solve(
                                    t => ComputeV(t, i, solveSpikeCount,
                                        buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                        targetNrn, nPost, kPrefactor, tauMem, tauSyn, vTh) - vTh,
                                    tLower, tmax);
                                if (tPost > tLower && tPost < tmax)
                                {
                                    double current = ComputeI(tPost, nSpikes, times,
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
                        finished = true;
                    }
                    else
                    {
                        double tInput = times[i];
                        double tmax = ComputeTmax(i, nSpikes, times, neuronSpikeCount,
                            buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                            targetNrn, nPost, tmaxPrefactor, tmaxSummand, tauMem, vTh);

                        if (tmax != double.PositiveInfinity)
                        {
                            // Check if max voltage exceeds threshold
                            double vmax = ComputeV(tmax, i, neuronSpikeCount,
                                buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                targetNrn, nPost, kPrefactor, tauMem, tauSyn, vTh);
                            if (vmax > vTh + 1e-14)
                            {
                                double tBefore = (i > 0) ? times[i - 1] : 0.0;
                                for (int s = 0; s < neuronSpikeCount; s++)
                                {
                                    double spikeTime = buffers.PostSpikeTimes[neuronOffset + s];
                                    if (spikeTime > tInput) break;
                                    if (tBefore < spikeTime)
                                        tBefore = spikeTime;
                                }

                                int solveSpikeCount = neuronSpikeCount;
                                double tPost = Toms748.Solve(
                                    t => ComputeV(t, i, solveSpikeCount,
                                        buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                        targetNrn, nPost, kPrefactor, tauMem, tauSyn, vTh) - vTh,
                                    tBefore, tmax);

                                double current = ComputeI(tPost, nSpikes, times,
                                    buffers.Sum0, targetNrn, nPost, tauSyn);
                                if (neuronOffset + neuronSpikeCount < buffers.PostSpikeTimes.Length)
                                {
                                    buffers.PostSpikeTimes[neuronOffset + neuronSpikeCount] = tPost;
                                    buffers.PostSpikeCurrents[neuronOffset + neuronSpikeCount] = current;
                                    neuronSpikeCount++;
                                    foundSpikeInThisPass = true;
                                }
                                processedUpTo = i;
                                break; // Re-process this interval with updated spike count
                            }
                        }

                        // Also check at the input spike time itself
                        double vAtInput = ComputeV(tInput, i, neuronSpikeCount,
                            buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                            targetNrn, nPost, kPrefactor, tauMem, tauSyn, vTh);
                        if (vAtInput > vTh + 1e-14)
                        {
                            double tBefore = (i > 0) ? times[i - 1] : 0.0;
                            for (int s = 0; s < neuronSpikeCount; s++)
                            {
                                double spikeTime = buffers.PostSpikeTimes[neuronOffset + s];
                                if (spikeTime > tInput) break;
                                if (tBefore < spikeTime)
                                    tBefore = spikeTime;
                            }

                            int solveSpikeCount = neuronSpikeCount;
                            double tPost = Toms748.Solve(
                                t => ComputeV(t, i, solveSpikeCount,
                                    buffers.PostSpikeTimes, neuronOffset, buffers.Sum0, buffers.Sum1,
                                    targetNrn, nPost, kPrefactor, tauMem, tauSyn, vTh) - vTh,
                                tBefore, tInput);

                            double current = ComputeI(tPost, nSpikes, times,
                                buffers.Sum0, targetNrn, nPost, tauSyn);
                            if (neuronOffset + neuronSpikeCount < buffers.PostSpikeTimes.Length)
                            {
                                buffers.PostSpikeTimes[neuronOffset + neuronSpikeCount] = tPost;
                                buffers.PostSpikeCurrents[neuronOffset + neuronSpikeCount] = current;
                                neuronSpikeCount++;
                                foundSpikeInThisPass = true;
                            }
                            processedUpTo = i;
                            break; // Re-process this interval with updated spike count
                        }
                    }
                }

                // If no spike was found in this pass, advance to the next interval
                if (!foundSpikeInThisPass)
                    processedUpTo = Math.Min(processedUpTo + 1, nSpikes);
            }

            buffers.SpikeCountPerNeuron[targetNrn] = neuronSpikeCount;
        }

        // Count total spikes and build output from scratch buffers (single allocation)
        int totalSpikes = 0;
        for (int n = 0; n < nPost; n++)
            totalSpikes += buffers.SpikeCountPerNeuron[n];

        if (totalSpikes == 0)
            return new SpikesData([], []);

        // Single struct array — sort in-place, then unpack to result arrays
        var entries = new SpikeEntry[totalSpikes];
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

        // Sort by time (stable sort to preserve neuron order for equal times)
        Array.Sort(entries);

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
        // Clearing 3.5 MB of scratch space is wasted work when every buffer is immediately overwritten.

        // Merge input + post-spike indices and sort by time descending.
        // Pre-compute times into a struct array so the comparer has no ref-capture dependency.
        var merged = new MergedSpike[totalSpikes];
        for (int i = 0; i < nInputSpikes; i++)
            merged[i] = new MergedSpike(i, true, inputTimes[i]);
        for (int i = 0; i < nPostSpikes; i++)
            merged[nInputSpikes + i] = new MergedSpike(i, false, postSpikes.Times[i]);

        Array.Sort(merged);

        // Combined error array: input spike errors [0..nInputSpikes-1] + post-spike errors [nInputSpikes..totalSpikes-1].
        // This is needed because spikeIdx in the merged timeline can refer to either input or post-spike indices,
        // and postSpikes.Errors is only sized for post-spikes.
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

        // Per-neuron jump counters — jumps stored in flat arrays with per-neuron offsets.
        // Sized for max(nPre, nPost) since spikeSource can be either a pre-synaptic
        // neuron index (input spikes) or post-synaptic neuron index (post-spikes).
        int[] jumpEnd = new int[maxNeurons];

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
            double decayFactor = Math.Exp(-dt / tauMem);
            SimdAlgorithms.ScaleInPlace(lambdaV.AsSpan(0, nPost), decayFactor);

            if (isInput)
            {
                // Input spike: compute gradient and outbound signal
                double[] lambdaI = buffers.LambdaI;
                Array.Fill(lambdaI, 0.0);

                // Compute λ_i from accumulated jumps for this source neuron
                int start = 0, end = jumpEnd[spikeSource];
                for (int j = start; j < end; j++)
                {
                    double jumpTimeOffset = buffers.LambdaIJumpsTimeOffset[j];
                    double jumpMag = buffers.LambdaIJumpsMagnitude[j];
                    lambdaI[spikeSource] += jumpMag * kBwdPrefactor * (Math.Exp(-(tBwd - jumpTimeOffset) / tauMem) - Math.Exp(-(tBwd - jumpTimeOffset) / tauSyn));
                }

                // Add gradient contribution: gradient[source, nrn] += -tau_syn * lambda_i[nrn]
                for (int nrn = 0; nrn < nPost; nrn++)
                    gradient[spikeSource * nPost + nrn] += -tauSyn * lambdaI[nrn];

                // Compute outbound signal: sum_nrn(w[source, nrn] * (lambda_v[nrn] - lambda_i[nrn]))
                double outboundSignal = 0.0;
                for (int nrn = 0; nrn < nPost; nrn++)
                {
                    double wVal = w[spikeSource * nPost + nrn]; // row-major access: w[source, nrn]
                    outboundSignal += wVal * (lambdaV[nrn] - lambdaI[nrn]);
                }

                // Add outbound signal to input spike error
                allErrors[spikeIdx] += outboundSignal;
            }
            else
            {
                // Post-synaptic spike: record λ_i jump and update λ_v
                double current = postSpikes.Currents[spikeIdx];
                double invICurrent = 1.0 / (current - vTh);

                // Record λ_i jump for this neuron in the flat array
                int jIdx = jumpEnd[spikeSource];
                double postError = allErrors[nInputSpikes + spikeIdx]; // post-spikes stored at offset nInputSpikes
                buffers.LambdaIJumpsMagnitude[jIdx] =
                    invICurrent * (vTh * lambdaV[spikeSource] + postError);
                buffers.LambdaIJumpsTimeOffset[jIdx] = tBwd;
                jumpEnd[spikeSource]++;

                // Update λ_v for this neuron
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
                double kBwdVal = kBwdPrefactor * (Math.Exp(-tBwd / tauMem) - Math.Exp(-tBwd / tauSyn));
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
        double kPrefactor, double tauMem, double tauSyn, double vTh)
    {
        double mem = kPrefactor * (sum1[i * nPost + targetNrn] * Math.Exp(-t / tauMem)
                        - sum0[i * nPost + targetNrn] * Math.Exp(-t / tauSyn));
        for (int s = 0; s < spikeCount; s++)
        {
            double spikeTime = postSpikeTimes[neuronOffset + s];
            if (spikeTime > t) break;
            mem -= vTh * Math.Exp(-(t - spikeTime) / tauMem);
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
            double tLower = postSpikeTimes[neuronOffset + spikeCount - 1];
            if (tLower < tUpper) return tLower;
        }
        return 0.0;
    }

    /// <summary>
    /// Merged spike entry for sorting input + post-spike spikes by time.
    /// Pre-computed Time avoids capturing ReadOnlySpan in the comparer.
    /// </summary>
    struct MergedSpike : IComparable<MergedSpike>
    {
        public readonly int Idx;
        public readonly bool IsInput;
        public readonly double Time;

        public MergedSpike(int idx, bool isInput, double time)
        {
            Idx = idx;
            IsInput = isInput;
            Time = time;
        }

        public int CompareTo(MergedSpike other)
            => other.Time.CompareTo(Time); // descending by time
    }

    /// <summary>
    /// Spike entry for sorting output spikes by time.
    /// </summary>
    struct SpikeEntry : IComparable<SpikeEntry>
    {
        public readonly double Time;
        public readonly int Source;
        public readonly double Current;

        public SpikeEntry(double time, int source, double current)
        {
            Time = time;
            Source = source;
            Current = current;
        }

        public int CompareTo(SpikeEntry other)
        {
            int cmp = Time.CompareTo(other.Time);
            return cmp != 0 ? cmp : Source.CompareTo(other.Source);
        }
    }
}
