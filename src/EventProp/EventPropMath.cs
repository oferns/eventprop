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
    public static void ComputeSums(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        double tauMem,
        double tauSyn,
        Span<double> sum0,
        Span<double> sum1,
        int nPost)
    {
        var expInputMem = SpanAlgorithms.ExpDivScalar(times, tauMem);
        var expInputSyn = SpanAlgorithms.ExpDivScalar(times, tauSyn);

        int nSpikes = times.Length;

        // Row 0 is all zeros (already zeroed by caller)
        for (int i = 0; i < nSpikes; i++)
        {
            int src = sources[i];
            double expSyn = expInputSyn[i];
            double expMem = expInputMem[i];

            int rowOut = i + 1;
            int rowIn = i;

            for (int j = 0; j < nPost; j++)
            {
                sum0[rowOut * nPost + j] = sum0[rowIn * nPost + j] + expSyn * w[src * nPost + j];
                sum1[rowOut * nPost + j] = sum1[rowIn * nPost + j] + expMem * w[src * nPost + j];
            }
        }
    }

    /// <summary>
    /// Compute the membrane potential maximum time and value for each neuron.
    /// </summary>
    public static MaximaData ComputeMaxima(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        int nPost,
        double tauMem,
        double tauSyn)
    {
        var maximaValues = new double[nPost];
        var maximaTimes = new double[nPost];
        for (int i = 0; i < nPost; i++) maximaTimes[i] = double.NaN;

        var (sum0, sum1) = PrecomputeSums(w, times, sources, nPost, tauMem, tauSyn);

        double kPrefactor = tauSyn / (tauMem - tauSyn);
        double tmaxPrefactor = 1.0 / (1.0 / tauMem - 1.0 / tauSyn);
        double tmaxSummand = Math.Log(tauSyn / tauMem);

        int nSpikes = times.Length;

        for (int targetNrn = 0; targetNrn < nPost; targetNrn++)
        {
            for (int i = 0; i <= nSpikes; i++)
            {
                double tInput = (i == nSpikes) ? double.PositiveInfinity : times[i];
                double tBefore = (i > 0) ? times[i - 1] : 0.0;

                double sum0Elem = sum0[(i) * nPost + targetNrn];
                double sum1Elem = sum1[i * nPost + targetNrn];

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
    /// Precompute sum0 and sum1 matrices for a single sample. Returns owned buffers.
    /// </summary>
    private static (double[] sum0, double[] sum1) PrecomputeSums(
        ReadOnlySpan<double> w,
        ReadOnlySpan<double> times,
        ReadOnlySpan<int> sources,
        int nPost,
        double tauMem,
        double tauSyn)
    {
        int nSpikes = times.Length;
        var sum0 = new double[(nSpikes + 1) * nPost];
        var sum1 = new double[(nSpikes + 1) * nPost];

        ComputeSums(w, times, sources, tauMem, tauSyn, sum0, sum1, nPost);

        return (sum0, sum1);
    }
}
