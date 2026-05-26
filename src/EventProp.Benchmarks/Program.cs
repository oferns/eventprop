using CW.EventProp;

// Debug: compute_spikes voltage analysis for various configurations

Console.WriteLine("=== Single Spike, w=5.0 ===");
AnalyzeSpikes([5.0], [0.01], [0], 1, 1.0, 0.02, 0.005);

Console.WriteLine("\n=== Single Spike, w=10.0 ===");
AnalyzeSpikes([10.0], [0.01], [0], 1, 1.0, 0.02, 0.005);

Console.WriteLine("\n=== Single Spike, w=20.0 ===");
AnalyzeSpikes([20.0], [0.01], [0], 1, 1.0, 0.02, 0.005);

Console.WriteLine("\n=== Three Spikes, w=2.0 ===");
AnalyzeSpikes([2.0], [0.01, 0.02, 0.03], [0, 0, 0], 1, 1.0, 0.02, 0.005);

Console.WriteLine("\n=== Three Spikes, w=3.0 ===");
AnalyzeSpikes([3.0], [0.01, 0.02, 0.03], [0, 0, 0], 1, 1.0, 0.02, 0.005);

Console.WriteLine("\n=== Two Neurons, w=[5, 0.01, 0.01, 0.01, 0.01, 5] ===");
AnalyzeSpikes([5.0, 0.01, 0.01, 0.01, 0.01, 5.0], [0.01, 0.02], [0, 1], 3, 1.0, 0.02, 0.005);

Console.WriteLine("\n=== Large Network ===");
int nPre = 50, nPost = 20;
double[] wLarge = new double[nPre * nPost];
for (int i = 0; i < wLarge.Length; i++) wLarge[i] = 1.0 + (i % 5) * 0.5;
int nSpikes = 200;
double[] tLarge = new double[nSpikes];
int[] sLarge = new int[nSpikes];
for (int i = 0; i < nSpikes; i++) { tLarge[i] = 0.001 * (i + 1); sLarge[i] = i % nPre; }
AnalyzeSpikes(wLarge, tLarge, sLarge, nPost, 1.0, 0.02, 0.005);

static void AnalyzeSpikes(double[] w, double[] times, int[] sources, int nPost,
    double vTh, double tauMem, double tauSyn)
{
    // Compute sums first to see the values
    var sum0 = new double[(times.Length + 1) * nPost];
    var sum1 = new double[(times.Length + 1) * nPost];
    EventPropMath.ComputeSums(w, times, sources, tauMem, tauSyn, sum0, sum1, nPost);

    double kPrefactor = tauSyn / (tauMem - tauSyn);
    double tmaxPrefactor = 1.0 / (1.0 / tauMem - 1.0 / tauSyn);
    double tmaxSummand = Math.Log(tauSyn / tauMem);

    Console.WriteLine($"  kPrefactor={kPrefactor:F4}, tmaxPrefactor={tmaxPrefactor:F6}, tmaxSummand={tmaxSummand:F4}");

    for (int targetNrn = 0; targetNrn < nPost && targetNrn < 3; targetNrn++)
    {
        Console.WriteLine($"\n  Neuron {targetNrn}:");
        for (int i = 0; i <= times.Length; i++)
        {
            double tInput = (i == times.Length) ? double.PositiveInfinity : times[i];
            double tBefore = (i > 0) ? times[i - 1] : 0.0;
            double sum0Elem = sum0[i * nPost + targetNrn];
            double sum1Elem = sum1[i * nPost + targetNrn];

            if (sum0Elem <= 0 || sum1Elem <= 0)
            {
                Console.WriteLine($"    i={i}: sum0={sum0Elem:F4}, sum1={sum1Elem:F4} → skip (zero/neg)");
                continue;
            }

            double tmax = tmaxPrefactor * (tmaxSummand + Math.Log(sum1Elem / sum0Elem));
            bool inBounds = tBefore <= tmax && (i == times.Length || tmax <= tInput);

            double vAtTmax = kPrefactor * (sum1Elem * Math.Exp(-tmax / tauMem)
                                          - sum0Elem * Math.Exp(-tmax / tauSyn));

            Console.WriteLine($"    i={i}: tInput={tInput:F4}, tmax={tmax:F6} " +
                $"inBounds={inBounds}, v(tmax)={vAtTmax:F4} " +
                $"vTh={vTh}, exceeds? {vAtTmax > vTh}");
        }
    }

    // Now run compute_spikes and show results
    var buffers = new ScratchBuffers();
    int maxSpk = Math.Max(times.Length + 1, 4);
    buffers.PostSpikeTimes = new double[maxSpk * nPost];
    buffers.PostSpikeCurrents = new double[maxSpk * nPost];
    buffers.SpikeCountPerNeuron = new int[nPost];
    buffers.MaxSpikesPerNeuron = maxSpk;
    var result = EventPropMath.ComputeSpikes(w, times, sources, nPost, vTh, tauMem, tauSyn, buffers);
    Console.WriteLine($"\n  → {result.NSpikes} output spike(s)");
    for (int i = 0; i < Math.Min(result.NSpikes, 10); i++)
    {
        Console.WriteLine($"    spike[{i}]: t={result.Times[i]:F6}, src={result.Sources[i]}, i={result.Currents[i]:F4}");
    }
}
