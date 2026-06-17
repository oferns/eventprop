namespace CW.EventProp;

/// <summary>
/// Weight initialization distribution.
/// </summary>
public abstract class WeightInitializer
{
    public int? Seed { get; set; }

    /// <summary>Create weights with shape (nIn, nOut).</summary>
    public abstract double[] GetWeights(int nIn, int nOut);
}

/// <summary>
/// Gaussian weight initialization.
/// </summary>
public class GaussianInitializer : WeightInitializer
{
    public double Mean { get; set; } = 0.0;
    public double StdDev { get; set; } = 1.0;

    private Random? _rng;

    public GaussianInitializer() { }

    public override double[] GetWeights(int nIn, int nOut)
    {
        var rng = _rng ??= Seed is not null ? new Random(Seed.Value) : new Random();
        var weights = new double[nIn * nOut];

        // Box-Muller transform for normal distribution
        for (int i = 0; i < nIn * nOut;)
        {
            double u1 = rng.NextDouble();
            double u2 = rng.NextDouble();
            // Avoid log(0)
            while (u1 < 1e-300) u1 = rng.NextDouble();
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            weights[i++] = Mean + StdDev * z0;
            if (i < nIn * nOut)
            {
                double z1 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                weights[i++] = Mean + StdDev * z1;
            }
        }
        return weights;
    }
}

/// <summary>
/// Uniform weight initialization.
/// </summary>
public class UniformInitializer : WeightInitializer
{
    public double Lower { get; set; } = -1.0;
    public double Upper { get; set; } = 1.0;

    private Random? _rng;

    public UniformInitializer() { }

    public override double[] GetWeights(int nIn, int nOut)
    {
        var rng = _rng ??= Seed is not null ? new Random(Seed.Value) : new Random();
        var weights = new double[nIn * nOut];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = Lower + rng.NextDouble() * (Upper - Lower);
        return weights;
    }
}

/// <summary>
/// Diagonal identity weight initialization (nIn must equal nOut).
/// </summary>
public class DiagonalInitializer : WeightInitializer
{
    public override double[] GetWeights(int nIn, int nOut)
    {
        if (nIn != nOut)
            throw new ArgumentException($"Diagonal weights require nIn == nOut, got {nIn} != {nOut}", nameof(nIn));
        var weights = new double[nIn * nOut];
        for (int i = 0; i < nIn; i++)
            weights[i * nOut + i] = 1.0;
        return weights;
    }
}
