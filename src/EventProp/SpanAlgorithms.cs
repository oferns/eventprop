namespace CW.EventProp;

/// <summary>
/// Zero-allocation span-based math operations for the EventProp hot paths.
/// </summary>
public static class SpanAlgorithms
{
    /// <summary>Element-wise exponential: returns new array with exp(x[i]).</summary>
    public static double[] Exp(ReadOnlySpan<double> input)
    {
        var result = new double[input.Length];
        Exp(input, result.AsSpan());
        return result;
    }

    /// <summary>Element-wise exponential: writes exp(x[i]) to output span (zero-allocation).</summary>
    public static void Exp(ReadOnlySpan<double> input, Span<double> output)
    {
        for (int i = 0; i < input.Length; i++)
            output[i] = Math.Exp(input[i]);
    }

    /// <summary>Element-wise exp(x[i] / scalar): returns new array.</summary>
    public static double[] ExpDivScalar(ReadOnlySpan<double> input, double scalar)
    {
        var result = new double[input.Length];
        ExpDivScalar(input, scalar, result.AsSpan());
        return result;
    }

    /// <summary>Element-wise exp(x[i] / scalar): writes to output span (zero-allocation).</summary>
    public static void ExpDivScalar(ReadOnlySpan<double> input, double scalar, Span<double> output)
    {
        for (int i = 0; i < input.Length; i++)
            output[i] = Math.Exp(input[i] / scalar);
    }

    /// <summary>Element-wise exponential in-place.</summary>
    public static void ExpInPlace(Span<double> span)
    {
        for (int i = 0; i < span.Length; i++)
            span[i] = Math.Exp(span[i]);
    }

    /// <summary>Element-wise division by scalar: returns new array.</summary>
    public static double[] DivideScalar(ReadOnlySpan<double> input, double scalar)
    {
        var result = new double[input.Length];
        DivideScalar(input, scalar, result.AsSpan());
        return result;
    }

    /// <summary>Element-wise division by scalar: writes to output span (zero-allocation).</summary>
    public static void DivideScalar(ReadOnlySpan<double> input, double scalar, Span<double> output)
    {
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] / scalar;
    }

    /// <summary>Element-wise multiplication of two spans, stored in output.</summary>
    public static void Multiply(Span<double> dest, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = a[i] * b[i];
    }

    /// <summary>Element-wise multiply by scalar: dest = a * scalar.</summary>
    public static void MultiplyScalar(Span<double> dest, ReadOnlySpan<double> a, double scalar)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = a[i] * scalar;
    }

    /// <summary>Element-wise addition: dest = a + b.</summary>
    public static void Add(Span<double> dest, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = a[i] + b[i];
    }

    /// <summary>Element-wise addition with scalar: dest = a + scalar.</summary>
    public static void AddScalar(Span<double> dest, ReadOnlySpan<double> a, double scalar)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = a[i] + scalar;
    }

    /// <summary>Sum of all elements.</summary>
    public static double Sum(ReadOnlySpan<double> span)
    {
        double sum = 0.0;
        for (int i = 0; i < span.Length; i++)
            sum += span[i];
        return sum;
    }

    /// <summary>Sum of element-wise product: dot(a, b).</summary>
    public static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /// <summary>Compute exp(-x / tau) for all elements: returns new array.</summary>
    public static double[] ExpNegDivTau(ReadOnlySpan<double> input, double tau)
    {
        var result = new double[input.Length];
        ExpNegDivTau(input, tau, result.AsSpan());
        return result;
    }

    /// <summary>Compute exp(-x / tau) for all elements: writes to output span (zero-allocation).</summary>
    public static void ExpNegDivTau(ReadOnlySpan<double> input, double tau, Span<double> output)
    {
        for (int i = 0; i < input.Length; i++)
            output[i] = Math.Exp(-input[i] / tau);
    }

    /// <summary>Element-wise subtraction: dest = a - b.</summary>
    public static void Subtract(Span<double> dest, ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = a[i] - b[i];
    }

    /// <summary>Element-wise subtraction with scalar: dest = a - scalar.</summary>
    public static void SubtractScalar(Span<double> dest, ReadOnlySpan<double> a, double scalar)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = a[i] - scalar;
    }

    /// <summary>Scale in-place: span *= scalar.</summary>
    public static void ScaleInPlace(Span<double> span, double scalar)
    {
        for (int i = 0; i < span.Length; i++)
            span[i] *= scalar;
    }

    /// <summary>Copy and scale: dest = src * scalar.</summary>
    public static void CopyScale(Span<double> dest, ReadOnlySpan<double> src, double scalar)
    {
        for (int i = 0; i < dest.Length; i++)
            dest[i] = src[i] * scalar;
    }
}
