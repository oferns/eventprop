using CW.EventProp;

namespace CW.EventProp.Tests;

public class SpanAlgorithmsTests
{
    [Test]
    [MatrixDataSource]
    public async Task Exp_InputRanges_ReturnsCorrectValues([Matrix("zero", "positive", "negative")] string range)
    {
        double[] input = range switch
        {
            "zero" => [0, 0, 0],
            "positive" => [0.5, 1.0, 2.0],
            "negative" => [-1, -2, -3],
            _ => throw new ArgumentException($"Unknown range: {range}")
        };

        var result = SpanAlgorithms.Exp(input);
        await Assert.That(result.Length).IsEqualTo(3);

        for (int i = 0; i < input.Length; i++)
            await Assert.That(result[i]).IsCloseTo(Math.Exp(input[i]), 1e-12);

        if (range == "zero")
        {
            for (int i = 0; i < result.Length; i++)
                await Assert.That(result[i]).IsCloseTo(1.0, 1e-12);
        }
        else if (range == "positive")
        {
            for (int i = 0; i < input.Length; i++)
                await Assert.That(result[i]).IsGreaterThan(1.0);
        }
        else
        {
            for (int i = 0; i < input.Length; i++)
                await Assert.That(result[i]).IsLessThan(1.0);
        }
    }

    [Test]
    public async Task Exp_EmptyInput_ReturnsEmptyArray()
    {
        var result = SpanAlgorithms.Exp([]);
        await Assert.That(result.Length).IsEqualTo(0);
    }

    [Test]
    public async Task Exp_LargeInput_ReturnsInfinity()
    {
        double[] input = [710, 800]; // exp(710) ≈ overflow threshold for double
        var result = SpanAlgorithms.Exp(input);

        await Assert.That(double.IsInfinity(result[0])).IsTrue();
        await Assert.That(double.IsInfinity(result[1])).IsTrue();
    }

    [Test]
    [MatrixDataSource]
    public async Task ExpDivScalar_ScalarVariants_ReturnsCorrectValues([Matrix("negative", "positive")] string variant)
    {
        double[] input;
        double scalar;

        switch (variant)
        {
            case "negative":
                input = [1.0, 2.0, 3.0]; scalar = -2.0;
                break;
            case "positive":
                input = [0, 0]; scalar = 5.0;
                break;
            default:
                throw new ArgumentException($"Unknown variant: {variant}");
        }

        var result = SpanAlgorithms.ExpDivScalar(input, scalar);

        for (int i = 0; i < input.Length; i++)
            await Assert.That(result[i]).IsCloseTo(Math.Exp(input[i] / scalar), 1e-12);
    }

    [Test]
    public async Task ExpInPlace_ModifiesInputInPlace()
    {
        double[] data = [0, 1, -1];
        SpanAlgorithms.ExpInPlace(data);

        await Assert.That(data[0]).IsCloseTo(Math.Exp(0), 1e-12);
        await Assert.That(data[1]).IsCloseTo(Math.Exp(1), 1e-12);
        await Assert.That(data[2]).IsCloseTo(Math.Exp(-1), 1e-12);
    }

    [Test]
    public async Task DivideScalar_ZeroInput_ReturnsZeros()
    {
        double[] input = [0, 0, 0];
        var result = SpanAlgorithms.DivideScalar(input, 2.5);

        for (int i = 0; i < result.Length; i++)
            await Assert.That(result[i]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public async Task DivideScalar_NonZeroInput_ReturnsCorrectDivision()
    {
        double[] input = [1.0, 2.0, 3.0];
        var result = SpanAlgorithms.DivideScalar(input, 2.0);

        for (int i = 0; i < result.Length; i++)
            await Assert.That(result[i]).IsCloseTo(input[i] / 2.0, 1e-12);
    }

    [Test]
    public async Task DivideScalar_ZeroScalar_ReturnsInfinity()
    {
        double[] input = [1.0, -2.0];
        var result = SpanAlgorithms.DivideScalar(input, 0.0);

        await Assert.That(double.IsPositiveInfinity(result[0])).IsTrue();
        await Assert.That(double.IsNegativeInfinity(result[1])).IsTrue();
    }

    [Test]
    public async Task Multiply_ElementWiseCorrect()
    {
        double[] a = [1, 2, 3];
        double[] b = [4, 5, 6];
        var dest = new double[3];

        SpanAlgorithms.Multiply(dest, a, b);

        await Assert.That(dest[0]).IsCloseTo(4.0, 1e-12);
        await Assert.That(dest[1]).IsCloseTo(10.0, 1e-12);
        await Assert.That(dest[2]).IsCloseTo(18.0, 1e-12);
    }

    [Test]
    public async Task Multiply_WithZeros_ReturnsZeros()
    {
        double[] a = [0, 1, 2];
        double[] b = [3, 4, 5];
        var dest = new double[3];

        SpanAlgorithms.Multiply(dest, a, b);
        await Assert.That(dest[0]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task MultiplyScalar_ScalarVariants_ReturnsCorrectScaling([Matrix(3.0, 0.0, -1.0, 0.5)] double scalar)
    {
        double[] a = [1, 2, 3];
        var dest = new double[3];

        SpanAlgorithms.MultiplyScalar(dest, a, scalar);

        for (int i = 0; i < dest.Length; i++)
            await Assert.That(dest[i]).IsCloseTo(a[i] * scalar, 1e-12);
    }

    [Test]
    public async Task Add_ElementWiseCorrect()
    {
        double[] a = [1, 2, 3];
        double[] b = [4, 5, 6];
        var dest = new double[3];

        SpanAlgorithms.Add(dest, a, b);

        await Assert.That(dest[0]).IsCloseTo(5.0, 1e-12);
        await Assert.That(dest[1]).IsCloseTo(7.0, 1e-12);
        await Assert.That(dest[2]).IsCloseTo(9.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task AddScalar_ScalarVariants_AddsCorrectly([Matrix(10.0, -5.0, 0.0)] double scalar)
    {
        double[] a = [1, 2, 3];
        var dest = new double[3];

        SpanAlgorithms.AddScalar(dest, a, scalar);

        for (int i = 0; i < dest.Length; i++)
            await Assert.That(dest[i]).IsCloseTo(a[i] + scalar, 1e-12);
    }

    [Test]
    public async Task Subtract_ElementWiseCorrect()
    {
        double[] a = [5, 6, 7];
        double[] b = [1, 2, 3];
        var dest = new double[3];

        SpanAlgorithms.Subtract(dest, a, b);

        await Assert.That(dest[0]).IsCloseTo(4.0, 1e-12);
        await Assert.That(dest[1]).IsCloseTo(4.0, 1e-12);
        await Assert.That(dest[2]).IsCloseTo(4.0, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task SubtractScalar_ScalarVariants_SubtractsCorrectly([Matrix(5.0, -3.0, 0.0)] double scalar)
    {
        double[] a = [10, 20, 30];
        var dest = new double[3];

        SpanAlgorithms.SubtractScalar(dest, a, scalar);

        for (int i = 0; i < dest.Length; i++)
            await Assert.That(dest[i]).IsCloseTo(a[i] - scalar, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task Sum_ValueVariants_ReturnsCorrectSum([Matrix("positive", "mixed", "empty")] string variant)
    {
        double[] data;
        double expected;

        switch (variant)
        {
            case "positive":
                data = [1, 2, 3, 4, 5]; expected = 15.0;
                break;
            case "mixed":
                data = [-1, 2, -3, 4]; expected = 2.0;
                break;
            case "empty":
                data = []; expected = 0.0;
                break;
            default:
                throw new ArgumentException($"Unknown variant: {variant}");
        }

        var sum = SpanAlgorithms.Sum(data);
        await Assert.That(sum).IsCloseTo(expected, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task Dot_VariantInputs_ReturnsCorrectProduct([Matrix("standard", "withZeros", "empty")] string variant)
    {
        double[] a, b;
        double expected;

        switch (variant)
        {
            case "standard":
                a = [1, 2, 3]; b = [4, 5, 6]; expected = 32.0; // 4+10+18
                break;
            case "withZeros":
                a = [1, 0, 3]; b = [4, 5, 6]; expected = 22.0; // 4+0+18
                break;
            case "empty":
                a = []; b = []; expected = 0.0;
                break;
            default:
                throw new ArgumentException($"Unknown variant: {variant}");
        }

        var dot = SpanAlgorithms.Dot(a, b);
        await Assert.That(dot).IsCloseTo(expected, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task ExpNegDivTau_InputVariants_ReturnsCorrectDecay([Matrix("standard", "zeroInput")] string variant)
    {
        double[] input;
        double tau;

        switch (variant)
        {
            case "standard":
                input = [0, 1, 2]; tau = 0.5;
                break;
            case "zeroInput":
                input = [0, 0]; tau = 1.0;
                break;
            default:
                throw new ArgumentException($"Unknown variant: {variant}");
        }

        var result = SpanAlgorithms.ExpNegDivTau(input, tau);

        for (int i = 0; i < input.Length; i++)
            await Assert.That(result[i]).IsCloseTo(Math.Exp(-input[i] / tau), 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task ScaleInPlace_ScalarVariants_ModifiesCorrectly([Matrix(2.0, 0.0, -1.0, 0.5)] double scalar)
    {
        double[] data = [1, 2, 3];
        SpanAlgorithms.ScaleInPlace(data, scalar);

        for (int i = 0; i < data.Length; i++)
            await Assert.That(data[i]).IsCloseTo((i + 1) * scalar, 1e-12);
    }

    [Test]
    [MatrixDataSource]
    public async Task CopyScale_ScalarVariants_CopiesAndScalesCorrectly([Matrix(3.0, 2.0, -1.0)] double scalar)
    {
        double[] src = [1, 2, 3];
        var dest = new double[3];

        SpanAlgorithms.CopyScale(dest, src, scalar);

        for (int i = 0; i < src.Length; i++)
            await Assert.That(dest[i]).IsCloseTo(src[i] * scalar, 1e-12);

        // Source unchanged
        await Assert.That(src[0]).IsCloseTo(1.0, 1e-12);
    }

    [Test]
    public async Task SpanAlgorithms_LargeArray_NoPrecisionLoss()
    {
        int n = 10000;
        double[] data = new double[n];
        for (int i = 0; i < n; i++)
            data[i] = 1.0 / (i + 1);

        var sum = SpanAlgorithms.Sum(data);
        await Assert.That(sum).IsGreaterThan(9.0);
        await Assert.That(sum).IsLessThan(10.5);
    }

    [Test]
    public async Task SpanAlgorithms_Dot_ProductOfVectorWithItself_IsNormSquared()
    {
        double[] a = [3, 4];
        var dot = SpanAlgorithms.Dot(a, a);

        await Assert.That(dot).IsCloseTo(25.0, 1e-12); // 9 + 16
    }
}
