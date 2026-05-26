using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;

namespace CW.EventProp.Tests;

public class Toms748Tests
{
    [Test]
    public void Solve_SimpleLinearRoot_ReturnsCorrectValue()
    {
        double result = Toms748.Solve(x => x - 3.0, 0.0, 5.0);
        SyncAssert.That(result).IsCloseTo(3.0, 1e-10);
    }

    [Test]
    public void Solve_QuadraticRoot_ReturnsCorrectValue()
    {
        double result = Toms748.Solve(x => x * x - 4.0, 1.0, 3.0);
        SyncAssert.That(result).IsCloseTo(2.0, 1e-10);
    }

    [Test]
    public void Solve_SameSignInputs_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => Toms748.Solve(x => x * x + 1.0, 0.0, 5.0));
    }

    [Test]
    public void Solve_ReversedBracket_SwapsAndReturnsCorrectValue()
    {
        double result = Toms748.Solve(x => x - 3.0, 5.0, 0.0);
        SyncAssert.That(result).IsCloseTo(3.0, 1e-10);
    }

    [Test]
    public void Solve_IQIPath_UsesInverseQuadraticInterpolation()
    {
        double result = Toms748.Solve(x => x * x * x - 2.0 * x - 5.0, 2.0, 3.0);
        SyncAssert.That(result).IsCloseTo(2.0945514815423265, 1e-10);
    }

    [Test]
    public void Solve_StepSizeConvergence_ExitsEarly()
    {
        double result = Toms748.Solve(x => Math.Pow(x - 1.0, 3), 0.999, 1.001);
        SyncAssert.That(result).IsCloseTo(1.0, 1e-6);
    }

    [Test]
    public void Solve_MaxIterations_ExitsGracefully()
    {
        double result = Toms748.Solve(
            x => Math.Sign(x) * Math.Pow(Math.Abs(x), 1.0 / 3.0),
            -1e-6,
            1e-6);

        SyncAssert.That(result).IsCloseTo(0.0, 1e-4);
    }

    [Test]
    public void Solve_NegativeRoot_ReturnsCorrectValue()
    {
        double result = Toms748.Solve(x => x + 5.0, -10.0, 0.0);
        SyncAssert.That(result).IsCloseTo(-5.0, 1e-10);
    }

    [Test]
    public void Solve_SmallBracket_ReturnsCorrectValue()
    {
        double result = Toms748.Solve(x => x - 0.001, 0.0, 0.002);
        SyncAssert.That(result).IsCloseTo(0.001, 1e-12);
    }

    [Test]
    public void Solve_LargeBracket_ReturnsCorrectValue()
    {
        double result = Toms748.Solve(x => x - 1e6, 0.0, 2e6);
        SyncAssert.That(result).IsCloseTo(1e6, 1e-4);
    }

    [Test]
    public void Solve_ZeroTolerance_UsesDefault()
    {
        double result = Toms748.Solve(x => x - 3.0, 0.0, 5.0, tolerance: 0.0);
        SyncAssert.That(result).IsCloseTo(3.0, 1e-10);
    }

    [Test]
    public void Solve_NegativeTolerance_UsesDefault()
    {
        double result = Toms748.Solve(x => x - 3.0, 0.0, 5.0, tolerance: -1.0);
        SyncAssert.That(result).IsCloseTo(3.0, 1e-10);
    }

    [Test]
    public void Solve_ExactlyZeroAtEndpoint_ReturnsEndpoint()
    {
        double result = Toms748.Solve(x => x - 5.0, 0.0, 5.0);
        SyncAssert.That(result).IsCloseTo(5.0, 1e-10);
    }

    [Test]
    public void Solve_SineRoot_ReturnsCorrectValue()
    {
        double result = Toms748.Solve(Math.Sin, 3.0, 3.2);
        SyncAssert.That(result).IsCloseTo(Math.PI, 1e-10);
    }

    [Test]
    public void Solve_ExponentialRoot_ReturnsCorrectValue()
    {
        double result = Toms748.Solve(x => Math.Exp(x) - 2.0, 0.0, 1.0);
        SyncAssert.That(result).IsCloseTo(Math.Log(2), 1e-10);
    }

    /// <summary>
    /// Force the IQI acceptance path (line 79: useBisection = false).
    /// The cubic x^3 - 2x - 5 on a wide bracket [0.1, 10] produces IQI estimates
    /// that land inside the bracket with small enough step size.
    /// </summary>
    [Test]
    public void Solve_IQIAccepted_UseBisectionSetToFalse()
    {
        // Real root ≈ 2.0946. Wide bracket lets IQI converge well after initial bisections.
        double result = Toms748.Solve(x => x * x * x - 2.0 * x - 5.0, 0.1, 10.0);
        SyncAssert.That(result).IsCloseTo(2.0945514815423265, 1e-12);
    }

    /// <summary>
    /// Additional IQI acceptance test with log(x) on [0.1, 3].
    /// Root at x=1 (log(1)=0). Wide bracket enables IQI path.
    /// </summary>
    [Test]
    public void Solve_IQIAccepted_LogFunction()
    {
        double result = Toms748.Solve(Math.Log, 0.1, 3.0);
        SyncAssert.That(result).IsCloseTo(1.0, 1e-12);
    }
}
