using CW.EventProp;

namespace CW.EventProp.Tests;

public class SpikesDataTests
{
    [Test]
    public async Task Constructor_SetsTimesAndSources()
    {
        double[] times = [0.1, 0.2, 0.3];
        int[] sources = [0, 1, 0];

        var data = new SpikesData(times, sources);

        await Assert.That(data.Times).IsEqualTo(times);
        await Assert.That(data.Sources).IsEqualTo(sources);
    }

    [Test]
    public async Task Constructor_NSPikes_ReturnsCorrectCount()
    {
        double[] times = [0.1, 0.2, 0.3];
        int[] sources = [0, 1, 0];

        var data = new SpikesData(times, sources);

        await Assert.That(data.NSpikes).IsEqualTo(3);
    }

    [Test]
    public async Task Constructor_InitializesErrorsArray()
    {
        double[] times = [0.1, 0.2];
        int[] sources = [0, 1];

        var data = new SpikesData(times, sources);

        await Assert.That(data.Errors.Length).IsEqualTo(2);
    }

    [Test]
    public async Task Constructor_InitializesErrorsToZero()
    {
        double[] times = [0.1, 0.2];
        int[] sources = [0, 1];

        var data = new SpikesData(times, sources);

        for (int i = 0; i < data.Errors.Length; i++)
            await Assert.That(data.Errors[i]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public async Task Constructor_InitializesCurrentsSameSizeAsTimes()
    {
        double[] times = [0.1];
        int[] sources = [0];

        var data = new SpikesData(times, sources);

        await Assert.That(data.Currents.Length).IsEqualTo(1);
    }

    [Test]
    [MatrixDataSource]
    public async Task SetError_SetsCorrectIndex([Matrix(42.0, -5.0, 1e6)] double errorValue)
    {
        double[] times = [0.1, 0.2, 0.3];
        int[] sources = [0, 1, 0];

        var data = new SpikesData(times, sources);
        data.SetError(1, errorValue);

        await Assert.That(data.Errors[1]).IsCloseTo(errorValue, Math.Abs(errorValue) * 1e-10 + 1e-12);
        // Other indices unaffected
        await Assert.That(data.Errors[0]).IsCloseTo(0.0, 1e-12);
        await Assert.That(data.Errors[2]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public async Task SetTime_ModifiesSpikeTime()
    {
        double[] times = [0.1, 0.2, 0.3];
        int[] sources = [0, 1, 0];

        var data = new SpikesData(times, sources);
        data.SetTime(1, 0.99);

        await Assert.That(data.Times[1]).IsCloseTo(0.99, 1e-12);
    }

    [Test]
    public async Task SetTime_DoesNotAffectOtherTimes()
    {
        double[] times = [0.1, 0.2, 0.3];
        int[] sources = [0, 1, 0];

        var data = new SpikesData(times, sources);
        data.SetTime(1, 0.99);

        await Assert.That(data.Times[0]).IsCloseTo(0.1, 1e-12);
        await Assert.That(data.Times[2]).IsCloseTo(0.3, 1e-12);
    }

    [Test]
    public async Task NSpikes_EmptyData_ReturnsZero()
    {
        var data = new SpikesData([], []);

        await Assert.That(data.NSpikes).IsEqualTo(0);
    }

    [Test]
    public async Task SpikesData_LargeNumberOfSpikes()
    {
        int n = 10000;
        double[] times = new double[n];
        int[] sources = new int[n];

        for (int i = 0; i < n; i++)
        {
            times[i] = 0.001 * (i + 1);
            sources[i] = i % 50;
        }

        var data = new SpikesData(times, sources);

        await Assert.That(data.NSpikes).IsEqualTo(n);
        await Assert.That(data.Errors.Length).IsEqualTo(n);
    }

    [Test]
    public async Task SpikesData_TimesAndSourcesAreSameReference()
    {
        double[] times = [0.1, 0.2];
        int[] sources = [0, 1];

        var data = new SpikesData(times, sources);

        await Assert.That(data.Times).IsSameReferenceAs(times);
        await Assert.That(data.Sources).IsSameReferenceAs(sources);
    }

    [Test]
    public async Task SpikesData_IsSealedClass()
    {
        var data = new SpikesData([0.1], [0]);

        // Verify it's a sealed class by checking its type
        Type t = data.GetType();
        await Assert.That(t.IsSealed).IsTrue();
    }

    [Test]
    public async Task SetTime_ZeroValue()
    {
        double[] times = [0.1, 0.2];
        int[] sources = [0, 1];

        var data = new SpikesData(times, sources);
        data.SetTime(0, 0.0);

        await Assert.That(data.Times[0]).IsCloseTo(0.0, 1e-12);
    }

    [Test]
    public async Task SpikesData_CurrentsCanBeSet()
    {
        double[] times = [0.1];
        int[] sources = [0];

        var data = new SpikesData(times, sources);
        data.Currents = [1.0, 2.0];

        await Assert.That(data.Currents.Length).IsEqualTo(2);
    }
}
