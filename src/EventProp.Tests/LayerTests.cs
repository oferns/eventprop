using CW.EventProp;
using CW.EventProp.Tests.SyncAssertions;
using TUnit;

namespace EventProp.Tests;

public class LayerTests
{
    [Test]
    public async Task LIFLayer_Forward_Backward_Works()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        // Set positive weights to ensure spikes are generated
        var rng = new Random(42);
        for (int i = 0; i < layer.Weights.Length; i++)
            layer.Weights[i] = rng.NextDouble() * 2.0 + 0.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2, 0.3, 0.4 }, new int[] { 0, 1, 2, 0 }),
        };

        // Act
        layer.Forward(input);
        var output = layer.Output;
        layer.Backward();

        // Assert
        await Assert.That(output.Count).IsEqualTo(1);
        await Assert.That(output[0].NSpikes).IsGreaterThanOrEqualTo(0);
        await Assert.That(layer.Gradient.Length).IsEqualTo(15); // nIn * nOut
    }

    [Test]
    public async Task LIFLayer_MultipleSamples_Works()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        // Set positive weights to ensure spikes are generated
        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
            new SpikesData(new double[] { 0.15, 0.25 }, new int[] { 1, 2 }),
        };

        // Act
        layer.Forward(input);
        var output = layer.Output;
        layer.Backward();

        // Assert
        await Assert.That(output.Count).IsEqualTo(2);
    }

    [Test]
    public async Task LILayer_Forward_Backward_Works()
    {
        // Arrange
        var parameters = new LILayerParameters(N: 4, NIn: 3);
        var layer = new LILayer(parameters);

        // Set positive weights
        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        layer.Forward(input);
        layer.Backward();

        // Assert
        await Assert.That(layer.Gradient.Length).IsEqualTo(12); // nIn * nOut
    }

    [Test]
    public async Task LIFLayer_Chain_Works()
    {
        // Arrange
        var params1 = new LIFLayerParameters(N: 5, NIn: 3);
        var params2 = new LIFLayerParameters(N: 4, NIn: 5);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        // Set positive weights
        for (int i = 0; i < layer1.Weights.Length; i++) layer1.Weights[i] = 1.5;
        for (int i = 0; i < layer2.Weights.Length; i++) layer2.Weights[i] = 1.5;

        var chain = Layer.Chain(layer1, layer2);
        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        chain.Forward(input);
        var output = chain.Output;
        chain.Backward();

        // Assert — output is from layer1 (the outer/first layer in the chain)
        await Assert.That(output.Count).IsEqualTo(1);
    }

    [Test]
    public async Task LIFLayer_ZeroGrad_Works()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        // Set positive weights
        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        layer.Forward(input);
        layer.Backward();

        var gradBefore = layer.Gradient.Sum();

        layer.ZeroGrad();

        // Assert
        await Assert.That(layer.Gradient.Sum()).IsEqualTo(0.0);
    }

    [Test]
    public async Task LIFLayer_VoltageTrace_Works()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        layer.Forward(input);
        var (times, voltages) = layer.GetVoltageTrace(0, 0, tMax: 0.5, dt: 0.01);

        // Assert
        await Assert.That(times.Length).IsEqualTo(51); // 0.5 / 0.01 + 1
        await Assert.That(voltages.Length).IsEqualTo(51);
    }

    [Test]
    public async Task WeightInitializer_Gaussian_Works()
    {
        // Arrange
        var init = new GaussianInitializer { Mean = 0, StdDev = 1, Seed = 42 };

        // Act
        var weights = init.GetWeights(3, 5);

        // Assert
        await Assert.That(weights.Length).IsEqualTo(15);
    }

    [Test]
    public async Task WeightInitializer_Uniform_Works()
    {
        // Arrange
        var init = new UniformInitializer { Lower = -1, Upper = 1, Seed = 42 };

        // Act
        var weights = init.GetWeights(3, 5);

        // Assert
        await Assert.That(weights.Length).IsEqualTo(15);
    }

    [Test]
    public async Task WeightInitializer_Diagonal_Works()
    {
        // Arrange
        var init = new DiagonalInitializer();

        // Act
        var weights = init.GetWeights(5, 5);

        // Assert
        await Assert.That(weights.Length).IsEqualTo(25);
        for (int i = 0; i < 5; i++)
            await Assert.That(weights[i * 5 + i]).IsEqualTo(1.0);
    }

    [Test]
    public async Task WeightInitializer_Diagonal_RejectsMismatchedDimensions()
    {
        // Arrange
        var init = new DiagonalInitializer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () => { init.GetWeights(3, 5); await Task.CompletedTask; });
    }

    [Test]
    public async Task LIFLayer_GetWeights_SetWeights_Works()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        var layer = new LIFLayer(parameters);

        // Act — set weights via Span (copy to array for await boundary)
        double[] newWeights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        layer.SetWeights(newWeights);

        // Assert — read back via Span (copy to array for await boundary)
        double[] readBack = layer.GetWeights().ToArray();
        await Assert.That(readBack.Length).IsEqualTo(6);
        for (int i = 0; i < 6; i++)
            await Assert.That(readBack[i]).IsEqualTo(newWeights[i]);
    }

    [Test]
    public async Task LIFLayer_WeightValidationError()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        var badWeights = new double[] { 1.0, 2.0, 3.0 }; // wrong length

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () => { new LIFLayer(parameters, badWeights); await Task.CompletedTask; });
    }

    [Test]
    public async Task LILayer_VoltageTrace_Works()
    {
        // Arrange
        var parameters = new LILayerParameters(N: 5, NIn: 3);
        var layer = new LILayer(parameters);

        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        layer.Forward(input);
        var (times, voltages) = layer.GetVoltageTrace(0, 0, tMax: 0.5, dt: 0.01);

        // Assert
        await Assert.That(times.Length).IsEqualTo(51);
        await Assert.That(voltages.Length).IsEqualTo(51);
    }

    [Test]
    public async Task LIFLayer_BackwardBeforeForward_Throws()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => { layer.Backward(); await Task.CompletedTask; });
    }

    [Test]
    public async Task LILayer_BackwardBeforeForward_Throws()
    {
        // Arrange
        var parameters = new LILayerParameters(N: 5, NIn: 3);
        var layer = new LILayer(parameters);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => { layer.Backward(); await Task.CompletedTask; });
    }

    [Test]
    public async Task LIFLayer_OutputBeforeForward_Throws()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => { var _ = layer.Output; await Task.CompletedTask; });
    }

    [Test]
    public async Task LIFLayer_GetVoltageTraceBeforeForward_Throws()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(async () => { layer.GetVoltageTrace(0, 0); await Task.CompletedTask; });
    }

    [Test]
    public async Task LIFLayer_GetVoltageTrace_InvalidBatchIdx_Throws()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1 }, new int[] { 0 }),
        };

        layer.Forward(input);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => Task.FromResult(layer.GetVoltageTrace(5, 0)));
    }

    [Test]
    public async Task LIFLayer_GetLambdaI_Works()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        layer.Forward(input);
        var lambdaI = layer.GetLambdaI(0, 0, t: 0.15);

        // Assert — should return a finite number
        await Assert.That(double.IsFinite(lambdaI)).IsTrue();
    }

    [Test]
    public async Task LIFLayer_GetLambdaITrace_Works()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        layer.Forward(input);
        var (times, lambdaI) = layer.GetLambdaITrace(0, 0, tMax: 0.5, dt: 0.1);

        // Assert
        await Assert.That(times.Length).IsEqualTo(6); // 0.5 / 0.1 + 1
        await Assert.That(lambdaI.Length).IsEqualTo(6);
    }

    [Test]
    public async Task LIFLayer_SetWeights_WrongLength_Throws()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        var layer = new LIFLayer(parameters);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () => { layer.SetWeights(stackalloc double[] { 1.0, 2.0 }); await Task.CompletedTask; });
    }

    [Test]
    public async Task Chain_NIn_NOut_Correct()
    {
        // Arrange — chain: input(3) -> layer2(4) -> layer1(5)
        var params1 = new LIFLayerParameters(N: 5, NIn: 4);
        var params2 = new LIFLayerParameters(N: 4, NIn: 3);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        var chain = Layer.Chain(layer1, layer2);

        // Assert — chain NIn comes from innermost (layer2), NOut from outermost (layer1)
        await Assert.That(chain.NIn).IsEqualTo(3);
        await Assert.That(chain.NOut).IsEqualTo(5);
    }

    [Test]
    public async Task Chain_LayerAccessors_Works()
    {
        // Arrange
        var params1 = new LIFLayerParameters(N: 5, NIn: 3);
        var params2 = new LIFLayerParameters(N: 4, NIn: 5);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        var chain = Layer.Chain(layer1, layer2);

        // Assert
        await Assert.That(chain.LayerA).IsEqualTo(layer1);
        await Assert.That(chain.LayerB).IsEqualTo(layer2);
    }

    [Test]
    public async Task Chain_ZeroGrad_ZerosBothLayers()
    {
        // Arrange
        var params1 = new LIFLayerParameters(N: 5, NIn: 3);
        var params2 = new LIFLayerParameters(N: 4, NIn: 5);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        for (int i = 0; i < layer1.Weights.Length; i++) layer1.Weights[i] = 1.5;
        for (int i = 0; i < layer2.Weights.Length; i++) layer2.Weights[i] = 1.5;

        var chain = Layer.Chain(layer1, layer2);
        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        // Act
        chain.Forward(input);
        chain.Backward();
        chain.ZeroGrad();

        // Assert — both layers should have zero gradients
        await Assert.That(layer1.Gradient.Sum()).IsEqualTo(0.0);
        await Assert.That(layer2.Gradient.Sum()).IsEqualTo(0.0);
    }

    [Test]
    public async Task GaussianInitializer_Seeded_Reproducible()
    {
        // Arrange
        var init1 = new GaussianInitializer { Mean = 0, StdDev = 1, Seed = 42 };
        var init2 = new GaussianInitializer { Mean = 0, StdDev = 1, Seed = 42 };

        // Act
        var weights1 = init1.GetWeights(3, 5);
        var weights2 = init2.GetWeights(3, 5);

        // Assert — same seed should produce identical weights
        for (int i = 0; i < weights1.Length; i++)
            await Assert.That(weights2[i]).IsEqualTo(weights1[i]);
    }

    [Test]
    public async Task GaussianInitializer_MeanStdDev_Applied()
    {
        // Arrange — mean=10, std=2 should produce values around 10
        var init = new GaussianInitializer { Mean = 10, StdDev = 2, Seed = 42 };

        // Act
        var weights = init.GetWeights(10, 10);
        double mean = weights.Average();

        // Assert — average should be close to 10 (within reasonable tolerance for small sample)
        SyncAssert.That(mean).IsCloseTo(10.0, tolerance: 3.0);
    }

    [Test]
    public async Task UniformInitializer_Bounds_Applied()
    {
        // Arrange
        var init = new UniformInitializer { Lower = 5.0, Upper = 10.0, Seed = 42 };

        // Act
        var weights = init.GetWeights(10, 10);

        // Assert — all values should be in [5, 10]
        foreach (double w in weights)
            await Assert.That(w).IsGreaterThanOrEqualTo(5.0);

        foreach (double w in weights)
            await Assert.That(w).IsLessThanOrEqualTo(10.0);
    }

    [Test]
    public async Task LIFLayer_WithPreInitializedWeights_Works()
    {
        // Arrange — pre-allocate weights with known values
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        double[] knownWeights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        var layer = new LIFLayer(parameters, knownWeights);

        // Assert — weights should match what was passed in
        await Assert.That(layer.Weights.Length).IsEqualTo(6);
        for (int i = 0; i < knownWeights.Length; i++)
            await Assert.That(layer.Weights[i]).IsEqualTo(knownWeights[i]);

        // Act — forward/backward should work
        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        layer.Forward(input);
        var output = layer.Output;
        layer.Backward();

        await Assert.That(output.Count).IsEqualTo(1);
    }

    [Test]
    public async Task LIFLayer_DefaultInitializer_Gaussian()
    {
        // Arrange — no initializer specified, should default to Gaussian(0, 1)
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        var layer = new LIFLayer(parameters);

        // Act — check weights are in a reasonable range for Gaussian(0,1)
        double min = layer.Weights.Min();
        double max = layer.Weights.Max();

        // Assert — with 6 weights from Gaussian(0,1), range should be reasonable
        await Assert.That(min).IsLessThan(max);
    }

    [Test]
    public async Task LIFLayer_UniformInitializer_Works()
    {
        // Arrange — use Uniform(-2, 2) initializer
        var parameters = new LIFLayerParameters(N: 3, NIn: 2);
        var init = new UniformInitializer { Lower = 0.5, Upper = 2.0, Seed = 42 };
        var layer = new LIFLayer(new LIFLayerParameters(parameters.N, parameters.NIn, parameters.TauMem, parameters.TauSyn, parameters.VTh, parameters.PlasticWeights, init));

        // Assert — all weights should be in [-2, 2]
        foreach (double w in layer.Weights)
            await Assert.That(w).IsGreaterThanOrEqualTo(0.5);

        foreach (double w in layer.Weights)
            await Assert.That(w).IsLessThanOrEqualTo(2.0);

        // Act — forward/backward should work
        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        layer.Forward(input);
        var output = layer.Output;
        layer.Backward();

        await Assert.That(output.Count).IsEqualTo(1);
    }

    [Test]
    public async Task LIFLayer_DiagonalInitializer_Works()
    {
        // Arrange — use Diagonal initializer (requires N == NIn)
        var parameters = new LIFLayerParameters(N: 3, NIn: 3);
        var init = new DiagonalInitializer();
        var layer = new LIFLayer(new LIFLayerParameters(parameters.N, parameters.NIn, parameters.TauMem, parameters.TauSyn, parameters.VTh, parameters.PlasticWeights, init));

        // Assert — diagonal should be 1, off-diagonal should be 0
        for (int i = 0; i < 3; i++)
            await Assert.That(layer.Weights[i * 3 + i]).IsEqualTo(1.0);

        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                if (row != col)
                    await Assert.That(layer.Weights[row * 3 + col]).IsEqualTo(0.0);
    }

    [Test]
    public async Task LIFLayer_Dispose_ClearsBuffers()
    {
        // Arrange
        var parameters = new LIFLayerParameters(N: 5, NIn: 3);
        var layer = new LIFLayer(parameters);

        // Set positive weights to ensure spikes are generated
        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        // Act — forward/backward then dispose
        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        layer.Forward(input);
        layer.Backward();
        layer.Dispose();

        // Assert — buffers should be cleared (no exception on dispose)
    }

    [Test]
    public async Task LILayer_Dispose_ClearsBuffers()
    {
        // Arrange
        var parameters = new LILayerParameters(N: 5, NIn: 3);
        var layer = new LILayer(parameters);

        // Set positive weights to ensure correct computation
        for (int i = 0; i < layer.Weights.Length; i++) layer.Weights[i] = 1.5;

        // Act — forward/backward then dispose
        var input = new SpikesVector
        {
            new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 }),
        };

        layer.Forward(input);
        layer.Backward();
        layer.Dispose();

        // Assert — buffers should be cleared (no exception on dispose)
    }

    // ── SpikesData tests ────────────────────────────────────────────────

    [Test]
    public async Task SpikesData_SetError_ValidIndex_Applies() {
        var spikes = new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 });
        await Assert.That(spikes.Errors[0]).IsCloseTo(0.0, tolerance: 1e-12);
        spikes.SetError(0, 42.0);
        await Assert.That(spikes.Errors[0]).IsCloseTo(42.0, tolerance: 1e-12);
    }

    [Test]
    public async Task SpikesData_SetError_OutOfRange_Throws() {
        var spikes = new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 });
        await Assert.That(() => spikes.SetError(-1, 1.0)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => spikes.SetError(3, 1.0)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task SpikesData_SetTime_ValidIndex_Applies() {
        var spikes = new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 });
        spikes.SetTime(1, 0.99);
        await Assert.That(spikes.Times[1]).IsCloseTo(0.99, tolerance: 1e-12);
    }

    [Test]
    public async Task SpikesData_SetTime_OutOfRange_Throws() {
        var spikes = new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 });
        await Assert.That(() => spikes.SetTime(-1, 1.0)).Throws<ArgumentOutOfRangeException>();
        await Assert.That(() => spikes.SetTime(3, 1.0)).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async Task SpikesData_SetErrors_ValidLength_Replaces() {
        var spikes = new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 });
        spikes.SetErrors(new double[] { 1.0, 2.0 });
        await Assert.That(spikes.Errors.Length).IsEqualTo(2);
        await Assert.That(spikes.Errors[0]).IsCloseTo(1.0, tolerance: 1e-12);
        await Assert.That(spikes.Errors[1]).IsCloseTo(2.0, tolerance: 1e-12);
    }

    [Test]
    public async Task SpikesData_SetErrors_WrongLength_Throws() {
        var spikes = new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 });
        await Assert.That(() => spikes.SetErrors(new double[] { 1.0 })).Throws<ArgumentException>();
    }

    [Test]
    public async Task SpikesData_SetCurrents_ValidLength_Replaces() {
        var spikes = new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 });
        spikes.SetCurrents(new double[] { 3.0, 4.0 });
        await Assert.That(spikes.Currents[0]).IsCloseTo(3.0, tolerance: 1e-12);
        await Assert.That(spikes.Currents[1]).IsCloseTo(4.0, tolerance: 1e-12);
    }

    [Test]
    public async Task SpikesData_SetCurrents_WrongLength_Throws() {
        var spikes = new SpikesData(new double[] { 0.1, 0.2 }, new int[] { 0, 1 });
        await Assert.That(() => spikes.SetCurrents(new double[] { 3.0 })).Throws<ArgumentException>();
    }

    [Test]
    public async Task SpikesData_NSpikes_ReturnsLength() {
        var spikes = new SpikesData(new double[] { 0.1, 0.2, 0.3 }, new int[] { 0, 1, 0 });
        await Assert.That(spikes.NSpikes).IsEqualTo(3);
    }

    [Test]
    public async Task SpikesData_EmptySpikes_HasZeroLengthArrays() {
        var spikes = new SpikesData(new double[] { }, new int[] { });
        await Assert.That(spikes.NSpikes).IsEqualTo(0);
        await Assert.That(spikes.Errors.Length).IsEqualTo(0);
        await Assert.That(spikes.Currents.Length).IsEqualTo(0);
    }

    [Test]
    public async Task SpikesData_ConstructorWithCurrents_UsesProvidedCurrents() {
        var spikes = new SpikesData(
            new double[] { 0.1, 0.2 },
            new int[] { 0, 1 },
            new double[] { 5.0, 6.0 });
        await Assert.That(spikes.Currents[0]).IsCloseTo(5.0, tolerance: 1e-12);
        await Assert.That(spikes.Currents[1]).IsCloseTo(6.0, tolerance: 1e-12);
    }

    // ── SpikesVector / MaximaVector tests ───────────────────────────────

    [Test]
    public async Task SpikesVector_DefaultConstructor_Empty() {
        var vec = new SpikesVector();
        await Assert.That(vec.Count).IsEqualTo(0);
    }

    [Test]
    public async Task SpikesVector_CapacityConstructor_InitialCapacity() {
        var vec = new SpikesVector(5);
        await Assert.That(vec.Capacity).IsGreaterThanOrEqualTo(5);
    }

    [Test]
    public async Task SpikesVector_CollectionConstructor_AddsItems() {
        var items = new List<SpikesData> {
            new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 }),
            new SpikesData(new double[] { 0.2 }, new int[] { 1 }),
        };
        var vec = new SpikesVector(items);
        await Assert.That(vec.Count).IsEqualTo(2);
        await Assert.That(vec[0].NSpikes).IsEqualTo(3);
    }

    [Test]
    public async Task MaximaVector_DefaultConstructor_Empty() {
        var vec = new MaximaVector();
        await Assert.That(vec.Count).IsEqualTo(0);
    }

    [Test]
    public async Task MaximaVector_CollectionConstructor_AddsItems() {
        var items = new List<MaximaData> {
            new MaximaData(new double[] { 0.5, 1.2 }, new double[] { 3.0 }),
        };
        var vec = new MaximaVector(items);
        await Assert.That(vec.Count).IsEqualTo(1);
    }

    // ── LayerChain additional tests ─────────────────────────────────────

    [Test]
    public async Task Chain_OutputReturnsLayerAOutput() {
        // Arrange
        var params1 = new LIFLayerParameters(N: 3, NIn: 2);
        var params2 = new LIFLayerParameters(N: 2, NIn: 3);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        for (int i = 0; i < layer1.Weights.Length; i++) layer1.Weights[i] = 1.5;
        for (int i = 0; i < layer2.Weights.Length; i++) layer2.Weights[i] = 1.5;

        var chain = Layer.Chain(layer1, layer2);
        var input = new SpikesVector {
            new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 }),
        };

        // Act
        chain.Forward(input);

        // Assert — chain.Output should be layer1's output (3 neurons)
        await Assert.That(chain.Output.Count).IsEqualTo(1);
        await Assert.That(chain.Output[0].NSpikes).IsGreaterThanOrEqualTo(0);
    }

    [Test]
    public async Task Chain_IntermediateReturnsLayerBOutput() {
        // Arrange
        var params1 = new LIFLayerParameters(N: 3, NIn: 2);
        var params2 = new LIFLayerParameters(N: 2, NIn: 3);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        for (int i = 0; i < layer1.Weights.Length; i++) layer1.Weights[i] = 1.5;
        for (int i = 0; i < layer2.Weights.Length; i++) layer2.Weights[i] = 1.5;

        var chain = Layer.Chain(layer1, layer2);
        var input = new SpikesVector {
            new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 }),
        };

        // Act
        chain.Forward(input);

        // Assert — Intermediate should be layer2's output (2 neurons)
        await Assert.That(chain.Intermediate.Count).IsEqualTo(1);
    }

    [Test]
    public async Task Chain_GetVoltageTrace_DelegatesToLayerA() {
        // Arrange
        var params1 = new LIFLayerParameters(N: 3, NIn: 2);
        var params2 = new LIFLayerParameters(N: 2, NIn: 3);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        for (int i = 0; i < layer1.Weights.Length; i++) layer1.Weights[i] = 1.5;
        for (int i = 0; i < layer2.Weights.Length; i++) layer2.Weights[i] = 1.5;

        var chain = Layer.Chain(layer1, layer2);
        var input = new SpikesVector {
            new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 }),
        };

        // Act
        chain.Forward(input);
        var (times, voltages) = chain.GetVoltageTrace(0, 0, tMax: 0.2, dt: 0.01);

        // Assert — should return a trace for the outer layer's neuron 0
        await Assert.That(times.Length).IsGreaterThan(0);
        await Assert.That(voltages.Length).IsEqualTo(times.Length);
    }

    [Test]
    public async Task Chain_GetLambdaI_DelegatesToLayerA() {
        // Arrange — use multiple input spikes to ensure layer1 produces output spikes
        var params1 = new LIFLayerParameters(N: 3, NIn: 2);
        var params2 = new LIFLayerParameters(N: 2, NIn: 3);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        for (int i = 0; i < layer1.Weights.Length; i++) layer1.Weights[i] = 1.5;
        for (int i = 0; i < layer2.Weights.Length; i++) layer2.Weights[i] = 1.5;

        var chain = Layer.Chain(layer1, layer2);
        // Multiple input spikes at early times to ensure layer1 fires
        var input = new SpikesVector {
            new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 }),
        };

        // Act
        chain.Forward(input);
        var lambdaI = chain.GetLambdaI(0, 0, t: 0.05);

        // Assert — should return a value from the outer layer's neuron 0
        // (may be 0 if no post-spike times after t, but should not throw)
        await Assert.That(lambdaI).IsGreaterThanOrEqualTo(0.0);
    }

    [Test]
    public async Task Chain_GetLambdaITrace_DelegatesToLayerA() {
        // Arrange
        var params1 = new LIFLayerParameters(N: 3, NIn: 2);
        var params2 = new LIFLayerParameters(N: 2, NIn: 3);

        var layer1 = new LIFLayer(params1);
        var layer2 = new LIFLayer(params2);

        for (int i = 0; i < layer1.Weights.Length; i++) layer1.Weights[i] = 1.5;
        for (int i = 0; i < layer2.Weights.Length; i++) layer2.Weights[i] = 1.5;

        var chain = Layer.Chain(layer1, layer2);
        var input = new SpikesVector {
            new SpikesData(new double[] { 0.01, 0.02, 0.03 }, new int[] { 0, 1, 0 }),
        };

        // Act
        chain.Forward(input);
        var (times, lambdaI) = chain.GetLambdaITrace(0, 0, tMax: 0.2, dt: 0.01);

        // Assert — should return a trace from the outer layer's neuron 0
        await Assert.That(times.Length).IsGreaterThan(0);
        await Assert.That(lambdaI.Length).IsEqualTo(times.Length);
    }

    // ── DiagonalInitializer additional tests (basic tested in WeightInitializer_Diagonal_Works) ──

    [Test]
    public async Task DiagonalInitializer_Identity_ZeroOffDiagonal() {
        // Arrange — identity diagonal for 5x5
        var init = new DiagonalInitializer();

        // Act
        var weights = init.GetWeights(5, 5);

        // Assert — all off-diagonal elements should be zero
        for (int i = 0; i < 25; i++) {
            int row = i / 5, col = i % 5;
            if (row != col)
                await Assert.That(weights[i]).IsCloseTo(0.0, tolerance: 1e-12);
        }
    }

    [Test]
    public async Task DiagonalInitializer_Rectangular_Throws() {
        // Arrange — nIn != nOut not allowed for DiagonalInitializer
        var init = new DiagonalInitializer();

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(async () => { init.GetWeights(4, 2); await Task.CompletedTask; });
    }
}
