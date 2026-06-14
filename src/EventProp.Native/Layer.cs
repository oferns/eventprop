using System.Diagnostics;

namespace CW.EventProp;

/// <summary>
/// Base interface for spiking neural network layers.
/// Native-backed — delegates to the C library via P/Invoke batch operations.
/// </summary>
public abstract class Layer
{
    /// <summary>The input batch stored during forward pass.</summary>
    protected SpikesVector? _inputBatch;

    /// <summary>The input batch for this layer.</summary>
    protected internal SpikesVector? InputBatch => _inputBatch;

    /// <summary>The output of the forward pass (set by derived classes).</summary>
    protected SpikesVector? _output;

    /// <summary>The output spikes from the forward pass.</summary>
    public SpikesVector Output => _output ?? throw new InvalidOperationException("Forward has not been run.");

    /// <summary>Whether forward has been run (for debugging).</summary>
    protected bool _ranForward = false;

    /// <summary>Whether backward has been run (for debugging).</summary>
    protected bool _ranBackward = false;

    /// <summary>The accumulated weight gradient.</summary>
    public double[] Gradient { get; protected set; } = Array.Empty<double>();

    /// <summary>
    /// Run forward pass. Stores the input batch for use in backward.
    /// </summary>
    public abstract void Forward(SpikesVector inputBatch);

    /// <summary>
    /// Run backward pass. Computes gradients and propagates errors to ancestors.
    /// </summary>
    public abstract void Backward();

    /// <summary>Zero the weight gradient.</summary>
    public abstract void ZeroGrad();

    /// <summary>
    /// Compute the voltage trace for a single neuron during forward pass.
    /// </summary>
    public virtual (double[] times, double[] voltages) GetVoltageTrace(
        int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
    {
        throw new InvalidOperationException("Voltage trace not supported for this layer type.");
    }

    /// <summary>
    /// Compute λ_i at a specific time for a single neuron.
    /// </summary>
    public virtual double GetLambdaI(int batchIdx, int targetNrnIdx, double t)
    {
        throw new InvalidOperationException("λ_i not supported for this layer type.");
    }

    /// <summary>
    /// Compute the λ_i trace over time for a single neuron.
    /// </summary>
    public virtual (double[] times, double[] lambdaI) GetLambdaITrace(
        int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
    {
        throw new InvalidOperationException("λ_i trace not supported for this layer type.");
    }

    /// <summary>Get the weight matrix as a Span (row-major, nIn × nOut).</summary>
    public abstract ReadOnlySpan<double> GetWeights();

    /// <summary>Set the weight matrix from a Span (row-major, nIn × nOut).</summary>
    public abstract void SetWeights(ReadOnlySpan<double> weights);

    /// <summary>Number of input neurons.</summary>
    public abstract int NIn { get; }

    /// <summary>Number of output neurons.</summary>
    public abstract int NOut { get; }

    /// <summary>
    /// Chain layers together: layerA(layerB(input)) propagates backward through both.
    /// </summary>
    public static LayerChain<TLayerA, TLayerB> Chain<TLayerA, TLayerB>(TLayerA layerA, TLayerB layerB)
        where TLayerA : Layer
        where TLayerB : Layer => new(layerA, layerB);

    /// <summary>
    /// Chain two layers: input → layerB → layerA, backward propagates through both.
    /// </summary>
    public sealed class LayerChain<TLayerA, TLayerB> : Layer
        where TLayerA : Layer
        where TLayerB : Layer
    {
        private readonly TLayerA _layerA;
        private readonly TLayerB _layerB;

        public LayerChain(TLayerA layerA, TLayerB layerB)
        {
            _layerA = layerA;
            _layerB = layerB;
        }

        public TLayerA LayerA => _layerA;
        public TLayerB LayerB => _layerB;

        public override void Forward(SpikesVector inputBatch)
        {
            _layerB.Forward(inputBatch);
            var intermediate = _layerB.Output;
            _layerA.Forward(intermediate);
            _ranForward = true;
        }

        public override void Backward()
        {
            _layerA.Backward();
            _layerB.Backward();
            _ranBackward = true;
        }

        public override void ZeroGrad()
        {
            _layerA.ZeroGrad();
            _layerB.ZeroGrad();
        }

        public override (double[] times, double[] voltages) GetVoltageTrace(
            int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
            => _layerA.GetVoltageTrace(batchIdx, targetNrnIdx, tMax, dt);

        public override double GetLambdaI(int batchIdx, int targetNrnIdx, double t)
            => _layerA.GetLambdaI(batchIdx, targetNrnIdx, t);

        public override (double[] times, double[] lambdaI) GetLambdaITrace(
            int batchIdx, int targetNrnIdx, double tMax = 1.0, double dt = 1e-4)
            => _layerA.GetLambdaITrace(batchIdx, targetNrnIdx, tMax, dt);

        public override ReadOnlySpan<double> GetWeights() => _layerA.GetWeights();
        public override void SetWeights(ReadOnlySpan<double> weights) => _layerA.SetWeights(weights);
        public override int NIn => _layerB.NIn;
        public override int NOut => _layerA.NOut;

        /// <summary>Output spikes from the chain (output of layerA).</summary>
        public new SpikesVector Output => _layerA.Output;

        /// <summary>Intermediate spikes between layerB and layerA.</summary>
        public SpikesVector Intermediate => _layerB.Output;
    }
}
