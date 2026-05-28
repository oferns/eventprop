namespace CW.EventProp;

/// <summary>
/// Parameters for a Leaky Integrate-and-Fire (LIF) layer.
/// </summary>
public readonly record struct LIFLayerParameters(
    int N,                    // number of post-synaptic neurons
    int NIn,                  // number of pre-synaptic neurons (input features)
    double TauMem = 20e-3,    // membrane time constant (seconds)
    double TauSyn = 5e-3,     // synaptic time constant (seconds)
    double VTh = 1.0,         // spike threshold voltage
    bool PlasticWeights = true,
    WeightInitializer? Weights = null  // null = default Gaussian(0, 1)
);
