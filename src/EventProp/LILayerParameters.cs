namespace CW.EventProp;

/// <summary>
/// Parameters for a Leaky Integrate (LI) layer (no spike threshold, uses maxima-based loss).
/// </summary>
public readonly record struct LILayerParameters(
    int N,                    // number of post-synaptic neurons
    int NIn,                  // number of pre-synaptic neurons (input features)
    double TauMem = 20e-3,    // membrane time constant (seconds)
    double TauSyn = 5e-3,     // synaptic time constant (seconds)
    bool PlasticWeights = true,
    WeightInitializer? Weights = null  // null = default Gaussian(0, 1)
);
