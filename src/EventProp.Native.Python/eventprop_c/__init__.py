"""EventProp ctypes bindings — zero-copy access to C library.

Provides Python wrappers around the EventProp C alloc functions with
zero-copy memory access to C-allocated arrays. No numpy required.

Example:
    from eventprop_ctypes import EventPropCtypes
    
    lib = EventPropCtypes()
    
    # Forward pass
    output = lib.compute_spikes(w, times, sources, v_th=1.0)
    print(f"Generated {output.n_spikes} spikes")
    
    # Zero-copy access to spike times (returns memoryview)
    times_view = output.times  # No copy!
    
    # Cleanup (frees C memory)
    del output
"""

from .native import EventPropCtypes as NativeApi, SpikeOutput, SumsOutput, MaximaOutput
from .native import BackwardOutput
from .native import VoltageTraceOutput, LambdaITraceOutput
from .spikes import Spikes, SpikesVector, Maxima, MaximaVector

# Layer classes
from .layer import (
    SpikeDataset,
    WeightDistribution,
    GaussianDistribution,
    UniformDistribution,
    DiagonalWeights,
    Layer
)
from .lif_layer import LIFLayerParameters, LIFLayer
from .li_layer import LILayerParameters, LILayer
from .loss_layer import (
    TTFSCrossEntropyLossParameters,
    VMaxCrossEntropyLossParameters,
    TTFSCrossEntropyLoss,
    VMaxCrossEntropyLoss
)

# Optimizer classes
from .optimizer import (
    GradientDescentParameters,
    Optimizer,
    GradientDescent,
    Adam,
)

# Training base classes
from .training import (
    dropout_spikes,
    AbstractTraining,
    AbstractTwoLayer,
)

# TTFS training
from .ttfs_training import (
    TTFSMixin,
    TwoLayerTTFS,
)

# VMax training
from .vmax_training import (
    OneLayerVMax,
    TwoLayerVMax,
)

__all__ = [
    # Native API and output wrappers
    'NativeApi',
    'SpikeOutput',
    'SumsOutput',
    'MaximaOutput',
    'BackwardOutput',
    'VoltageTraceOutput',
    'LambdaITraceOutput',
    # Python data classes (matching pybind11 interface)
    'Spikes',
    'SpikesVector',
    'Maxima',
    'MaximaVector',
    # Layer base classes
    'SpikeDataset',
    'WeightDistribution',
    'GaussianDistribution',
    'UniformDistribution',
    'DiagonalWeights',
    'Layer',
    # Layer classes
    'LIFLayerParameters',
    'LIFLayer',
    'LILayerParameters',
    'LILayer',
    # Loss layer classes
    'TTFSCrossEntropyLossParameters',
    'VMaxCrossEntropyLossParameters',
    'TTFSCrossEntropyLoss',
    'VMaxCrossEntropyLoss',
    # Optimizer classes
    'GradientDescentParameters',
    'Optimizer',
    'GradientDescent',
    'Adam',
    # Training utilities
    'dropout_spikes',
    'AbstractTraining',
    'AbstractTwoLayer',
    # TTFS training
    'TTFSMixin',
    'TwoLayerTTFS',
    # VMax training
    'OneLayerVMax',
    'TwoLayerVMax',
]
