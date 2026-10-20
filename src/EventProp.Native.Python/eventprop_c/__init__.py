"""EventProp Cython bindings — zero-copy access to C library.

Provides Python wrappers around the EventProp C alloc functions with
zero-copy memory access via compiled Cython extensions. No ctypes overhead.

Example:
    from eventprop_c import compute_spikes_batch, backward_spikes_batch
    
    # Forward pass (batch)
    output = compute_spikes_batch(times, sources, n_per_sample, w, ...)
    
    # Zero-copy access to spike times (returns numpy arrays referencing C memory)
    for i in range(batch_size):
        times_i, sources_i = output[i]
"""

# Windows: register the directory holding eventprop_c_api.dll BEFORE the extension
# is imported. Python 3.8+ does not search PATH for a pyd's dependencies, so
# os.add_dll_directory is the only mechanism that works from Python; see
# _library_path.py for the measured failure this fixes.
# Linux: the same call preloads libeventprop_c_api.so, which is what makes
# EVENTPROP_C_LIB_DIR work there. macOS: no-op — setup.py links an rpath.
from ._library_path import register_c_library_dirs as _register_c_library_dirs

_register_c_library_dirs()

del _register_c_library_dirs  # keep the package namespace to the public API

from ._native import (
    compute_spikes_batch,
    backward_spikes_batch_accumulate as backward_spikes_batch,
    compute_maxima_batch,
    backward_maxima_batch_accumulate as backward_maxima_batch,
    dropout_spikes_batch,
    abi_check,
    abi_struct_sizes,
)
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
    # Native batch functions
    'compute_spikes_batch',
    'backward_spikes_batch',  # alias for backward_spikes_batch_accumulate
    'compute_maxima_batch',
    'backward_maxima_batch',  # alias for backward_maxima_batch_accumulate
    'dropout_spikes_batch',
    # ABI self-check (it also runs at import; exposed for tests and diagnostics)
    'abi_check',
    'abi_struct_sizes',
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