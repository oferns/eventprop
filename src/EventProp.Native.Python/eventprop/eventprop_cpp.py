"""Shim module providing pybind11-compatible interface via ctypes.

This module mirrors the EventProp.Python pybind11 interface but calls through
the C library via ctypes. Allows Python code to switch backends just by changing imports.

Example:
    # Original pybind11 code:
    from eventprop.eventprop_cpp import compute_spikes_batch_cpp, SpikesVector
    
    # With ctypes (same API):
    from eventprop.eventprop_cpp import compute_spikes_batch_cpp, SpikesVector
    
    # Code works identically
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

from .native import EventPropCtypes
from .spikes import Spikes, SpikesVector, Maxima, MaximaVector

# Re-export for compatibility with original pybind11 module
__all__ = [
    'Spikes',
    'SpikesVector', 
    'Maxima',
    'MaximaVector',
    # Functions
    'compute_spikes_batch_cpp',
    'backward_spikes_batch_cpp',
    'compute_sums_batch_cpp',
    'compute_maxima_batch_cpp',
    'backward_maxima_batch_cpp',
    'compute_voltage_trace_cpp',
    'compute_lambda_i_cpp',
    'compute_lambda_i_trace_cpp',
    'dropout_spikes_cpp',
]

# Global library instance (cached)
_lib: EventPropCtypes | None = None


def _get_lib() -> EventPropCtypes:
    """Get or create the global ctypes library instance."""
    global _lib
    if _lib is None:
        # Try to find the library automatically
        import os
        lib_paths = [
            os.path.join(os.path.dirname(__file__), "libeventprop_c_api.dylib"),
            os.path.join(os.path.dirname(__file__), "libeventprop_c_api.so"),
        ]
        
        # Try project root (common case when running from src/EventProp.Python.Benchmarks)
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent.parent  # EventProp.Python.Benchmarks
        lib_paths.extend([
            project_root / "EventProp.C" / "build" / "libeventprop_c_api.dylib",
            project_root.parent / "EventProp.C" / "build" / "libeventprop_c_api.dylib",
        ])
        
        for path in lib_paths:
            if pathlib.Path(path).exists():
                _lib = EventPropCtypes(str(path))
                return _lib
        
        raise FileNotFoundError(
            "Could not find libeventprop_c_api.dylib/.so. "
            "Run ./build.sh c first, or provide path to EventPropCtypes."
        )
    
    return _lib


def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous (required for ctypes)."""
    if not arr.flags.c_contiguous:
        return np.ascontiguousarray(arr)
    return arr


def _to_flat_list(arr: np.ndarray) -> list:
    """Convert numpy array to flat Python list."""
    return arr.flatten().tolist()


def _to_times_sources_list(spikes: Spikes) -> tuple[list, list]:
    """Extract times and sources as flat lists."""
    return spikes.times.flatten().tolist(), spikes.sources.flatten().tolist()


def compute_spikes_batch_cpp(w: np.ndarray, batch: SpikesVector,
                            v_th: float = 1.0, tau_mem: float = 0.02,
                            tau_syn: float = 0.005) -> SpikesVector:
    """Compute post-synaptic spikes for a batch of inputs.

    Args:
        w: Weight matrix (n_in x n_out), row-major numpy array
        batch: SpikesVector of input spikes
        v_th: Spike threshold voltage
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)

    Returns:
        SpikesVector of post-synaptic spikes for each input in the batch.
    """
    lib = _get_lib()
    
    n_pre, n_post = w.shape
    
    # Flatten weight matrix to row-major list
    w_flat = _to_flat_list(w)
    
    results = []
    for input_spikes in batch:
        times_list, sources_list = _to_times_sources_list(input_spikes)
        
        # Call C library
        c_output = lib.compute_spikes(
            w=w_flat, times=times_list, sources=sources_list,
            n_pre=n_pre, n_post=n_post,
            v_th=v_th, tau_mem=tau_mem, tau_syn=tau_syn
        )
        
        # Wrap in Spikes object (zero-copy)
        results.append(Spikes.from_c_output(c_output))
    
    return SpikesVector(results)


def backward_spikes_batch_cpp(input_batch: SpikesVector, post_batch: SpikesVector,
                             w: np.ndarray, gradient: np.ndarray,
                             v_th: float = 1.0, tau_mem: float = 0.02,
                             tau_syn: float = 0.005):
    """Compute weight gradients via backward pass for spikes.

    Args:
        input_batch: Pre-synaptic spike times and sources
        post_batch: Post-synaptic spikes with errors set
        w: Weight matrix (n_in x n_out)
        gradient: Output buffer for weight gradients (mutated in-place)
        v_th: Spike threshold voltage
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)
    """
    lib = _get_lib()
    
    n_pre, n_post = w.shape
    
    # Flatten weight matrix
    w_flat = _to_flat_list(w)
    
    for i in range(len(input_batch)):
        input_spikes = input_batch[i]
        post_spikes = post_batch[i]
        
        # Extract arrays as lists
        input_times, input_sources = _to_times_sources_list(input_spikes)
        post_times, post_sources = _to_times_sources_list(post_spikes)
        
        # Post-synaptic currents and errors (approximate if not available)
        post_currents = [1.0] * len(post_times)  # TODO: get from forward pass
        post_errors = list(post_spikes.errors) if hasattr(post_spikes, 'errors') else [0.0] * len(post_times)
        
        # Call C library
        backward_output = lib.backward_spikes(
            times=input_times, sources=input_sources,
            fwd_times=post_times, fwd_sources=post_sources,
            fwd_currents=post_currents, post_errors=post_errors,
            w=w_flat, n_pre=n_pre, n_post=n_post,
            v_th=v_th, tau_mem=tau_mem, tau_syn=tau_syn
        )
        
        # Copy gradient to output buffer (zero-copy via numpy view)
        grad_view = np.ctypeslib.as_array(backward_output.grad, shape=(n_pre * n_post,))
        gradient += grad_view.reshape(n_pre, n_post)


def compute_sums_batch_cpp(w: np.ndarray, batch: SpikesVector,
                          tau_mem: float = 0.02, tau_syn: float = 0.005) -> Tuple[SpikesVector, SpikesVector]:
    """Compute cumulative exponential-weighted sums.

    Args:
        w: Weight matrix (n_in x n_out)
        batch: SpikesVector of input spikes
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)

    Returns:
        Tuple of sum0 and sum1 vectors.
    """
    lib = _get_lib()
    
    n_pre, n_post = w.shape
    w_flat = _to_flat_list(w)
    
    results_sum0, results_sum1 = [], []
    for input_spikes in batch:
        times_list, sources_list = _to_times_sources_list(input_spikes)
        
        c_output = lib.compute_sums(
            w=w_flat, times=times_list, sources=sources_list,
            n_pre=n_pre, n_post=n_post,
            tau_mem=tau_mem, tau_syn=tau_syn
        )
        
        # Return as tuples for now (sums don't map to Spikes)
        results_sum0.append(c_output.sum0.to_list())
        results_sum1.append(c_output.sum1.to_list())
    
    return results_sum0, results_sum1


def compute_maxima_batch_cpp(w: np.ndarray, batch: SpikesVector,
                            tau_mem: float = 0.02, tau_syn: float = 0.005) -> MaximaVector:
    """Compute membrane potential maxima for each post-synaptic neuron.

    Args:
        w: Weight matrix (n_in x n_out)
        batch: SpikesVector of input spikes
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)

    Returns:
        MaximaVector of maxima times and values.
    """
    lib = _get_lib()
    
    n_pre, n_post = w.shape
    w_flat = _to_flat_list(w)
    
    results = []
    for input_spikes in batch:
        times_list, sources_list = _to_times_sources_list(input_spikes)
        
        c_output = lib.compute_maxima(
            w=w_flat, times=times_list, sources=sources_list,
            n_pre=n_pre, n_post=n_post,
            tau_mem=tau_mem, tau_syn=tau_syn
        )
        
        results.append(Maxima.from_c_output(c_output))
    
    return MaximaVector(results)


def backward_maxima_batch_cpp(input_batch: SpikesVector, maxima: MaximaVector,
                             w: np.ndarray, gradient: np.ndarray,
                             tau_mem: float = 0.02, tau_syn: float = 0.005):
    """Compute weight gradients via backward pass for maxima-based loss.

    Args:
        input_batch: Pre-synaptic spike times and sources
        maxima: Maxima times, values, and errors
        w: Weight matrix (n_in x n_out) — not used but required for API compatibility
        gradient: Output buffer for weight gradients (mutated in-place)
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)
    """
    lib = _get_lib()
    
    n_pre, n_post = w.shape
    
    # Flatten weight matrix (not used by C function but required)
    w_flat = _to_flat_list(w)
    
    for i in range(len(input_batch)):
        input_spikes = input_batch[i]
        max_data = maxima[i]
        
        # Extract arrays as lists
        input_times, input_sources = _to_times_sources_list(input_spikes)
        
        # Call C library
        backward_output = lib.backward_maxima(
            input_times=input_times, input_sources=input_sources,
            maxima_times=list(max_data.times),
            maxima_values=list(max_data.values),
            maxima_errors=list(max_data.errors),
            w=w_flat, n_pre=n_pre, n_post=n_post,
            tau_mem=tau_mem, tau_syn=tau_syn
        )
        
        # Copy gradient to output buffer
        grad_view = np.ctypeslib.as_array(backward_output.gradient, shape=(n_pre * n_post,))
        gradient += grad_view.reshape(n_pre, n_post)


def compute_voltage_trace_cpp(t_max: float, dt: float, target_nrn_idx: int,
                             w: np.ndarray, spikes: Spikes, v_th: float = 1.0,
                             tau_mem: float = 0.02, tau_syn: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """Compute voltage trace for a single post-synaptic neuron.

    Args:
        t_max: Maximum simulation time
        dt: Time step
        target_nrn_idx: Post-synaptic neuron index
        w: Weight matrix (n_in x n_out)
        spikes: Input spikes
        v_th: Spike threshold voltage
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)

    Returns:
        Tuple of (times, voltages) numpy arrays.
    """
    lib = _get_lib()
    
    n_pre, n_post = w.shape
    
    trace_output = lib.compute_voltage_trace(
        t_max=t_max, dt=dt, target_nrn_idx=target_nrn_idx,
        w=_to_flat_list(w), n_pre=n_pre, n_post=n_post,
        times=spikes.times.flatten().tolist(),
        sources=spikes.sources.flatten().tolist(),
        v_th=v_th, tau_mem=tau_mem, tau_syn=tau_syn
    )
    
    return np.array(list(trace_output.ts)), np.array(list(trace_output.v_trace))


def compute_lambda_i_cpp(t: float, target_nrn_idx: int, post_spikes: Spikes,
                        v_th: float = 1.0, tau_mem: float = 0.02,
                        tau_syn: float = 0.005) -> float:
    """Compute synaptic adjoint at a specific time.

    Args:
        t: Time at which to compute lambda_i
        target_nrn_idx: Post-synaptic neuron index
        post_spikes: Post-synaptic spikes
        v_th: Spike threshold voltage
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)

    Returns:
        Lambda-i value at the given time
    """
    lib = _get_lib()
    
    return lib.compute_lambda_i(
        t=t, target_nrn_idx=target_nrn_idx,
        fwd_times=post_spikes.times.flatten().tolist(),
        n_post_spikes=post_spikes.n_spikes,
        v_th=v_th, tau_mem=tau_mem, tau_syn=tau_syn
    )


def compute_lambda_i_trace_cpp(t_max: float, dt: float, target_nrn_idx: int,
                              post_spikes: Spikes, v_th: float = 1.0,
                              tau_mem: float = 0.02, tau_syn: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lambda-I trace over time.

    Args:
        t_max: Maximum simulation time
        dt: Time step
        target_nrn_idx: Post-synaptic neuron index
        post_spikes: Post-synaptic spikes
        v_th: Spike threshold voltage
        tau_mem: Membrane time constant (seconds)
        tau_syn: Synaptic time constant (seconds)

    Returns:
        Tuple of (times, lambda-I values) numpy arrays.
    """
    lib = _get_lib()
    
    trace_output = lib.compute_lambda_i_trace(
        t_max=t_max, dt=dt, target_nrn_idx=target_nrn_idx,
        fwd_times=post_spikes.times.flatten().tolist(),
        fwd_sources=post_spikes.sources.flatten().tolist(),
        v_th=v_th, tau_mem=tau_mem, tau_syn=tau_syn
    )
    
    return np.array(list(trace_output.ts)), np.array(list(trace_output.lambda_trace))


def dropout_spikes_cpp(spikes: SpikesVector, p_drop: float = 0.5, random_seed: int = None) -> SpikesVector:
    """Dropout spikes with given probability.

    For each spike in the vector, randomly decides whether to keep it or drop it
    using a binomial distribution.

    Args:
        spikes: SpikesVector of input spikes
        p_drop: Dropout probability (0 = keep all, 1 = drop all)
        random_seed: Random seed for reproducibility

    Returns:
        SpikesVector with some spikes dropped.
    """
    import random
    
    if random_seed is not None:
        random.seed(random_seed)
    
    dropped_spikes = []
    for spike in spikes:
        times, sources = [], []
        for i in range(spike.n_spikes):
            # Keep spike if random number > p_drop
            if random.random() > p_drop:
                times.append(spike.times[i])
                sources.append(spike.sources[i])
        
        if times:
            dropped_spikes.append(Spikes(
                np.array(times, dtype=np.float64),
                np.array(sources, dtype=np.int32)
            ))
        else:
            dropped_spikes.append(Spikes())
    
    return SpikesVector(dropped_spikes)
