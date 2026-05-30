"""EventProp C bindings — loads _native.so and provides helper functions."""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

from .spikes import Spikes, Maxima, SpikesVector, MaximaVector


# ── Native function bindings (loaded from _native.so) ───────────────────────

try:
    from ._native import (
        _compute_spikes_batch_raw,
        _backward_spikes_batch_raw,
        _compute_maxima_batch_raw,
        _backward_maxima_batch_raw,
        _compute_voltage_trace_raw,
        _compute_lambda_i_raw,
        _compute_lambda_i_trace_raw,
        _dropout_spikes_raw,
    )
except ImportError:
    _compute_spikes_batch_raw = None
    _backward_spikes_batch_raw = None
    _compute_maxima_batch_raw = None
    _backward_maxima_batch_raw = None
    _compute_voltage_trace_raw = None
    _compute_lambda_i_raw = None
    _compute_lambda_i_trace_raw = None
    _dropout_spikes_raw = None
    _backward_spikes_batch_raw = None
    _compute_maxima_batch_raw = None
    _backward_maxima_batch_raw = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _infer_n(w):
    """Infer (n_pre, n_post) from weight matrix shape."""
    if hasattr(w, 'shape'):
        return w.shape[0], w.shape[1]
    return len(w), len(w[0]) if w else 0


def _spikes_to_array_tuple(spikes_obj):
    """Convert Spikes → (times, sources, currents, errors) for zero-copy C access."""
    return (spikes_obj.times, spikes_obj.sources, spikes_obj.currents, spikes_obj.errors)


def _maxima_to_array_tuple(max_obj):
    """Convert Maxima → (times_array, values_array) for zero-copy C++ access."""
    return (max_obj.times, max_obj.values)


def _from_tuple(tup, n_post):
    """Convert (times_array, sources_array, currents_array, first_spike_times, first_spike_idxs) to Spikes.
    
    The C library now computes first_spike per neuron natively.
    """
    times = np.asarray(tup[0], dtype=np.float64)
    sources = np.asarray(tup[1], dtype=np.int32)
    currents = np.asarray(tup[2], dtype=np.float64) if len(tup) > 2 else np.zeros_like(times)
    
    # Extract first_spike data (5-tuple from C library, 3-tuple for backward compat)
    if len(tup) >= 5:
        first_spike_times = list(tup[3])
        first_spike_idxs = list(tup[4])
    else:
        # Fallback: compute from raw spike data (for backward compat or manual Spikes construction)
        first_spike_times = None
        first_spike_idxs = None
    
    return Spikes(times, sources, currents,
                  first_spike_times=first_spike_times,
                  first_spike_idxs=first_spike_idxs)


def _from_maxima_tuple(tup):
    """Convert (times_array, values_array) to Maxima."""
    return Maxima(np.asarray(tup[0], dtype=np.float64), np.asarray(tup[1], dtype=np.float64))


# ── Public API (matches original eventprop_cpp interface) ───────────────────

def compute_spikes_batch_cpp(w, batch, v_th, tau_mem, tau_syn):
    """Forward pass: predict post-synaptic spike times."""
    if _compute_spikes_batch_raw is None:
        raise RuntimeError("eventprop._native not found")
    n_pre, n_post = _infer_n(w)
    array_tuples = [_spikes_to_array_tuple(s) for s in batch]
    raw_result = _compute_spikes_batch_raw(w, n_pre, n_post, array_tuples, v_th, tau_mem, tau_syn)
    return SpikesVector([_from_tuple(t, n_post) for t in raw_result])


def backward_spikes_batch_cpp(input_batch, post_batch, w, gradient, v_th, tau_mem, tau_syn):
    """Backward pass for spikes loss."""
    if _backward_spikes_batch_raw is None:
        raise RuntimeError("eventprop._native not found")
    n_pre, n_post = _infer_n(w)
    input_tuples = [_spikes_to_array_tuple(s) for s in input_batch]
    post_tuples = [_spikes_to_array_tuple(s) for s in post_batch]
    _backward_spikes_batch_raw(input_tuples, post_tuples, w, n_pre, n_post,
                               gradient, v_th, tau_mem, tau_syn)


def compute_maxima_batch_cpp(w, batch, tau_mem, tau_syn):
    """Compute membrane potential maxima per neuron."""
    if _compute_maxima_batch_raw is None:
        raise RuntimeError("eventprop._native not found")
    n_pre, n_post = _infer_n(w)
    array_tuples = [_spikes_to_array_tuple(s) for s in batch]
    raw_result = _compute_maxima_batch_raw(w, n_pre, n_post, array_tuples, tau_mem, tau_syn)
    return MaximaVector([_from_maxima_tuple(t) for t in raw_result])


def backward_maxima_batch_cpp(input_batch, maxima_list, w, gradient, tau_mem, tau_syn):
    """Backward pass for maxima loss."""
    if _backward_maxima_batch_raw is None:
        raise RuntimeError("eventprop._native not found")
    n_pre, n_post = _infer_n(w)
    input_tuples = [_spikes_to_array_tuple(s) for s in input_batch]
    max_tuples = [_maxima_to_array_tuple(m) for m in maxima_list]
    _backward_maxima_batch_raw(input_tuples, max_tuples, w, n_pre, n_post,
                               gradient, tau_mem, tau_syn)


def compute_voltage_trace_cpp(t_max, dt, target_nrn_idx, w,
                               input_spikes, v_th, tau_mem, tau_syn):
    """Compute voltage trace for a target neuron."""
    if _compute_voltage_trace_raw is None:
        raise RuntimeError("eventprop._native not found")
    n_pre, n_post = _infer_n(w)
    array_tuple = _spikes_to_array_tuple(input_spikes)
    return _compute_voltage_trace_raw(t_max, dt, target_nrn_idx,
                                       w, n_pre, n_post, array_tuple,
                                       v_th, tau_mem, tau_syn)


def compute_lambda_i_cpp(t, target_nrn_idx, post_spikes,
                          v_th, tau_mem, tau_syn):
    """Compute lambda_i for a target neuron at time t."""
    if _compute_lambda_i_raw is None:
        raise RuntimeError("eventprop._native not found")
    array_tuple = _spikes_to_array_tuple(post_spikes)
    return _compute_lambda_i_raw(t, target_nrn_idx,
                                  array_tuple, v_th, tau_mem, tau_syn)


def compute_lambda_i_trace_cpp(t_max, dt, target_nrn_idx, post_spikes,
                                v_th, tau_mem, tau_syn):
    """Compute lambda_i trace for a target neuron."""
    if _compute_lambda_i_trace_raw is None:
        raise RuntimeError("eventprop._native not found")
    array_tuple = _spikes_to_array_tuple(post_spikes)
    return _compute_lambda_i_trace_raw(t_max, dt, target_nrn_idx,
                                        array_tuple, v_th, tau_mem, tau_syn)


def dropout_spikes_cpp(batch, p_drop=0.0, random_seed=42):
    """Randomly drop spikes from batch."""
    if _dropout_spikes_raw is None:
        raise RuntimeError("eventprop._native not found")
    array_tuples = [_spikes_to_array_tuple(s) for s in batch]
    raw_result = _dropout_spikes_raw(array_tuples, p_drop, random_seed)
    return [Spikes(t[0], t[1]) for t in raw_result]



