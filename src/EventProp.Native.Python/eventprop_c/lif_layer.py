"""LIF layer — Cython bindings, zero-copy numpy arrays."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from ._native import (
    compute_spikes_batch_ptrs, backward_spikes_batch_accumulate_ptrs,
)
from .spikes import Spikes, SpikesVector


# fmt: off
class LIFLayerParameters(NamedTuple):
    n               : int   = None
    n_in            : int   = None
    tau_mem         : float = 20e-3 # s
    tau_syn         : float = 5e-3  # s
    v_th            : float = 1.
    v_leak          : float = 0
    plastic_weights : bool  = True
    w_dist          : WeightDistribution = GaussianDistribution()
# fmt: on


class LIFLayer(Layer):
    def __init__(self, parameters: LIFLayerParameters, w_in: np.ndarray = None):
        super().__init__()
        self.parameters = parameters
        if w_in is not None:
            assert isinstance(w_in, np.ndarray)
            assert w_in.shape == (self.parameters.n_in, self.parameters.n)
            self.w_in = w_in
        else:
            self.w_in = self.parameters.w_dist.get_weights(
                self.parameters.n_in, self.parameters.n
            )

        self.post_batch: SpikesVector | None = None
        self.gradient = np.zeros_like(self.w_in)

    # ── Forward / Backward ───────────────────────────────────────────

    def forward(self, input_batch: SpikesVector) -> SpikesVector:
        super().forward(input_batch)

        # Extract per-sample arrays — C iterates via pointer array (no Python concatenation)
        times_list = [inp.times for inp in input_batch]
        sources_list = [inp.sources for inp in input_batch]

        # Forward-only path: skip error buffer allocation.
        # Errors are allocated lazily by backward() if/when needed.
        c_output = compute_spikes_batch_ptrs(
            times_list, sources_list,
            self.w_in_flat, self.parameters.n_in, self.parameters.n,
            self.parameters.tau_mem, self.parameters.tau_syn, self.parameters.v_th,
            alloc_errors=False,
        )

        # Wrap in SpikesVector (zero-copy, keeps C memory alive via _batch_output)
        self.post_batch = SpikesVector._from_batch_output(c_output)
        self._ran_forward = True

        return self.post_batch

    @property
    def w_in(self) -> np.ndarray:
        return self._w_in

    @w_in.setter
    def w_in(self, value: np.ndarray):
        self._w_in = value
        self.w_in_flat = np.ascontiguousarray(value).ravel().reshape(self.parameters.n_in, self.parameters.n)

    def backward(self):
        batch_size = len(self.input_batch)

        # Extract per-sample arrays — C iterates via pointer array (no Python concatenation)
        times_list = [inp.times for inp in self.input_batch]
        sources_list = [inp.sources for inp in self.input_batch]

        # Errors are already in native memory (BatchSpikeOutput._errors)
        # backward_spikes_batch_accumulate_ptrs reads them directly — zero concatenation
        result = backward_spikes_batch_accumulate_ptrs(
            times_list, sources_list,
            self.post_batch._batch_output,
            self.w_in_flat, self.parameters.n_in, self.parameters.n,
            self.parameters.tau_mem, self.parameters.tau_syn, self.parameters.v_th,
        )

        self.gradient[:] = result.grad.reshape(self.w_in.shape)

        # Propagate spike errors back to input spikes (for layer chains)
        if result.spike_errors is not None:
            offset = 0
            for i in range(batch_size):
                inp = self.input_batch[i]
                n_spikes_i = len(inp.times) if hasattr(inp, 'times') else 0
                if n_spikes_i > 0 and offset + n_spikes_i <= len(result.spike_errors):
                    inp._errors = np.array(result.spike_errors[offset:offset + n_spikes_i])
                offset += n_spikes_i

        self._ran_backward = True
        super().backward()

    def zero_grad(self):
        self.gradient[:] = 0
