"""LIF layer — Cython bindings, zero-copy numpy arrays."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from ._native import (
    compute_spikes_batch, backward_spikes_batch_accumulate,
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

        # Use flat array API — SpikesVector already has _flat_times, _flat_sources,
        # and _n_spikes_per_sample pre-built at construction.
        times, sources, n_per_sample = SpikesVector._extract_flat_buffers(input_batch)

        c_output = compute_spikes_batch(
            times, sources, n_per_sample,
            self.w_in_flat, self.parameters.n_in, self.parameters.n,
            self.parameters.tau_mem, self.parameters.tau_syn, self.parameters.v_th,
        )

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

        times, sources, n_per_sample = SpikesVector._extract_flat_buffers(self.input_batch)
        errors = self.post_batch._batch_output.get_flat_errors()

        result = backward_spikes_batch_accumulate(
            times, sources, n_per_sample,
            self.post_batch._batch_output, errors,
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
