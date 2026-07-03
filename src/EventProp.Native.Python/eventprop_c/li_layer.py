"""LI layer — Cython bindings, zero-copy numpy arrays."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from ._native import (
    compute_maxima_batch_flat, backward_maxima_batch_accumulate,
)
from .spikes import MaximaVector, SpikesVector


# fmt: off
class LILayerParameters(NamedTuple):
    n               : int   = None
    n_in            : int   = None
    tau_mem         : float = 20e-3 # s
    tau_syn         : float = 5e-3  # s
    v_leak          : float = 0
    plastic_weights : bool  = True
    w_dist          : WeightDistribution = GaussianDistribution()
# fmt: on


class LILayer(Layer):
    def __init__(self, parameters: LILayerParameters, w_in: np.ndarray = None):
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

        self.maxima_batch: MaximaVector | None = None  # C-owned via Cython BatchMaximaOutput
        self.gradient = np.zeros_like(self.w_in)

    # ── Forward / Backward ───────────────────────────────────────────

    def forward(self, input_batch: SpikesVector):
        """Batch forward pass — returns Cython BatchMaximaOutput (zero-copy, C-owned)."""
        super().forward(input_batch)

        times, sources, n_per_sample = SpikesVector._extract_flat_buffers(input_batch)

        c_output = compute_maxima_batch_flat(
            times, sources, n_per_sample,
            self.w_in_flat, self.parameters.n_in, self.parameters.n,
            self.parameters.tau_mem, self.parameters.tau_syn, float('inf'),  # LI: no threshold
        )

        self.maxima_batch = MaximaVector._from_batch_output(c_output)
        self._ran_forward = True
        return self.maxima_batch

    @property
    def w_in(self) -> np.ndarray:
        return self._w_in

    @w_in.setter
    def w_in(self, value: np.ndarray):
        self._w_in = value
        self.w_in_flat = np.ascontiguousarray(value).ravel().reshape(self.parameters.n_in, self.parameters.n)

    def backward(self):
        # C owns inputs internally — no Python-side buffer extraction needed
        result = backward_maxima_batch_accumulate(
            self.maxima_batch._batch_output,
        )

        self.gradient[:] = result.grad.reshape(self.w_in.shape)

        # Propagate spike errors back to input spikes (per-sample, zero-copy)
        for i in range(result.batch_size):
            inp = self.input_batch[i]
            if len(inp.times) > 0:
                inp._errors = result.get_spike_errors(i)

        self._ran_backward = True
        super().backward()

    def zero_grad(self):
        self.gradient[:] = 0
