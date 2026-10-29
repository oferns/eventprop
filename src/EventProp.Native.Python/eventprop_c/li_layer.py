"""LI layer — Cython bindings, zero-copy numpy arrays."""

import numpy as np
from typing import NamedTuple, Optional, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from ._native import (
    compute_maxima_batch, backward_maxima_batch_accumulate,
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

        c_output = compute_maxima_batch(
            times, sources, n_per_sample,
            self._w_in, self.parameters.n_in, self.parameters.n,
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
        # Ensure row-contiguous for zero-copy Cython transfer (double[:, ::1])
        self._w_in = np.ascontiguousarray(value)

    def backward(self):
        # C owns inputs internally — no Python-side buffer extraction needed
        result = backward_maxima_batch_accumulate(
            self.maxima_batch._batch_output,
        )

        self.gradient[:] = result.grad.reshape(self.w_in.shape)

        # Propagate spike errors back to input spikes (per-sample, zero-copy).
        # Same lifetime rule as LIFLayer.backward: get_spike_errors() views into
        # the native result, which __dealloc__ frees, and the ancestor's backward
        # below is the consumer. Keep the result alive on the layer.
        self._backward_result = result
        for i in range(result.batch_size):
            inp = self.input_batch[i]
            if len(inp.times) > 0:
                inp._errors = result.get_spike_errors(i)

        self._ran_backward = True
        super().backward()

    def regression(self, decoder: np.ndarray, bias: float, targets: np.ndarray,
                   huber_delta: float = np.inf, *,
                   decoder_grad: Optional[np.ndarray] = None,
                   bias_grad: Optional[np.ndarray] = None) -> dict:
        """Continuous readout over the maxima from the last ``forward()``, plus its loss.

            y_hat = bias + Σ decoder[n] · maxima[n]

        This is the ergonomic entry to ``eventprop_maxima_regression``: it writes dL/dmaxima into
        the maxima error buffers, so the step is two calls and no loss arithmetic appears in
        Python at all —

            loss = layer.regression(decoder, bias, targets)
            layer.backward()              # dL/dw_in, summed over the batch

        which is the same two-step the shipped ``VMaxCrossEntropyLoss`` performs internally; the
        difference is that here you own the readout and the target.

        Returns ``predictions``, ``losses``, ``n_skipped``, ``decoder_grad``, ``bias_grad``.
        ``huber_delta`` non-finite selects least squares, finite and positive smooth Huber.

        Nothing is zeroed anywhere: ``decoder_grad``/``bias_grad`` accumulate into the buffers you
        pass (allocated when omitted), and ``backward()`` accumulates into ``self.gradient``, so
        clearing is the caller's job — ``zero_grad()`` for ``w_in``, your own buffers for the
        readout. A neuron that never peaked is excluded from the readout and counted in
        ``n_skipped``, which makes a fully silent population a valid sample rather than an error.

        Requires a prior ``forward()``; the errors have to land in the buffers that forward
        allocated.
        """
        if not self._ran_forward or self.maxima_batch is None:
            raise RuntimeError(
                "regression() requires a prior forward(): dL/dmaxima is written into the maxima "
                "error buffers that forward allocated"
            )
        return self.maxima_batch.regress(decoder, bias, targets, huber_delta,
                                        decoder_grad=decoder_grad, bias_grad=bias_grad)

    def zero_grad(self):
        self.gradient[:] = 0
