"""LIF layer — Cython bindings, zero-copy numpy arrays."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from ._native import (
    compute_spikes_batch, backward_spikes_batch_accumulate,
    compute_voltage_trace, compute_lambda_i_trace, compute_lambda_i_at_time,
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
            self._w_in, self.parameters.n_in, self.parameters.n,
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
        # Ensure row-contiguous for zero-copy Cython transfer (double[:, ::1])
        self._w_in = np.ascontiguousarray(value)

    def backward(self):
        # C owns inputs internally — no Python-side buffer extraction needed
        result = backward_spikes_batch_accumulate(
            self.post_batch._batch_output,
            SpikesVector._error_arrays(self.post_batch),
        )

        self.gradient[:] = result.grad.reshape(self.w_in.shape)

        # Propagate spike errors back to input spikes (per-sample, zero-copy).
        # get_spike_errors() is a view into the native AccumulatedBackwardResult,
        # which __dealloc__ frees — so the result has to outlive this call. The
        # consumer is the *ancestor* layer's backward, which runs below in
        # super().backward(), and the views stay live on the input spikes after
        # that. Dropping the last reference here frees the buffers those views
        # point at, and the ancestor then reads freed heap memory (observed as
        # identical 1.2e+79 garbage in every sample of the batch).
        self._backward_result = result
        for i in range(result.batch_size):
            inp = self.input_batch[i]
            if len(inp.times) > 0:
                inp._errors = result.get_spike_errors(i)

        self._ran_backward = True
        super().backward()

    # ── Diagnostics ──────────────────────────────────────────────────
    #
    # Same names and argument order as the reference LIFLayer, so a notebook written
    # against either backend reads the same. Each one traces the state this layer already
    # has: the input sample it was given and the output spikes its own forward pass
    # produced. Recomputing a forward pass here instead would describe a network state the
    # layer does not have, and would look identical when it is wrong.

    def get_voltage_trace_for_neuron(
        self,
        batch_idx: int,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Membrane potential of one neuron over time. Returns (ts, v_trace)."""
        input_spikes, post_spikes = self._trace_spikes(batch_idx)
        return compute_voltage_trace(
            input_spikes.times, input_spikes.sources,
            self._w_in, self.parameters.n_in, self.parameters.n,
            self.parameters.tau_mem, self.parameters.tau_syn, self.parameters.v_th,
            post_spikes.times, post_spikes.sources,
            t_max, dt, target_nrn_idx,
        )

    def get_lambda_i_trace_for_neuron(
        self,
        batch_idx: int,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synaptic adjoint of one neuron over time. Returns (ts, lambda_trace).

        Reads this layer's post-spike error arrays, the same ones backward() hands to the
        C library — so the adjoint is driven by the error signal the training pass used. If
        nothing has set those errors they are zero, and a zero adjoint is the correct answer
        rather than a bug: lambda_i is loss-driven.
        """
        input_spikes, post_spikes = self._trace_spikes(batch_idx)
        return compute_lambda_i_trace(
            input_spikes.times, input_spikes.sources,
            self._w_in, self.parameters.n_in, self.parameters.n,
            self.parameters.tau_mem, self.parameters.tau_syn, self.parameters.v_th,
            post_spikes.times, post_spikes.sources, post_spikes.errors,
            t_max, dt, target_nrn_idx,
        )

    def get_lambda_i_for_neuron(
        self, batch_idx: int, target_nrn_idx: int, t: float
    ) -> float:
        """Synaptic adjoint at one time — the figure-1 style query.

        The same sweep the trace samples on a grid, evaluated at t.
        """
        input_spikes, post_spikes = self._trace_spikes(batch_idx)
        return compute_lambda_i_at_time(
            input_spikes.times, input_spikes.sources,
            self._w_in, self.parameters.n_in, self.parameters.n,
            self.parameters.tau_mem, self.parameters.tau_syn, self.parameters.v_th,
            post_spikes.times, post_spikes.sources, post_spikes.errors,
            t, target_nrn_idx,
        )

    def _trace_spikes(self, batch_idx: int):
        """(input spikes, output spikes) for one sample, from the state forward() left."""
        if not self._ran_forward or self.post_batch is None:
            raise RuntimeError(
                f"{type(self).__name__} has no forward-pass state to trace: call forward() "
                f"before requesting a trace"
            )
        if batch_idx >= len(self.post_batch):
            raise IndexError(
                f"batch_idx {batch_idx} out of range for a batch of {len(self.post_batch)}"
            )
        return self.input_batch[batch_idx], self.post_batch[batch_idx]

    def zero_grad(self):
        self.gradient[:] = 0
