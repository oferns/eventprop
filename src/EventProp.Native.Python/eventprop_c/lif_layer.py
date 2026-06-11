"""LIF layer for ctypes library — calls C library directly via NativeApi."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from .native import EventPropCtypes as NativeApi
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

        # Cache flattened view for zero-copy access (avoids flatten() per call)
        self.w_in_flat = np.ascontiguousarray(self.w_in).ravel()

        self.post_batch = None
        self.gradient = np.zeros_like(self.w_in)

    def forward(self, input_batch: SpikesVector) -> SpikesVector:
        super().forward(input_batch)

        # Collect per-sample spike data as numpy arrays (kept alive for C call)
        times_list = [np.asarray(s.times, dtype=np.float64) for s in input_batch]
        sources_list = [np.asarray(s.sources, dtype=np.int32) for s in input_batch]

        # Batch forward pass via convenience wrapper
        lib = NativeApi()
        output = lib.compute_spikes_batch_kw(
            w=self.w_in_flat,
            n_pre=self.parameters.n_in,
            n_post=self.parameters.n,
            times_list=times_list,
            sources_list=sources_list,
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )

        # Collect per-sample results — store numpy copies (C memory freed on del)
        post_spikes_list = []
        for i in range(output.batch_size):
            sample = output[i]
            post_spikes_list.append(Spikes(
                times=sample.times.to_numpy(),
                sources=sample.sources.to_numpy(),
            ))

        self.post_batch = SpikesVector(post_spikes_list)
        self._ran_forward = True
        return self.post_batch

    def backward(self):
        lib = NativeApi()

        # Initialize gradient to zero
        self.gradient[:] = 0.0

        n_samples = len(self.input_batch)

        # Process each sample individually (avoids complex batch struct construction)
        for i in range(n_samples):
            inp = self.input_batch[i]
            post = self.post_batch[i]

            times = np.asarray(inp.times, dtype=np.float64)
            sources = np.asarray(inp.sources, dtype=np.int32)
            fwd_times = np.asarray(post.times, dtype=np.float64)
            fwd_sources = np.asarray(post.sources, dtype=np.int32)

            n_fwd = len(fwd_times)
            currents = np.zeros(n_fwd, dtype=np.float64)

            errors = getattr(post, 'errors', None)
            if errors is None or len(errors) == 0:
                post_errors = np.ones(n_fwd, dtype=np.float64)
            else:
                post_errors = np.asarray(errors, dtype=np.float64)

            output = lib.backward_spikes_kw(
                w=self.w_in_flat,
                n_pre=self.parameters.n_in,
                n_post=self.parameters.n,
                times=times,
                sources=sources,
                fwd_times=fwd_times,
                fwd_sources=fwd_sources,
                fwd_currents=currents,
                post_errors=post_errors,
                v_th=self.parameters.v_th,
                tau_mem=self.parameters.tau_mem,
                tau_syn=self.parameters.tau_syn,
            )

            self.gradient += output.grad.to_numpy().reshape(
                self.parameters.n_in, self.parameters.n
            ) / n_samples

        self._ran_backward = True
        super().backward()

    def get_voltage_trace_for_neuron(
        self,
        batch_idx: int,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")

        lib = NativeApi()
        input_spikes = self.input_batch[batch_idx]
        post_spikes = self.post_batch[batch_idx]

        output = lib.compute_voltage_trace_kw(
            w=self.w_in_flat,
            n_pre=self.parameters.n_in,
            n_post=self.parameters.n,
            times=input_spikes.times,
            sources=input_spikes.sources,
            target_nrn_idx=target_nrn_idx,
            fwd_times=post_spikes.times,
            fwd_sources=post_spikes.sources,
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
            t_max=t_max,
            dt=dt,
        )

        return output.ts.to_numpy(), output.v_trace.to_numpy()

    def zero_grad(self):
        self.gradient[:] = 0

    def get_lambda_i_trace_for_neuron(
        self,
        batch_idx: int,
        target_nrn_idx: int,
        t_max: float = 1.0,
        dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lib = NativeApi()
        post_spikes = self.post_batch[batch_idx]

        output = lib.compute_lambda_i_trace_kw(
            fwd_times=post_spikes.times,
            target_nrn_idx=target_nrn_idx,
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
            t_max=t_max,
            dt=dt,
        )

        return output.ts.to_numpy(), output.lambda_trace.to_numpy()
