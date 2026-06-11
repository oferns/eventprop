"""LI layer for ctypes library — calls C library directly via NativeApi."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from .native import EventPropCtypes as NativeApi
from .spikes import SpikesVector, Maxima, MaximaVector


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

        # Cache flattened view for zero-copy access
        self.w_in_flat = np.ascontiguousarray(self.w_in).ravel()

        self.input_batch = None
        self.maxima_batch = None
        self.gradient = np.zeros_like(self.w_in)

    def forward(self, input_batch: SpikesVector) -> MaximaVector:
        super().forward(input_batch)

        lib = NativeApi()

        # Process each sample in batch
        maxima_list = []
        for spikes in input_batch:
            output = lib.compute_maxima_kw(
                w=self.w_in_flat,
                times=spikes.times,
                sources=spikes.sources,
                n_pre=self.parameters.n_in,
                n_post=self.parameters.n,
                tau_mem=self.parameters.tau_mem,
                tau_syn=self.parameters.tau_syn,
            )

            maxima_list.append(Maxima(
                times=output.times.to_numpy(),
                values=output.values.to_numpy(),
            ))

        self.maxima_batch = MaximaVector(maxima_list)
        self._ran_forward = True
        return self.maxima_batch

    def backward(self):
        lib = NativeApi()

        # Collect all data for batch processing
        times_list = []
        sources_list = []
        maxima_times_list = []
        maxima_values_list = []
        maxima_errors_list = []

        for i in range(len(self.input_batch)):
            inp = self.input_batch[i]
            mx = self.maxima_batch[i]

            times_list.append(np.asarray(inp.times, dtype=np.float64))
            sources_list.append(np.asarray(inp.sources, dtype=np.int32))
            maxima_times_list.append(np.asarray(mx.times, dtype=np.float64))
            maxima_values_list.append(np.asarray(mx.values, dtype=np.float64))

            errors = getattr(mx, 'errors', None)
            if errors is None or len(errors) == 0:
                maxima_errors_list.append(np.ones(len(mx.times), dtype=np.float64))
            else:
                maxima_errors_list.append(np.asarray(errors, dtype=np.float64))

        # Process each sample and accumulate gradients
        n_samples = len(self.input_batch)
        self.gradient[:] = 0.0

        for i in range(n_samples):
            output = lib.backward_maxima_kw(
                w=self.w_in_flat,
                n_pre=self.parameters.n_in,
                n_post=self.parameters.n,
                times=times_list[i],
                sources=sources_list[i],
                maxima_times=maxima_times_list[i],
                maxima_values=maxima_values_list[i],
                maxima_errors=maxima_errors_list[i],
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
        maxima = self.maxima_batch[batch_idx]

        output = lib.compute_voltage_trace_kw(
            w=self.w_in_flat,
            n_pre=self.parameters.n_in,
            n_post=self.parameters.n,
            times=input_spikes.times,
            sources=input_spikes.sources,
            target_nrn_idx=target_nrn_idx,
            fwd_times=maxima.times,
            fwd_sources=np.zeros(len(maxima.times), dtype=np.int32),
            v_th=np.inf,  # No threshold for LI layer
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
            t_max=t_max,
            dt=dt,
        )

        return output.ts.to_numpy(), output.v_trace.to_numpy()

    def zero_grad(self):
        self.gradient[:] = 0
