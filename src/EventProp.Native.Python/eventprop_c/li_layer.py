"""LI layer for ctypes library — calls C library directly via NativeApi."""

import numpy as np
from typing import NamedTuple, Tuple, List

from .layer import GaussianDistribution, Layer, WeightDistribution
from .native import EventPropCtypes as NativeApi
from .spikes import SpikesVector, MaximaVector


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

    def forward(self, input_batch: List[Spikes]) -> MaximaVector:
        super().forward(input_batch)
        
        # Call C library directly via NativeApi
        lib = NativeApi()
        
        # Process each sample in batch
        maxima_list = []
        for spikes in input_batch:
            output = lib.compute_maxima(
                w=self.w_in_flat,  # Use pre-flattened view (zero-copy)
                times=spikes.times,
                sources=spikes.sources,
                n_pre=self.parameters.n_in,
                n_post=self.parameters.n,
                tau_mem=self.parameters.tau_mem,
                tau_syn=self.parameters.tau_syn,
            )
            
            # Zero-copy conversion to numpy arrays
            times = output.times.to_numpy()
            values = output.values.to_numpy()
            
            maxima_list.append(Maxima(times, values))
        
        self.maxima_batch = MaximaVector(maxima_list)
        self._ran_forward = True
        return self.maxima_batch

    def backward(self):
        lib = NativeApi()
        
        # Collect all data for single call (assuming batch size 1)
        input_spikes = self.input_batch[0] if len(self.input_batch) == 1 else None
        maxima = self.maxima_batch[0] if len(self.maxima_batch) == 1 else None
        
        # Build flat lists from batch
        all_times = []
        all_sources = []
        all_maxima_times = []
        all_maxima_values = []
        all_maxima_errors = []
        
        for i in range(len(self.input_batch)):
            inp = self.input_batch[i]
            mx = self.maxima_batch[i]
            
            all_times.extend(inp.times)
            all_sources.extend(inp.sources)
            all_maxima_times.extend(mx.times)
            all_maxima_values.extend(mx.values)
            all_maxima_errors.extend(getattr(mx, 'errors', [1.0] * len(mx.times)))
        
        # Call C library directly
        output = lib.backward_maxima(
            input_times=all_times,
            input_sources=all_sources,
            maxima_times=all_maxima_times,
            maxima_values=all_maxima_values,
            maxima_errors=all_maxima_errors,
            w=self.w_in_flat,  # Use pre-flattened view (zero-copy)
            n_pre=self.parameters.n_in,
            n_post=self.parameters.n,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )
        
        # Zero-copy gradient conversion
        self.gradient[:] = output.gradient.to_numpy().reshape(
            self.parameters.n_in, self.parameters.n
        )
        
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
        
        output = lib.compute_voltage_trace(
            t_max=t_max, dt=dt, target_nrn_idx=target_nrn_idx,
            w=self.w_in_flat,  # Use pre-flattened view (zero-copy)
            n_pre=self.parameters.n_in,
            n_post=self.parameters.n,
            times=input_spikes.times,
            sources=input_spikes.sources,
            v_th=np.inf,  # No threshold for LI layer
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )
        
        ts = output.ts.to_numpy()
        v_trace = output.v_trace.to_numpy()
        
        return ts, v_trace

    def zero_grad(self):
        self.gradient[:] = 0
