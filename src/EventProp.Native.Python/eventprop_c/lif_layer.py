"""LIF layer for ctypes library — calls C library directly via NativeApi."""

import numpy as np
from typing import NamedTuple, Tuple, List

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

    def forward(self, input_batch: List[Spikes]) -> SpikesVector:
        super().forward(input_batch)
        
        # Call C library directly via NativeApi
        lib = NativeApi()
        
        # Concatenate all spikes from batch into single arrays
        all_times = np.concatenate([spikes.times for spikes in input_batch])
        all_sources = np.concatenate([spikes.sources for spikes in input_batch])
        
        # Track spike counts per sample
        n_spikes_per_sample = [len(spikes.times) for spikes in input_batch]
        
        # Use batch function (single C call, parallel processing)
        output = lib.compute_spikes_batch(
            w=self.w_in_flat,  # Use pre-flattened view (zero-copy)
            times=all_times,
            sources=all_sources,
            n_spikes_per_sample=n_spikes_per_sample,
            batch_size=len(input_batch),
            n_pre=self.parameters.n_in,
            n_post=self.parameters.n,
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )
        
        # Collect per-sample results
        post_spikes_list = []
        for i in range(output.batch_size):
            sample = output[i]
            post_spikes_list.append(Spikes(
                times=sample.times.to_numpy(),
                sources=sample.sources.to_numpy()
            ))
        
        self.post_batch = SpikesVector(post_spikes_list)
        self._ran_forward = True
        return self.post_batch

    def backward(self):
        lib = NativeApi()
        
        # Use numpy for fast concatenation (avoids slow Python lists)
        all_times = np.concatenate([inp.times for inp in self.input_batch])
        all_sources = np.concatenate([inp.sources for inp in self.input_batch])
        all_fwd_times = np.concatenate([post.times for post in self.post_batch])
        all_fwd_sources = np.concatenate([post.sources for post in self.post_batch])
        
        # Get currents and errors as numpy arrays (with defaults)
        fwd_currents_list = []
        errors_list = []
        for post in self.post_batch:
            n_spikes = len(post.times)
            # Get currents (default to zeros if not available)
            currents = getattr(post, 'currents', None)
            if currents is None or len(currents) == 0:
                fwd_currents_list.append(np.zeros(n_spikes))
            else:
                fwd_currents_list.append(np.asarray(currents, dtype=np.float64))
            
            # Get errors (default to ones if not available)
            errors = getattr(post, 'errors', None)
            if errors is None or len(errors) == 0:
                errors_list.append(np.ones(n_spikes))
            else:
                errors_list.append(np.asarray(errors, dtype=np.float64))
        
        all_fwd_currents = np.concatenate(fwd_currents_list)
        all_errors = np.concatenate(errors_list)
        
        # Call C library directly with numpy arrays (zero-copy via ctypes)
        output = lib.backward_spikes(
            times=all_times,
            sources=all_sources,
            fwd_times=all_fwd_times,
            fwd_sources=all_fwd_sources,
            fwd_currents=all_fwd_currents,
            post_errors=all_errors,
            w=self.w_in_flat,  # Use pre-flattened view (zero-copy)
            n_pre=self.parameters.n_in,
            n_post=self.parameters.n,
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )
        
        # Zero-copy gradient conversion
        self.gradient[:] = output.grad.to_numpy().reshape(
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
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )
        
        ts = output.ts.to_numpy()
        v_trace = output.v_trace.to_numpy()
        
        return ts, v_trace

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
        
        output = lib.compute_lambda_i_trace(
            t_max=t_max, dt=dt, target_nrn_idx=target_nrn_idx,
            post_times=post_spikes.times,
            post_sources=post_spikes.sources,
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )
        
        ts = output.ts.to_numpy()
        lambda_i = output.lambda_trace.to_numpy()  # Note: corrected attribute name
        
        return ts, lambda_i

    def get_lambda_i_for_neuron(
        self, batch_idx: int, target_nrn_idx: int, t: float
    ) -> float:
        lib = NativeApi()
        post_spikes = self.post_batch[batch_idx]
        
        return lib.compute_lambda_i(
            t=t, target_nrn_idx=target_nrn_idx,
            post_times=post_spikes.times,
            post_sources=post_spikes.sources,
            v_th=self.parameters.v_th,
            tau_mem=self.parameters.tau_mem,
            tau_syn=self.parameters.tau_syn,
        )
