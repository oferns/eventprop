"""LIF layer — constructs C structs directly, uses batch functions with zero-copy."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from .native import (
    EventPropCtypes as NativeApi,
    _EventPropInput, _BatchEventPropInput, _BackwardSpikesInput,
    _VoltageTraceInput, _LambdaITraceInput,
    POINTER, c_double, c_int, byref, pointer, cast,
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

        # Cache flattened contiguous view for zero-copy ctypes access
        self.w_in_flat = np.ascontiguousarray(self.w_in).ravel()

        self.post_batch: SpikesVector | None = None
        self.gradient = np.zeros_like(self.w_in)

    # ── Helpers for struct construction ──────────────────────────────
    # These return (ptr, arr) tuples. The caller MUST keep `arr` alive
    # for as long as the C pointer is valid, otherwise it dangles.

    @staticmethod
    def _np_to_ptr(array):
        """Convert numpy array to ctypes pointer. Returns (ptr, arr) — keep arr alive."""
        if array is None or len(array) == 0:
            return None, None
        arr = np.asarray(array, dtype=np.float64)
        return arr.ctypes.data_as(POINTER(c_double)), arr

    @staticmethod
    def _np_to_int_ptr(array):
        """Convert numpy int array to ctypes pointer. Returns (ptr, arr) — keep arr alive."""
        if array is None or len(array) == 0:
            return None, None
        arr = np.asarray(array, dtype=np.int32)
        return arr.ctypes.data_as(POINTER(c_int)), arr

    def _make_input(self, times: np.ndarray | None, sources: np.ndarray | None) -> Tuple[_EventPropInput, dict]:
        """Construct _EventPropInput. Returns (struct, refs_dict) to keep arrays alive."""
        inp = _EventPropInput()
        w_ptr, w_arr = self._np_to_ptr(self.w_in_flat)
        inp.w = w_ptr
        inp.n_pre = self.parameters.n_in
        inp.n_post = self.parameters.n

        t_ptr, t_arr = self._np_to_ptr(times) if times is not None and len(times) > 0 else (None, None)
        inp.times = t_ptr

        s_ptr, s_arr = self._np_to_int_ptr(sources) if sources is not None and len(sources) > 0 else (None, None)
        inp.sources = s_ptr

        inp.n_spikes = len(times) if times is not None else 0
        inp.tau_mem = self.parameters.tau_mem
        inp.tau_syn = self.parameters.tau_syn
        inp.v_th = self.parameters.v_th

        # Keep all arrays alive — C pointers reference their memory
        refs = {'w': w_arr}
        if t_arr is not None: refs['times'] = t_arr
        if s_arr is not None: refs['sources'] = s_arr
        return inp, refs

    # ── Forward / Backward ───────────────────────────────────────────

    def forward(self, input_batch: SpikesVector) -> SpikesVector:
        super().forward(input_batch)

        lib = NativeApi()
        batch_size = len(input_batch)

        # Flatten per-sample arrays into contiguous buffers for C
        times_list = [np.asarray(s.times, dtype=np.float64) for s in input_batch]
        sources_list = [np.asarray(s.sources, dtype=np.int32) for s in input_batch]

        all_times = np.concatenate(times_list).astype(np.float64, copy=True) if times_list else np.array([], dtype=np.float64)
        all_sources = np.concatenate(sources_list).astype(np.int32, copy=True) if sources_list else np.array([], dtype=np.int32)

        inp, refs = self._make_input(all_times, all_sources)
        # _make_input may have created new arrays via asarray — use those refs

        # Build batch input struct
        n_spikes_per_sample = (c_int * batch_size)(*[len(t) for t in times_list])

        batch_inp = _BatchEventPropInput()
        batch_inp.input = pointer(inp)
        batch_inp.n_spikes_per_sample = n_spikes_per_sample
        batch_inp.batch_size = batch_size

        # Call C library — returns BatchSpikeOutput owning the C memory
        c_output = lib.compute_spikes_batch(batch_inp)

        # Wrap in SpikesVector (zero-copy, keeps C memory alive via _batch_output)
        self.post_batch = SpikesVector._from_batch_output(c_output)
        self._ran_forward = True

        # Keep Python arrays AND ctypes structs alive (pointer fields don't keep targets alive)
        self._forward_refs = {**refs, 'inp': inp, 'batch_inp': batch_inp, 'n_spikes_per_sample': n_spikes_per_sample}

        return self.post_batch

    def backward(self):
        lib = NativeApi()
        batch_size = len(self.input_batch)

        # Collect per-sample input data into flat arrays
        times_list = [np.asarray(inp.times, dtype=np.float64) for inp in self.input_batch]
        sources_list = [np.asarray(inp.sources, dtype=np.int32) for inp in self.input_batch]

        all_times = np.concatenate(times_list).astype(np.float64, copy=True) if times_list else np.array([], dtype=np.float64)
        all_sources = np.concatenate(sources_list).astype(np.int32, copy=True) if sources_list else np.array([], dtype=np.int32)

        inp, refs = self._make_input(all_times, all_sources)

        # Collect forward output data and errors into flat arrays
        fwd_times_list = []
        fwd_sources_list = []
        fwd_currents_list = []
        errors_list = []

        for i in range(batch_size):
            fwd = self.post_batch[i]
            fwd_times_list.append(fwd.times)
            fwd_sources_list.append(fwd.sources)
            fwd_currents_list.append(fwd.currents)

            errors = getattr(fwd, '_errors', None)
            if errors is None:
                errors_list.append(np.ones(fwd.n_spikes, dtype=np.float64))
            else:
                errors_list.append(np.asarray(errors, dtype=np.float64))

        all_fwd_times = np.concatenate(fwd_times_list).astype(np.float64, copy=True) if fwd_times_list else np.array([], dtype=np.float64)
        all_fwd_sources = np.concatenate(fwd_sources_list).astype(np.int32, copy=True) if fwd_sources_list else np.array([], dtype=np.int32)
        all_fwd_currents = np.concatenate(fwd_currents_list).astype(np.float64, copy=True) if fwd_currents_list else np.array([], dtype=np.float64)
        all_errors = np.concatenate(errors_list).astype(np.float64, copy=True) if errors_list else np.array([], dtype=np.float64)

        # Build batch input struct
        n_spikes_per_sample = (c_int * batch_size)(*[len(t) for t in times_list])

        batch_inp = _BatchEventPropInput()
        batch_inp.input = pointer(inp)
        batch_inp.n_spikes_per_sample = n_spikes_per_sample
        batch_inp.batch_size = batch_size

        # Build backward input struct (pointers to fwd data)
        fwd_t_ptr, fwd_t_arr = self._np_to_ptr(all_fwd_times)
        fwd_s_ptr, fwd_s_arr = self._np_to_int_ptr(all_fwd_sources)
        fwd_c_ptr, fwd_c_arr = self._np_to_ptr(all_fwd_currents)
        err_ptr, err_arr = self._np_to_ptr(all_errors)

        backward_inp = _BackwardSpikesInput()
        backward_inp.input = pointer(inp)
        backward_inp.fwd_times = fwd_t_ptr
        backward_inp.fwd_sources = fwd_s_ptr
        backward_inp.fwd_currents = fwd_c_ptr
        backward_inp.post_errors = err_ptr
        backward_inp.n_post_spikes = len(all_fwd_times)

        # Call C library batch backward
        c_output = lib.backward_spikes_batch(
            batch_inp, self.post_batch._batch_output._c, err_ptr
        )

        # Accumulate gradients (average over batch)
        self.gradient[:] = c_output.grads.mean(axis=0)

        # Keep all arrays alive
        self._backward_refs = {**refs, 'batch_inp': batch_inp, 'n_spikes_per_sample': n_spikes_per_sample,
                               'fwd_times': fwd_t_arr, 'fwd_sources': fwd_s_arr,
                               'fwd_currents': fwd_c_arr, 'errors': err_arr}

        self._ran_backward = True
        super().backward()

    # ── Diagnostic functions ─────────────────────────────────────────

    def get_voltage_trace_for_neuron(
        self, batch_idx: int, target_nrn_idx: int,
        t_max: float = 1.0, dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._ran_forward:
            raise RuntimeError("Run forward first!")

        lib = NativeApi()
        input_spikes = self.input_batch[batch_idx]
        post_spikes = self.post_batch[batch_idx]

        inp, _ = self._make_input(
            np.asarray(input_spikes.times, dtype=np.float64),
            np.asarray(input_spikes.sources, dtype=np.int32),
        )

        fwd_t_ptr, _ = self._np_to_ptr(np.asarray(post_spikes.times, dtype=np.float64))
        fwd_s_ptr, _ = self._np_to_int_ptr(np.asarray(post_spikes.sources, dtype=np.int32))

        trace_inp = _VoltageTraceInput()
        trace_inp.input = pointer(inp)
        trace_inp.target_nrn_idx = int(target_nrn_idx)
        trace_inp.fwd_times = fwd_t_ptr
        trace_inp.fwd_sources = fwd_s_ptr
        trace_inp.n_post_spikes = post_spikes.n_spikes
        trace_inp.t_max = float(t_max)
        trace_inp.dt = float(dt)

        output = lib.compute_voltage_trace(trace_inp)
        return output.ts, output.v_trace

    def get_lambda_i_trace_for_neuron(
        self, batch_idx: int, target_nrn_idx: int,
        t_max: float = 1.0, dt: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lib = NativeApi()
        post_spikes = self.post_batch[batch_idx]

        fwd_t_ptr, _ = self._np_to_ptr(np.asarray(post_spikes.times, dtype=np.float64))

        trace_inp = _LambdaITraceInput()
        trace_inp.fwd_times = fwd_t_ptr
        trace_inp.n_post_spikes = post_spikes.n_spikes
        trace_inp.target_nrn_idx = int(target_nrn_idx)
        trace_inp.v_th = self.parameters.v_th
        trace_inp.tau_mem = self.parameters.tau_mem
        trace_inp.tau_syn = self.parameters.tau_syn
        trace_inp.t_max = float(t_max)
        trace_inp.dt = float(dt)

        output = lib.compute_lambda_i_trace(trace_inp)
        return output.ts, output.lambda_trace

    def zero_grad(self):
        self.gradient[:] = 0
