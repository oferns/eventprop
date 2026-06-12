"""LI layer for ctypes library — batch functions, zero-copy forward output."""

import numpy as np
from typing import NamedTuple, Tuple

from .layer import GaussianDistribution, Layer, WeightDistribution
from .native import (
    EventPropCtypes as NativeApi,
    _EventPropInput, _BatchEventPropInput,
    POINTER, c_double, c_int, byref, pointer,
)
from .spikes import SpikesVector


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

        # Cache flattened contiguous view for zero-copy ctypes access
        self.w_in_flat = np.ascontiguousarray(self.w_in).ravel()

        self.maxima_batch = None  # BatchMaximaOutput (C-owned, zero-copy)
        self.gradient = np.zeros_like(self.w_in)

    # ── Helpers for struct construction ──────────────────────────────

    @staticmethod
    def _np_to_ptr(array):
        """Convert numpy array to ctypes pointer. Returns (ptr, arr) — keep arr alive."""
        if array is None or len(array) == 0:
            return None, None
        arr = np.asarray(array, dtype=np.float64)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr.ctypes.data_as(POINTER(c_double)), arr

    @staticmethod
    def _np_to_int_ptr(array):
        """Convert numpy int array to ctypes pointer. Returns (ptr, arr) — keep arr alive."""
        if array is None or len(array) == 0:
            return None, None
        arr = np.asarray(array, dtype=np.int32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr.ctypes.data_as(POINTER(c_int)), arr

    def _make_input(self, times: np.ndarray | None, sources: np.ndarray | None):
        """Construct _EventPropInput. Returns (struct, refs_dict) to keep arrays alive."""
        from .native import _EventPropInput as EI
        inp = EI()
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
        inp.v_th = float('inf')  # LI layer: no threshold

        refs = {'w': w_arr}
        if t_arr is not None: refs['times'] = t_arr
        if s_arr is not None: refs['sources'] = s_arr
        return inp, refs

    # ── Forward / Backward ───────────────────────────────────────────

    def forward(self, input_batch: SpikesVector):
        """Batch forward pass — returns BatchMaximaOutput (zero-copy, C-owned)."""
        super().forward(input_batch)

        lib = NativeApi()
        batch_size = len(input_batch)

        # Flatten per-sample arrays into contiguous buffers for C
        times_list = [np.asarray(s.times, dtype=np.float64) for s in input_batch]
        sources_list = [np.asarray(s.sources, dtype=np.int32) for s in input_batch]

        all_times = np.concatenate(times_list) if times_list else np.array([], dtype=np.float64)
        all_sources = np.concatenate(sources_list) if sources_list else np.array([], dtype=np.int32)

        inp, refs = self._make_input(all_times, all_sources)

        # Build batch input struct
        n_spikes_per_sample = (c_int * batch_size)(*[len(t) for t in times_list])

        batch_inp = _BatchEventPropInput()
        batch_inp.input = pointer(inp)
        batch_inp.n_spikes_per_sample = n_spikes_per_sample
        batch_inp.batch_size = batch_size

        # Call C library — returns BatchMaximaOutput owning the C memory
        from .native import BatchMaximaOutput
        c_output = lib.lib.eventprop_compute_maxima_batch(byref(batch_inp))
        self.maxima_batch = BatchMaximaOutput(c_output, batch_size)

        # Keep Python arrays AND ctypes structs alive
        self._forward_refs = {**refs, 'inp': inp, 'batch_inp': batch_inp,
                              'n_spikes_per_sample': n_spikes_per_sample}

        self._ran_forward = True
        return self.maxima_batch

    def backward(self):
        lib = NativeApi()
        batch_size = len(self.input_batch)

        # Collect per-sample input data into flat arrays (necessary: Python-owned -> C)
        times_list = [np.asarray(inp.times, dtype=np.float64) for inp in self.input_batch]
        sources_list = [np.asarray(inp.sources, dtype=np.int32) for inp in self.input_batch]

        all_times = np.concatenate(times_list) if times_list else np.array([], dtype=np.float64)
        all_sources = np.concatenate(sources_list) if sources_list else np.array([], dtype=np.int32)

        inp, refs = self._make_input(all_times, all_sources)

        # Collect errors into flat array (necessary: Python computes loss)
        # C accesses fwd maxima data directly from MaximaOutput* — no concatenation needed
        errors_list = []
        for i in range(batch_size):
            mx = self.maxima_batch[i]
            errors = getattr(mx, '_errors', None)
            if errors is None:
                errors_list.append(np.ones(mx.n_post, dtype=np.float64))
            else:
                errors_list.append(np.asarray(errors, dtype=np.float64))

        all_errors = np.concatenate(errors_list) if errors_list else np.array([], dtype=np.float64)
        err_ptr, err_arr = self._np_to_ptr(all_errors)

        # Build batch input struct
        n_spikes_per_sample = (c_int * batch_size)(*[len(t) for t in times_list])

        batch_inp = _BatchEventPropInput()
        batch_inp.input = pointer(inp)
        batch_inp.n_spikes_per_sample = n_spikes_per_sample
        batch_inp.batch_size = batch_size

        # Call C library — passes MaximaOutput* directly, C reads fwd data from it
        c_output = lib.lib.eventprop_backward_maxima_batch(
            byref(batch_inp), self.maxima_batch._c, err_ptr
        )

        # Accumulate gradients (average over batch)
        from .native import BatchBackwardOutput
        backward_output = BatchBackwardOutput(c_output, batch_size)
        self.gradient[:] = backward_output.grads.mean(axis=0)

        # Keep arrays alive
        self._backward_refs = {**refs, 'inp': inp, 'batch_inp': batch_inp,
                               'n_spikes_per_sample': n_spikes_per_sample, 'errors': err_arr}

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

        inp, _ = self._make_input(
            np.asarray(input_spikes.times, dtype=np.float64),
            np.asarray(input_spikes.sources, dtype=np.int32),
        )

        from .native import _VoltageTraceInput
        fwd_t_ptr, _ = self._np_to_ptr(np.asarray(maxima.times, dtype=np.float64))

        trace_inp = _VoltageTraceInput()
        trace_inp.input = pointer(inp)
        trace_inp.target_nrn_idx = int(target_nrn_idx)
        trace_inp.fwd_times = fwd_t_ptr
        trace_inp.fwd_sources = None  # LI layer: no sources for maxima
        trace_inp.n_post_spikes = maxima.n_post
        trace_inp.t_max = float(t_max)
        trace_inp.dt = float(dt)

        output = lib.lib.eventprop_compute_voltage_trace(byref(trace_inp))
        return output.ts, output.v_trace

    def zero_grad(self):
        self.gradient[:] = 0
