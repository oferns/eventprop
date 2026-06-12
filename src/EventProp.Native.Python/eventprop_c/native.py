"""EventProp ctypes bindings — thin wrapper matching C signatures."""

from __future__ import annotations
import ctypes
from ctypes import cdll, c_double, c_int, Structure, POINTER, byref, cast, pointer
import numpy as np


# ── Input structs (borrowed data) matching eventprop.h ─────────────────



class _EventPropInput(Structure):
    _fields_ = [
        ('w', POINTER(c_double)),
        ('n_pre', c_int),
        ('n_post', c_int),
        ('times', POINTER(c_double)),
        ('sources', POINTER(c_int)),
        ('n_spikes', c_int),
        ('tau_mem', c_double),
        ('tau_syn', c_double),
        ('v_th', c_double),
    ]






class _BatchEventPropInput(Structure):
    _fields_ = [
        ('input', POINTER(_EventPropInput)),
        ('n_spikes_per_sample', POINTER(c_int)),
        ('batch_size', c_int),
    ]


class _BackwardSpikesInput(Structure):
    _fields_ = [
        ('input', POINTER(_EventPropInput)),
        ('fwd_times', POINTER(c_double)),
        ('fwd_sources', POINTER(c_int)),
        ('fwd_currents', POINTER(c_double)),
        ('post_errors', POINTER(c_double)),
        ('n_post_spikes', c_int),
    ]


class _BackwardMaximaInput(Structure):
    _fields_ = [
        ('input', POINTER(_EventPropInput)),
        ('maxima_times', POINTER(c_double)),
        ('maxima_values', POINTER(c_double)),
        ('maxima_errors', POINTER(c_double)),
        ('n_maxima', c_int),
    ]


class _VoltageTraceInput(Structure):
    _fields_ = [
        ('input', POINTER(_EventPropInput)),
        ('target_nrn_idx', c_int),
        ('fwd_times', POINTER(c_double)),
        ('fwd_sources', POINTER(c_int)),
        ('n_post_spikes', c_int),
        ('t_max', c_double),
        ('dt', c_double),
    ]


class _LambdaITraceInput(Structure):
    _fields_ = [
        ('fwd_times', POINTER(c_double)),
        ('n_post_spikes', c_int),
        ('target_nrn_idx', c_int),
        ('v_th', c_double),
        ('tau_mem', c_double),
        ('tau_syn', c_double),
        ('t_max', c_double),
        ('dt', c_double),
    ]


class _SpikeOutput(Structure):
    _fields_ = [
        ('n_spikes', c_int),
        ('times', POINTER(c_double)),
        ('sources', POINTER(c_int)),
        ('currents', POINTER(c_double)),
    ]


class _SumsOutput(Structure):
    _fields_ = [
        ('n_rows', c_int),
        ('n_post', c_int),
        ('sum0', POINTER(c_double)),
        ('sum1', POINTER(c_double)),
    ]


class _MaximaOutput(Structure):
    _fields_ = [
        ('n_post', c_int),
        ('times', POINTER(c_double)),
        ('values', POINTER(c_double)),
    ]


class _BackwardOutput(Structure):
    _fields_ = [
        ('n_pre', c_int),
        ('n_post', c_int),
        ('grad', POINTER(c_double)),
        ('spike_errors', POINTER(c_double)),
    ]


class _VoltageTraceOutput(Structure):
    _fields_ = [
        ('n_t', c_int),
        ('v_trace', POINTER(c_double)),
        ('ts', POINTER(c_double)),
    ]


class _LambdaITraceOutput(Structure):
    _fields_ = [
        ('n_t', c_int),
        ('lambda_trace', POINTER(c_double)),
        ('ts', POINTER(c_double)),
    ]
# ── Zero-copy array wrappers ─────────────────────────────────────────────

class DoubleArrayView:
    """Zero-copy view of C double array."""
    __slots__ = ('_ptr', '_length')

    def __init__(self, ptr: POINTER(c_double), length: int):
        self._ptr = ptr
        self._length = length

    def __len__(self) -> int:
        return self._length

    def to_numpy(self) -> np.ndarray:
        if self._ptr is None or self._length == 0:
            return np.array([], dtype=np.float64)
        return np.frombuffer(
            ctypes.string_at(self._ptr, self._length * 8),
            dtype=np.float64, count=self._length,
        )

    def __array__(self, dtype=None) -> np.ndarray:
        """Allow np.asarray() to work on DoubleArrayView."""
        arr = self.to_numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    @property
    def ptr(self) -> POINTER[c_double]:
        """Raw ctypes pointer for passing to C structs."""
        return self._ptr


class Int32ArrayView:
    """Zero-copy view of C int32 array."""
    __slots__ = ('_ptr', '_length')

    def __init__(self, ptr: POINTER(c_int), length: int):
        self._ptr = ptr
        self._length = length

    def __len__(self) -> int:
        return self._length

    def to_numpy(self) -> np.ndarray:
        if self._ptr is None or self._length == 0:
            return np.array([], dtype=np.int32)
        return np.frombuffer(
            ctypes.string_at(self._ptr, self._length * 4),
            dtype=np.int32, count=self._length,
        )

    def __array__(self, dtype=None) -> np.ndarray:
        """Allow np.asarray() to work on Int32ArrayView."""
        arr = self.to_numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    @property
    def ptr(self) -> POINTER[c_int]:
        """Raw ctypes pointer for passing to C structs."""
        return self._ptr


# ── Output wrapper classes with auto-cleanup ───────────────────────────

class SpikeOutput:
    """Wrapper for C SpikeOutput with zero-copy array access and cleanup."""

    def __init__(self, c_struct: _SpikeOutput):
        self._c = c_struct

    @property
    def n_spikes(self) -> int:
        return self._c.n_spikes

    @property
    def times(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.times, self.n_spikes)

    @property
    def sources(self) -> Int32ArrayView:
        return Int32ArrayView(self._c.sources, self.n_spikes)

    @property
    def currents(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.currents, self.n_spikes)

    def __del__(self):
        if EventPropCtypes._global_lib is not None:
            EventPropCtypes._global_lib.eventprop_free_spike_output(byref(self._c))


class SumsOutput:
    """Wrapper for C SumsOutput with zero-copy array access and cleanup."""

    def __init__(self, c_struct: _SumsOutput):
        self._c = c_struct

    @property
    def n_rows(self) -> int:
        return self._c.n_rows

    @property
    def n_post(self) -> int:
        return self._c.n_post

    @property
    def sum0(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.sum0, self.n_rows * self.n_post)

    @property
    def sum1(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.sum1, self.n_rows * self.n_post)

    def __del__(self):
        if EventPropCtypes._global_lib is not None:
            EventPropCtypes._global_lib.eventprop_free_sums_output(byref(self._c))


class MaximaOutput:
    """Wrapper for C MaximaOutput with zero-copy array access and cleanup."""

    def __init__(self, c_struct: _MaximaOutput):
        self._c = c_struct

    @property
    def n_post(self) -> int:
        return self._c.n_post

    @property
    def times(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.times, self.n_post)

    @property
    def values(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.values, self.n_post)

    def __del__(self):
        if EventPropCtypes._global_lib is not None:
            EventPropCtypes._global_lib.eventprop_free_maxima_output(byref(self._c))


class BackwardOutput:
    """Wrapper for C BackwardOutput with zero-copy array access."""

    def __init__(self, c_struct: _BackwardOutput):
        self._c = c_struct

    @property
    def n_pre(self) -> int:
        return self._c.n_pre

    @property
    def n_post(self) -> int:
        return self._c.n_post

    @property
    def grad(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.grad, self.n_pre * self.n_post)

    @property
    def spike_errors(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.spike_errors, self.n_pre)

    def __del__(self):
        if EventPropCtypes._global_lib is not None:
            EventPropCtypes._global_lib.eventprop_free_backward_output(byref(self._c))


class VoltageTraceOutput:
    """Wrapper for C VoltageTraceOutput with zero-copy array access."""

    def __init__(self, c_struct: _VoltageTraceOutput):
        self._c = c_struct

    @property
    def n_t(self) -> int:
        return self._c.n_t

    @property
    def v_trace(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.v_trace, self.n_t)

    @property
    def ts(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.ts, self.n_t)

    def __del__(self):
        if EventPropCtypes._global_lib is not None:
            EventPropCtypes._global_lib.eventprop_free_voltage_trace_output(byref(self._c))


class LambdaITraceOutput:
    """Wrapper for C LambdaITraceOutput with zero-copy array access."""

    def __init__(self, c_struct: _LambdaITraceOutput):
        self._c = c_struct

    @property
    def n_t(self) -> int:
        return self._c.n_t

    @property
    def lambda_trace(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.lambda_trace, self.n_t)

    @property
    def ts(self) -> DoubleArrayView:
        return DoubleArrayView(self._c.ts, self.n_t)

    def __del__(self):
        if EventPropCtypes._global_lib is not None:
            EventPropCtypes._global_lib.eventprop_free_lambda_i_trace_output(byref(self._c))


class BatchSpikeOutput:
    """Wrapper for batch spike output (array of SpikeOutput structs)."""

    def __init__(self, c_ptr: ctypes.POINTER(_SpikeOutput), batch_size: int):
        self._c = c_ptr
        self.batch_size = batch_size

    def __getitem__(self, idx: int) -> 'SpikeOutputView':
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range [0, {self.batch_size})")
        return SpikeOutputView(self._c[idx])

    def __len__(self) -> int:
        return self.batch_size

    def __del__(self):
        # TEMP: disabled to isolate crash
        pass


class SpikeOutputView(SpikeOutput):
    """View of a SpikeOutput — doesn't free memory on deletion."""

    def __del__(self):
        pass

class BatchBackwardOutput:
    """Wrapper for batch backward spikes output (array of BackwardOutput structs)."""

    def __init__(self, c_ptr: ctypes.POINTER(_BackwardOutput), batch_size: int):
        self._c = c_ptr
        self.batch_size = batch_size

    def __getitem__(self, idx: int) -> 'BackwardOutputView':
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range [0, {self.batch_size})")
        return BackwardOutputView(self._c[idx])

    def __len__(self) -> int:
        return self.batch_size

    @property
    def grads(self) -> np.ndarray:
        """Aggregate per-sample gradients into (batch_size, n_pre, n_post) array."""
        results = []
        for i in range(self.batch_size):
            sample = self._c[i]
            grad_arr = ctypes.cast(sample.grad, POINTER(c_double * (sample.n_pre * sample.n_post)))
            grad_np = np.frombuffer(grad_arr.contents, dtype=np.float64, count=sample.n_pre * sample.n_post)
            grad_np = grad_np.reshape(sample.n_pre, sample.n_post)
            results.append(grad_np)
        return np.stack(results)

    def __del__(self):
        if EventPropCtypes._global_lib is not None and self._c:
            EventPropCtypes._global_lib.eventprop_free_backward_batch_output(
                self._c, self.batch_size
            )


class BackwardOutputView(BackwardOutput):
    """View of a BackwardOutput — doesn't free memory on deletion."""

    def __del__(self):
        pass
# ── Library loader and function bindings ─────────────────────────────────


class EventPropCtypes:
    """EventProp C library wrapper with ctypes."""

    _LIB_NAME = "libeventprop_c_api"
    _global_lib = None

    def __init__(self, lib_path=None):
        if EventPropCtypes._global_lib is not None:
            self.lib = EventPropCtypes._global_lib
            return

        if lib_path:
            self.lib = cdll.LoadLibrary(lib_path)
        else:
            import os
            for suffix in ('.so', '.dylib'):
                path = os.path.join(os.path.dirname(__file__), f"{EventPropCtypes._LIB_NAME}{suffix}")
                try:
                    self.lib = cdll.LoadLibrary(path)
                    break
                except OSError:
                    continue
            else:
                raise FileNotFoundError(
                    f"Could not load {EventPropCtypes._LIB_NAME}. Provide lib_path explicitly."
                )

        self._setup_signatures()
        EventPropCtypes._global_lib = self.lib

    def _setup_signatures(self):
        """Set up ctypes function signatures."""
        lib = self.lib

        # Forward pass (struct input) — uses toms748 root-finder which sets errno=EDOM
        lib.eventprop_compute_spikes.restype = _SpikeOutput
        lib.eventprop_compute_spikes.argtypes = [POINTER(_EventPropInput)]
        lib.eventprop_compute_spikes.use_errno = True

        lib.eventprop_free_spike_output.restype = None
        lib.eventprop_free_spike_output.argtypes = [POINTER(_SpikeOutput)]

        lib.eventprop_compute_sums.restype = _SumsOutput
        lib.eventprop_compute_sums.argtypes = [POINTER(_EventPropInput)]

        lib.eventprop_free_sums_output.restype = None
        lib.eventprop_free_sums_output.argtypes = [POINTER(_SumsOutput)]

        lib.eventprop_compute_maxima.restype = _MaximaOutput
        lib.eventprop_compute_maxima.argtypes = [POINTER(_EventPropInput)]

        lib.eventprop_free_maxima_output.restype = None
        lib.eventprop_free_maxima_output.argtypes = [POINTER(_MaximaOutput)]

        # Backward pass (struct input)
        lib.eventprop_backward_spikes.restype = _BackwardOutput
        lib.eventprop_backward_spikes.argtypes = [POINTER(_BackwardSpikesInput)]

        lib.eventprop_free_backward_output.restype = None
        lib.eventprop_free_backward_output.argtypes = [POINTER(_BackwardOutput)]

        lib.eventprop_backward_maxima.restype = _BackwardOutput
        lib.eventprop_backward_maxima.argtypes = [POINTER(_BackwardMaximaInput)]

        lib.eventprop_free_backward_output.restype = None
        lib.eventprop_free_backward_output.argtypes = [POINTER(_BackwardOutput)]

        # Diagnostic functions (struct input)
        lib.eventprop_compute_voltage_trace.restype = _VoltageTraceOutput
        lib.eventprop_compute_voltage_trace.argtypes = [POINTER(_VoltageTraceInput)]

        lib.eventprop_free_voltage_trace_output.restype = None
        lib.eventprop_free_voltage_trace_output.argtypes = [POINTER(_VoltageTraceOutput)]

        lib.eventprop_compute_lambda_i_trace.restype = _LambdaITraceOutput
        lib.eventprop_compute_lambda_i_trace.argtypes = [POINTER(_LambdaITraceInput)]

        lib.eventprop_free_lambda_i_trace_output.restype = None
        lib.eventprop_free_lambda_i_trace_output.argtypes = [POINTER(_LambdaITraceOutput)]

        # Batch operations — compute_spikes_batch uses toms748 per sample
        lib.eventprop_compute_spikes_batch.restype = POINTER(_SpikeOutput)
        lib.eventprop_compute_spikes_batch.argtypes = [
            POINTER(_BatchEventPropInput),
        ]
        lib.eventprop_compute_spikes_batch.use_errno = True

        lib.eventprop_free_batch_spike_output.restype = None
        lib.eventprop_free_batch_spike_output.argtypes = [POINTER(_SpikeOutput), c_int]

        lib.eventprop_backward_spikes_batch.restype = POINTER(_BackwardOutput)
        lib.eventprop_backward_spikes_batch.argtypes = [
            POINTER(_BatchEventPropInput),
            POINTER(_SpikeOutput),
            POINTER(c_double),
        ]

        lib.eventprop_free_backward_batch_output.restype = None
        lib.eventprop_free_backward_batch_output.argtypes = [POINTER(_BackwardOutput), c_int]
   # ── Forward pass ────────────────────────────────────────────────

    def compute_spikes(self, input: _EventPropInput) -> SpikeOutput:
        """Forward pass: predict post-synaptic spike times."""
        return SpikeOutput(self.lib.eventprop_compute_spikes(byref(input)))

    def free_spike_output(self, output: POINTER(_SpikeOutput)):
        """Free a SpikeOutput returned by compute_spikes()."""
        self.lib.eventprop_free_spike_output(output)

    def compute_sums(self, input: _EventPropInput) -> SumsOutput:
        """Compute cumulative exponential-weighted sums."""
        return SumsOutput(self.lib.eventprop_compute_sums(byref(input)))

    def free_sums_output(self, output: POINTER(_SumsOutput)):
        """Free a SumsOutput returned by compute_sums()."""
        self.lib.eventprop_free_sums_output(output)

    def compute_maxima(self, input: _EventPropInput) -> MaximaOutput:
        """Compute membrane potential maxima per neuron."""
        return MaximaOutput(self.lib.eventprop_compute_maxima(byref(input)))

    def free_maxima_output(self, output: POINTER(_MaximaOutput)):
        """Free a MaximaOutput returned by compute_maxima()."""
        self.lib.eventprop_free_maxima_output(output)

    # ── Backward pass ───────────────────────────────────────────────

    def backward_spikes(self, input: _BackwardSpikesInput) -> BackwardOutput:
        """Backward pass: compute weight gradients from post-synaptic spike errors."""
        return BackwardOutput(self.lib.eventprop_backward_spikes(byref(input)))

    def free_backward_spikes_output(self, output: POINTER(_BackwardOutput)):
        """Free a BackwardOutput returned by backward_spikes()."""
        self.lib.eventprop_free_backward_output(output)

    def backward_maxima(self, input: _BackwardMaximaInput) -> BackwardOutput:
        """Backward pass: compute weight gradients from membrane potential maxima."""
        return BackwardOutput(self.lib.eventprop_backward_maxima(byref(input)))

    def free_backward_maxima_output(self, output: POINTER(_BackwardOutput)):
        """Free a BackwardOutput returned by backward_maxima()."""
        self.lib.eventprop_free_backward_output(output)

    # ── Diagnostic functions ────────────────────────────────────────

    def compute_voltage_trace(self, input: _VoltageTraceInput) -> VoltageTraceOutput:
        """Compute membrane potential voltage trace for a single neuron."""
        return VoltageTraceOutput(self.lib.eventprop_compute_voltage_trace(byref(input)))

    def free_voltage_trace_output(self, output: POINTER(_VoltageTraceOutput)):
        """Free a VoltageTraceOutput returned by compute_voltage_trace()."""
        self.lib.eventprop_free_voltage_trace_output(output)

    def compute_lambda_i_trace(self, input: _LambdaITraceInput) -> LambdaITraceOutput:
        """Compute lambda-I trace over time for a single neuron."""
        return LambdaITraceOutput(self.lib.eventprop_compute_lambda_i_trace(byref(input)))

    def free_lambda_i_trace_output(self, output: POINTER(_LambdaITraceOutput)):
        """Free a LambdaITraceOutput returned by compute_lambda_i_trace()."""
        self.lib.eventprop_free_lambda_i_trace_output(output)

    # ── Batch operations ────────────────────────────────────────────

    def compute_spikes_batch(
        self, input: _BatchEventPropInput
    ) -> BatchSpikeOutput:
        """Batch forward pass: predict post-synaptic spikes for multiple samples."""
        c_ptr = self.lib.eventprop_compute_spikes_batch(byref(input))
        return BatchSpikeOutput(c_ptr, input.batch_size)

    def free_batch_spike_output(self, output: POINTER(_SpikeOutput), batch_size: int):
        """Free a batch SpikeOutput returned by compute_spikes_batch()."""
        self.lib.eventprop_free_batch_spike_output(output, batch_size)

    def backward_spikes_batch(
        self, input: _BatchEventPropInput,
        fwd_batch: POINTER(_SpikeOutput), post_errors: POINTER(c_double)
    ) -> BatchBackwardOutput:
        """Batch backward pass: compute weight gradients for multiple samples."""
        c_ptr = self.lib.eventprop_backward_spikes_batch(
            byref(input), fwd_batch, post_errors
        )
        return BatchBackwardOutput(c_ptr, input.batch_size)

    def free_backward_spikes_batch_output(self, output: POINTER(_BackwardOutput), batch_size: int):
        """Free a batch BackwardOutput returned by backward_spikes_batch()."""
        self.lib.eventprop_free_backward_batch_output(output, batch_size)

    # ── Convenience wrappers (keyword-argument API) ────────────────
    # These construct the ctypes Structures internally so callers can pass
    # numpy arrays directly instead of manually building structs.

    @staticmethod
    def _np_to_ptr(array):
        """Convert a numpy array to a ctypes pointer (no copy)."""
        if array is None or len(array) == 0:
            return None
        return np.asarray(array, dtype=np.float64).ctypes.data_as(POINTER(c_double))

    @staticmethod
    def _np_to_int_ptr(array):
        """Convert a numpy int array to a ctypes pointer (no copy)."""
        if array is None or len(array) == 0:
            return None
        return np.asarray(array, dtype=np.int32).ctypes.data_as(POINTER(c_int))

    def _make_input(self, w, n_pre, n_post, times, sources, v_th=1.0,
                    tau_mem=20e-3, tau_syn=5e-3):
        """Construct an _EventPropInput from keyword arguments."""
        inp = _EventPropInput()
        inp.w = self._np_to_ptr(w)
        inp.n_pre = int(n_pre)
        inp.n_post = int(n_post)
        inp.times = self._np_to_ptr(times)
        inp.sources = self._np_to_int_ptr(sources)
        inp.n_spikes = len(times) if times is not None else 0
        inp.tau_mem = float(tau_mem)
        inp.tau_syn = float(tau_syn)
        inp.v_th = float(v_th)
        return inp

    def compute_spikes_kw(self, w, n_pre, n_post, times, sources,
                          v_th=1.0, tau_mem=20e-3, tau_syn=5e-3) -> SpikeOutput:
        """Forward pass with keyword arguments."""
        return self.compute_spikes(self._make_input(w, n_pre, n_post, times, sources,
                                                     v_th, tau_mem, tau_syn))

    def compute_maxima_kw(self, w, n_pre, n_post, times, sources,
                          tau_mem=20e-3, tau_syn=5e-3) -> MaximaOutput:
        """Compute maxima with keyword arguments (v_th unused for LI layer)."""
        return self.compute_maxima(self._make_input(w, n_pre, n_post, times, sources,
                                                     v_th=np.inf, tau_mem=tau_mem, tau_syn=tau_syn))

    def backward_spikes_kw(self, w, n_pre, n_post, times, sources,
                           fwd_times, fwd_sources, fwd_currents, post_errors,
                           v_th=1.0, tau_mem=20e-3, tau_syn=5e-3) -> BackwardOutput:
        """Backward pass with keyword arguments."""
        inp = self._make_input(w, n_pre, n_post, times, sources, v_th, tau_mem, tau_syn)
        backward_inp = _BackwardSpikesInput()
        backward_inp.input = pointer(inp)
        backward_inp.fwd_times = self._np_to_ptr(fwd_times)
        backward_inp.fwd_sources = self._np_to_int_ptr(fwd_sources)
        backward_inp.fwd_currents = self._np_to_ptr(fwd_currents)
        backward_inp.post_errors = self._np_to_ptr(post_errors)
        backward_inp.n_post_spikes = len(fwd_times) if fwd_times is not None else 0
        return self.backward_spikes(backward_inp)

    def backward_maxima_kw(self, w, n_pre, n_post, times, sources,
                           maxima_times, maxima_values, maxima_errors,
                           tau_mem=20e-3, tau_syn=5e-3) -> BackwardOutput:
        """Backward maxima with keyword arguments."""
        inp = self._make_input(w, n_pre, n_post, times, sources,
                               v_th=np.inf, tau_mem=tau_mem, tau_syn=tau_syn)
        backward_inp = _BackwardMaximaInput()
        backward_inp.input = pointer(inp)
        backward_inp.maxima_times = self._np_to_ptr(maxima_times)
        backward_inp.maxima_values = self._np_to_ptr(maxima_values)
        backward_inp.maxima_errors = self._np_to_ptr(maxima_errors)
        backward_inp.n_maxima = len(maxima_times) if maxima_times is not None else 0
        return self.backward_maxima(backward_inp)

    def compute_spikes_batch_kw(self, w, n_pre, n_post,
                                times_list, sources_list,
                                v_th=1.0, tau_mem=20e-3, tau_syn=5e-3) -> BatchSpikeOutput:
        """Batch forward pass with keyword arguments.

        Concatenates per-sample times/sources into flat arrays for the C library,
        which expects a single EventPropInput with offset-based sample access.

        Parameters
        ----------
        times_list, sources_list : list of np.ndarray
            One (times, sources) pair per sample.
        """
        batch_size = len(times_list)
        n_spikes_per_sample = (c_int * batch_size)(*[len(t) for t in times_list])

        # C library expects flat concatenated arrays — it computes offsets internally
        all_times = np.concatenate(times_list) if times_list else np.array([], dtype=np.float64)
        all_sources = np.concatenate(sources_list) if sources_list else np.array([], dtype=np.int32)

        inp = self._make_input(
            w, n_pre, n_post,
            all_times, all_sources,
            v_th=v_th, tau_mem=tau_mem, tau_syn=tau_syn,
        )

        batch_inp = _BatchEventPropInput()
        batch_inp.input = pointer(inp)
        batch_inp.n_spikes_per_sample = n_spikes_per_sample
        batch_inp.batch_size = batch_size

        output = self.compute_spikes_batch(batch_inp)
        # Keep references alive so C pointers remain valid during OpenMP execution
        output._refs = (inp, all_times, all_sources, n_spikes_per_sample, batch_inp)
        return output

    def backward_spikes_batch_kw(self, w, n_pre, n_post,
                                 times_list, sources_list,
                                 fwd_batch, post_errors_list,
                                 v_th=1.0, tau_mem=20e-3, tau_syn=5e-3) -> BatchBackwardOutput:
        """Batch backward pass with keyword arguments.

        Concatenates per-sample times/sources into flat arrays for the C library.

        Parameters
        ----------
        fwd_batch : BatchSpikeOutput
            Forward outputs from compute_spikes_batch_kw.
        post_errors_list : list of np.ndarray
            Error values per forward spike, one array per sample.
        """
        batch_size = len(times_list)
        n_spikes_per_sample = (c_int * batch_size)(*[len(t) for t in times_list])

        # C library expects flat concatenated arrays
        all_times = np.concatenate(times_list) if times_list else np.array([], dtype=np.float64)
        all_sources = np.concatenate(sources_list) if sources_list else np.array([], dtype=np.int32)

        inp = self._make_input(
            w, n_pre, n_post,
            all_times, all_sources,
            v_th=v_th, tau_mem=tau_mem, tau_syn=tau_syn,
        )

        batch_inp = _BatchEventPropInput()
        batch_inp.input = pointer(inp)
        batch_inp.n_spikes_per_sample = n_spikes_per_sample
        batch_inp.batch_size = batch_size

        # Flatten post_errors into single array for C
        all_errors = np.concatenate(post_errors_list) if post_errors_list else np.array([], dtype=np.float64)
        errors_ptr = self._np_to_ptr(all_errors)

        # Keep references alive so C pointers remain valid during OpenMP execution
        output = self.backward_spikes_batch(batch_inp, fwd_batch._c, errors_ptr)
        output._refs = (inp, all_times, all_sources, n_spikes_per_sample, batch_inp, all_errors)
        return output

    def compute_voltage_trace_kw(self, w, n_pre, n_post,
                                 times, sources,
                                 target_nrn_idx, fwd_times, fwd_sources,
                                 v_th=1.0, tau_mem=20e-3, tau_syn=5e-3,
                                 t_max=1.0, dt=1e-4) -> VoltageTraceOutput:
        """Voltage trace with keyword arguments."""
        inp = self._make_input(w, n_pre, n_post, times, sources, v_th, tau_mem, tau_syn)
        trace_inp = _VoltageTraceInput()
        trace_inp.input = pointer(inp)
        trace_inp.target_nrn_idx = int(target_nrn_idx)
        trace_inp.fwd_times = self._np_to_ptr(fwd_times)
        trace_inp.fwd_sources = self._np_to_int_ptr(fwd_sources)
        trace_inp.n_post_spikes = len(fwd_times) if fwd_times is not None else 0
        trace_inp.t_max = float(t_max)
        trace_inp.dt = float(dt)
        return self.compute_voltage_trace(trace_inp)

    def compute_lambda_i_trace_kw(self, fwd_times, target_nrn_idx,
                                  v_th=1.0, tau_mem=20e-3, tau_syn=5e-3,
                                  t_max=1.0, dt=1e-4) -> LambdaITraceOutput:
        """Lambda-I trace with keyword arguments."""
        trace_inp = _LambdaITraceInput()
        trace_inp.fwd_times = self._np_to_ptr(fwd_times)
        trace_inp.n_post_spikes = len(fwd_times) if fwd_times is not None else 0
        trace_inp.target_nrn_idx = int(target_nrn_idx)
        trace_inp.v_th = float(v_th)
        trace_inp.tau_mem = float(tau_mem)
        trace_inp.tau_syn = float(tau_syn)
        trace_inp.t_max = float(t_max)
        trace_inp.dt = float(dt)
        return self.compute_lambda_i_trace(trace_inp)


# ── Convenience function for quick access ────────────────────────────────

def load_library(lib_path=None) -> EventPropCtypes:
    """Load the EventProp C library.

    Args:
        lib_path: Optional path to shared library (.so/.dylib)

    Returns:
        EventPropCtypes instance with zero-copy access
    """
    return EventPropCtypes(lib_path)
