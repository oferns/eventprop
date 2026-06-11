"""EventProp ctypes bindings — thin wrapper matching C signatures."""

from __future__ import annotations
import ctypes
from ctypes import cdll, c_double, c_int32, c_int, Structure, POINTER, byref
import numpy as np


# ── Input structs (borrowed data) matching eventprop.h ─────────────────



class _ForwardInputBase(Structure):
    _fields_ = [
        ('w', POINTER(c_double)),
        ('n_pre', c_int),
        ('n_post', c_int),
        ('times', POINTER(c_double)),
        ('sources', POINTER(c_int32)),
        ('n_spikes', c_int),
        ('tau_mem', c_double),
        ('tau_syn', c_double),
    ]




class _BackwardInputBase(Structure):
    _fields_ = [
        ('base', POINTER(_ForwardInputBase)),
    ]


class _BackwardSpikesInput(Structure):
    _fields_ = [
        ('base', _BackwardInputBase),
        ('fwd_times', POINTER(c_double)),
        ('fwd_sources', POINTER(c_int32)),
        ('fwd_currents', POINTER(c_double)),
        ('post_errors', POINTER(c_double)),
        ('n_post_spikes', c_int),
    ]


class _BackwardMaximaInput(Structure):
    _fields_ = [
        ('base', _BackwardInputBase),
        ('maxima_times', POINTER(c_double)),
        ('maxima_values', POINTER(c_double)),
        ('maxima_errors', POINTER(c_double)),
        ('n_maxima', c_int),
    ]


class _VoltageTraceInput(Structure):
    _fields_ = [
        ('input', POINTER(_ForwardInputBase)),
        ('target_nrn_idx', c_int),
        ('fwd_times', POINTER(c_double)),
        ('fwd_sources', POINTER(c_int32)),
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
        ('sources', POINTER(c_int32)),
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

    @property
    def ptr(self) -> POINTER[c_double]:
        """Raw ctypes pointer for passing to C structs."""
        return self._ptr


class Int32ArrayView:
    """Zero-copy view of C int32 array."""
    __slots__ = ('_ptr', '_length')

    def __init__(self, ptr: POINTER(c_int32), length: int):
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

    @property
    def ptr(self) -> POINTER[c_int32]:
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
        if EventPropCtypes._global_lib is not None and self._c:
            EventPropCtypes._global_lib.eventprop_free_batch_spike_output(
                self._c, self.batch_size
            )


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

        # Forward pass (struct input)
        lib.eventprop_compute_spikes.restype = _SpikeOutput
        lib.eventprop_compute_spikes.argtypes = [POINTER(_ForwardInputBase)]

        lib.eventprop_free_spike_output.restype = None
        lib.eventprop_free_spike_output.argtypes = [POINTER(_SpikeOutput)]

        lib.eventprop_compute_sums.restype = _SumsOutput
        lib.eventprop_compute_sums.argtypes = [POINTER(_ForwardInputBase)]

        lib.eventprop_free_sums_output.restype = None
        lib.eventprop_free_sums_output.argtypes = [POINTER(_SumsOutput)]

        lib.eventprop_compute_maxima.restype = _MaximaOutput
        lib.eventprop_compute_maxima.argtypes = [POINTER(_ForwardInputBase)]

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

        # Batch operations
        lib.eventprop_compute_spikes_batch.restype = POINTER(_SpikeOutput)
        lib.eventprop_compute_spikes_batch.argtypes = [
            POINTER(_ForwardInputBase),
            POINTER(c_int),
            c_int,
        ]

        lib.eventprop_free_batch_spike_output.restype = None
        lib.eventprop_free_batch_spike_output.argtypes = [POINTER(_SpikeOutput), c_int]

        lib.eventprop_backward_spikes_batch.restype = POINTER(_BackwardOutput)
        lib.eventprop_backward_spikes_batch.argtypes = [
            POINTER(_BackwardInputBase),
            POINTER(c_int),
            POINTER(_SpikeOutput),
            POINTER(c_double),
            c_int,
        ]

        lib.eventprop_free_backward_batch_output.restype = None
        lib.eventprop_free_backward_batch_output.argtypes = [POINTER(_BackwardOutput), c_int]
   # ── Forward pass ────────────────────────────────────────────────

    def compute_spikes(self, input: _ForwardInputBase) -> SpikeOutput:
        """Forward pass: predict post-synaptic spike times."""
        return SpikeOutput(self.lib.eventprop_compute_spikes(byref(input)))

    def free_spike_output(self, output: POINTER(_SpikeOutput)):
        """Free a SpikeOutput returned by compute_spikes()."""
        self.lib.eventprop_free_spike_output(output)

    def compute_sums(self, input: _ForwardInputBase) -> SumsOutput:
        """Compute cumulative exponential-weighted sums."""
        return SumsOutput(self.lib.eventprop_compute_sums(byref(input)))

    def free_sums_output(self, output: POINTER(_SumsOutput)):
        """Free a SumsOutput returned by compute_sums()."""
        self.lib.eventprop_free_sums_output(output)

    def compute_maxima(self, input: _ForwardInputBase) -> MaximaOutput:
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
        self, input: _ForwardInputBase, n_spikes_per_sample: POINTER(c_int), batch_size: int
    ) -> BatchSpikeOutput:
        """Batch forward pass: predict post-synaptic spikes for multiple samples."""
        c_ptr = self.lib.eventprop_compute_spikes_batch(byref(input), n_spikes_per_sample, batch_size)
        return BatchSpikeOutput(c_ptr, batch_size)

    def free_batch_spike_output(self, output: POINTER(_SpikeOutput), batch_size: int):
        """Free a batch SpikeOutput returned by compute_spikes_batch()."""
        self.lib.eventprop_free_batch_spike_output(output, batch_size)

    def backward_spikes_batch(
        self, base: _BackwardInputBase, n_spikes_per_sample: POINTER(c_int),
        fwd_batch: POINTER(_SpikeOutput), post_errors: POINTER(c_double), batch_size: int
    ) -> BatchBackwardOutput:
        """Batch backward pass: compute weight gradients for multiple samples."""
        c_ptr = self.lib.eventprop_backward_spikes_batch(
            byref(base), n_spikes_per_sample, fwd_batch, post_errors, batch_size
        )
        return BatchBackwardOutput(c_ptr, batch_size)

    def free_backward_spikes_batch_output(self, output: POINTER(_BackwardOutput), batch_size: int):
        """Free a batch BackwardOutput returned by backward_spikes_batch()."""
        self.lib.eventprop_free_backward_batch_output(output, batch_size)


# ── Convenience function for quick access ────────────────────────────────

def load_library(lib_path=None) -> EventPropCtypes:
    """Load the EventProp C library.

    Args:
        lib_path: Optional path to shared library (.so/.dylib)

    Returns:
        EventPropCtypes instance with zero-copy access
    """
    return EventPropCtypes(lib_path)
