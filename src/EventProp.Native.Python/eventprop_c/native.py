"""EventProp ctypes bindings — zero-copy access to C alloc functions."""

from __future__ import annotations
import ctypes
from ctypes import cdll, c_double, c_int32, c_int, Structure, POINTER
from ctypes import byref, addressof
import numpy as np


# ── C struct definitions matching eventprop_alloc.h ───────────────────────

class _SpikeOutput(Structure):
    """C SpikeOutput struct."""
    _fields_ = [
        ('n_spikes', c_int),
        ('times', POINTER(c_double)),
        ('sources', POINTER(c_int32)),
        ('currents', POINTER(c_double))
    ]


class _SumsOutput(Structure):
    """C SumsOutput struct."""
    _fields_ = [
        ('n_rows', c_int),
        ('n_post', c_int),
        ('sum0', POINTER(c_double)),
        ('sum1', POINTER(c_double))
    ]


class _MaximaOutput(Structure):
    """C MaximaOutput struct."""
    _fields_ = [
        ('n_post', c_int),
        ('times', POINTER(c_double)),
        ('values', POINTER(c_double))
    ]


class _BackwardSpikesOutput(Structure):
    """C BackwardSpikesOutput struct."""
    _fields_ = [
        ('n_pre', c_int),
        ('n_post', c_int),
        ('grad', POINTER(c_double)),
        ('spike_errors', POINTER(c_double))
    ]


class _BackwardMaximaOutput(Structure):
    """C BackwardMaximaOutput struct."""
    _fields_ = [
        ('n_pre', c_int),
        ('n_post', c_int),
        ('gradient', POINTER(c_double)),
        ('spike_errors', POINTER(c_double))
    ]


class _VoltageTraceOutput(Structure):
    """C VoltageTraceOutput struct."""
    _fields_ = [
        ('n_t', c_int),
        ('v_trace', POINTER(c_double)),
        ('ts', POINTER(c_double))
    ]


class _LambdaITraceOutput(Structure):
    """C LambdaITraceOutput struct."""
    _fields_ = [
        ('n_t', c_int),
        ('lambda_trace', POINTER(c_double)),
        ('ts', POINTER(c_double))
    ]


# ── Zero-copy array wrappers ─────────────────────────────────────────────

class DoubleArrayView:
    """Zero-copy view of C double array. No allocation — just pointer access."""
    
    __slots__ = ('_ptr', '_length')
    
    def __init__(self, ptr: POINTER(c_double), length: int):
        self._ptr = ptr
        self._length = length
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index: int) -> float:
        if self._ptr is None or index >= self._length:
            raise IndexError("Index out of range")
        return float(self._ptr[index])
    
    def __iter__(self):
        for i in range(self._length):
            yield self[i]
    
    def to_list(self) -> list:
        """Convert to Python list (copy)."""
        return [self[i] for i in range(self._length)]
    
    def to_numpy(self) -> np.ndarray:
        """Create numpy array directly from C memory (zero-copy view)."""
        if self._ptr is None or self._length == 0:
            return np.array([], dtype=np.float64)
        return np.frombuffer(
            ctypes.string_at(self._ptr, self._length * 8),
            dtype=np.float64,
            count=self._length
        )
    
    @property
    def memoryview_obj(self):
        """Return Python memoryview for zero-copy access."""
        if self._ptr is None:
            return memoryview(b'')
        # Create memoryview directly on C memory (no copy!)
        return memoryview(ctypes.string_at(self._ptr, self._length * 8))


class Int32ArrayView:
    """Zero-copy view of C int32 array. No allocation — just pointer access."""
    
    __slots__ = ('_ptr', '_length')
    
    def __init__(self, ptr: POINTER(c_int32), length: int):
        self._ptr = ptr
        self._length = length
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index: int) -> int:
        if self._ptr is None or index >= self._length:
            raise IndexError("Index out of range")
        return int(self._ptr[index])
    
    def __iter__(self):
        for i in range(self._length):
            yield self[i]
    
    def to_list(self) -> list:
        """Convert to Python list (copy)."""
        return [self[i] for i in range(self._length)]
    
    def to_numpy(self) -> np.ndarray:
        """Create numpy array directly from C memory (zero-copy view)."""
        if self._ptr is None or self._length == 0:
            return np.array([], dtype=np.int32)
        return np.frombuffer(
            ctypes.string_at(self._ptr, self._length * 4),
            dtype=np.int32,
            count=self._length
        )


# ── Output classes with zero-copy access and automatic cleanup ───────────

class SpikeOutput:
    """Wrapper for C SpikeOutput with zero-copy array access and cleanup."""
    
    def __init__(self, c_struct: _SpikeOutput):
        self._c = c_struct
    
    @property
    def n_spikes(self) -> int:
        return self._c.n_spikes
    
    @property
    def times(self) -> DoubleArrayView:
        """Zero-copy view of spike times."""
        return DoubleArrayView(self._c.times, self.n_spikes)
    
    @property
    def sources(self) -> Int32ArrayView:
        """Zero-copy view of source neurons."""
        return Int32ArrayView(self._c.sources, self.n_spikes)
    
    @property
    def currents(self) -> DoubleArrayView:
        """Zero-copy view of synaptic currents."""
        return DoubleArrayView(self._c.currents, self.n_spikes)
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c'):
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_spike_output(byref(self._c))


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
        """Zero-copy view of sum0 array (n_rows × n_post)."""
        return DoubleArrayView(self._c.sum0, self.n_rows * self.n_post)
    
    @property
    def sum1(self) -> DoubleArrayView:
        """Zero-copy view of sum1 array (n_rows × n_post)."""
        return DoubleArrayView(self._c.sum1, self.n_rows * self.n_post)
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c'):
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_sums_output(byref(self._c))


class MaximaOutput:
    """Wrapper for C MaximaOutput with zero-copy array access and cleanup."""
    
    def __init__(self, c_struct: _MaximaOutput):
        self._c = c_struct
    
    @property
    def n_post(self) -> int:
        return self._c.n_post
    
    @property
    def times(self) -> DoubleArrayView:
        """Zero-copy view of maxima times."""
        return DoubleArrayView(self._c.times, self.n_post)
    
    @property
    def values(self) -> DoubleArrayView:
        """Zero-copy view of maxima values."""
        return DoubleArrayView(self._c.values, self.n_post)
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c'):
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_maxima_output(byref(self._c))


class BackwardSpikesOutput:
    """Wrapper for C BackwardSpikesOutput with zero-copy array access."""
    
    def __init__(self, c_struct: _BackwardSpikesOutput):
        self._c = c_struct
    
    @property
    def n_pre(self) -> int:
        return self._c.n_pre
    
    @property
    def n_post(self) -> int:
        return self._c.n_post
    
    @property
    def grad(self) -> DoubleArrayView:
        """Zero-copy view of weight gradient (n_pre × n_post)."""
        return DoubleArrayView(self._c.grad, self.n_pre * self.n_post)
    
    @property
    def spike_errors(self) -> DoubleArrayView:
        """Zero-copy view of input spike errors."""
        return DoubleArrayView(self._c.spike_errors, self.n_pre)  # Approximation
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c'):
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_backward_spikes_output(byref(self._c))


class BackwardMaximaOutput:
    """Wrapper for C BackwardMaximaOutput with zero-copy array access."""
    
    def __init__(self, c_struct: _BackwardMaximaOutput):
        self._c = c_struct
    
    @property
    def n_pre(self) -> int:
        return self._c.n_pre
    
    @property
    def n_post(self) -> int:
        return self._c.n_post
    
    @property
    def gradient(self) -> DoubleArrayView:
        """Zero-copy view of weight gradient (n_pre × n_post)."""
        return DoubleArrayView(self._c.gradient, self.n_pre * self.n_post)
    
    @property
    def spike_errors(self) -> DoubleArrayView:
        """Zero-copy view of input spike errors."""
        return DoubleArrayView(self._c.spike_errors, self.n_pre)  # Approximation
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c'):
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_backward_maxima_output(byref(self._c))


class VoltageTraceOutput:
    """Wrapper for C VoltageTraceOutput with zero-copy array access."""
    
    def __init__(self, c_struct: _VoltageTraceOutput):
        self._c = c_struct
    
    @property
    def n_t(self) -> int:
        return self._c.n_t
    
    @property
    def v_trace(self) -> DoubleArrayView:
        """Zero-copy view of voltage trace values."""
        return DoubleArrayView(self._c.v_trace, self.n_t)
    
    @property
    def ts(self) -> DoubleArrayView:
        """Zero-copy view of time array."""
        return DoubleArrayView(self._c.ts, self.n_t)
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c'):
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_voltage_trace_output(byref(self._c))


class LambdaITraceOutput:
    """Wrapper for C LambdaITraceOutput with zero-copy array access."""
    
    def __init__(self, c_struct: _LambdaITraceOutput):
        self._c = c_struct
    
    @property
    def n_t(self) -> int:
        return self._c.n_t
    
    @property
    def lambda_trace(self) -> DoubleArrayView:
        """Zero-copy view of lambda trace values."""
        return DoubleArrayView(self._c.lambda_trace, self.n_t)
    
    @property
    def ts(self) -> DoubleArrayView:
        """Zero-copy view of time array."""
        return DoubleArrayView(self._c.ts, self.n_t)
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c'):
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_lambda_i_trace_output(byref(self._c))


class BatchSpikeOutput:
    """Wrapper for batch spike output (array of SpikeOutput structs).
    
    Provides per-sample access to output spikes from batch forward pass.
    """
    
    def __init__(self, c_ptr: ctypes.POINTER(_SpikeOutput), batch_size: int):
        self._c = c_ptr
        self._batch_size = batch_size
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def outputs(self) -> list[SpikeOutput]:
        """Access per-sample SpikeOutput objects."""
        return [SpikeOutput(self._c[i]) for i in range(self.batch_size)]
    
    def __getitem__(self, idx: int) -> SpikeOutput:
        """Get output for sample at index (doesn't own memory)."""
        return _SpikeOutputView(self._c[idx])
    
    def __len__(self) -> int:
        return self.batch_size
    
    @property
    def n_spikes_per_sample(self) -> list[int]:
        """Get spike count per sample."""
        return [self._c[i].n_spikes for i in range(self.batch_size)]
    
    def __del__(self):
        """Free C memory on garbage collection."""
        if hasattr(self, '_c') and self._c:
            lib = EventPropCtypes._get_global_lib()
            if lib:
                lib.eventprop_free_batch_spike_output(self._c, self._batch_size)


class _SpikeOutputView(SpikeOutput):
    """View of a SpikeOutput (doesn't free memory on deletion)."""
    
    def __del__(self):
        """Don't free memory - owned by parent BatchSpikeOutput."""
        pass


# ── Library loader and function bindings ─────────────────────────────────

def _get_array_pointer(arr, ctype=c_double):
    """Get ctypes pointer from numpy array (zero-copy) or create new ctypes array.
    
    Args:
        arr: numpy array (zero-copy) or list/tuple (creates new ctypes array)
        ctype: ctypes type (c_double for float64, c_int32 for int32)
    
    Returns:
        ctypes pointer to array data
    """
    if hasattr(arr, 'ctypes'):  # numpy array - zero-copy
        return arr.ctypes.data_as(ctypes.POINTER(ctype))
    else:  # list/tuple - create ctypes array
        return (ctype * len(arr))(*arr)


class EventPropCtypes:
    """EventProp C library wrapper with ctypes.
    
    Provides zero-copy access to all alloc functions without numpy dependency.
    
    Example:
        lib = EventPropCtypes()
        
        # Forward pass with zero-copy access
        output = lib.compute_spikes(w, times, sources)
        
        # Access spike data without copying
        for t in output.times:  # Zero-copy iteration
            print(f"Spike at {t:.6f}")
        
        # Cleanup is automatic on garbage collection
    """
    
    _LIB_NAME = "libeventprop_c_api"
    _global_lib = None
    
    def __init__(self, lib_path: str | None = None):
        """Load the EventProp C library.
        
        Args:
            lib_path: Optional path to shared library (.so/.dylib)
        """
        if EventPropCtypes._global_lib:
            self.lib = EventPropCtypes._global_lib
            return
        
        if lib_path:
            self.lib = cdll.LoadLibrary(lib_path)
        else:
            # Try standard paths
            import os
            lib_paths = [
                f"./{EventPropCtypes._LIB_NAME}.so",
                f"./{EventPropCtypes._LIB_NAME}.dylib",
                os.path.join(os.path.dirname(__file__), f"{EventPropCtypes._LIB_NAME}.so"),
                os.path.join(os.path.dirname(__file__), f"{EventPropCtypes._LIB_NAME}.dylib"),
            ]
            
            for path in lib_paths:
                try:
                    self.lib = cdll.LoadLibrary(path)
                    break
                except OSError:
                    continue
            else:
                raise FileNotFoundError(
                    f"Could not load {EventPropCtypes._LIB_NAME}. "
                    "Provide lib_path explicitly."
                )
        
        self._setup_function_signatures()
        EventPropCtypes._global_lib = self.lib
    
    @classmethod
    def _get_global_lib(cls):
        return cls._global_lib
    
    def _setup_function_signatures(self):
        """Set up ctypes function signatures."""
        lib = self.lib
        
        # compute_spikes_alloc
        lib.eventprop_compute_spikes_alloc.restype = _SpikeOutput
        lib.eventprop_compute_spikes_alloc.argtypes = [
            POINTER(c_double), c_int, c_int,  # w, n_pre, n_post
            POINTER(c_double), POINTER(c_int32), c_int,  # times, sources, n_spikes
            c_double, c_double, c_double  # v_th, tau_mem, tau_syn
        ]
        
        # free_spike_output
        lib.eventprop_free_spike_output.restype = None
        lib.eventprop_free_spike_output.argtypes = [POINTER(_SpikeOutput)]
        
        # compute_sums_alloc
        lib.eventprop_compute_sums_alloc.restype = _SumsOutput
        lib.eventprop_compute_sums_alloc.argtypes = [
            POINTER(c_double), c_int, c_int,  # w, n_pre, n_post
            POINTER(c_double), POINTER(c_int32), c_int,  # times, sources, n_spikes
            c_double, c_double  # tau_mem, tau_syn
        ]
        
        # free_sums_output
        lib.eventprop_free_sums_output.restype = None
        lib.eventprop_free_sums_output.argtypes = [POINTER(_SumsOutput)]
        
        # compute_maxima_alloc
        lib.eventprop_compute_maxima_alloc.restype = _MaximaOutput
        lib.eventprop_compute_maxima_alloc.argtypes = [
            POINTER(c_double), c_int, c_int,  # w, n_pre, n_post
            POINTER(c_double), POINTER(c_int32), c_int,  # times, sources, n_spikes
            c_double, c_double  # tau_mem, tau_syn
        ]
        
        # free_maxima_output
        lib.eventprop_free_maxima_output.restype = None
        lib.eventprop_free_maxima_output.argtypes = [POINTER(_MaximaOutput)]
        
        # backward_spikes_alloc
        lib.eventprop_backward_spikes_alloc.restype = _BackwardSpikesOutput
        lib.eventprop_backward_spikes_alloc.argtypes = [
            POINTER(c_double), POINTER(c_int32), c_int,  # times, sources, n_spikes
            POINTER(c_double), POINTER(c_int32), POINTER(c_double),  # fwd_times, sources, currents
            POINTER(c_double), c_int,  # post_errors, n_post_spikes
            POINTER(c_double), c_int, c_int,  # w, n_pre, n_post
            c_double, c_double, c_double  # v_th, tau_mem, tau_syn
        ]
        
        # free_backward_spikes_output
        lib.eventprop_free_backward_spikes_output.restype = None
        lib.eventprop_free_backward_spikes_output.argtypes = [POINTER(_BackwardSpikesOutput)]
        
        # backward_maxima_alloc
        lib.eventprop_backward_maxima_alloc.restype = _BackwardMaximaOutput
        lib.eventprop_backward_maxima_alloc.argtypes = [
            POINTER(c_double), POINTER(c_int32), c_int,  # input_times, sources, n_input_spikes
            POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int,  # maxima_times, values, errors, n_maxima
            POINTER(c_double), c_int, c_int,  # w, n_pre, n_post
            c_double, c_double  # tau_mem, tau_syn
        ]
        
        # free_backward_maxima_output
        lib.eventprop_free_backward_maxima_output.restype = None
        lib.eventprop_free_backward_maxima_output.argtypes = [POINTER(_BackwardMaximaOutput)]
        
        # ── Diagnostic functions (alloc versions) ───────────────────────
        
        # compute_voltage_trace_alloc
        lib.eventprop_compute_voltage_trace_alloc.restype = _VoltageTraceOutput
        lib.eventprop_compute_voltage_trace_alloc.argtypes = [
            c_double, c_double,  # t_max, dt
            c_int,  # target_nrn_idx
            POINTER(c_double), c_int, c_int,  # w, n_pre, n_post
            POINTER(c_double), POINTER(c_int32), c_int,  # times, sources, n_spikes
            POINTER(c_double), POINTER(c_int32), c_int,  # fwd_times, fwd_sources, n_post_spikes
            c_double, c_double, c_double  # v_th, tau_mem, tau_syn
        ]
        
        # free_voltage_trace_output
        lib.eventprop_free_voltage_trace_output.restype = None
        lib.eventprop_free_voltage_trace_output.argtypes = [POINTER(_VoltageTraceOutput)]
        
        # compute_lambda_i_trace_alloc
        lib.eventprop_compute_lambda_i_trace_alloc.restype = _LambdaITraceOutput
        lib.eventprop_compute_lambda_i_trace_alloc.argtypes = [
            c_double, c_double,  # t_max, dt
            c_int,  # target_nrn_idx
            POINTER(c_double), c_int,  # fwd_times, n_post_spikes
            c_double, c_double, c_double  # v_th, tau_mem, tau_syn
        ]
        
        # free_lambda_i_trace_output
        lib.eventprop_free_lambda_i_trace_output.restype = None
        lib.eventprop_free_lambda_i_trace_output.argtypes = [POINTER(_LambdaITraceOutput)]
        
        # compute_lambda_i (returns single double, no alloc needed)
        lib.eventprop_compute_lambda_i.restype = c_double
        lib.eventprop_compute_lambda_i.argtypes = [
            c_double,  # t
            c_int,     # target_nrn_idx
            POINTER(c_double), c_int,  # fwd_times, n_post_spikes
            c_double, c_double, c_double  # v_th, tau_mem, tau_syn
        ]
        
        # ── Batch operations ───────────────────────────────────────────────
        
        # compute_spikes_batch_alloc (returns pointer to array of SpikeOutput)
        lib.eventprop_compute_spikes_batch_alloc.restype = POINTER(_SpikeOutput)
        lib.eventprop_compute_spikes_batch_alloc.argtypes = [
            POINTER(c_double), c_int, c_int,  # w, n_pre, n_post
            POINTER(c_double), POINTER(c_int32),  # times, sources (flat arrays)
            POINTER(c_int), c_int,  # n_spikes_per_sample, batch_size
            c_double, c_double, c_double  # v_th, tau_mem, tau_syn
        ]
        
        # free_batch_spike_output
        lib.eventprop_free_batch_spike_output.restype = None
        lib.eventprop_free_batch_spike_output.argtypes = [
            POINTER(_SpikeOutput), c_int  # output array, batch_size
        ]
    
    # ── High-level Python API ─────────────────────────────────────────────
    
    def compute_spikes(self, w: list | tuple, times: list | tuple,
                       sources: list | tuple,
                       n_pre: int, n_post: int,
                       v_th: float = 1.0, tau_mem: float = 0.02,
                       tau_syn: float = 0.005) -> SpikeOutput:
        """Forward pass: predict post-synaptic spike times.
        
        Args:
            w: Weight matrix (flat list, row-major)
            times: Pre-synaptic spike times
            sources: Pre-synaptic source neurons
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            v_th: Spike threshold voltage
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
            
        Returns:
            SpikeOutput with zero-copy access to times, sources, currents
        """
        
        # Zero-copy access for numpy arrays
        w_ptr = _get_array_pointer(w, c_double)
        times_ptr = _get_array_pointer(times, c_double)
        sources_ptr = _get_array_pointer(sources, c_int32)
        
        c_struct = self.lib.eventprop_compute_spikes_alloc(
            w_ptr, n_pre, n_post,
            times_ptr, sources_ptr, len(times),
            c_double(v_th), c_double(tau_mem), c_double(tau_syn)
        )
        
        return SpikeOutput(c_struct)
    
    def compute_spikes_batch(self, w: list | tuple,
                             times: list | tuple,
                             sources: list | tuple,
                             n_spikes_per_sample: list | tuple,
                             batch_size: int,
                             n_pre: int, n_post: int,
                             v_th: float = 1.0, tau_mem: float = 0.02,
                             tau_syn: float = 0.005) -> BatchSpikeOutput:
        """Batch forward pass: predict post-synaptic spikes for multiple samples.
        
        Args:
            w: Weight matrix (flat list, row-major)
            times: Flat array of all input spike times (sorted per sample)
            sources: Flat array of all source neurons
            n_spikes_per_sample: Spike count per sample (length: batch_size)
            batch_size: Number of samples in batch
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            v_th: Spike threshold voltage
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
            
        Returns:
            BatchSpikeOutput with per-sample access to times, sources, currents
        """
        
        # Zero-copy access for numpy arrays
        w_ptr = _get_array_pointer(w, c_double)
        times_ptr = _get_array_pointer(times, c_double)
        sources_ptr = _get_array_pointer(sources, c_int32)
        n_spikes_ptr = _get_array_pointer(n_spikes_per_sample, c_int)
        
        # Call batch function (returns pointer to array of SpikeOutput)
        c_ptr = self.lib.eventprop_compute_spikes_batch_alloc(
            w_ptr, n_pre, n_post,
            times_ptr, sources_ptr,
            n_spikes_ptr, batch_size,
            c_double(v_th), c_double(tau_mem), c_double(tau_syn)
        )
        
        return BatchSpikeOutput(c_ptr, batch_size)
    
    def compute_sums(self, w: list | tuple, times: list | tuple,
                     sources: list | tuple,
                     n_pre: int, n_post: int,
                     tau_mem: float = 0.02, tau_syn: float = 0.005) -> SumsOutput:
        """Compute cumulative exponential-weighted sums.
        
        Args:
            w: Weight matrix (flat list, row-major)
            times: Pre-synaptic spike times
            sources: Pre-synaptic source neurons
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
            
        Returns:
            SumsOutput with zero-copy access to sum0, sum1 arrays
        """
        
        # Zero-copy access for numpy arrays
        w_ptr = _get_array_pointer(w, c_double)
        times_ptr = _get_array_pointer(times, c_double)
        sources_ptr = _get_array_pointer(sources, c_int32)
        
        c_struct = self.lib.eventprop_compute_sums_alloc(
            w_ptr, n_pre, n_post,
            times_ptr, sources_ptr, len(times),
            c_double(tau_mem), c_double(tau_syn)
        )
        
        return SumsOutput(c_struct)
    
    def compute_maxima(self, w: list | tuple, times: list | tuple,
                       sources: list | tuple,
                       n_pre: int, n_post: int,
                       tau_mem: float = 0.02, tau_syn: float = 0.005) -> MaximaOutput:
        """Compute membrane potential maxima per neuron.
        
        Args:
            w: Weight matrix (flat list, row-major)
            times: Pre-synaptic spike times
            sources: Pre-synaptic source neurons
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
            
        Returns:
            MaximaOutput with zero-copy access to times, values
        """
        
        # Zero-copy access for numpy arrays
        w_ptr = _get_array_pointer(w, c_double)
        times_ptr = _get_array_pointer(times, c_double)
        sources_ptr = _get_array_pointer(sources, c_int32)
        
        c_struct = self.lib.eventprop_compute_maxima_alloc(
            w_ptr, n_pre, n_post,
            times_ptr, sources_ptr, len(times),
            c_double(tau_mem), c_double(tau_syn)
        )
        
        return MaximaOutput(c_struct)
    
    # ── Backward pass functions ─────────────────────────────────────────
    
    def backward_spikes(self, times: list | tuple, sources: list | tuple,
                        fwd_times: list | tuple, fwd_sources: list | tuple,
                        fwd_currents: list | tuple, post_errors: list | tuple,
                        w: list | tuple, n_pre: int, n_post: int,
                        v_th: float = 1.0, tau_mem: float = 0.02,
                        tau_syn: float = 0.005) -> BackwardSpikesOutput:
        """Backward pass: compute weight gradients from post-synaptic spike errors.
        
        Args:
            times: Pre-synaptic spike times
            sources: Pre-synaptic source neurons
            fwd_times: Forward pass post-synaptic spike times
            fwd_sources: Forward pass post-synaptic source neurons
            fwd_currents: Forward pass synaptic currents at spike times
            post_errors: Post-synaptic spike errors (same length as fwd_times)
            w: Weight matrix (flat list, row-major)
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            v_th: Spike threshold voltage
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
        
        Returns:
            BackwardSpikesOutput with zero-copy access to grad and spike_errors
        """
        # Zero-copy access for numpy arrays
        times_ptr = _get_array_pointer(times, c_double)
        sources_ptr = _get_array_pointer(sources, c_int32)
        fwd_times_ptr = _get_array_pointer(fwd_times, c_double)
        fwd_sources_ptr = _get_array_pointer(fwd_sources, c_int32)
        fwd_currents_ptr = _get_array_pointer(fwd_currents, c_double)
        post_errors_ptr = _get_array_pointer(post_errors, c_double)
        w_ptr = _get_array_pointer(w, c_double)
        
        c_struct = self.lib.eventprop_backward_spikes_alloc(
            times_ptr, sources_ptr, len(times),
            fwd_times_ptr, fwd_sources_ptr, fwd_currents_ptr,
            post_errors_ptr, len(post_errors),
            w_ptr, n_pre, n_post,
            c_double(v_th), c_double(tau_mem), c_double(tau_syn)
        )
        
        return BackwardSpikesOutput(c_struct)
    
    def backward_maxima(self, input_times: list | tuple, input_sources: list | tuple,
                        maxima_times: list | tuple, maxima_values: list | tuple,
                        maxima_errors: list | tuple,
                        w: list | tuple, n_pre: int, n_post: int,
                        tau_mem: float = 0.02, tau_syn: float = 0.005) -> BackwardMaximaOutput:
        """Backward pass: compute weight gradients from membrane potential maxima.
        
        Args:
            input_times: Pre-synaptic spike times
            input_sources: Pre-synaptic source neurons
            maxima_times: Maxima times per neuron (length n_post)
            maxima_values: Maxima values per neuron (length n_post)
            maxima_errors: Maxima errors per neuron (length n_post)
            w: Weight matrix (flat list, row-major) — not used but required
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
        
        Returns:
            BackwardMaximaOutput with zero-copy access to gradient and spike_errors
        """
        # Zero-copy access for numpy arrays
        input_times_ptr = _get_array_pointer(input_times, c_double)
        input_sources_ptr = _get_array_pointer(input_sources, c_int32)
        maxima_times_ptr = _get_array_pointer(maxima_times, c_double)
        maxima_values_ptr = _get_array_pointer(maxima_values, c_double)
        maxima_errors_ptr = _get_array_pointer(maxima_errors, c_double)
        w_ptr = _get_array_pointer(w, c_double)
        
        c_struct = self.lib.eventprop_backward_maxima_alloc(
            input_times_ptr, input_sources_ptr, len(input_times),
            maxima_times_ptr, maxima_values_ptr, maxima_errors_ptr, len(maxima_times),
            w_ptr, n_pre, n_post,
            c_double(tau_mem), c_double(tau_syn)
        )
        
        return BackwardMaximaOutput(c_struct)
    
    # ── Diagnostic functions ─────────────────────────────────────────────
    
    def compute_voltage_trace(self, t_max: float, dt: float,
                              target_nrn_idx: int,
                              w: list | tuple, n_pre: int, n_post: int,
                              times: list | tuple, sources: list | tuple,
                              fwd_times: list | tuple = None,
                              fwd_sources: list | tuple = None,
                              v_th: float = 1.0, tau_mem: float = 0.02,
                              tau_syn: float = 0.005) -> VoltageTraceOutput:
        """Compute membrane potential voltage trace for a single neuron.
        
        Args:
            t_max: Maximum simulation time
            dt: Time step
            target_nrn_idx: Post-synaptic neuron index
            w: Weight matrix (flat list, row-major)
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            times: Pre-synaptic spike times
            sources: Pre-synaptic source neurons
            fwd_times: Post-synaptic spike times (optional)
            fwd_sources: Post-synaptic source neurons (optional)
            v_th: Spike threshold voltage
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
        
        Returns:
            VoltageTraceOutput with zero-copy access to v_trace and ts
        """
        if fwd_times is None:
            fwd_times = []
        if fwd_sources is None:
            fwd_sources = []
        
        w_arr = (c_double * len(w))(*w)
        times_arr = (c_double * len(times))(*times) if times else None
        sources_arr = (c_int32 * len(sources))(*sources) if sources else None
        fwd_times_arr = (c_double * len(fwd_times))(*fwd_times) if fwd_times else None
        fwd_sources_arr = (c_int32 * len(fwd_sources))(*fwd_sources) if fwd_sources else None
        
        c_struct = self.lib.eventprop_compute_voltage_trace_alloc(
            c_double(t_max), c_double(dt), target_nrn_idx,
            w_arr, n_pre, n_post,
            times_arr, sources_arr, len(times),
            fwd_times_arr, fwd_sources_arr, len(fwd_times),
            c_double(v_th), c_double(tau_mem), c_double(tau_syn)
        )
        
        return VoltageTraceOutput(c_struct)
    
    def compute_lambda_i_trace(self, t_max: float, dt: float,
                               target_nrn_idx: int,
                               fwd_times: list | tuple, fwd_sources: list | tuple,
                               v_th: float = 1.0, tau_mem: float = 0.02,
                               tau_syn: float = 0.005) -> LambdaITraceOutput:
        """Compute lambda-I trace over time for a single neuron.
        
        Args:
            t_max: Maximum simulation time
            dt: Time step
            target_nrn_idx: Post-synaptic neuron index
            fwd_times: Post-synaptic spike times
            fwd_sources: Post-synaptic source neurons
            v_th: Spike threshold voltage
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
        
        Returns:
            LambdaITraceOutput with zero-copy access to lambda_trace and ts
        """
        fwd_times_arr = (c_double * len(fwd_times))(*fwd_times)
        
        c_struct = self.lib.eventprop_compute_lambda_i_trace_alloc(
            c_double(t_max), c_double(dt), target_nrn_idx,
            fwd_times_arr, len(fwd_times),
            c_double(v_th), c_double(tau_mem), c_double(tau_syn)
        )
        
        return LambdaITraceOutput(c_struct)
    
    def compute_lambda_i(self, t: float, target_nrn_idx: int,
                        fwd_times: list | tuple, n_post_spikes: int,
                        v_th: float = 1.0, tau_mem: float = 0.02,
                        tau_syn: float = 0.005) -> float:
        """Compute synaptic adjoint at a specific time.
        
        Args:
            t: Time at which to compute lambda_i
            target_nrn_idx: Post-synaptic neuron index
            fwd_times: Post-synaptic spike times
            n_post_spikes: Number of post-synaptic spikes
            v_th: Spike threshold voltage
            tau_mem: Membrane time constant (seconds)
            tau_syn: Synaptic time constant (seconds)
        
        Returns:
            Lambda-i value at the given time
        """
        fwd_times_arr = (c_double * len(fwd_times))(*fwd_times)
        
        return float(self.lib.eventprop_compute_lambda_i(
            c_double(t), target_nrn_idx,
            fwd_times_arr, n_post_spikes,
            c_double(v_th), c_double(tau_mem), c_double(tau_syn)
        ))


# ── Convenience function for quick access ────────────────────────────────

def load_library(lib_path: str | None = None) -> EventPropCtypes:
    """Load the EventProp C library.
    
    Args:
        lib_path: Optional path to shared library
        
    Returns:
        EventPropCtypes instance with zero-copy access
    """
    return EventPropCtypes(lib_path)
