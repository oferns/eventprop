# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython bindings for EventProp C library.

Zero-copy interface: numpy arrays → flat buffers → C structs → back to numpy views.
No data copies across the FFI boundary.

Philosophy:
    - compute_spikes_batch() constructs EventPropInput[] in Cython, calls non-flat C API
    - Wrapper classes own both input pointers (C-allocated) and output pointers (C-returned)
    - backward_*_batch_accumulate() combines parallel backward + sequential gradient sum in C
    - Managed wrapper classes keep C memory alive via __dealloc__
"""

import ctypes
import os
import sys
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free, malloc

cnp.import_array()


# ---------------------------------------------------------------------------
# 1. Load the shared library (once at import time)
# ---------------------------------------------------------------------------

def _load_lib():
    """Find and load libeventprop_c_api."""
    base = os.path.dirname(__file__)
    for suffix in ('.dll', '.so', '.dylib'):
        path = os.path.join(base, f"libeventprop_c_api{suffix}")
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError:
            pass
    # Fallback search paths
    if sys.platform == 'win32':
        for suffix in ('.dll', '.lib'):
            path = os.path.join(base, f"eventprop_c_api{suffix}")
            try:
                return ctypes.cdll.LoadLibrary(path)
            except OSError:
                pass
    for libdir in ('.', '/usr/local/lib', '/opt/homebrew/lib'):
        suffix = '.dylib' if sys.platform == 'darwin' else '.so'
        path = os.path.join(libdir, f"libeventprop_c_api{suffix}")
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError:
            pass
    raise FileNotFoundError(
        "Could not load libeventprop_c_api from:\n" +
        "\n".join(f"  {os.path.join(base, f'libeventprop_c_api{s}')}" for s in ('.dll', '.so', '.dylib'))
    )


_load_lib()


# ---------------------------------------------------------------------------
# 2. C declarations — eventprop.h (single-sample API)
# ---------------------------------------------------------------------------

cdef extern from "eventprop.h":

    # Error codes (enum maps to int in Cython)
    ctypedef enum EventPropError:
        EVENTPROP_OK                  = 0
        EVENTPROP_INVALID_DIMENSION   = 1
        EVENTPROP_NEGATIVE_SPIKES     = 2
        EVENTPROP_NULL_POINTER        = 3
        EVENTPROP_INVALID_SOURCE      = 4
        EVENTPROP_INVALID_TAU         = 5
        EVENTPROP_ALLOCATION_FAILED   = 6
        EVENTPROP_INVALID_CURRENT     = 7

    ctypedef struct MatrixView:
        const double* data
        int rows
        int cols

    ctypedef struct MutableMatrixView:
        double* data
        int rows
        int cols

    ctypedef struct EventPropInput:
        const double* w
        int n_pre
        int n_post
        const double* times
        const int* sources
        int n_spikes
        double tau_mem
        double tau_syn
        double v_th

    ctypedef struct SpikeOutput:
        EventPropError error_no
        int n_spikes
        double* times
        int* sources
        double* currents
        double* errors

    ctypedef struct SumsOutput:
        EventPropError error_no
        double* sum0
        double* sum1

    ctypedef struct MaximaOutput:
        EventPropError error_no
        double* times
        double* values
        double* errors

    ctypedef struct BackwardOutput:
        EventPropError error_no
        double* grad
        double* spike_errors

    # Free functions (take pointer to null out fields after free)
    void eventprop_free_spike_output(SpikeOutput* output)
    void eventprop_free_sums_output(SumsOutput* output)
    void eventprop_free_maxima_output(MaximaOutput* output)
    void eventprop_free_backward_output(BackwardOutput* output)

    # Core functions (return by value, heap-allocated internals)
    SpikeOutput     eventprop_compute_spikes(const EventPropInput input)
    SumsOutput      eventprop_compute_sums(const EventPropInput input)
    MaximaOutput    eventprop_compute_maxima(const EventPropInput input)
    BackwardOutput  eventprop_backward_spikes(const EventPropInput input, const SpikeOutput spike)
    # backward_maxima takes direct maxima_times + maxima_errors arrays (length n_post)
    BackwardOutput  eventprop_backward_maxima(
        const EventPropInput input,
        const double* maxima_times,
        const double* maxima_errors)

    # Input validation helper
    EventPropError eventprop_validate_input(const EventPropInput input)


# ---------------------------------------------------------------------------
# 3. C declarations — eventprop_batch.h (batch API)
# ---------------------------------------------------------------------------

cdef extern from "eventprop_batch.h":

    ctypedef struct SpikesBatch:
        int batch_size
        int has_errors
        SpikeOutput* spikes

    void eventprop_free_spikes_batch(SpikesBatch* batch)

    # Batch backward output — per-sample results
    ctypedef struct BackwardsBatch:
        int batch_size
        int has_errors
        BackwardOutput* outputs

    void eventprop_free_backwards_batch(BackwardsBatch* batch)

    ctypedef struct MaximaBatch:
        int batch_size
        int has_errors
        MaximaOutput* outputs

    void eventprop_free_maxima_batch(MaximaBatch* batch)

    # Dropout output — per-sample surviving spikes
    ctypedef struct DropoutOutput:
        int n_spikes
        double* times
        int* sources

    void eventprop_free_dropout_output(DropoutOutput* output)

    ctypedef struct DropoutBatch:
        int batch_size
        int has_errors
        DropoutOutput* outputs

    void eventprop_free_dropout_batch(DropoutBatch* batch)

    DropoutBatch eventprop_dropout_spikes_batch(
        const double* times,
        const int* sources,
        const int* n_per_sample,
        int batch_size,
        double p_drop,
        int seed)

    ctypedef struct GradientOutput:
        double* data
        int rows
        int cols

    void eventprop_free_gradient_output(GradientOutput* output)

    # Combined backward + accumulate result (primary FFI entry point)
    ctypedef struct AccumulatedBackwardResult:
        GradientOutput summed_grads   # accumulated gradient with dimensions
        int has_errors                # 1 if any sample returned an error
        BackwardOutput* outputs       # per-sample (.grad freed, .spike_errors kept)
        int batch_size

    void eventprop_free_accumulated_backward_result(AccumulatedBackwardResult* result)

    # Batch functions (return by value, heap-allocated internals)
    SpikesBatch  eventprop_compute_spikes_batch(const EventPropInput* inputs, int batch_size)

    BackwardsBatch eventprop_backward_spikes_batch(
        const EventPropInput* inputs, const SpikesBatch spikes_batch)

    MaximaBatch eventprop_compute_maxima_batch(
        const EventPropInput* inputs, const SpikesBatch spikes_batch)

    # backward_maxima_batch uses MaximaOutput.errors written by caller
    BackwardsBatch eventprop_backward_maxima_batch(
        const EventPropInput* inputs, const MaximaBatch maxima_batch)

    # Gradient accumulation helper (skips samples with errors)
    GradientOutput eventprop_accumulate_gradients(BackwardsBatch batch, int n_pre, int n_post)

    # Combined backward + accumulate result (primary FFI entry point)
    AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(
        const EventPropInput* inputs, const SpikesBatch spikes_batch)

    # backward_maxima_batch_accumulate takes inputs separately
    AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
        const EventPropInput* inputs, const MaximaBatch maxima_batch)


# ---------------------------------------------------------------------------
# 3b. C declarations — eventprop_diag.h (diagnostic helpers)
# ---------------------------------------------------------------------------

cdef extern from "eventprop_diag.h":

    ctypedef struct VoltageTraceInput:
        const EventPropInput* input
        int target_nrn_idx
        const double* fwd_times
        const int* fwd_sources
        int n_post_spikes
        double t_max
        double dt

    ctypedef struct VoltageTraceOutput:
        int n_t
        double* v_trace
        double* ts
        EventPropError error_no

    void eventprop_free_voltage_trace_output(VoltageTraceOutput* output)

    VoltageTraceOutput eventprop_compute_voltage_trace(const VoltageTraceInput input)

    ctypedef struct LambdaITraceInput:
        const double* fwd_times
        int n_post_spikes
        int target_nrn_idx
        double v_th
        double tau_mem
        double tau_syn
        double t_max
        double dt

    ctypedef struct LambdaITraceOutput:
        int n_t
        double* lambda_trace
        double* ts
        EventPropError error_no

    void eventprop_free_lambda_i_trace_output(LambdaITraceOutput* output)

    LambdaITraceOutput eventprop_compute_lambda_i_trace(const LambdaITraceInput input)


# ---------------------------------------------------------------------------
# 4. Zero-copy numpy view helpers
# ---------------------------------------------------------------------------

cdef inline object _double_view(double* ptr, ssize_t size):
    """Create a zero-copy writable numpy float64 view over C memory."""
    cdef cnp.npy_intp dims[1]
    dims[0] = <cnp.npy_intp>size
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = \
        cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_FLOAT64, <void*>ptr)
    return np.asarray(arr)


cdef inline object _int_view(int* ptr, ssize_t size):
    """Create a zero-copy writable numpy int32 view over C memory."""
    cdef cnp.npy_intp dims[1]
    dims[0] = <cnp.npy_intp>size
    cdef cnp.ndarray[cnp.int32_t, ndim=1] arr = \
        cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_INT32, <void*>ptr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# 5. Managed wrapper classes — keep C memory alive via __dealloc__
# ---------------------------------------------------------------------------

cdef class _SpikesBatchOutput:
    """Managed wrapper around SpikesBatch + EventPropInput[] (C-owned memory).

    Returned by compute_spikes_batch(). Keeps both the input array and spike outputs
    alive for the lifetime of this Python object. Provides zero-copy access to per-sample
    spike data and error buffers.

    The .inputs array is stored in C and reused directly by backward/maxima calls —
    no Python-side reconstruction needed.
    """
    cdef EventPropInput* _inputs      # pointer to heap-allocated input array (owned)
    cdef SpikesBatch* _spikes         # pointer to heap-allocated spikes batch (owned)
    cdef bint _owned                  # True if we own the memory (should free)
    # Keep flat arrays alive when inputs are constructed in Cython (not C-owned)
    cdef object _times_ref            # reference to original times array
    cdef object _sources_ref          # reference to original sources array

    def __cinit__(self):
        self._inputs = NULL
        self._spikes = NULL
        self._owned = False
        self._times_ref = None
        self._sources_ref = None

    @staticmethod
    cdef _SpikesBatchOutput from_c_result(EventPropInput* inputs, SpikesBatch result):
        """Wrap C return values by copying spikes to heap and managing lifetime."""
        cdef _SpikesBatchOutput obj = _SpikesBatchOutput.__new__(_SpikesBatchOutput)
        obj._inputs = inputs  # caller transfers ownership
        obj._spikes = <SpikesBatch*>malloc(sizeof(SpikesBatch))
        if obj._spikes == NULL:
            raise MemoryError("Failed to allocate SpikesBatch wrapper")
        obj._spikes[0] = result  # copy struct by value onto heap
        obj._owned = True
        return obj

    def __len__(self):
        if self._spikes == NULL:
            raise RuntimeError("SpikesBatchOutput has been freed")
        return self._spikes.batch_size

    def __getitem__(self, int idx):
        """Return (times, sources, currents) as zero-copy numpy views for sample idx."""
        cdef SpikeOutput* out = &self._spikes.spikes[idx]
        if out.n_spikes <= 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        return (
            _double_view(out.times, out.n_spikes),
            _int_view(out.sources, out.n_spikes),
            _double_view(out.currents, out.n_spikes),
        )

    def get_errors(self, int idx):
        """Return zero-copy writable view into sample idx's error buffer.

        Python writes target errors here before calling backward().
        C reads them directly — no copying needed.
        """
        cdef SpikeOutput* out = &self._spikes.spikes[idx]
        return _double_view(out.errors, out.n_spikes)

    def get_error_no(self, int idx):
        """Return the error code for sample idx (EVENTPROP_OK == 0 on success)."""
        cdef SpikeOutput* out = &self._spikes.spikes[idx]
        return out.error_no

    @property
    def batch_size(self):
        if self._spikes == NULL:
            return 0
        return self._spikes.batch_size

    @property
    def has_errors(self):
        """True if any sample in the batch returned an error."""
        if self._spikes == NULL:
            return False
        return bool(self._spikes.has_errors)

    @property
    def n_pre(self):
        if self._inputs == NULL:
            return 0
        return self._inputs[0].n_pre

    @property
    def n_post(self):
        if self._inputs == NULL:
            return 0
        return self._inputs[0].n_post

    def __dealloc__(self):
        if self._owned and self._spikes != NULL:
            eventprop_free_spikes_batch(self._spikes)
            free(self._spikes)
            self._spikes = NULL
        if self._owned and self._inputs != NULL:
            free(self._inputs)
            self._inputs = NULL


cdef class _MaximaBatchOutput:
    """Managed wrapper around MaximaBatch + EventPropInput[] (C-owned memory).

    Returned by compute_maxima_batch(). Keeps both the input array and maxima outputs
    alive for the lifetime of this Python object. Provides zero-copy access to per-sample
    maxima data.

    The .inputs array is stored in C and reused directly by backward calls —
    no Python-side reconstruction needed.
    """
    cdef EventPropInput* _inputs      # pointer to heap-allocated input array (owned)
    cdef MaximaBatch* _maxima         # pointer to heap-allocated maxima batch (owned)
    cdef bint _owned                  # True if we own the memory (should free)
    # Keep flat arrays alive when inputs are constructed in Cython (not C-owned)
    cdef object _times_ref            # reference to original times array
    cdef object _sources_ref          # reference to original sources array

    def __cinit__(self):
        self._inputs = NULL
        self._maxima = NULL
        self._owned = False
        self._times_ref = None
        self._sources_ref = None

    @staticmethod
    cdef _MaximaBatchOutput from_c_result(EventPropInput* inputs, MaximaBatch result):
        """Wrap C return values by copying maxima to heap and managing lifetime."""
        cdef _MaximaBatchOutput obj = _MaximaBatchOutput.__new__(_MaximaBatchOutput)
        obj._inputs = inputs  # caller transfers ownership
        obj._maxima = <MaximaBatch*>malloc(sizeof(MaximaBatch))
        if obj._maxima == NULL:
            raise MemoryError("Failed to allocate MaximaBatch wrapper")
        obj._maxima[0] = result  # copy struct by value onto heap
        obj._owned = True
        return obj

    def __len__(self):
        if self._maxima == NULL:
            raise RuntimeError("MaximaBatchOutput has been freed")
        return self._maxima.batch_size

    @property
    def batch_size(self):
        if self._maxima == NULL:
            return 0
        return self._maxima.batch_size

    @property
    def has_errors(self):
        """True if any sample in the batch returned an error."""
        if self._maxima == NULL:
            return False
        return bool(self._maxima.has_errors)

    @property
    def n_post(self):
        if self._inputs == NULL:
            return 0
        return self._inputs[0].n_post

    def __getitem__(self, int idx):
        """Return (times, values) as zero-copy numpy views for sample idx.

        Used by spikes.py MaximaVector._batch_output.__getitem__.
        """
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return (
            _double_view(out.times, self.n_post),
            _double_view(out.values, self.n_post),
        )

    def get_times(self, int idx):
        """Zero-copy view into sample idx's maxima times."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.times, self.n_post)

    def get_values(self, int idx):
        """Zero-copy view into sample idx's maxima values."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.values, self.n_post)

    def get_maxima_errors(self, int idx):
        """Return zero-copy writable view into sample idx's maxima error buffer.

        Python writes target errors here before calling backward_maxima().
        C reads them directly — no copying needed.
        """
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.errors, self.n_post)

    def get_error_no(self, int idx):
        """Return the error code for sample idx (EVENTPROP_OK == 0 on success)."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return out.error_no

    def __dealloc__(self):
        if self._owned and self._maxima != NULL:
            eventprop_free_maxima_batch(self._maxima)
            free(self._maxima)
            self._maxima = NULL
        if self._owned and self._inputs != NULL:
            free(self._inputs)
            self._inputs = NULL


cdef class _AccumulatedBackwardResult:
    """Managed wrapper around AccumulatedBackwardResult (C-owned memory).

    Returned by backward_*_batch_accumulate(). Holds summed gradient and per-sample
    outputs (.grad + .spike_errors both kept alive for drill-down access).
    """
    cdef AccumulatedBackwardResult* _ptr   # pointer to heap-allocated struct
    cdef bint _owned                       # True if we own the memory (should free)
    cdef object _n_spikes_arr              # kept alive for spike error sizing

    def __cinit__(self):
        self._ptr = NULL
        self._owned = False

    @staticmethod
    cdef _AccumulatedBackwardResult from_c_result(AccumulatedBackwardResult result, int[:] n_spikes_per_sample):
        """Wrap a C return value by copying it to heap and managing lifetime."""
        cdef _AccumulatedBackwardResult obj = _AccumulatedBackwardResult.__new__(_AccumulatedBackwardResult)
        obj._ptr = <AccumulatedBackwardResult*>malloc(sizeof(AccumulatedBackwardResult))
        if obj._ptr == NULL:
            raise MemoryError("Failed to allocate AccumulatedBackwardResult wrapper")
        obj._ptr[0] = result  # copy struct by value onto heap
        obj._owned = True
        # Keep n_spikes array alive for spike error sizing (prevents GC)
        obj._n_spikes_arr = n_spikes_per_sample
        return obj

    @property
    def grad(self):
        """Zero-copy flat gradient array. Use .reshape(rows, cols) for weight update."""
        if self._ptr == NULL:
            raise RuntimeError("AccumulatedBackwardResult has been freed")
        return _double_view(self._ptr.summed_grads.data,
                            self._ptr.summed_grads.rows * self._ptr.summed_grads.cols)

    @property
    def rows(self):
        if self._ptr == NULL:
            return 0
        return self._ptr.summed_grads.rows

    @property
    def cols(self):
        if self._ptr == NULL:
            return 0
        return self._ptr.summed_grads.cols

    @property
    def batch_size(self):
        if self._ptr == NULL:
            return 0
        return self._ptr.batch_size

    @property
    def has_errors(self):
        """True if any sample in the batch returned an error."""
        if self._ptr == NULL:
            return False
        return bool(self._ptr.has_errors)

    def get_spike_errors(self, int idx):
        """Get spike errors for sample idx (for layer chain propagation)."""
        cdef BackwardOutput* out = &self._ptr.outputs[idx]
        if out.spike_errors == NULL:
            return None
        return _double_view(out.spike_errors, self._n_spikes_arr[idx])

    def get_grad(self, int idx):
        """Get per-sample gradient for sample idx (for drill-down visualization)."""
        cdef BackwardOutput* out = &self._ptr.outputs[idx]
        if out.grad == NULL:
            return None
        return _double_view(out.grad,
                            self._ptr.summed_grads.rows * self._ptr.summed_grads.cols)

    def __dealloc__(self):
        if self._owned and self._ptr != NULL:
            eventprop_free_accumulated_backward_result(self._ptr)
            free(self._ptr)
            self._ptr = NULL


cdef class _DropoutBatchOutput:
    """Managed wrapper around DropoutBatch (C-owned memory).

    Returned by dropout_spikes_batch(). Keeps the C heap allocation alive
    for the lifetime of this Python object. Provides zero-copy access to per-sample
    surviving spike data.
    """
    cdef DropoutBatch* _ptr   # pointer to heap-allocated struct
    cdef bint _owned          # True if we own the memory (should free)

    def __cinit__(self):
        self._ptr = NULL
        self._owned = False

    @staticmethod
    cdef _DropoutBatchOutput from_c_result(DropoutBatch result):
        """Wrap a C return value by copying it to heap and managing lifetime."""
        cdef _DropoutBatchOutput obj = _DropoutBatchOutput.__new__(_DropoutBatchOutput)
        obj._ptr = <DropoutBatch*>malloc(sizeof(DropoutBatch))
        if obj._ptr == NULL:
            raise MemoryError("Failed to allocate DropoutBatch wrapper")
        obj._ptr[0] = result  # copy struct by value onto heap
        obj._owned = True
        return obj

    def __len__(self):
        if self._ptr == NULL:
            raise RuntimeError("DropoutBatchOutput has been freed")
        return self._ptr.batch_size

    def __getitem__(self, int idx):
        """Return (times, sources) as zero-copy numpy views for sample idx."""
        cdef DropoutOutput* out = &self._ptr.outputs[idx]
        if out.n_spikes <= 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int32)
        return (
            _double_view(out.times, out.n_spikes),
            _int_view(out.sources, out.n_spikes),
        )

    @property
    def batch_size(self):
        if self._ptr == NULL:
            return 0
        return self._ptr.batch_size

    @property
    def has_errors(self):
        """True if any sample in the batch returned an error."""
        if self._ptr == NULL:
            return False
        return bool(self._ptr.has_errors)

    def __dealloc__(self):
        if self._owned and self._ptr != NULL:
            eventprop_free_dropout_batch(self._ptr)
            free(self._ptr)
            self._ptr = NULL


# ---------------------------------------------------------------------------
# 6. Internal helper — construct EventPropInput[] from flat numpy arrays
# ---------------------------------------------------------------------------

cdef EventPropInput* _build_inputs(
        double[:] times, int[:] sources, int[:] n_per_sample,
        double[:, ::1] w, int n_pre, int n_post,
        double tau_mem, double tau_syn, double v_th):
    """Construct a heap-allocated EventPropInput[] from flat numpy arrays.

    Returns pointer to caller-owned memory (caller must free via libc.free).
    Pointers into times/sources are valid as long as the original numpy arrays live.
    """
    cdef int batch_size = n_per_sample.shape[0]
    cdef EventPropInput* inputs = <EventPropInput*>malloc(batch_size * sizeof(EventPropInput))
    if inputs == NULL:
        raise MemoryError("Failed to allocate EventPropInput array")

    cdef const double* w_ptr = &w[0, 0]
    cdef const double* t_ptr
    cdef const int* s_ptr
    cdef int b, k
    for b in range(batch_size):
        inputs[b].w = w_ptr
        inputs[b].n_pre = n_pre
        inputs[b].n_post = n_post
        inputs[b].tau_mem = tau_mem
        inputs[b].tau_syn = tau_syn
        inputs[b].v_th = v_th

        # Walk flat arrays to find this sample's slice
        t_ptr = &times[0]
        s_ptr = &sources[0]
        for k in range(b):
            t_ptr += n_per_sample[k]
            s_ptr += n_per_sample[k]
        inputs[b].times = t_ptr
        inputs[b].sources = s_ptr
        inputs[b].n_spikes = n_per_sample[b]

    return inputs


# ---------------------------------------------------------------------------
# 7. Public API — Python-callable functions
# ---------------------------------------------------------------------------

def compute_spikes_batch(
        double[:] times, int[:] sources, int[:] n_per_sample,
        double[:, ::1] w, int n_pre, int n_post,
        double tau_mem, double tau_syn, double v_th):
    """Batch forward pass — construct EventPropInput[] in Cython, call non-flat C API.

    Returns _SpikesBatchOutput that owns both the input array and spike outputs.
    Keeps references to original flat arrays so input pointers stay valid for backward passes.
    """
    cdef int batch_size = n_per_sample.shape[0]
    cdef EventPropInput* inputs = _build_inputs(
        times, sources, n_per_sample, w, n_pre, n_post, tau_mem, tau_syn, v_th)

    cdef SpikesBatch result = eventprop_compute_spikes_batch(inputs, batch_size)

    # Wrap: transfer input ownership to wrapper, copy spikes onto heap
    cdef _SpikesBatchOutput obj = _SpikesBatchOutput.from_c_result(inputs, result)
    # Keep flat arrays alive so input pointers stay valid
    obj._times_ref = times
    obj._sources_ref = sources
    return obj


def compute_maxima_batch(
        double[:] times, int[:] sources, int[:] n_per_sample,
        double[:, ::1] w, int n_pre, int n_post,
        double tau_mem, double tau_syn, double v_th):
    """Batch maxima pass — construct inputs in Cython, compute spikes then maxima.

    Returns _MaximaBatchOutput that owns both the input array and maxima outputs.
    Keeps references to original flat arrays so input pointers stay valid for backward passes.
    """
    cdef int batch_size = n_per_sample.shape[0]
    cdef EventPropInput* inputs = _build_inputs(
        times, sources, n_per_sample, w, n_pre, n_post, tau_mem, tau_syn, v_th)

    # Compute spikes first (needed for maxima)
    cdef SpikesBatch spikes_result = eventprop_compute_spikes_batch(inputs, batch_size)

    # Compute maxima using the same inputs and spike results
    cdef MaximaBatch maxima_result = eventprop_compute_maxima_batch(inputs, spikes_result)

    # Free intermediate spikes batch (maxima doesn't need it after computation)
    eventprop_free_spikes_batch(&spikes_result)

    # Wrap: transfer input ownership to wrapper, copy maxima onto heap
    cdef _MaximaBatchOutput obj = _MaximaBatchOutput.from_c_result(inputs, maxima_result)
    # Keep flat arrays alive so input pointers stay valid
    obj._times_ref = times
    obj._sources_ref = sources
    return obj


def backward_spikes_batch_accumulate(_SpikesBatchOutput batch_output):
    """Backward pass (spike loss) + gradient accumulation.

    Uses batch_output's internal .inputs and .spikes directly — no Python-side work.
    Returns _AccumulatedBackwardResult with summed gradients and per-sample spike errors.
    """
    cdef EventPropInput* inputs = batch_output._inputs
    cdef SpikesBatch* spikes_ptr = batch_output._spikes
    if inputs == NULL or spikes_ptr == NULL:
        raise RuntimeError("SpikesBatchOutput has been freed")

    # Build a plain SpikesBatch view for the C API (shared with wrapper)
    cdef AccumulatedBackwardResult result = eventprop_backward_spikes_batch_accumulate(
        inputs, spikes_ptr[0])  # copy struct by value onto C stack

    # Build n_spikes_per_sample from inputs (needed for spike error sizing)
    cdef int batch_size = spikes_ptr.batch_size
    cdef cnp.ndarray[cnp.int32_t, ndim=1] n_spikes_arr = np.empty(batch_size, dtype=np.int32)
    cdef int b
    for b in range(batch_size):
        n_spikes_arr[b] = inputs[b].n_spikes

    return _AccumulatedBackwardResult.from_c_result(result, n_spikes_arr)


def backward_maxima_batch_accumulate(_MaximaBatchOutput batch_output):
    """Backward pass (maxima loss) + gradient accumulation.

    Uses batch_output's internal .inputs and .maxima directly — no Python-side work.
    Returns _AccumulatedBackwardResult with summed gradients and per-sample spike errors.
    """
    cdef EventPropInput* inputs = batch_output._inputs
    cdef MaximaBatch* maxima_ptr = batch_output._maxima
    if inputs == NULL or maxima_ptr == NULL:
        raise RuntimeError("MaximaBatchOutput has been freed")

    # Pass inputs and maxima separately (new API)
    cdef AccumulatedBackwardResult result = eventprop_backward_maxima_batch_accumulate(
        inputs, maxima_ptr[0])  # copy struct by value onto C stack

    # Build n_spikes_per_sample from inputs (needed for spike error sizing)
    cdef int batch_size = maxima_ptr.batch_size
    cdef cnp.ndarray[cnp.int32_t, ndim=1] n_spikes_arr = np.empty(batch_size, dtype=np.int32)
    cdef int b
    for b in range(batch_size):
        n_spikes_arr[b] = inputs[b].n_spikes

    return _AccumulatedBackwardResult.from_c_result(result, n_spikes_arr)


def dropout_spikes_batch(
        double[:] times, int[:] sources, int[:] n_per_sample,
        double p_drop, int seed):
    """Batch dropout — randomly drop spikes from each sample.

    Accepts flat numpy arrays (concatenated across samples).
    Returns _DropoutBatchOutput that owns the C memory.
    """
    cdef int batch_size = n_per_sample.shape[0]
    cdef DropoutBatch result = eventprop_dropout_spikes_batch(
        &times[0], &sources[0], &n_per_sample[0],
        batch_size, p_drop, seed)
    return _DropoutBatchOutput.from_c_result(result)
