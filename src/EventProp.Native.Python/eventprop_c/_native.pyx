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

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free, malloc
from libc.stddef cimport size_t

cnp.import_array()


# The shared library is found by the platform's own mechanism, and they differ:
#   macOS / Linux  an rpath is embedded in the extension at build time (setup.py);
#                  on Linux that is RUNPATH, and without it a plain `import` cannot find
#                  the bare soname unless something exports LD_LIBRARY_PATH
#   Linux only     eventprop_c/_library_path.py also preloads libeventprop_c_api.so with
#                  RTLD_GLOBAL, which is what makes EVENTPROP_C_LIB_DIR work and survives a
#                  moved checkout, where an absolute rpath cannot
#   Windows        os.add_dll_directory, registered by that same module (PATH is excluded
#                  from CPython's extension dependency search since 3.8)
# No manual ctypes.LoadLibrary is needed here beyond that Linux preload.


# ---------------------------------------------------------------------------
# 2. C declarations — eventprop.h (single-sample API)
# ---------------------------------------------------------------------------

cdef extern from "eventprop.h":

    # ABI self-check. The library reports the version and struct sizes it was
    # COMPILED with; this extension compares them against its own compiled layouts
    # at import (section 3c).
    int eventprop_abi_version()
    size_t eventprop_abi_struct_size(const char* struct_name)

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
        int error_no
        int n_spikes
        double* times
        int* sources

    ctypedef struct SumsOutput:
        int error_no
        int n_rows
        int n_post
        double* sum0
        double* sum1

    ctypedef struct MaximaOutput:
        int error_no
        int n_post
        double* times
        double* values
        double* errors

    ctypedef struct BackwardOutput:
        int error_no
        int n_pre
        int n_post
        int n_spikes
        double* grad
        double* spike_errors

    # Free functions (take struct by value — no pointer needed)
    void eventprop_free_spike_output(SpikeOutput output)
    void eventprop_free_sums_output(SumsOutput output)
    void eventprop_free_maxima_output(MaximaOutput output)
    void eventprop_free_backward_output(BackwardOutput output)

    # Core functions (return by value, heap-allocated internals)
    SpikeOutput     eventprop_compute_spikes(const EventPropInput input)
    SumsOutput      eventprop_compute_sums(const EventPropInput input)
    MaximaOutput    eventprop_compute_maxima(const EventPropInput input)
    BackwardOutput  eventprop_backward_spikes(const EventPropInput input, const SpikeOutput spike,
                                              const double* post_spike_errors)
    # backward_maxima takes direct maxima_times + maxima_errors arrays (length n_post)
    BackwardOutput  eventprop_backward_maxima(
        const EventPropInput input,
        const double* maxima_times,
        const double* maxima_errors)

    # Input validation helper
    int eventprop_validate_input(const EventPropInput input)

    # Human-readable error message for an error code
    const char* eventprop_error_string(int error_no)


# ---------------------------------------------------------------------------
# 3. C declarations — eventprop_batch.h (batch API)
# ---------------------------------------------------------------------------

cdef extern from "eventprop_batch.h":

    ctypedef struct SpikesBatch:
        int error_no
        int batch_size
        SpikeOutput* spikes

    void eventprop_free_spikes_batch(SpikesBatch batch)

    # Batch backward output — per-sample results
    ctypedef struct BackwardsBatch:
        int error_no
        int batch_size
        BackwardOutput* outputs

    void eventprop_free_backwards_batch(BackwardsBatch batch)

    ctypedef struct MaximaBatch:
        int error_no
        int batch_size
        MaximaOutput* outputs

    void eventprop_free_maxima_batch(MaximaBatch batch)

    # Dropout output — per-sample surviving spikes
    ctypedef struct DropoutOutput:
        int n_spikes
        double* times
        int* sources

    void eventprop_free_dropout_output(DropoutOutput output)

    ctypedef struct DropoutBatch:
        int error_no
        int batch_size
        int has_errors
        DropoutOutput* outputs

    void eventprop_free_dropout_batch(DropoutBatch batch)

    DropoutBatch eventprop_dropout_spikes_batch(
        const double* times,
        const int* sources,
        const int* n_per_sample,
        int batch_size,
        double p_drop,
        int seed)

    ctypedef struct GradientOutput:
        int error_no
        double* data
        int rows
        int cols

    void eventprop_free_gradient_output(GradientOutput output)

    # Combined backward + accumulate result (primary FFI entry point)
    ctypedef struct AccumulatedBackwardResult:
        int error_no                  # Batch error code
        GradientOutput* summed_grads  # heap-allocated gradient with dimensions
        BackwardOutput* outputs       # per-sample (.grad freed, .spike_errors kept)
        int batch_size

    void eventprop_free_accumulated_backward_result(AccumulatedBackwardResult result)

    # Batch functions (return by value, heap-allocated internals)
    SpikesBatch  eventprop_compute_spikes_batch(const EventPropInput* inputs, int batch_size)

    BackwardsBatch eventprop_backward_spikes_batch(
        const EventPropInput* inputs, const SpikesBatch spikes_batch,
        const double* const* post_spike_errors)

    MaximaBatch eventprop_compute_maxima_batch(
        const EventPropInput* inputs, int batch_size)

    # backward_maxima_batch uses MaximaOutput.errors written by caller
    BackwardsBatch eventprop_backward_maxima_batch(
        const EventPropInput* inputs, const MaximaBatch maxima_batch)

    # Gradient accumulation helper (skips samples with errors)
    GradientOutput eventprop_accumulate_gradients(BackwardsBatch batch, int n_pre, int n_post)

    # Combined backward + accumulate result (primary FFI entry point)
    AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(
        const EventPropInput* inputs, const SpikesBatch spikes_batch,
        const double* const* post_spike_errors)

    # backward_maxima_batch_accumulate takes inputs separately
    AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
        const EventPropInput* inputs, const MaximaBatch maxima_batch)


# ---------------------------------------------------------------------------
# 3b. C declarations — eventprop_diag.h (diagnostic helpers)
# ---------------------------------------------------------------------------

cdef extern from "eventprop_diag.h":

    ctypedef struct VoltageTraceInput:
        EventPropInput input
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
        int error_no

    void eventprop_free_voltage_trace_output(VoltageTraceOutput output)

    VoltageTraceOutput eventprop_compute_voltage_trace(const VoltageTraceInput input)

    ctypedef struct LambdaITraceInput:
        EventPropInput input
        int target_nrn_idx
        const double* fwd_times
        const int* fwd_sources
        const double* post_errors
        int n_post_spikes
        double t_max
        double dt

    ctypedef struct LambdaITraceOutput:
        int n_t
        double* lambda_trace
        double* ts
        int error_no

    void eventprop_free_lambda_i_trace_output(LambdaITraceOutput output)

    LambdaITraceOutput eventprop_compute_lambda_i_trace(const LambdaITraceInput input)


# ---------------------------------------------------------------------------
# 3c. ABI self-check — refuse to import on a header/binary layout mismatch
# ---------------------------------------------------------------------------

cdef extern from *:
    """
    /* The ABI version as compiled INTO this extension — i.e. read out of the copy of
       eventprop.h this extension was built from. Defined here rather than mirrored as
       a hand-maintained Cython constant so there is no second value that can drift
       from the header. */
    #include "eventprop.h"
    static int eventprop_compiled_abi_version(void) {
        return EVENTPROP_ABI_VERSION;
    }
    """
    int eventprop_compiled_abi_version()


def abi_struct_sizes():
    """sizeof of every public struct, as compiled INTO this extension.

    The extension embeds these layouts (the cdef extern structs above), so these are
    the offsets its own code reads. abi_check() compares them against the library's.
    """
    return {
        "EventPropInput": <Py_ssize_t>sizeof(EventPropInput),
        "SpikeOutput": <Py_ssize_t>sizeof(SpikeOutput),
        "SumsOutput": <Py_ssize_t>sizeof(SumsOutput),
        "MaximaOutput": <Py_ssize_t>sizeof(MaximaOutput),
        "BackwardOutput": <Py_ssize_t>sizeof(BackwardOutput),
        "SpikesBatch": <Py_ssize_t>sizeof(SpikesBatch),
        "MaximaBatch": <Py_ssize_t>sizeof(MaximaBatch),
        "BackwardsBatch": <Py_ssize_t>sizeof(BackwardsBatch),
        "GradientOutput": <Py_ssize_t>sizeof(GradientOutput),
        "DropoutOutput": <Py_ssize_t>sizeof(DropoutOutput),
        "DropoutBatch": <Py_ssize_t>sizeof(DropoutBatch),
        "AccumulatedBackwardResult": <Py_ssize_t>sizeof(AccumulatedBackwardResult),
        "VoltageTraceInput": <Py_ssize_t>sizeof(VoltageTraceInput),
        "VoltageTraceOutput": <Py_ssize_t>sizeof(VoltageTraceOutput),
        "LambdaITraceInput": <Py_ssize_t>sizeof(LambdaITraceInput),
        "LambdaITraceOutput": <Py_ssize_t>sizeof(LambdaITraceOutput),
    }


def abi_check():
    """(ok, details) — this extension's compiled layouts vs the loaded library's.

    Returns, never raises: the caller decides what to do. Exposed for tests and
    diagnostics; the import-time check below is what protects users of the package.
    """
    cdef int compiled_v = eventprop_compiled_abi_version()
    cdef int library_v = eventprop_abi_version()
    cdef size_t library_size
    mismatches = []
    if compiled_v != library_v:
        mismatches.append(f"abi_version: extension compiled={compiled_v}, library={library_v}")
    own_sizes = abi_struct_sizes()
    for name, own_size in own_sizes.items():
        library_size = eventprop_abi_struct_size(name.encode("utf-8"))
        if <Py_ssize_t>library_size != own_size:
            mismatches.append(f"{name}: extension compiled sizeof={own_size}, "
                              f"library sizeof={library_size}")
    return (not mismatches), {
        "compiled_abi_version": compiled_v,
        "library_abi_version": library_v,
        "sizes": own_sizes,
        "mismatches": mismatches,
    }


# Runs at import. A layout mismatch is not a soft condition: the extension reads struct
# fields at the offsets it compiled, so a mismatch means every number it returns comes
# from the wrong offsets. Measured, not theoretical — a _native.cp311-win_amd64.pyd
# built against July headers, loaded against a library rebuilt from October headers,
# produced an access violation inside spikes.py __getitem__ via loss_layer._find_first_spikes,
# 40 minutes into a benchmark run, in a frame with no hint that a header/binary offset
# mismatch was the cause. Nothing in the build checked that the extension was newer than
# the headers it was compiled from. Refusing to import names both values at the moment
# the mismatch exists.
_abi_ok, _abi_details = abi_check()
if not _abi_ok:
    raise ImportError(
        "eventprop_c._native: ABI mismatch between this extension and the EventProp C "
        "library it loaded. The extension compiled struct layouts that the loaded "
        "library does not have, so every value it returns would be read from the wrong "
        "offsets. Rebuild the extension against the current headers: "
        "`pip install -e src/EventProp.Native.Python --no-build-isolation`. "
        "Mismatches: " + "; ".join(_abi_details["mismatches"])
    )
del _abi_ok, _abi_details


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
        """Return (times, sources) as zero-copy numpy views for sample idx.

        The synaptic current is not part of the spike output any more — the C
        backward pass rebuilds it from the input spikes, so there is no third
        array to hand back.
        """
        cdef SpikeOutput* out = &self._spikes.spikes[idx]
        if out.n_spikes <= 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int32)
        return (
            _double_view(out.times, out.n_spikes),
            _int_view(out.sources, out.n_spikes),
        )

    def new_errors(self, int idx):
        """Return a fresh Python-owned error array for sample idx.

        The C forward pass no longer carries a derivative buffer, so the array
        is allocated here and handed back to backward_spikes_batch_accumulate.
        """
        cdef SpikeOutput* out = &self._spikes.spikes[idx]
        return np.zeros(out.n_spikes, dtype=np.float64)

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
            eventprop_free_spikes_batch(self._spikes[0])
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
            _double_view(out.times, out.n_post),
            _double_view(out.values, out.n_post),
        )

    def get_times(self, int idx):
        """Zero-copy view into sample idx's maxima times."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.times, out.n_post)

    def get_values(self, int idx):
        """Zero-copy view into sample idx's maxima values."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.values, out.n_post)

    def get_maxima_errors(self, int idx):
        """Return zero-copy writable view into sample idx's maxima error buffer.

        Python writes target errors here before calling backward_maxima().
        C reads them directly — no copying needed.
        """
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.errors, out.n_post)

    def get_error_no(self, int idx):
        """Return the error code for sample idx (EVENTPROP_OK == 0 on success)."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return out.error_no

    def __dealloc__(self):
        if self._owned and self._maxima != NULL:
            eventprop_free_maxima_batch(self._maxima[0])
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
        if self._ptr == NULL or self._ptr.summed_grads == NULL:
            raise RuntimeError("AccumulatedBackwardResult has been freed")
        return _double_view(self._ptr.summed_grads[0].data,
                            self._ptr.summed_grads[0].rows * self._ptr.summed_grads[0].cols)

    @property
    def rows(self):
        if self._ptr == NULL or self._ptr.summed_grads == NULL:
            return 0
        return self._ptr.summed_grads[0].rows

    @property
    def cols(self):
        if self._ptr == NULL or self._ptr.summed_grads == NULL:
            return 0
        return self._ptr.summed_grads[0].cols

    @property
    def batch_size(self):
        if self._ptr == NULL:
            return 0
        return self._ptr.batch_size

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
                            self._ptr.summed_grads[0].rows * self._ptr.summed_grads[0].cols)

    def __dealloc__(self):
        if self._owned and self._ptr != NULL:
            eventprop_free_accumulated_backward_result(self._ptr[0])
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

    def __dealloc__(self):
        if self._owned and self._ptr != NULL:
            eventprop_free_dropout_batch(self._ptr[0])
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
    cdef const double* t_ptr = &times[0]
    cdef const int* s_ptr = &sources[0]
    cdef int b, n
    for b in range(batch_size):
        inputs[b].w = w_ptr
        inputs[b].n_pre = n_pre
        inputs[b].n_post = n_post
        inputs[b].tau_mem = tau_mem
        inputs[b].tau_syn = tau_syn
        inputs[b].v_th = v_th

        # Running offset: each sample's slice starts where the previous ended.
        # (The old code re-walked all prior samples per b — O(batch^2).)
        inputs[b].times = t_ptr
        inputs[b].sources = s_ptr
        n = n_per_sample[b]
        inputs[b].n_spikes = n
        t_ptr += n
        s_ptr += n

    return inputs


# ---------------------------------------------------------------------------
# 6. Error checking helpers
# ---------------------------------------------------------------------------

cdef void check_spikes_batch_error(SpikesBatch result):
    """Check SpikesBatch for errors, raise RuntimeError with message."""
    cdef int i
    for i in range(result.batch_size):
        if result.spikes[i].error_no != 0:
            raise RuntimeError(f"EventProp error {result.spikes[i].error_no}: "
                               f"{eventprop_error_string(result.spikes[i].error_no).decode('utf-8')}")


cdef void check_maxima_batch_error(MaximaBatch result):
    """Check MaximaBatch for errors, raise RuntimeError with message."""
    cdef int i
    for i in range(result.batch_size):
        if result.outputs[i].error_no != 0:
            raise RuntimeError(f"EventProp error {result.outputs[i].error_no}: "
                               f"{eventprop_error_string(result.outputs[i].error_no).decode('utf-8')}")


cdef void check_backward_result_error(AccumulatedBackwardResult result):
    """Check AccumulatedBackwardResult for errors, raise RuntimeError with message."""
    cdef int i
    for i in range(result.batch_size):
        if result.outputs[i].error_no != 0:
            raise RuntimeError(f"EventProp error {result.outputs[i].error_no}: "
                               f"{eventprop_error_string(result.outputs[i].error_no).decode('utf-8')}")


cdef void check_dropout_batch_error(DropoutBatch result):
    """Check DropoutBatch for errors, raise RuntimeError with message."""
    if result.error_no != 0:
        raise RuntimeError(f"EventProp error {result.error_no}: "
                           f"{eventprop_error_string(result.error_no).decode('utf-8')}")


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
    check_spikes_batch_error(result)

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
    check_spikes_batch_error(spikes_result)

    # Compute maxima from the same inputs (maxima need only the sample count)
    cdef MaximaBatch maxima_result = eventprop_compute_maxima_batch(inputs, batch_size)
    check_maxima_batch_error(maxima_result)

    # Free intermediate spikes batch (maxima doesn't need it after computation)
    eventprop_free_spikes_batch(spikes_result)

    # Wrap: transfer input ownership to wrapper, copy maxima onto heap
    cdef _MaximaBatchOutput obj = _MaximaBatchOutput.from_c_result(inputs, maxima_result)
    # Keep flat arrays alive so input pointers stay valid
    obj._times_ref = times
    obj._sources_ref = sources
    return obj


def backward_spikes_batch_accumulate(_SpikesBatchOutput batch_output, object post_spike_errors):
    """Backward pass (spike loss) + gradient accumulation.

    Uses batch_output's internal .inputs and .spikes directly — no Python-side
    work. post_spike_errors is one array per sample: the derivatives the C
    forward pass no longer carries, so the caller owns them here.
    Returns _AccumulatedBackwardResult with summed gradients and per-sample spike errors.
    """
    cdef EventPropInput* inputs = batch_output._inputs
    cdef SpikesBatch* spikes_ptr = batch_output._spikes
    if inputs == NULL or spikes_ptr == NULL:
        raise RuntimeError("SpikesBatchOutput has been freed")

    cdef int batch_size = spikes_ptr.batch_size
    if post_spike_errors is None or len(post_spike_errors) != batch_size:
        raise ValueError("post_spike_errors must hold one array per sample")

    # One contiguous double* per sample, straight into the C batch call. The
    # numpy arrays stay alive in `keep` until the call returns.
    cdef size_t nbytes = <size_t>batch_size * sizeof(double *)
    cdef double** ptrs = <double**>malloc(nbytes)
    cdef list keep = []
    cdef double[:] view
    cdef int b
    if ptrs == NULL:
        raise MemoryError("Failed to allocate post-spike error pointers")
    for b in range(batch_size):
        arr = np.ascontiguousarray(post_spike_errors[b], dtype=np.float64)
        if arr.shape[0] != spikes_ptr.spikes[b].n_spikes:
            free(<void*>ptrs)
            raise ValueError("post_spike_errors[%d] must have one entry per spike" % b)
        keep.append(arr)
        if arr.shape[0] > 0:
            view = arr
            ptrs[b] = &view[0]
        else:
            ptrs[b] = NULL

    cdef AccumulatedBackwardResult result = eventprop_backward_spikes_batch_accumulate(
        inputs, spikes_ptr[0], <const double* const*>ptrs)  # copy struct by value onto C stack
    free(<void*>ptrs)
    check_backward_result_error(result)

    # Build n_spikes_per_sample from inputs (needed for spike error sizing)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] n_spikes_arr = np.empty(batch_size, dtype=np.int32)
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
    check_backward_result_error(result)

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
    check_dropout_batch_error(result)
    return _DropoutBatchOutput.from_c_result(result)
