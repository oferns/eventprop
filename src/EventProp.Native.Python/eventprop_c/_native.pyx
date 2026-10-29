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

    ctypedef struct LambdaIAtTimeOutput:
        double lambda_i
        int error_no

    LambdaIAtTimeOutput eventprop_compute_lambda_i_at_time(const LambdaITraceInput input, double t)



# ---------------------------------------------------------------------------
# 3b2. eventprop_loss.h — continuous readout and regression losses
# ---------------------------------------------------------------------------

cdef extern from "eventprop_loss.h":
    # --- eventprop_loss.h: continuous readout and regression losses -----------------

    ctypedef struct MaximaRegressionInput:
        const double* maxima_values
        const double* decoder
        double bias
        double target
        double huber_delta
        int n_post

    ctypedef struct MaximaRegressionOutput:
        int error_no
        double prediction
        double loss
        double residual_derivative
        double* errors
        double* decoder_grad
        double bias_grad
        int n_post
        int n_skipped

    ctypedef struct MaximaRegressionResult:
        int error_no
        double prediction
        double loss
        double residual_derivative
        int n_skipped

    void eventprop_free_maxima_regression_output(MaximaRegressionOutput output)

    MaximaRegressionOutput eventprop_maxima_regression(MaximaRegressionInput input)

    MaximaRegressionResult eventprop_maxima_regression_into(
        MaximaOutput maxima, const double* decoder, double bias, double target,
        double huber_delta)

    int eventprop_maxima_regression_batch(
        MaximaBatch maxima_batch, const double* decoder, double bias, const double* targets,
        double huber_delta, double* predictions, double* losses, int* n_skipped,
        double* decoder_grad_accum, double* bias_grad_accum)


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
        "LambdaIAtTimeOutput": <Py_ssize_t>sizeof(LambdaIAtTimeOutput),
        "MaximaRegressionInput": <Py_ssize_t>sizeof(MaximaRegressionInput),
        "MaximaRegressionOutput": <Py_ssize_t>sizeof(MaximaRegressionOutput),
        "MaximaRegressionResult": <Py_ssize_t>sizeof(MaximaRegressionResult),
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

cdef inline object _retain(cnp.ndarray arr, object owner):
    """Make `owner` the array's base object, so the array keeps the C memory's owner alive.

    `PyArray_SimpleNewFromData` returns an array whose base is NULL: the buffer is tied to
    nothing. A view like that can outlive the wrapper that owns the malloc'd memory it points
    at, and the next read is a read of freed heap — which does not crash, it returns plausible
    garbage (measured before this existed: -3.6e+178 and a denormal where zeros belonged, from
    `np.array(backward_maxima_batch_accumulate(b).grad)`, a one-liner whose only crime is
    letting the result object go out of scope).

    Setting the base is how numpy says "this memory belongs to that object" — it is what
    `np.frombuffer` does with the buffer it wraps — and it costs one pointer store: no copy,
    no ndarray subclass, the returned object stays a plain `numpy.ndarray`.

    `owner` is a REQUIRED argument rather than an optional one. A defaulted parameter is how a
    new view helper gets to forget; making it mandatory means forgetting is a compile error.
    Pass None only where the data is copied out of C memory before that memory is freed.
    """
    if owner is not None:
        cnp.set_array_base(arr, owner)   # INCREFs; numpy's own wrapper around SetBaseObject
    return arr


cdef inline object _double_view(double* ptr, ssize_t size, object owner):
    """Create a zero-copy writable numpy float64 view over C memory, kept alive by `owner`."""
    cdef cnp.npy_intp dims[1]
    dims[0] = <cnp.npy_intp>size
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = \
        cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_FLOAT64, <void*>ptr)
    return _retain(arr, owner)


cdef inline object _int_view(int* ptr, ssize_t size, object owner):
    """Create a zero-copy writable numpy int32 view over C memory, kept alive by `owner`."""
    cdef cnp.npy_intp dims[1]
    dims[0] = <cnp.npy_intp>size
    cdef cnp.ndarray[cnp.int32_t, ndim=1] arr = \
        cnp.PyArray_SimpleNewFromData(1, dims, cnp.NPY_INT32, <void*>ptr)
    return _retain(arr, owner)


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
            _double_view(out.times, out.n_spikes, self),
            _int_view(out.sources, out.n_spikes, self),
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

    def regress(self, double[::1] decoder, double bias, double[::1] targets,
                double huber_delta, decoder_grad=None, bias_grad=None):
        """Write this batch's regression errors into the maxima outputs, in place.

        This is the training-loop path: zero allocation on the C side, because the errors
        land in the `errors` arrays compute_maxima_batch already allocated and the readout
        gradients accumulate into caller-owned buffers. `backward_maxima_batch()` then reads
        the errors written here, which is the whole backward pass for a continuous target.

        `decoder_grad` and `bias_grad` are ACCUMULATED into when supplied (float64 buffers of
        length n_post and 1), matching the library's accumulate-never-zero gradient rule; pass
        the same buffers across steps to sum them, or omit them for one step's gradient.

        `huber_delta`: non-finite (numpy.inf) for least squares, a positive scale for the
        smooth Huber loss. Samples whose neurons never peaked contribute zero gradient and are
        reported per sample in `n_skipped` — they are not failures.
        """
        if self._maxima == NULL:
            raise RuntimeError("MaximaBatchOutput has been freed")
        cdef int n = self._maxima.batch_size
        cdef int n_post = self.n_post
        if decoder.shape[0] != n_post:
            raise ValueError(f"decoder has {decoder.shape[0]} entries, the layer has {n_post}")
        if targets.shape[0] != n:
            raise ValueError(f"targets has {targets.shape[0]} entries, the batch has {n}")

        cdef double[::1] grad_a
        if decoder_grad is None:
            grad_a = np.zeros(n_post if n_post > 0 else 1, dtype=np.float64)
        else:
            grad_a = decoder_grad
            if grad_a.shape[0] != n_post:
                raise ValueError(f"decoder_grad has {grad_a.shape[0]} entries, the layer "
                                 f"has {n_post}")
        cdef double[::1] grad_b
        if bias_grad is None:
            grad_b = np.zeros(1, dtype=np.float64)
        else:
            grad_b = bias_grad
            if grad_b.shape[0] != 1:
                raise ValueError("bias_grad must be a length-1 float64 buffer")

        cdef double[::1] preds = np.empty(n, dtype=np.float64)
        cdef double[::1] lossv = np.empty(n, dtype=np.float64)
        cdef int[::1] skipped = np.empty(n, dtype=np.int32)
        cdef int err = 0

        if n > 0:
            err = eventprop_maxima_regression_batch(
                self._maxima[0], &decoder[0], bias, &targets[0], huber_delta,
                &preds[0], &lossv[0], &skipped[0], &grad_a[0], &grad_b[0])
            _check_diag_error(err, "maxima regression batch")

        return {
            "predictions": np.asarray(preds),
            "losses": np.asarray(lossv),
            "n_skipped": np.asarray(skipped),
            "decoder_grad": np.asarray(grad_a),
            "bias_grad": np.asarray(grad_b),
        }

    def regress_sample(self, int idx, double[::1] decoder, double bias, double target,
                       double huber_delta):
        """Write ONE sample's regression errors into its own maxima buffer, in place.

        The per-sample counterpart of `regress()`. It calls
        `eventprop_maxima_regression_into`, which writes into the `errors` array
        compute_maxima_batch already allocated for that sample and allocates nothing — not
        even the three per-call numpy arrays the batch path hands back, which is the point if
        you are calling it once per sample in a loop.

        It does NOT compute the readout gradients: that C entry point has no parameters to
        put them in. Use `regress()` when the decoder is being trained (it sums dL/da across
        the batch in one call), and this when the readout is fixed, or when one sample needs
        its own target or loss scale.

        Only this sample's buffer is written. Every neuron of it is overwritten, skipped ones
        with 0, so a reused batch cannot carry a previous step's error forward — but the other
        samples keep whatever they hold, and the forward pass only zeroes them once.

        Returns {"prediction", "loss", "residual_derivative", "n_skipped"}.
        """
        if self._maxima == NULL:
            raise RuntimeError("MaximaBatchOutput has been freed")
        if idx < 0 or idx >= self._maxima.batch_size:
            raise IndexError(
                f"sample {idx} out of range for a batch of {self._maxima.batch_size}")
        if decoder.shape[0] == 0:
            raise ValueError("decoder must not be empty")

        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        if out.n_post != decoder.shape[0]:
            raise ValueError(f"decoder has {decoder.shape[0]} entries, sample {idx} has "
                             f"{out.n_post} maxima")

        cdef MaximaRegressionResult r = eventprop_maxima_regression_into(
            out[0], &decoder[0], bias, target, huber_delta)
        _check_diag_error(r.error_no, f"maxima regression for sample {idx}")

        return {
            "prediction": r.prediction,
            "loss": r.loss,
            "residual_derivative": r.residual_derivative,
            "n_skipped": r.n_skipped,
        }

    def __getitem__(self, int idx):
        """Return (times, values) as zero-copy numpy views for sample idx.

        Used by spikes.py MaximaVector._batch_output.__getitem__.
        """
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return (
            _double_view(out.times, out.n_post, self),
            _double_view(out.values, out.n_post, self),
        )

    def get_times(self, int idx):
        """Zero-copy view into sample idx's maxima times."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.times, out.n_post, self)

    def get_values(self, int idx):
        """Zero-copy view into sample idx's maxima values."""
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.values, out.n_post, self)

    def get_maxima_errors(self, int idx):
        """Return zero-copy writable view into sample idx's maxima error buffer.

        Python writes target errors here before calling backward_maxima().
        C reads them directly — no copying needed.
        """
        cdef MaximaOutput* out = &self._maxima.outputs[idx]
        return _double_view(out.errors, out.n_post, self)

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
                            self._ptr.summed_grads[0].rows * self._ptr.summed_grads[0].cols,
                            self)

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
        return _double_view(out.spike_errors, self._n_spikes_arr[idx], self)

    def get_grad(self, int idx):
        """Get per-sample gradient for sample idx (for drill-down visualization)."""
        cdef BackwardOutput* out = &self._ptr.outputs[idx]
        if out.grad == NULL:
            return None
        return _double_view(out.grad,
                            self._ptr.summed_grads[0].rows * self._ptr.summed_grads[0].cols,
                            self)

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
            _double_view(out.times, out.n_spikes, self),
            _int_view(out.sources, out.n_spikes, self),
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


# ---------------------------------------------------------------------------
# 8. Diagnostic entry points — voltage trace, adjoint trace, pointwise adjoint
#
# These COPY instead of handing out zero-copy views. The C buffers are freed before the
# call returns, and holding them open the way the batch outputs do would mean a wrapper
# class per trace — for arrays whose only purpose is to be plotted. The batch paths stay
# zero-copy because there the copies would happen every forward pass, not every click.
# ---------------------------------------------------------------------------

cdef inline EventPropInput _single_input(
        double[::1] times, int[::1] sources,
        double[:, ::1] w, int n_pre, int n_post,
        double tau_mem, double tau_syn, double v_th):
    """EventPropInput pointing into the caller's arrays.

    Valid because the C call retains nothing and the memoryviews keep the caller's arrays
    alive for the duration of it. The [::1] declarations are load-bearing rather than
    decorative: a strided buffer would leave &times[0] the start of an array whose
    elements are not adjacent, and the C cursor would read across the gaps with nothing
    to report.
    """
    cdef EventPropInput inp
    inp.w = &w[0, 0]
    inp.n_pre = n_pre
    inp.n_post = n_post
    inp.n_spikes = times.shape[0]
    inp.tau_mem = tau_mem
    inp.tau_syn = tau_syn
    inp.v_th = v_th
    if times.shape[0] > 0:
        inp.times = &times[0]
        inp.sources = &sources[0]
    else:
        inp.times = NULL
        inp.sources = NULL
    return inp


cdef inline void _check_diag_error(int error_no, str what):
    if error_no != 0:
        raise RuntimeError(f"EventProp {what} failed: error {error_no} "
                           f"({eventprop_error_string(error_no).decode('utf-8')})")


def compute_voltage_trace(
        double[::1] times, int[::1] sources,
        double[:, ::1] w, int n_pre, int n_post,
        double tau_mem, double tau_syn, double v_th,
        double[::1] fwd_times, int[::1] fwd_sources,
        double t_max, double dt, int target_nrn_idx):
    """Membrane-potential trace for one neuron of one sample.

    fwd_times / fwd_sources are that sample's OUTPUT spikes. Pass the layer's own forward
    result rather than recomputing it — a trace built from a second forward pass describes
    a network state the layer does not have.

    Returns (ts, v_trace), both float64 of length int(t_max / dt).
    """
    cdef VoltageTraceInput vin
    vin.input = _single_input(times, sources, w, n_pre, n_post, tau_mem, tau_syn, v_th)
    vin.target_nrn_idx = target_nrn_idx
    vin.n_post_spikes = fwd_times.shape[0]
    vin.t_max = t_max
    vin.dt = dt
    if fwd_times.shape[0] > 0:
        vin.fwd_times = &fwd_times[0]
        vin.fwd_sources = &fwd_sources[0]
    else:
        vin.fwd_times = NULL
        vin.fwd_sources = NULL

    cdef VoltageTraceOutput out = eventprop_compute_voltage_trace(vin)
    _check_diag_error(out.error_no, "voltage trace")
    try:
        # None as owner on purpose: copied out immediately, and the C buffer is freed in
        # this same block, so there is nothing for a view to keep alive.
        ts = _double_view(out.ts, out.n_t, None).copy()
        v_trace = _double_view(out.v_trace, out.n_t, None).copy()
    finally:
        eventprop_free_voltage_trace_output(out)
    return ts, v_trace


def compute_lambda_i_trace(
        double[::1] times, int[::1] sources,
        double[:, ::1] w, int n_pre, int n_post,
        double tau_mem, double tau_syn, double v_th,
        double[::1] fwd_times, int[::1] fwd_sources, double[::1] post_errors,
        double t_max, double dt, int target_nrn_idx):
    """Synaptic adjoint (lambda_i) trace for one neuron of one sample.

    fwd_times, fwd_sources and post_errors must cover EVERY post-synaptic neuron's spikes
    for that sample, not only the traced neuron's: the adjoint decays lambda_v for all
    neurons at every spike, so a pre-filtered list yields a trace of the right shape and
    wrong values. post_errors is the loss-driven error signal at each of those spikes —
    all-zero errors give an all-zero adjoint, which is why the backward pass has to have
    run before this is interesting.

    Returns (ts, lambda_trace), both float64 of length int(t_max / dt).
    """
    cdef LambdaITraceInput lin
    lin.input = _single_input(times, sources, w, n_pre, n_post, tau_mem, tau_syn, v_th)
    lin.target_nrn_idx = target_nrn_idx
    lin.n_post_spikes = fwd_times.shape[0]
    lin.t_max = t_max
    lin.dt = dt
    if fwd_times.shape[0] > 0:
        lin.fwd_times = &fwd_times[0]
        lin.fwd_sources = &fwd_sources[0]
    else:
        lin.fwd_times = NULL
        lin.fwd_sources = NULL
    lin.post_errors = &post_errors[0] if post_errors.shape[0] > 0 else NULL

    cdef LambdaITraceOutput out = eventprop_compute_lambda_i_trace(lin)
    _check_diag_error(out.error_no, "lambda-I trace")
    try:
        ts = _double_view(out.ts, out.n_t, None).copy()
        lambda_trace = _double_view(out.lambda_trace, out.n_t, None).copy()
    finally:
        eventprop_free_lambda_i_trace_output(out)
    return ts, lambda_trace


def compute_lambda_i_at_time(
        double[::1] times, int[::1] sources,
        double[:, ::1] w, int n_pre, int n_post,
        double tau_mem, double tau_syn, double v_th,
        double[::1] fwd_times, int[::1] fwd_sources, double[::1] post_errors,
        double t, int target_nrn_idx):
    """Synaptic adjoint for one neuron at one time — the figure-1 style query.

    The same sweep compute_lambda_i_trace samples on a grid, evaluated at a single t, so
    the two agree by construction. Cheaper than a trace when a handful of values is wanted.
    """
    cdef LambdaITraceInput lin
    lin.input = _single_input(times, sources, w, n_pre, n_post, tau_mem, tau_syn, v_th)
    lin.target_nrn_idx = target_nrn_idx
    lin.n_post_spikes = fwd_times.shape[0]
    lin.t_max = 0.0   # unused by the pointwise query
    lin.dt = 0.0      # unused by the pointwise query
    if fwd_times.shape[0] > 0:
        lin.fwd_times = &fwd_times[0]
        lin.fwd_sources = &fwd_sources[0]
    else:
        lin.fwd_times = NULL
        lin.fwd_sources = NULL
    lin.post_errors = &post_errors[0] if post_errors.shape[0] > 0 else NULL

    cdef LambdaIAtTimeOutput out = eventprop_compute_lambda_i_at_time(lin, t)
    _check_diag_error(out.error_no, "lambda-I query")
    return out.lambda_i


def maxima_regression(double[::1] maxima_values, double[::1] decoder,
                      double bias, double target, double huber_delta):
    """Continuous readout and regression loss for one sample.

        y_hat = bias + Σ decoder[n] · maxima_values[n]

    The returned `errors` is dL/dmaxima, laid out exactly as eventprop_backward_maxima wants
    its maxima_errors argument, so compute_maxima → maxima_regression → backward_maxima is a
    whole regression step with no per-binding loss arithmetic anywhere.

    `huber_delta`: non-finite (numpy.inf) for least squares, a positive scale for the smooth
    Huber loss; NaN is refused. A non-finite `maxima_values` entry means that neuron never
    peaked — it is excluded from y_hat, given zero error, and counted in `n_skipped` instead
    of failing the call.
    """
    cdef Py_ssize_t n = maxima_values.shape[0]
    if decoder.shape[0] != n:
        raise ValueError(f"decoder has {decoder.shape[0]} entries, maxima_values has {n}")

    cdef MaximaRegressionInput inp
    inp.maxima_values = &maxima_values[0] if n > 0 else NULL
    inp.decoder = &decoder[0] if n > 0 else NULL
    inp.bias = bias
    inp.target = target
    inp.huber_delta = huber_delta
    inp.n_post = <int>n

    cdef MaximaRegressionOutput out = eventprop_maxima_regression(inp)
    _check_diag_error(out.error_no, "maxima regression")
    # Copied before freeing: these are one-off gradients rather than a batch a caller wants
    # to hold a view into, and the C arrays are freed on the way out.
    try:
        # Copied out before eventprop_free_maxima_regression_output below, so no owner needed.
        errors = _double_view(out.errors, out.n_post, None).copy()
        decoder_grad = _double_view(out.decoder_grad, out.n_post, None).copy()
    finally:
        eventprop_free_maxima_regression_output(out)

    return {
        "prediction": out.prediction,
        "loss": out.loss,
        "residual_derivative": out.residual_derivative,
        "bias_grad": out.bias_grad,
        "n_skipped": out.n_skipped,
        "errors": errors,
        "decoder_grad": decoder_grad,
    }
