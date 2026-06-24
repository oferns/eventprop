# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython bindings for EventProp C library."""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, calloc

cdef object _load_lib(object lib_path):
    import ctypes, os, sys
    from ctypes import cdll
    if lib_path is not None:
        return cdll.LoadLibrary(lib_path)
    for suffix in ('.dll', '.so', '.dylib'):
        path = os.path.join(os.path.dirname(__file__), f"libeventprop_c_api{suffix}")
        try:
            return cdll.LoadLibrary(path)
        except OSError:
            continue
    # Windows: DLL may be named eventprop_c_api.dll (no lib prefix)
    if sys.platform == 'win32':
        path = os.path.join(os.path.dirname(__file__), "eventprop_c_api.dll")
        try:
            return cdll.LoadLibrary(path)
        except OSError:
            pass
    for dir_ in ('.', '/usr/local/lib', '/opt/homebrew/lib'):
        path = os.path.join(dir_, 'libeventprop_c_api.dylib')
        try:
            return cdll.LoadLibrary(path)
        except OSError:
            continue
    raise FileNotFoundError("Could not load libeventprop_c_api")

_load_lib(None)

cdef object _double_view(double* ptr, ssize_t length):
    if ptr == NULL or length == 0:
        return np.array([], dtype=np.float64)
    cdef object arr = cnp.PyArray_SimpleNewFromData(1, [length], cnp.NPY_FLOAT64, ptr)
    return arr  # Return numpy array directly — zero-copy view

cdef object _int_view(int* ptr, ssize_t length):
    if ptr == NULL or length == 0:
        return np.array([], dtype=np.int32)
    cdef object arr = cnp.PyArray_SimpleNewFromData(1, [length], cnp.NPY_INT32, ptr)
    return arr  # Return numpy array directly — zero-copy view

# ================================================================
# ForwardSpikeOutput — lightweight forward-only output class.
#
# Uses Cython memoryviews for zero-copy access (no numpy array creation).
# No error buffer allocation (forward pass doesn't need errors).
# ~10x less Python overhead than BatchSpikeOutput per call.
# ================================================================

cdef class ForwardSpikeOutput:
    """Lightweight forward-only spike output — memoryviews, no numpy arrays."""
    cdef ssize_t _ptr
    cdef int batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, int idx):
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range")
        cdef SpikeOutput* s = <SpikeOutput*>self._ptr + idx
        # Handle zero-spike case (NULL pointer from C library)
        if s.times == NULL or s.n_spikes <= 0:
            return (np.array([], dtype=np.float64), np.array([], dtype=np.int32))
        # Typed memoryviews from raw pointers — zero-copy, no numpy overhead
        cdef double[:] times_mv = <double[:s.n_spikes]>s.times
        cdef int[:] sources_mv = <int[:s.n_spikes]>(<int*>s.sources)
        return (times_mv, sources_mv)

    def __dealloc__(self):
        if self._ptr != 0:
            eventprop_free_batch_spike_output(<SpikeOutput*>self._ptr, self.batch_size)
            self._ptr = 0

cdef ForwardSpikeOutput _compute_spikes_forward(list tl, list sl,
                                                 double[:,:] w, int np_, int npost,
                                                 double tm, double ts, double vt):
    """Forward-only compute — no error buffers, memoryview output."""
    cdef int bs = len(tl)
    # Allocate pointer arrays for C library
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    # Build pointers directly — zero Python overhead
    cdef int b
    cdef const double[:] tv
    cdef const int[:] sv
    for b in range(bs):
        tv = tl[b]
        sv = sl[b]
        pt[b] = &tv[0] if tv.shape[0] > 0 else NULL
        ps[b] = &sv[0] if sv.shape[0] > 0 else NULL
        pn[b] = tv.shape[0]
    
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    cdef ForwardSpikeOutput o = ForwardSpikeOutput.__new__(ForwardSpikeOutput)
    o._ptr = <ssize_t>eventprop_compute_spikes_batch_ptrs(&ip)
    o.batch_size = bs
    free(pt); free(ps); free(pn)
    return o

cdef class BatchSpikeOutput:
    cdef ssize_t _ptr, batch_size
    cdef double* _errors          # native error buffer (total_post_spikes)
    cdef int* _error_offsets      # per-sample error offsets (batch_size+1)

    def __getitem__(self, int idx):
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range")
        cdef SpikeOutput* s = <SpikeOutput*>self._ptr + idx
        return (_double_view(s.times, s.n_spikes), _int_view(s.sources, s.n_spikes),
                _double_view(s.currents, s.n_spikes))

    def __len__(self):
        return self.batch_size

    cdef SpikeOutput* get_raw_ptr(self):
        return <SpikeOutput*>self._ptr

    # Return numpy view into native error buffer for sample idx (Python-accessible)
    def get_error_view(self, int idx):
        self.ensure_errors()  # Lazy allocation on first access
        if self._errors == NULL:
            return np.array([], dtype=np.float64)
        cdef SpikeOutput* s = <SpikeOutput*>self._ptr + idx
        cdef double* err_ptr = self._errors + self._error_offsets[idx]
        return _double_view(err_ptr, s.n_spikes)

    # Return flat numpy view of entire error buffer (for struct-based batch API)
    def get_flat_errors(self):
        self.ensure_errors()
        if self._errors == NULL:
            return np.array([], dtype=np.float64)
        cdef int tot = self.total_post_spikes()
        return _double_view(self._errors, tot)

    # Total post-spikes across batch (for error buffer sizing)
    cdef int total_post_spikes(self):
        cdef int total = 0, i
        cdef SpikeOutput* base = <SpikeOutput*>self._ptr
        for i in range(self.batch_size):
            total += base[i].n_spikes
        return total

    # Lazy error buffer allocation — called by backward() if needed.
    cdef void ensure_errors(self):
        if self._errors != NULL:
            return  # Already allocated
        cdef int tot_spikes = self.total_post_spikes()
        cdef int i, offset
        if tot_spikes > 0:
            self._errors = <double*>calloc(tot_spikes, sizeof(double))
            self._error_offsets = <int*>malloc((self.batch_size + 1) * sizeof(int))
            offset = 0
            for i in range(self.batch_size):
                self._error_offsets[i] = offset
                offset += (<SpikeOutput*>self._ptr)[i].n_spikes
            self._error_offsets[self.batch_size] = offset


    def __dealloc__(self):
        if self._errors != NULL:
            free(self._errors)
            self._errors = NULL
        if self._error_offsets != NULL:
            free(self._error_offsets)
            self._error_offsets = NULL
        if self._ptr != 0:
            eventprop_free_batch_spike_output(<SpikeOutput*>self._ptr, self.batch_size)

cdef class BatchMaximaOutput:
    cdef ssize_t _ptr, batch_size
    cdef double* _errors          # native error buffer (batch_size * n_post)
    cdef int n_post               # neurons per sample

    def __getitem__(self, int idx):
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range")
        cdef MaximaOutput* m = <MaximaOutput*>self._ptr + idx
        return (_double_view(m.times, m.n_post), _double_view(m.values, m.n_post))

    def __len__(self):
        return self.batch_size

    cdef MaximaOutput* get_raw_ptr(self):
        return <MaximaOutput*>self._ptr

    # Lazy error buffer allocation — called by backward() if needed.
    cdef void ensure_errors(self):
        if self._errors != NULL:
            return  # Already allocated
        cdef int total_errs = self.batch_size * self.n_post
        if total_errs > 0:
            self._errors = <double*>calloc(total_errs, sizeof(double))

    # Return numpy view into native error buffer for sample idx (Python-accessible)
    def get_error_view(self, int idx):
        self.ensure_errors()  # Lazy allocation on first access
        if self._errors == NULL or self.n_post <= 0:
            return np.array([], dtype=np.float64)
        cdef double* err_ptr = self._errors + idx * self.n_post
        return _double_view(err_ptr, self.n_post)

    # Return flat numpy view of entire error buffer (for struct-based batch API)
    def get_flat_errors(self):
        self.ensure_errors()
        if self._errors == NULL or self.n_post <= 0:
            return np.array([], dtype=np.float64)
        cdef int total_errs = self.batch_size * self.n_post
        return _double_view(self._errors, total_errs)

    def __dealloc__(self):
        if self._errors != NULL:
            free(self._errors)
            self._errors = NULL
        if self._ptr != 0:
            eventprop_free_batch_maxima_output(<MaximaOutput*>self._ptr, self.batch_size)

cdef class BatchBackwardOutput:
    cdef ssize_t _ptr, batch_size
    def __getitem__(self, int idx):
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range")
        cdef BackwardOutput* b = <BackwardOutput*>self._ptr + idx
        return (_double_view(b.grad, b.n_pre * b.n_post), _double_view(b.spike_errors, b.n_pre))
    def __len__(self):
        return self.batch_size
    def __dealloc__(self):
        if self._ptr != 0:
            eventprop_free_backward_batch_output(<BackwardOutput*>self._ptr, self.batch_size)

cdef class DropoutBatchOutput:
    cdef ssize_t _ptr, batch_size
    cdef bint _freed
    def __getitem__(self, int idx):
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range")
        cdef DropoutOutput* d = <DropoutOutput*>self._ptr + idx
        return (_double_view(d.times, d.n_spikes), _int_view(d.sources, d.n_spikes))
    def __len__(self):
        return self.batch_size
    cpdef free(self):
        if not self._freed and self._ptr != 0:
            eventprop_free_dropout_output(<DropoutOutput*>self._ptr, self.batch_size)
            self._freed = True
    def __dealloc__(self):
        if not self._freed and self._ptr != 0:
            eventprop_free_dropout_output(<DropoutOutput*>self._ptr, self.batch_size)

# ================================================================
# BatchInput — pre-extracted C pointers, zero validation on hot path.
#
# Construct once from numpy arrays (validates + extracts pointers).
# Reuse across many calls: only 1 Python object to pass instead of 5.
# Keeps references to original arrays alive (prevents GC).
# ================================================================

cdef class BatchInput:
    """Pre-extracted C pointers for batch compute functions.

    Construct once, reuse across many calls. Eliminates per-call
    memory view validation overhead (~800 us at tiny scales).

    Usage:
        inp = BatchInput(times, sources, n_per_sample, w, n_pre, n_post,
                         tau_mem, tau_syn, v_th)
        output = compute_spikes_batch_obj(inp)   # ~9 us instead of ~850 us
    """
    cdef const double* _w
    cdef const double* _times
    cdef const int* _sources
    cdef const int* _n_spikes_per_sample
    cdef int n_pre, n_post, batch_size
    cdef double tau_mem, tau_syn, v_th
    # Keep references alive (prevent GC while pointers are valid)
    cdef object _w_arr, _times_arr, _sources_arr, _n_spikes_arr

    def __cinit__(self,
                  const double[:,:] w,
                  const double[:] times,
                  const int[:] sources,
                  const int[:] n_spikes_per_sample,
                  int n_pre, int n_post,
                  double tau_mem, double tau_syn, double v_th):
        self._w_arr = w
        self._times_arr = times
        self._sources_arr = sources
        self._n_spikes_arr = n_spikes_per_sample
        self._w = &w[0,0] if w.shape[0] > 0 and w.shape[1] > 0 else NULL
        self._times = &times[0] if times.shape[0] > 0 else NULL
        self._sources = &sources[0] if sources.shape[0] > 0 else NULL
        self._n_spikes_per_sample = <const int*>(&n_spikes_per_sample[0])
        self.n_pre = n_pre
        self.n_post = n_post
        self.batch_size = n_spikes_per_sample.shape[0]
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.v_th = v_th

    cdef void fill_input(self, EventPropInput* inp) noexcept:
        inp.w = self._w
        inp.n_pre = self.n_pre
        inp.n_post = self.n_post
        inp.times = self._times
        inp.sources = self._sources
        inp.n_spikes = <int>self._times_arr.shape[0] if self._times != NULL else 0
        inp.tau_mem = self.tau_mem
        inp.tau_syn = self.tau_syn
        inp.v_th = self.v_th

    cdef void fill_batch_input(self, BatchEventPropInput* batch_inp) noexcept:
        # Reuse a stack-allocated EventPropInput — caller must provide it
        pass  # Handled inline in each function for performance

cdef void _fill_input(EventPropInput* inp, const double[:,:] w, int n_pre, int n_post,
                      const double[:] times, const int[:] sources,
                      double tau_mem, double tau_syn, double v_th):
    inp.w = &w[0,0] if w.shape[0] > 0 and w.shape[1] > 0 else NULL
    inp.n_pre = n_pre
    inp.n_post = n_post
    inp.times = &times[0] if times.shape[0] > 0 else NULL
    inp.sources = &sources[0] if sources.shape[0] > 0 else NULL
    inp.n_spikes = times.shape[0]
    inp.tau_mem = tau_mem
    inp.tau_syn = tau_syn
    inp.v_th = v_th

cdef ssize_t _total_spikes(list tl):
    cdef ssize_t t = 0
    for item in tl:
        t += len(item)
    return t

cdef BatchSpikeOutput _compute_spikes_batch_ptrs(list tl, list sl, double[:,:] w, int np_, int npost,
                                                  double tm, double ts, double vt,
                                                  bint alloc_errors=True):
    cdef int bs = len(tl)
    # Allocate pointer arrays for C library
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    # Build pointers directly — zero Python overhead
    cdef int b
    cdef const double[:] tv
    cdef const int[:] sv
    for b in range(bs):
        tv = tl[b]
        sv = sl[b]
        pt[b] = &tv[0] if tv.shape[0] > 0 else NULL
        ps[b] = &sv[0] if sv.shape[0] > 0 else NULL
        pn[b] = tv.shape[0]
    
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    cdef BatchSpikeOutput o = BatchSpikeOutput.__new__(BatchSpikeOutput)
    o._ptr = <ssize_t>eventprop_compute_spikes_batch_ptrs(&ip)
    o.batch_size = bs
    free(pt); free(ps); free(pn)

    # Allocate native error buffer + offsets — zero-copy for backward pass.
    # Lazy allocation: skip if alloc_errors=False (forward-only path).
    cdef int tot_spikes, i, offset
    if alloc_errors:
        tot_spikes = o.total_post_spikes()
        if tot_spikes > 0:
            o._errors = <double*>calloc(tot_spikes, sizeof(double))
            o._error_offsets = <int*>malloc((bs + 1) * sizeof(int))
            offset = 0
            for i in range(bs):
                o._error_offsets[i] = offset
                offset += (<SpikeOutput*>o._ptr)[i].n_spikes
            o._error_offsets[bs] = offset  # sentinel
        else:
            o._errors = NULL
            o._error_offsets = NULL
    else:
        o._errors = NULL
        o._error_offsets = NULL

    return o

cdef BatchMaximaOutput _compute_maxima_batch_ptrs(list tl, list sl, double[:,:] w, int np_, int npost,
                                                   double tm, double ts, double vt,
                                                   bint alloc_errors=True):
    cdef int bs = len(tl)
    # Allocate pointer arrays for C library
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    # Build pointers directly — zero Python overhead
    cdef int b
    cdef const double[:] tv
    cdef const int[:] sv
    for b in range(bs):
        tv = tl[b]
        sv = sl[b]
        pt[b] = &tv[0] if tv.shape[0] > 0 else NULL
        ps[b] = &sv[0] if sv.shape[0] > 0 else NULL
        pn[b] = tv.shape[0]
    
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    cdef BatchMaximaOutput o = BatchMaximaOutput.__new__(BatchMaximaOutput)
    o._ptr = <ssize_t>eventprop_compute_maxima_batch_ptrs(&ip)
    o.batch_size = bs
    o.n_post = npost
    free(pt); free(ps); free(pn)

    # Lazy error buffer allocation — skip if alloc_errors=False.
    cdef int total_errs
    if alloc_errors:
        total_errs = bs * npost
        if total_errs > 0:
            o._errors = <double*>calloc(total_errs, sizeof(double))
        else:
            o._errors = NULL
    else:
        o._errors = NULL

    return o

cdef class AccumulatedGradient:
    cdef ssize_t _ptr
    cdef object _grad
    cdef object _spike_errors
    @property
    def grad(self):
        return self._grad
    @property
    def spike_errors(self):
        return self._spike_errors
    def __cinit__(self):
        self._ptr = 0; self._grad = None; self._spike_errors = None
    @staticmethod
    cdef AccumulatedGradient from_result(BackwardOutput* r, int np_, int npost, ssize_t ts_=0):
        cdef AccumulatedGradient o = AccumulatedGradient.__new__(AccumulatedGradient)
        o._ptr = <ssize_t>r
        o._grad = _double_view(r.grad, np_ * npost)
        if r.spike_errors != NULL and ts_ > 0:
            o._spike_errors = _double_view(r.spike_errors, ts_)
        return o
    def __dealloc__(self):
        if self._ptr != 0:
            eventprop_free_backward_output(<BackwardOutput*>self._ptr)
            self._ptr = 0

cdef AccumulatedGradient _backward_spikes_batch_accumulate_ptrs(
        list tl, list sl, BatchSpikeOutput fb,
        double[:,:] w, int np_, int npost, double tm, double ts, double vt):
    cdef int bs = len(tl)
    # Allocate pointer arrays for C library
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    # Build pointers directly — zero Python overhead
    cdef int b
    cdef const double[:] tv
    cdef const int[:] sv
    for b in range(bs):
        tv = tl[b]
        sv = sl[b]
        pt[b] = &tv[0] if tv.shape[0] > 0 else NULL
        ps[b] = &sv[0] if sv.shape[0] > 0 else NULL
        pn[b] = tv.shape[0]
    cdef ssize_t tis = _total_spikes(tl)
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    # Lazy error buffer allocation — allocate on first backward() call
    fb.ensure_errors()
    # Read error pointers directly from native buffer — zero concatenation
    cdef const double** pe2 = <const double**>malloc(bs * sizeof(const double*))
    cdef int* pne = <int*>malloc(bs * sizeof(int))
    cdef SpikeOutput* fwd_base = fb.get_raw_ptr()
    cdef double* err_base = fb._errors
    for b in range(bs):
        pe2[b] = (<const double*>(err_base + fb._error_offsets[b])) if err_base != NULL else NULL
        pne[b] = fwd_base[b].n_spikes
    cdef BackwardOutput* r = eventprop_backward_spikes_batch_accumulate_ptrs(
        &ip, fb.get_raw_ptr(), pe2, pne)
    free(pt); free(ps); free(pn); free(pe2); free(pne)
    return AccumulatedGradient.from_result(r, np_, npost, tis)

cdef AccumulatedGradient _backward_maxima_batch_accumulate_ptrs(
        list tl, list sl, BatchMaximaOutput fb,
        double[:,:] w, int np_, int npost, double tm, double ts, double vt):
    cdef int bs = len(tl)
    # Allocate pointer arrays for C library
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    # Build pointers directly — zero Python overhead
    cdef int b
    cdef const double[:] tv
    cdef const int[:] sv
    for b in range(bs):
        tv = tl[b]
        sv = sl[b]
        pt[b] = &tv[0] if tv.shape[0] > 0 else NULL
        ps[b] = &sv[0] if sv.shape[0] > 0 else NULL
        pn[b] = tv.shape[0]
    cdef ssize_t tis = _total_spikes(tl)
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    # Lazy error buffer allocation — allocate on first backward() call
    fb.ensure_errors()
    # Read error pointers directly from native buffer — zero concatenation
    cdef const double** me2 = <const double**>malloc(bs * sizeof(const double*))
    for b in range(bs):
        me2[b] = fb._errors + b * npost if fb._errors != NULL else NULL
    cdef BackwardOutput* r = eventprop_backward_maxima_batch_accumulate_ptrs(
        &ip, fb.get_raw_ptr(), me2, npost)
    free(pt); free(ps); free(pn); free(me2)
    return AccumulatedGradient.from_result(r, np_, npost, tis)

# ================================================================
# Public API — Spikes list direct binding (zero Python overhead)
# Takes list of Spikes objects directly, extracts pointers in Cython
# ================================================================

def compute_spikes_batch_from_spikes_list(list spikes_list,
                                          double[:,:] w, int n_pre, int n_post,
                                          double tau_mem, double tau_syn, double v_th):
    """Batch forward pass — takes list of Spikes objects directly.
    
    Extracts numpy array pointers from each Spikes object in Cython,
    without creating Python lists or C structs. Most efficient binding.
    """
    cdef int bs = len(spikes_list)
    # Allocate pointer arrays for C library
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    
    # Extract pointers directly from Spikes objects — zero Python overhead
    cdef int b
    cdef object spike
    cdef double[:] tv
    cdef int[:] sv
    for b in range(bs):
        spike = spikes_list[b]
        tv = spike._times
        sv = spike._sources
        pt[b] = &tv[0] if tv.shape[0] > 0 else NULL
        ps[b] = &sv[0] if sv.shape[0] > 0 else NULL
        pn[b] = tv.shape[0]
    
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = n_pre; ip.n_post = n_post; ip.batch_size = bs
    ip.tau_mem = tau_mem; ip.tau_syn = tau_syn; ip.v_th = v_th
    
    cdef BatchSpikeOutput output = BatchSpikeOutput.__new__(BatchSpikeOutput)
    output._ptr = <ssize_t>eventprop_compute_spikes_batch_ptrs(&ip)
    output.batch_size = bs
    free(pt); free(ps); free(pn)
    
    # Allocate native error buffer + offsets
    cdef int tot_spikes, i, offset
    cdef SpikeOutput* s_ptr
    tot_spikes = 0
    for b in range(bs):
        tot_spikes += pn[b]
    
    if tot_spikes > 0:
        output._errors = <double*>calloc(tot_spikes, sizeof(double))
        output._error_offsets = <int*>malloc((bs + 1) * sizeof(int))
        offset = 0
        for b in range(bs):
            output._error_offsets[b] = offset
            s_ptr = <SpikeOutput*>output._ptr + b
            offset += s_ptr.n_spikes
        output._error_offsets[bs] = offset
    else:
        output._errors = NULL
        output._error_offsets = NULL
    
    return output

# ================================================================
# Public API — Direct spikes list binding (zero Python iteration overhead)
# Takes list of Spikes objects, extracts pointers via C-level slots
# ================================================================

def compute_spikes_batch_from_spikes_list_direct(list spikes_list,
                                                 double[:,:] w, int n_pre, int n_post,
                                                 double tau_mem, double tau_syn, double v_th):
    """Batch forward pass — takes list of Spikes objects directly.
    
    Extracts numpy array pointers from each Spikes object via C-level slots,
    without Python iteration overhead. Most efficient binding option.
    """
    cdef int bs = len(spikes_list)
    
    # Allocate array of EventPropInput structs
    cdef EventPropInput* inputs = <EventPropInput*>malloc(bs * sizeof(EventPropInput))
    
    # Fill EventPropInput structs directly from Spikes objects
    cdef int b
    cdef object spike_obj
    cdef object times_obj, sources_obj
    cdef double[:] tv
    cdef int[:] sv
    for b in range(bs):
        spike_obj = spikes_list[b]
        times_obj = spike_obj._times
        sources_obj = spike_obj._sources
        
        # Convert to memoryviews for pointer extraction
        tv = times_obj if isinstance(times_obj, np.ndarray) else np.asarray(times_obj, dtype=np.float64)
        sv = sources_obj if isinstance(sources_obj, np.ndarray) else np.asarray(sources_obj, dtype=np.int32)
        
        inputs[b].w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
        inputs[b].n_pre = n_pre
        inputs[b].n_post = n_post
        inputs[b].times = &tv[0] if tv.shape[0] > 0 else NULL
        inputs[b].sources = &sv[0] if sv.shape[0] > 0 else NULL
        inputs[b].n_spikes = tv.shape[0]
        inputs[b].tau_mem = tau_mem
        inputs[b].tau_syn = tau_syn
        inputs[b].v_th = v_th
    
    # Build direct input struct
    cdef BatchEventPropInputDirect ip
    ip.inputs = inputs
    ip.batch_size = bs
    
    cdef BatchSpikeOutput output = BatchSpikeOutput.__new__(BatchSpikeOutput)
    output._ptr = <ssize_t>eventprop_compute_spikes_batch_direct(&ip)
    output.batch_size = bs
    free(inputs)
    
    # Allocate native error buffer + offsets
    cdef int tot_spikes, i, offset
    cdef SpikeOutput* s_ptr
    tot_spikes = 0
    for b in range(bs):
        # Get n_spikes from the C output
        s_ptr = <SpikeOutput*>output._ptr + b
        tot_spikes += s_ptr.n_spikes
    
    if tot_spikes > 0:
        output._errors = <double*>calloc(tot_spikes, sizeof(double))
        output._error_offsets = <int*>malloc((bs + 1) * sizeof(int))
        offset = 0
        for b in range(bs):
            output._error_offsets[b] = offset
            s_ptr = <SpikeOutput*>output._ptr + b
            offset += s_ptr.n_spikes
        output._error_offsets[bs] = offset
    else:
        output._errors = NULL
        output._error_offsets = NULL
    
    return output

# ================================================================
# Public API — SpikesVector direct binding (zero Python overhead)
# Takes SpikesVector directly, uses pre-built flat buffers
# ================================================================

def compute_spikes_batch_from_vector(object vector,
                                     double[:,:] w, int n_pre, int n_post,
                                     double tau_mem, double tau_syn, double v_th):
    """Batch forward pass — takes SpikesVector directly, uses pre-built flat buffers.
    
    This is the most efficient binding option: SpikesVector already has
    _flat_times, _flat_sources, and _n_spikes_per_sample pre-built at construction.
    No Python list creation or array concatenation overhead.
    """
    cdef EventPropInput inp
    # Convert numpy arrays to memoryviews for C access
    cdef double[:] flat_times = vector._flat_times
    cdef int[:] flat_sources = vector._flat_sources
    cdef int[:] n_spikes_per_sample = vector._n_spikes_per_sample
    _fill_input(&inp, w, n_pre, n_post, flat_times, flat_sources,
                tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef BatchSpikeOutput output = BatchSpikeOutput.__new__(BatchSpikeOutput)
    output._ptr = <ssize_t>eventprop_compute_spikes_batch(&batch_inp)
    output.batch_size = batch_inp.batch_size
    return output

# ================================================================
# Public API — pointer-based batch functions (hot path, zero-copy)
# ================================================================

def compute_spikes_batch(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                         double[:,:] w, int n_pre, int n_post,
                         double tau_mem, double tau_syn, double v_th):
    """Batch forward pass — flat arrays (legacy API)."""
    cdef EventPropInput inp
    _fill_input(&inp, w, n_pre, n_post, times, sources, tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef BatchSpikeOutput output = BatchSpikeOutput.__new__(BatchSpikeOutput)
    output._ptr = <ssize_t>eventprop_compute_spikes_batch(&batch_inp)
    output.batch_size = batch_inp.batch_size
    return output

def compute_spikes_batch_obj(BatchInput inp):
    """Batch forward pass — pre-extracted pointers (zero validation overhead).

    Construct BatchInput once, call this many times. ~100x faster at tiny scales.
    """
    cdef EventPropInput ei
    ei.w = inp._w
    ei.n_pre = inp.n_pre
    ei.n_post = inp.n_post
    ei.times = inp._times
    ei.sources = inp._sources
    ei.n_spikes = <int>inp._times_arr.shape[0] if inp._times != NULL else 0
    ei.tau_mem = inp.tau_mem
    ei.tau_syn = inp.tau_syn
    ei.v_th = inp.v_th
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &ei
    batch_inp.n_spikes_per_sample = <const int*>inp._n_spikes_per_sample
    batch_inp.batch_size = inp.batch_size
    cdef BatchSpikeOutput output = BatchSpikeOutput.__new__(BatchSpikeOutput)
    output._ptr = <ssize_t>eventprop_compute_spikes_batch(&batch_inp)
    output.batch_size = inp.batch_size
    return output

def compute_maxima_batch_from_vector(object vector,
                                     double[:,:] w, int n_pre, int n_post,
                                     double tau_mem, double tau_syn, double v_th):
    """Batch forward pass (maxima) — takes SpikesVector directly, uses pre-built flat buffers."""
    cdef EventPropInput inp
    # Convert numpy arrays to memoryviews for C access
    cdef double[:] flat_times = vector._flat_times
    cdef int[:] flat_sources = vector._flat_sources
    cdef int[:] n_spikes_per_sample = vector._n_spikes_per_sample
    _fill_input(&inp, w, n_pre, n_post, flat_times, flat_sources,
                tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef BatchMaximaOutput output = BatchMaximaOutput.__new__(BatchMaximaOutput)
    output._ptr = <ssize_t>eventprop_compute_maxima_batch(&batch_inp)
    output.batch_size = batch_inp.batch_size
    return output

def compute_maxima_batch(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                         double[:,:] w, int n_pre, int n_post,
                         double tau_mem, double tau_syn, double v_th):
    """Batch forward pass (maxima) — flat arrays (legacy API)."""
    cdef EventPropInput inp
    _fill_input(&inp, w, n_pre, n_post, times, sources, tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef BatchMaximaOutput output = BatchMaximaOutput.__new__(BatchMaximaOutput)
    output._ptr = <ssize_t>eventprop_compute_maxima_batch(&batch_inp)
    output.batch_size = batch_inp.batch_size
    return output

def backward_spikes_batch(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                          BatchSpikeOutput fwd_batch, double[:] post_errors,
                          double[:,:] w, int n_pre, int n_post,
                          double tau_mem, double tau_syn, double v_th):
    """Batch backward pass — flat arrays (legacy API)."""
    cdef EventPropInput inp
    _fill_input(&inp, w, n_pre, n_post, times, sources, tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef BatchBackwardOutput output = BatchBackwardOutput.__new__(BatchBackwardOutput)
    output._ptr = <ssize_t>eventprop_backward_spikes_batch(
        &batch_inp, fwd_batch.get_raw_ptr(), &post_errors[0])
    output.batch_size = batch_inp.batch_size
    return output

def backward_spikes_batch_accumulate_from_vector(object vector,
                                                   BatchSpikeOutput fwd_batch, double[:] post_errors,
                                                   double[:,:] w, int n_pre, int n_post,
                                                   double tau_mem, double tau_syn, double v_th):
    """Batch backward pass with gradient accumulation — takes SpikesVector directly."""
    cdef EventPropInput inp
    # Convert numpy arrays to memoryviews for C access
    cdef double[:] flat_times = vector._flat_times
    cdef int[:] flat_sources = vector._flat_sources
    cdef int[:] n_spikes_per_sample = vector._n_spikes_per_sample
    _fill_input(&inp, w, n_pre, n_post, flat_times, flat_sources,
                tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef const double* ep = &post_errors[0] if post_errors.shape[0] > 0 else NULL
    cdef BackwardOutput* r = eventprop_backward_spikes_batch_accumulate(
        &batch_inp, fwd_batch.get_raw_ptr(), ep)
    return AccumulatedGradient.from_result(r, n_pre, n_post, flat_times.shape[0])

def backward_spikes_batch_accumulate(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                                     BatchSpikeOutput fwd_batch, double[:] post_errors,
                                     double[:,:] w, int n_pre, int n_post,
                                     double tau_mem, double tau_syn, double v_th):
    """Batch backward pass with gradient accumulation — flat arrays (legacy API)."""
    cdef EventPropInput inp
    _fill_input(&inp, w, n_pre, n_post, times, sources, tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef const double* ep = &post_errors[0] if post_errors.shape[0] > 0 else NULL
    cdef BackwardOutput* r = eventprop_backward_spikes_batch_accumulate(
        &batch_inp, fwd_batch.get_raw_ptr(), ep)
    return AccumulatedGradient.from_result(r, n_pre, n_post, len(times))

def backward_maxima_batch(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                          BatchMaximaOutput fwd_batch, double[:] maxima_errors,
                          double[:,:] w, int n_pre, int n_post,
                          double tau_mem, double tau_syn, double v_th):
    """Batch backward pass (maxima) — flat arrays (legacy API)."""
    cdef EventPropInput inp
    _fill_input(&inp, w, n_pre, n_post, times, sources, tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef BatchBackwardOutput output = BatchBackwardOutput.__new__(BatchBackwardOutput)
    output._ptr = <ssize_t>eventprop_backward_maxima_batch(
        &batch_inp, fwd_batch.get_raw_ptr(), &maxima_errors[0])
    output.batch_size = batch_inp.batch_size
    return output

def backward_maxima_batch_accumulate_from_vector(object vector,
                                                 BatchMaximaOutput fwd_batch, double[:] maxima_errors,
                                                 double[:,:] w, int n_pre, int n_post,
                                                 double tau_mem, double tau_syn, double v_th):
    """Batch backward pass (maxima) with gradient accumulation — takes SpikesVector directly."""
    cdef EventPropInput inp
    # Convert numpy arrays to memoryviews for C access
    cdef double[:] flat_times = vector._flat_times
    cdef int[:] flat_sources = vector._flat_sources
    cdef int[:] n_spikes_per_sample = vector._n_spikes_per_sample
    _fill_input(&inp, w, n_pre, n_post, flat_times, flat_sources,
                tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef const double* ep = &maxima_errors[0] if maxima_errors.shape[0] > 0 else NULL
    cdef BackwardOutput* r = eventprop_backward_maxima_batch_accumulate(
        &batch_inp, fwd_batch.get_raw_ptr(), ep)
    return AccumulatedGradient.from_result(r, n_pre, n_post, flat_times.shape[0])

def backward_maxima_batch_accumulate(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                                     BatchMaximaOutput fwd_batch, double[:] maxima_errors,
                                     double[:,:] w, int n_pre, int n_post,
                                     double tau_mem, double tau_syn, double v_th):
    """Batch backward pass (maxima) with gradient accumulation — flat arrays (legacy API)."""
    cdef EventPropInput inp
    _fill_input(&inp, w, n_pre, n_post, times, sources, tau_mem, tau_syn, v_th)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef const double* ep = &maxima_errors[0] if maxima_errors.shape[0] > 0 else NULL
    cdef BackwardOutput* r = eventprop_backward_maxima_batch_accumulate(
        &batch_inp, fwd_batch.get_raw_ptr(), ep)
    return AccumulatedGradient.from_result(r, n_pre, n_post, len(times))

def dropout_spikes_batch(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                         double p_drop, int seed):
    cdef EventPropInput inp
    cdef double[:,:] dummy_w = np.array([[0.0]])
    _fill_input(&inp, dummy_w, 0, 0, times, sources, 0.0, 0.0, 0.0)
    cdef BatchEventPropInput batch_inp
    batch_inp.input = &inp
    batch_inp.n_spikes_per_sample = <int*>(&n_spikes_per_sample[0])
    batch_inp.batch_size = n_spikes_per_sample.shape[0]
    cdef DropoutBatchOutput output = DropoutBatchOutput.__new__(DropoutBatchOutput)
    output._ptr = <ssize_t>eventprop_dropout_spikes_batch(&batch_inp, p_drop, seed)
    output.batch_size = batch_inp.batch_size
    output._freed = False
    return output

# ================================================================
# Public API — pointer-based batch functions (new hot path)
# Accept per-sample lists of numpy arrays, iterate in C via _ptrs
# ================================================================

def compute_spikes_forward(list times_list, list sources_list,
                           double[:,:] w, int n_pre, int n_post,
                           double tau_mem, double tau_syn, double v_th):
    """Forward-only batch pass — no error buffers, memoryview output.

    Returns ForwardSpikeOutput (lightweight, ~10x less Python overhead).
    Use this for inference or when backward() is not needed.
    For training (forward + backward), use compute_spikes_batch_ptrs instead.
    """
    return _compute_spikes_forward(times_list, sources_list, w, n_pre, n_post,
                                   tau_mem, tau_syn, v_th)

def compute_spikes_batch_ptrs(list times_list, list sources_list,
                              double[:,:] w, int n_pre, int n_post,
                              double tau_mem, double tau_syn, double v_th,
                              bint alloc_errors=True):
    """Batch forward pass — per-sample arrays, iteration in C (no Python concatenation).

    Returns BatchSpikeOutput with error buffers for backward().
    Set alloc_errors=False to skip error buffer allocation (lazy, allocated on first backward()).
    For inference-only, use compute_spikes_forward instead.
    """
    return _compute_spikes_batch_ptrs(times_list, sources_list, w, n_pre, n_post,
                                      tau_mem, tau_syn, v_th, alloc_errors)

def compute_maxima_batch_ptrs(list times_list, list sources_list,
                              double[:,:] w, int n_pre, int n_post,
                              double tau_mem, double tau_syn, double v_th,
                              bint alloc_errors=True):
    """Batch forward pass (maxima) — per-sample arrays, iteration in C.

    Set alloc_errors=False to skip error buffer allocation (lazy, allocated on first backward()).
    """
    return _compute_maxima_batch_ptrs(times_list, sources_list, w, n_pre, n_post,
                                      tau_mem, tau_syn, v_th, alloc_errors)

def backward_spikes_batch_accumulate_ptrs(list times_list, list sources_list,
                                          BatchSpikeOutput fwd_batch,
                                          double[:,:] w, int n_pre, int n_post,
                                          double tau_mem, double tau_syn, double v_th):
    """Batch backward pass with accumulation — errors read from fwd_batch native buffer."""
    return _backward_spikes_batch_accumulate_ptrs(times_list, sources_list, fwd_batch, w, n_pre, n_post, tau_mem, tau_syn, v_th)

def backward_maxima_batch_accumulate_ptrs(list times_list, list sources_list,
                                          BatchMaximaOutput fwd_batch,
                                          double[:,:] w, int n_pre, int n_post,
                                          double tau_mem, double tau_syn, double v_th):
    """Batch backward pass (maxima) with accumulation — errors read from fwd_batch native buffer."""
    return _backward_maxima_batch_accumulate_ptrs(times_list, sources_list, fwd_batch, w, n_pre, n_post, tau_mem, tau_syn, v_th)

