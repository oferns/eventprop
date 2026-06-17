# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython bindings for EventProp C library."""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free, calloc

cdef object _load_lib(object lib_path):
    import ctypes, os
    from ctypes import cdll
    if lib_path is not None:
        return cdll.LoadLibrary(lib_path)
    for suffix in ('.so', '.dylib'):
        path = os.path.join(os.path.dirname(__file__), f"libeventprop_c_api{suffix}")
        try:
            return cdll.LoadLibrary(path)
        except OSError:
            continue
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
    return np.asarray(arr)

cdef object _int_view(int* ptr, ssize_t length):
    if ptr == NULL or length == 0:
        return np.array([], dtype=np.int32)
    cdef object arr = cnp.PyArray_SimpleNewFromData(1, [length], cnp.NPY_INT32, ptr)
    return np.asarray(arr)

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
        if self._errors == NULL:
            return np.array([], dtype=np.float64)
        cdef SpikeOutput* s = <SpikeOutput*>self._ptr + idx
        cdef double* err_ptr = self._errors + self._error_offsets[idx]
        return _double_view(err_ptr, s.n_spikes)

    # Total post-spikes across batch (for error buffer sizing)
    cdef int total_post_spikes(self):
        cdef int total = 0, i
        cdef SpikeOutput* base = <SpikeOutput*>self._ptr
        for i in range(self.batch_size):
            total += base[i].n_spikes
        return total

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

    # Return numpy view into native error buffer for sample idx (Python-accessible)
    def get_error_view(self, int idx):
        if self._errors == NULL or self.n_post <= 0:
            return np.array([], dtype=np.float64)
        cdef double* err_ptr = self._errors + idx * self.n_post
        return _double_view(err_ptr, self.n_post)

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

cdef void _build_ptrs(list tl, list sl, int bs, const double** pt, const int** ps, int* pn):
    cdef int b
    cdef const double[:] tv
    cdef const int[:] sv
    for b in range(bs):
        tv = tl[b]
        sv = sl[b]
        pt[b] = &tv[0] if tv.shape[0] > 0 else NULL
        ps[b] = &sv[0] if sv.shape[0] > 0 else NULL
        pn[b] = tv.shape[0]

cdef ssize_t _total_spikes(list tl):
    cdef ssize_t t = 0
    for item in tl:
        t += len(item)
    return t

cdef BatchSpikeOutput _compute_spikes_batch_ptrs(list tl, list sl, double[:,:] w, int np_, int npost,
                                                  double tm, double ts, double vt):
    cdef int bs = len(tl)
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    _build_ptrs(tl, sl, bs, pt, ps, pn)
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    cdef BatchSpikeOutput o = BatchSpikeOutput.__new__(BatchSpikeOutput)
    o._ptr = <ssize_t>eventprop_compute_spikes_batch_ptrs(&ip)
    o.batch_size = bs
    free(pt); free(ps); free(pn)

    # Allocate native error buffer + offsets — zero-copy for backward pass
    cdef int tot_spikes = o.total_post_spikes()
    cdef int i, offset
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

    return o

cdef BatchMaximaOutput _compute_maxima_batch_ptrs(list tl, list sl, double[:,:] w, int np_, int npost,
                                                   double tm, double ts, double vt):
    cdef int bs = len(tl)
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    _build_ptrs(tl, sl, bs, pt, ps, pn)
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

    # Allocate native error buffer — batch_size * n_post (maxima errors are per-neuron)
    if npost > 0 and bs > 0:
        o._errors = <double*>calloc(bs * npost, sizeof(double))
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
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    _build_ptrs(tl, sl, bs, pt, ps, pn)
    cdef ssize_t tis = _total_spikes(tl)
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    # Read error pointers directly from native buffer — zero concatenation
    cdef const double** pe2 = <const double**>malloc(bs * sizeof(const double*))
    cdef int* pne = <int*>malloc(bs * sizeof(int))
    cdef int b
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
    cdef const double** pt = <const double**>malloc(bs * sizeof(const double*))
    cdef const int** ps = <const int**>malloc(bs * sizeof(const int*))
    cdef int* pn = <int*>malloc(bs * sizeof(int))
    _build_ptrs(tl, sl, bs, pt, ps, pn)
    cdef ssize_t tis = _total_spikes(tl)
    cdef BatchEventPropInputPtrs ip
    ip.times = pt; ip.sources = ps; ip.n_spikes = pn
    ip.w = &w[0,0] if w.shape[0]>0 and w.shape[1]>0 else NULL
    ip.n_pre = np_; ip.n_post = npost; ip.batch_size = bs
    ip.tau_mem = tm; ip.tau_syn = ts; ip.v_th = vt
    # Read error pointers directly from native buffer — zero concatenation
    cdef const double** me2 = <const double**>malloc(bs * sizeof(const double*))
    cdef int b
    for b in range(bs):
        me2[b] = fb._errors + b * npost if fb._errors != NULL else NULL
    cdef BackwardOutput* r = eventprop_backward_maxima_batch_accumulate_ptrs(
        &ip, fb.get_raw_ptr(), me2, npost)
    free(pt); free(ps); free(pn); free(me2)
    return AccumulatedGradient.from_result(r, np_, npost, tis)

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

def compute_spikes_batch_ptrs(list times_list, list sources_list,
                              double[:,:] w, int n_pre, int n_post,
                              double tau_mem, double tau_syn, double v_th):
    """Batch forward pass — per-sample arrays, iteration in C (no Python concatenation)."""
    return _compute_spikes_batch_ptrs(times_list, sources_list, w, n_pre, n_post,
                                      tau_mem, tau_syn, v_th)

def compute_maxima_batch_ptrs(list times_list, list sources_list,
                              double[:,:] w, int n_pre, int n_post,
                              double tau_mem, double tau_syn, double v_th):
    """Batch forward pass (maxima) — per-sample arrays, iteration in C."""
    return _compute_maxima_batch_ptrs(times_list, sources_list, w, n_pre, n_post,
                                      tau_mem, tau_syn, v_th)

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

