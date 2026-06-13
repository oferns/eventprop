# cython: language_level=3
"""Cython bindings for EventProp C library."""

import numpy as np
cimport numpy as cnp

# ── Library loader ────────────────────────────────────────────────

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


# ── Helpers: zero-copy numpy arrays from C pointers ───────────────

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


# ── Batch output wrappers (C-owned, freed on GC) ─────────────────

cdef class BatchSpikeOutput:
    cdef ssize_t _ptr, batch_size

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

    def __dealloc__(self):
        if self._ptr != 0:
            eventprop_free_batch_spike_output(<SpikeOutput*>self._ptr, self.batch_size)


cdef class BatchMaximaOutput:
    cdef ssize_t _ptr, batch_size

    def __getitem__(self, int idx):
        if not 0 <= idx < self.batch_size:
            raise IndexError(f"index {idx} out of range")
        cdef MaximaOutput* m = <MaximaOutput*>self._ptr + idx
        return (_double_view(m.times, m.n_post), _double_view(m.values, m.n_post))

    def __len__(self):
        return self.batch_size

    cdef MaximaOutput* get_raw_ptr(self):
        return <MaximaOutput*>self._ptr

    def __dealloc__(self):
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


# ── Internal: build EventPropInput from numpy arrays (cdef, C speed) ─

cdef void _fill_input(EventPropInput* inp, const double[:,:] w,
                      int n_pre, int n_post,
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


# ================================================================
# Public API — batch functions (hot path, zero-copy)
# ================================================================

def compute_spikes_batch(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                         double[:,:] w, int n_pre, int n_post,
                         double tau_mem, double tau_syn, double v_th):
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


def backward_maxima_batch(double[:] times, int[:] sources, int[:] n_spikes_per_sample,
                          BatchMaximaOutput fwd_batch, double[:] maxima_errors,
                          double[:,:] w, int n_pre, int n_post,
                          double tau_mem, double tau_syn, double v_th):
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
