/**
 * @file eventprop_cffi.c
 * @brief Pure C bindings for EventProp.C library — exposed to Python via CPython API.
 *        No pybind11, no C++. Just C calling into libeventprop_c_api.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../EventProp.C/eventprop.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: create numpy array from data with npy_intp dimension */
#define ARR_D(type, ndim, dims, data) \
    PyArray_SimpleNewFromData((ndim), (npy_intp[]){dims}, (type), (data))

/* ── _compute_spikes_batch_raw(w, n_pre, n_post, batch_tuples, v_th, tau_mem, tau_syn) ── */

static PyObject* py_compute_spikes_batch(PyObject* self, PyObject* args) {
    PyObject* w_obj, *batch_obj;
    int n_pre, n_post;
    double v_th, tau_mem, tau_syn;

    if (!PyArg_ParseTuple(args, "OiiOddd", &w_obj, &n_pre, &n_post,
                           &batch_obj, &v_th, &tau_mem, &tau_syn)) {
        return NULL;
    }

    const double* w = (const double*)PyArray_DATA((PyArrayObject*)w_obj);

    if (!PyList_Check(batch_obj)) {
        PyErr_SetString(PyExc_TypeError, "batch must be a list of (times, sources) tuples");
        return NULL;
    }

    Py_ssize_t batch_size = PyList_Size(batch_obj);
    PyObject* result_list = PyList_New(0);

    /* Determine max spikes for scratch allocation */
    int max_spikes = 64;
    for (Py_ssize_t bi = 0; bi < batch_size; bi++) {
        PyObject* item = PyList_GetItem(batch_obj, bi);
        if (PyTuple_Check(item) && PyTuple_Size(item) >= 2) {
            PyObject* times_item = PyTuple_GetItem(item, 0);
            if (PyArray_Check(times_item)) {
                Py_ssize_t n_spikes = PyArray_SIZE((PyArrayObject*)times_item);
                if (n_spikes > max_spikes) max_spikes = (int)n_spikes;
            }
        }
    }

    double* fwd_times      = malloc(max_spikes * n_post * sizeof(double));
    int32_t* fwd_sources   = malloc(max_spikes * n_post * sizeof(int32_t));
    double* fwd_currents   = malloc(max_spikes * n_post * sizeof(double));
    double* first_spike_times = malloc(n_post * sizeof(double));
    int32_t* first_spike_idxs   = malloc(n_post * sizeof(int32_t));

    if (!fwd_times || !fwd_sources || !fwd_currents || !first_spike_times || !first_spike_idxs) {
        free(fwd_times); free(fwd_sources); free(fwd_currents);
        free(first_spike_times); free(first_spike_idxs);
        PyErr_NoMemory();
        Py_DECREF(result_list);
        return NULL;
    }

    for (Py_ssize_t bi = 0; bi < batch_size; bi++) {
        PyObject* item = PyList_GetItem(batch_obj, bi);
        if (!PyTuple_Check(item) || PyTuple_Size(item) < 2) {
            PyErr_SetString(PyExc_TypeError, "each batch item must be (times_array, sources_array)");
            goto cleanup;
        }

        PyObject* times_arr = PyTuple_GetItem(item, 0);
        PyObject* sources_arr = PyTuple_GetItem(item, 1);

        const double* times = NULL;
        const int32_t* sources = NULL;
        Py_ssize_t n_spikes = 0;

        if (PyArray_Check(times_arr)) {
            times = (const double*)PyArray_DATA((PyArrayObject*)times_arr);
            n_spikes = PyArray_SIZE((PyArrayObject*)times_arr);
        } else {
            PyErr_SetString(PyExc_TypeError, "times must be a numpy array");
            goto cleanup;
        }

        if (PyArray_Check(sources_arr)) {
            sources = (const int32_t*)PyArray_DATA((PyArrayObject*)sources_arr);
        } else {
            PyErr_SetString(PyExc_TypeError, "sources must be a numpy array");
            goto cleanup;
        }

        int n_post_spikes = eventprop_compute_spikes(
            w, n_pre, n_post,
            times, sources, (int)n_spikes,
            v_th, tau_mem, tau_syn,
            fwd_times, fwd_sources, fwd_currents);

        /* Compute first_spike per neuron from the sorted output */
        eventprop_compute_first_spike(n_post, fwd_times, fwd_sources,
                                       n_post_spikes, first_spike_times, first_spike_idxs);

        /* Build result tuple: (times_array, sources_array, currents_array,
                                first_spike_times_list, first_spike_idxs_list) */
        if (n_post_spikes > 0) {
            PyObject* t_arr = ARR_D(NPY_DOUBLE, 1, n_post_spikes, fwd_times);
            PyObject* s_arr = ARR_D(NPY_INT32, 1, n_post_spikes, fwd_sources);
            PyObject* c_arr = ARR_D(NPY_DOUBLE, 1, n_post_spikes, fwd_currents);
            if (t_arr && s_arr && c_arr) {
                PyObject* fst_list = PyList_New(n_post);
                PyObject* fsi_list = PyList_New(n_post);
                for (int n = 0; n < n_post; n++) {
                    PyList_SetItem(fst_list, n, PyFloat_FromDouble(first_spike_times[n]));
                    PyList_SetItem(fsi_list, n, PyLong_FromLong(first_spike_idxs[n]));
                }
                PyObject* tup = Py_BuildValue("(OOOOO)", t_arr, s_arr, c_arr, fst_list, fsi_list);
                if (tup) PyList_Append(result_list, tup);
                Py_XDECREF(t_arr); Py_XDECREF(s_arr); Py_XDECREF(c_arr);
                Py_XDECREF(fst_list); Py_XDECREF(fsi_list);
            } else { PyErr_NoMemory(); goto cleanup; }
        } else {
            PyObject* t_arr = ARR_D(NPY_DOUBLE, 1, 0, NULL);
            PyObject* s_arr = ARR_D(NPY_INT32, 1, 0, NULL);
            PyObject* c_arr = ARR_D(NPY_DOUBLE, 1, 0, NULL);
            if (t_arr && s_arr && c_arr) {
                PyObject* fst_list = PyList_New(n_post);
                PyObject* fsi_list = PyList_New(n_post);
                for (int n = 0; n < n_post; n++) {
                    PyList_SetItem(fst_list, n, PyFloat_FromDouble(first_spike_times[n]));
                    PyList_SetItem(fsi_list, n, PyLong_FromLong(first_spike_idxs[n]));
                }
                PyObject* tup = Py_BuildValue("(OOOOO)", t_arr, s_arr, c_arr, fst_list, fsi_list);
                if (tup) PyList_Append(result_list, tup);
                Py_XDECREF(t_arr); Py_XDECREF(s_arr); Py_XDECREF(c_arr);
                Py_XDECREF(fst_list); Py_XDECREF(fsi_list);
            } else { PyErr_NoMemory(); goto cleanup; }
        }
    }

cleanup:
    free(fwd_times); free(fwd_sources); free(fwd_currents);
    free(first_spike_times); free(first_spike_idxs);
    return result_list;
}

/* ── _backward_spikes_batch_raw(input_tuples, post_tuples, w, n_pre, n_post, gradient, v_th, tau_mem, tau_syn) ── */

static PyObject* py_backward_spikes_batch(PyObject* self, PyObject* args) {
    PyObject* input_obj, *post_obj, *w_obj, *gradient_obj;
    int n_pre, n_post;
    double v_th, tau_mem, tau_syn;

    if (!PyArg_ParseTuple(args, "OOOiiOddd", &input_obj, &post_obj,
                           &w_obj, &n_pre, &n_post, &gradient_obj,
                           &v_th, &tau_mem, &tau_syn)) {
        return NULL;
    }

    const double* w = (const double*)PyArray_DATA((PyArrayObject*)w_obj);
    double* grad = (double*)PyArray_DATA((PyArrayObject*)gradient_obj);

    Py_ssize_t batch_size = PyList_Size(input_obj);

    for (Py_ssize_t bi = 0; bi < batch_size; bi++) {
        PyObject* input_item = PyList_GetItem(input_obj, bi);
        PyObject* post_item  = PyList_GetItem(post_obj, bi);

        if (!PyTuple_Check(input_item) || !PyTuple_Check(post_item)) continue;

        const double* itimes = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(input_item, 0));
        const int32_t* isources = (const int32_t*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(input_item, 1));
        int n_input_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(input_item, 0));

        const double* ptimes = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(post_item, 0));
        const int32_t* psources = (const int32_t*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(post_item, 1));
        const double* pcurrents = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(post_item, 2));
        int n_post_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(post_item, 0));

        if (n_post_spikes == 0) continue;

        /* Get post_errors from Spikes.errors attribute */
        PyObject* errors_obj = PyObject_GetAttrString(post_item, "errors");
        const double* post_errors = NULL;
        if (errors_obj && PyArray_Check(errors_obj)) {
            post_errors = (const double*)PyArray_DATA((PyArrayObject*)errors_obj);
        } else {
            if (errors_obj) Py_DECREF(errors_obj);
        }

        double* grad_buf = calloc(n_pre * n_post, sizeof(double));
        if (!grad_buf) { PyErr_NoMemory(); return NULL; }

        eventprop_backward_spikes(
            itimes, isources, n_input_spikes,
            ptimes, psources, pcurrents, post_errors, n_post_spikes,
            w, n_pre, n_post, v_th, tau_mem, tau_syn,
            grad_buf, NULL);

        for (int i = 0; i < n_pre * n_post; i++) {
            grad[i] += grad_buf[i];
        }

        free(grad_buf);
    }

    Py_RETURN_NONE;
}

/* ── _compute_maxima_batch_raw(w, n_pre, n_post, batch_tuples, tau_mem, tau_syn) ── */

static PyObject* py_compute_maxima_batch(PyObject* self, PyObject* args) {
    PyObject* w_obj, *batch_obj;
    int n_pre, n_post;
    double tau_mem, tau_syn;

    if (!PyArg_ParseTuple(args, "OiiOOd", &w_obj, &n_pre, &n_post,
                           &batch_obj, &tau_mem, &tau_syn)) {
        return NULL;
    }

    const double* w = (const double*)PyArray_DATA((PyArrayObject*)w_obj);

    if (!PyList_Check(batch_obj)) {
        PyErr_SetString(PyExc_TypeError, "batch must be a list");
        return NULL;
    }

    Py_ssize_t batch_size = PyList_Size(batch_obj);
    PyObject* result_list = PyList_New(0);

    for (Py_ssize_t bi = 0; bi < batch_size; bi++) {
        PyObject* item = PyList_GetItem(batch_obj, bi);
        if (!PyTuple_Check(item) || PyTuple_Size(item) < 2) continue;

        const double* times = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(item, 0));
        const int32_t* sources = (const int32_t*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(item, 1));
        int n_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(item, 0));

        double* maxima_times = calloc(n_post, sizeof(double));
        double* maxima_values = calloc(n_post, sizeof(double));
        if (!maxima_times || !maxima_values) {
            free(maxima_times); free(maxima_values);
            PyErr_NoMemory();
            goto cleanup_maxima;
        }
        for (int i = 0; i < n_post; i++) {
            maxima_times[i] = NAN;
            maxima_values[i] = 0.0;
        }

        eventprop_compute_maxima(
            w, n_pre, n_post,
            times, sources, n_spikes,
            tau_mem, tau_syn,
            maxima_times, maxima_values);

        PyObject* t_arr = ARR_D(NPY_DOUBLE, 1, n_post, maxima_times);
        PyObject* v_arr = ARR_D(NPY_DOUBLE, 1, n_post, maxima_values);
        if (t_arr && v_arr) {
            PyObject* tup = Py_BuildValue("(OO)", t_arr, v_arr);
            if (tup) PyList_Append(result_list, tup);
            Py_XDECREF(t_arr); Py_XDECREF(v_arr);
        }

    cleanup_maxima:
        free(maxima_times); free(maxima_values);
    }

    return result_list;
}

/* ── _backward_maxima_batch_raw(input_tuples, maxima_tuples, w, n_pre, n_post, gradient, tau_mem, tau_syn) ── */

static PyObject* py_backward_maxima_batch(PyObject* self, PyObject* args) {
    PyObject* input_obj, *maxima_obj, *w_obj, *gradient_obj;
    int n_pre, n_post;
    double tau_mem, tau_syn;

    if (!PyArg_ParseTuple(args, "OOOiiOdd", &input_obj, &maxima_obj,
                           &w_obj, &n_pre, &n_post, &gradient_obj,
                           &tau_mem, &tau_syn)) {
        return NULL;
    }

    const double* w = (const double*)PyArray_DATA((PyArrayObject*)w_obj);
    double* grad = (double*)PyArray_DATA((PyArrayObject*)gradient_obj);

    Py_ssize_t batch_size = PyList_Size(input_obj);

    for (Py_ssize_t bi = 0; bi < batch_size; bi++) {
        PyObject* input_item = PyList_GetItem(input_obj, bi);
        PyObject* maxima_item  = PyList_GetItem(maxima_obj, bi);

        if (!PyTuple_Check(input_item) || !PyTuple_Check(maxima_item)) continue;

        const double* itimes = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(input_item, 0));
        const int32_t* isources = (const int32_t*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(input_item, 1));
        int n_input_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(input_item, 0));

        if (n_input_spikes == 0) continue;

        const double* mtimes = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(maxima_item, 0));
        const double* mvalues = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(maxima_item, 1));

        int n_valid = 0;
        for (int i = 0; i < n_post; i++) {
            if (!isnan(mtimes[i])) n_valid++;
        }
        if (n_valid == 0) continue;

        double* mt = malloc(n_valid * sizeof(double));
        double* mv = malloc(n_valid * sizeof(double));
        if (!mt || !mv) {
            free(mt); free(mv);
            PyErr_NoMemory();
            goto cleanup_maxima_bwd;
        }
        int idx = 0;
        for (int i = 0; i < n_post; i++) {
            if (!isnan(mtimes[i])) {
                mt[idx] = mtimes[i];
                mv[idx] = mvalues[i];
                idx++;
            }
        }

        double* grad_buf = calloc(n_pre * n_post, sizeof(double));
        if (!grad_buf) { free(mt); free(mv); PyErr_NoMemory(); return NULL; }

        eventprop_backward_maxima(
            itimes, isources, n_input_spikes,
            mt, mv, NULL, n_valid,
            w, n_pre, n_post, tau_mem, tau_syn,
            grad_buf, NULL);

        for (int i = 0; i < n_pre * n_post; i++) {
            grad[i] += grad_buf[i];
        }

        free(grad_buf);
    cleanup_maxima_bwd:
        free(mt); free(mv);
    }

    Py_RETURN_NONE;
}

/* ── _compute_voltage_trace_raw(t_max, dt, target_nrn_idx, w, n_pre, n_post, spikes, v_th, tau_mem, tau_syn) ── */

static PyObject* py_compute_voltage_trace(PyObject* self, PyObject* args) {
    double t_max, dt;
    int target_nrn_idx;
    PyObject* w_obj, *spikes_obj;
    int n_pre, n_post;
    double v_th, tau_mem, tau_syn;

    if (!PyArg_ParseTuple(args, "ddiOiiOddd", &t_max, &dt, &target_nrn_idx,
                           &w_obj, &n_pre, &n_post, &spikes_obj,
                           &v_th, &tau_mem, &tau_syn)) {
        return NULL;
    }

    const double* w = (const double*)PyArray_DATA((PyArrayObject*)w_obj);

    /* Parse spikes: could be tuple (times, sources) or just times array */
    const double* times = NULL;
    const int32_t* sources = NULL;
    int n_spikes = 0;

    if (PyTuple_Check(spikes_obj) && PyTuple_Size(spikes_obj) >= 2) {
        times = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(spikes_obj, 0));
        sources = (const int32_t*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(spikes_obj, 1));
        n_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(spikes_obj, 0));
    } else if (PyArray_Check(spikes_obj)) {
        times = (const double*)PyArray_DATA((PyArrayObject*)spikes_obj);
        n_spikes = (int)PyArray_SIZE((PyArrayObject*)spikes_obj);
        /* No sources — create dummy */
        int32_t* dummy_sources = malloc(n_spikes * sizeof(int32_t));
        if (dummy_sources) {
            for (int i = 0; i < n_spikes; i++) dummy_sources[i] = 0;
            sources = dummy_sources;
        }
    }

    if (!times) { PyErr_SetString(PyExc_TypeError, "spikes must be (times, sources) or times array"); return NULL; }

    /* We need post-synaptic spikes for voltage trace — call compute_spikes first */
    int max_spikes = n_spikes + 1;
    double* fwd_times = malloc(max_spikes * n_post * sizeof(double));
    int32_t* fwd_sources = malloc(max_spikes * n_post * sizeof(int32_t));
    double* fwd_currents = malloc(max_spikes * n_post * sizeof(double));

    if (!fwd_times || !fwd_sources || !fwd_currents) {
        free(fwd_times); free(fwd_sources); free(fwd_currents);
        PyErr_NoMemory();
        return NULL;
    }

    int n_post_spikes = eventprop_compute_spikes(
        w, n_pre, n_post, times, sources, n_spikes,
        v_th, tau_mem, tau_syn, fwd_times, fwd_sources, fwd_currents);

    int n_t = (int)(t_max / dt);
    double* v_trace = malloc(n_t * sizeof(double));
    double* ts = malloc(n_t * sizeof(double));

    if (!v_trace || !ts) {
        free(v_trace); free(ts);
        free(fwd_times); free(fwd_sources); free(fwd_currents);
        PyErr_NoMemory();
        return NULL;
    }

    eventprop_compute_voltage_trace(t_max, dt, target_nrn_idx,
                                     w, n_pre, n_post,
                                     times, sources, n_spikes,
                                     fwd_times, fwd_sources, n_post_spikes,
                                     v_th, tau_mem, tau_syn,
                                     v_trace, ts);

    PyObject* v_arr = ARR_D(NPY_DOUBLE, 1, n_t, v_trace);
    PyObject* t_arr = ARR_D(NPY_DOUBLE, 1, n_t, ts);
    PyObject* result = Py_BuildValue("(OO)", v_arr, t_arr);

    Py_XDECREF(v_arr); Py_XDECREF(t_arr);
    free(v_trace); free(ts);
    free(fwd_times); free(fwd_sources); free(fwd_currents);

    return result;
}

/* ── _compute_lambda_i_raw(t, target_nrn_idx, post_spikes, v_th, tau_mem, tau_syn) ── */

static PyObject* py_compute_lambda_i(PyObject* self, PyObject* args) {
    double t;
    int target_nrn_idx;
    PyObject* post_spikes_obj;
    double v_th, tau_mem, tau_syn;

    if (!PyArg_ParseTuple(args, "diOddd", &t, &target_nrn_idx,
                           &post_spikes_obj, &v_th, &tau_mem, &tau_syn)) {
        return NULL;
    }

    const double* fwd_times = NULL;
    int n_post_spikes = 0;

    if (PyTuple_Check(post_spikes_obj) && PyTuple_Size(post_spikes_obj) >= 1) {
        fwd_times = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(post_spikes_obj, 0));
        n_post_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(post_spikes_obj, 0));
    } else if (PyArray_Check(post_spikes_obj)) {
        fwd_times = (const double*)PyArray_DATA((PyArrayObject*)post_spikes_obj);
        n_post_spikes = (int)PyArray_SIZE((PyArrayObject*)post_spikes_obj);
    }

    if (!fwd_times) { PyErr_SetString(PyExc_TypeError, "post_spikes must have times array"); return NULL; }

    double result = eventprop_compute_lambda_i(t, target_nrn_idx, fwd_times, n_post_spikes,
                                                v_th, tau_mem, tau_syn);

    return PyFloat_FromDouble(result);
}

/* ── _compute_lambda_i_trace_raw(t_max, dt, target_nrn_idx, post_spikes, v_th, tau_mem, tau_syn) ── */

static PyObject* py_compute_lambda_i_trace(PyObject* self, PyObject* args) {
    double t_max, dt;
    int target_nrn_idx;
    PyObject* post_spikes_obj;
    double v_th, tau_mem, tau_syn;

    if (!PyArg_ParseTuple(args, "ddiOddd", &t_max, &dt, &target_nrn_idx,
                           &post_spikes_obj, &v_th, &tau_mem, &tau_syn)) {
        return NULL;
    }

    const double* fwd_times = NULL;
    int n_post_spikes = 0;

    if (PyTuple_Check(post_spikes_obj) && PyTuple_Size(post_spikes_obj) >= 1) {
        fwd_times = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(post_spikes_obj, 0));
        n_post_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(post_spikes_obj, 0));
    } else if (PyArray_Check(post_spikes_obj)) {
        fwd_times = (const double*)PyArray_DATA((PyArrayObject*)post_spikes_obj);
        n_post_spikes = (int)PyArray_SIZE((PyArrayObject*)post_spikes_obj);
    }

    if (!fwd_times) { PyErr_SetString(PyExc_TypeError, "post_spikes must have times array"); return NULL; }

    int n_t = (int)(t_max / dt);
    double* lambda_trace = malloc(n_t * sizeof(double));
    double* ts = malloc(n_t * sizeof(double));

    if (!lambda_trace || !ts) {
        free(lambda_trace); free(ts);
        PyErr_NoMemory();
        return NULL;
    }

    eventprop_compute_lambda_i_trace(t_max, dt, target_nrn_idx,
                                      fwd_times, n_post_spikes,
                                      v_th, tau_mem, tau_syn,
                                      lambda_trace, ts);

    PyObject* l_arr = ARR_D(NPY_DOUBLE, 1, n_t, lambda_trace);
    PyObject* t_arr = ARR_D(NPY_DOUBLE, 1, n_t, ts);
    PyObject* result = Py_BuildValue("(OO)", l_arr, t_arr);

    Py_XDECREF(l_arr); Py_XDECREF(t_arr);
    free(lambda_trace); free(ts);

    return result;
}

/* ── _dropout_spikes_raw(spikes, p_drop, random_seed) ── */

static PyObject* py_dropout_spikes(PyObject* self, PyObject* args) {
    PyObject* spikes_obj;
    double p_drop;
    unsigned int random_seed;

    if (!PyArg_ParseTuple(args, "Odi", &spikes_obj, &p_drop, &random_seed)) {
        return NULL;
    }

    if (!PyList_Check(spikes_obj)) {
        PyErr_SetString(PyExc_TypeError, "spikes must be a list of (times, sources) tuples");
        return NULL;
    }

    Py_ssize_t batch_size = PyList_Size(spikes_obj);
    PyObject* result_list = PyList_New(0);

    /* Simple LCG random number generator (no libc dependency) */
    unsigned int state = random_seed;

    for (Py_ssize_t bi = 0; bi < batch_size; bi++) {
        PyObject* item = PyList_GetItem(spikes_obj, bi);
        if (!PyTuple_Check(item) || PyTuple_Size(item) < 2) continue;

        const double* times = (const double*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(item, 0));
        const int32_t* sources = (const int32_t*)PyArray_DATA((PyArrayObject*)PyTuple_GetItem(item, 1));
        int n_spikes = (int)PyArray_SIZE((PyArrayObject*)PyTuple_GetItem(item, 0));

        /* Count kept spikes */
        int n_kept = 0;
        for (int i = 0; i < n_spikes; i++) {
            state = state * 1103515245 + 12345;
            if (((state >> 16) & 0x7fff) % 100 / 100.0 >= p_drop) n_kept++;
        }

        if (n_kept == 0) {
            PyObject* t_arr = ARR_D(NPY_DOUBLE, 0, 0, NULL);
            PyObject* s_arr = ARR_D(NPY_INT32, 0, 0, NULL);
            PyObject* tup = Py_BuildValue("(OO)", t_arr, s_arr);
            if (tup) PyList_Append(result_list, tup);
            Py_XDECREF(t_arr); Py_XDECREF(s_arr);
        } else {
            /* Create numpy arrays with proper allocation */
            npy_intp t_dim = n_kept;
            PyObject* t_arr = PyArray_SimpleNew(1, &t_dim, NPY_DOUBLE);
            npy_intp s_dim = n_kept;
            PyObject* s_arr = PyArray_SimpleNew(1, &s_dim, NPY_INT32);
            if (!t_arr || !s_arr) {
                Py_XDECREF(t_arr); Py_XDECREF(s_arr);
                PyErr_NoMemory();
                goto cleanup_dropout;
            }

            double* kept_times = (double*)PyArray_DATA((PyArrayObject*)t_arr);
            int32_t* kept_sources = (int32_t*)PyArray_DATA((PyArrayObject*)s_arr);

            int idx = 0;
            for (int i = 0; i < n_spikes; i++) {
                state = state * 1103515245 + 12345;
                if (((state >> 16) & 0x7fff) % 100 / 100.0 >= p_drop) {
                    kept_times[idx] = times[i];
                    kept_sources[idx] = sources[i];
                    idx++;
                }
            }

            if (t_arr && s_arr) {
                PyObject* tup = Py_BuildValue("(OO)", t_arr, s_arr);
                if (tup) PyList_Append(result_list, tup);
                Py_XDECREF(t_arr); Py_XDECREF(s_arr);
            }
        }
    }

cleanup_dropout:
    return result_list;
}

/* ── Module method table ─────────────────────────────────────────────── */

static PyMethodDef EventPropCffiMethods[] = {
    {"_compute_spikes_batch_raw",  py_compute_spikes_batch, METH_VARARGS, NULL},
    {"_backward_spikes_batch_raw", py_backward_spikes_batch, METH_VARARGS, NULL},
    {"_compute_maxima_batch_raw",  py_compute_maxima_batch, METH_VARARGS, NULL},
    {"_backward_maxima_batch_raw", py_backward_maxima_batch, METH_VARARGS, NULL},
    {"_compute_voltage_trace_raw", py_compute_voltage_trace, METH_VARARGS, NULL},
    {"_compute_lambda_i_raw",      py_compute_lambda_i, METH_VARARGS, NULL},
    {"_compute_lambda_i_trace_raw", py_compute_lambda_i_trace, METH_VARARGS, NULL},
    {"_dropout_spikes_raw",        py_dropout_spikes, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef eventprop_cffi_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "EventProp CFFI native bindings — calls EventProp.C library",
    -1,
    EventPropCffiMethods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();  /* Initialize numpy C API */
    return PyModule_Create(&eventprop_cffi_module);
}
