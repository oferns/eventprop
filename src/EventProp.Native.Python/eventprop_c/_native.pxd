# cython: language_level=3
"""Cython declarations for EventProp C library.

Mirrors eventprop.h and eventprop_batch.h — compiled, zero runtime overhead.
"""

from libc.stdlib cimport malloc, free

cdef extern from "eventprop.h":
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
        int n_spikes
        double* times
        int* sources
        double* currents

    ctypedef struct SumsOutput:
        int n_rows
        int n_post
        double* sum0
        double* sum1

    ctypedef struct MaximaOutput:
        int n_post
        double* times
        double* values

    ctypedef struct BackwardSpikesInput:
        const EventPropInput* input
        const double* fwd_times
        const int* fwd_sources
        const double* fwd_currents
        const double* post_errors
        int n_post_spikes

    ctypedef struct BackwardMaximaInput:
        const EventPropInput* input
        const double* maxima_times
        const double* maxima_values
        const double* maxima_errors
        int n_maxima

    ctypedef struct BackwardOutput:
        int n_pre
        int n_post
        double* grad
        double* spike_errors

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

    # Single-sample functions (now return pointers)
    SpikeOutput* eventprop_compute_spikes(const EventPropInput* input)
    SumsOutput*  eventprop_compute_sums(const EventPropInput* input)
    MaximaOutput* eventprop_compute_maxima(const EventPropInput* input)
    BackwardOutput* eventprop_backward_spikes(const BackwardSpikesInput* input)
    BackwardOutput* eventprop_backward_maxima(const BackwardMaximaInput* input)
    VoltageTraceOutput* eventprop_compute_voltage_trace(const VoltageTraceInput* input)
    LambdaITraceOutput* eventprop_compute_lambda_i_trace(const LambdaITraceInput* input)

    void eventprop_free_spike_output(SpikeOutput* output)
    void eventprop_free_sums_output(SumsOutput* output)
    void eventprop_free_maxima_output(MaximaOutput* output)
    void eventprop_free_backward_output(BackwardOutput* output)
    void eventprop_free_voltage_trace_output(VoltageTraceOutput* output)
    void eventprop_free_lambda_i_trace_output(LambdaITraceOutput* output)


cdef extern from "eventprop_batch.h":
    ctypedef struct BatchEventPropInput:
        const EventPropInput* input
        const int* n_spikes_per_sample
        int batch_size

    ctypedef struct BatchEventPropInputPtrs:
        const double** times
        const int** sources
        const int* n_spikes
        const double* w
        int n_pre
        int n_post
        int batch_size
        double tau_mem
        double tau_syn
        double v_th

    ctypedef struct BatchEventPropInputDirect:
        const EventPropInput* inputs
        int batch_size

    ctypedef struct DropoutOutput:
        int n_spikes
        double* times
        int* sources

    SpikeOutput* eventprop_compute_spikes_batch(const BatchEventPropInput* input)
    BackwardOutput* eventprop_backward_spikes_batch(
        const BatchEventPropInput* input, SpikeOutput* fwd_batch, const double* post_errors)
    BackwardOutput* eventprop_backward_spikes_batch_accumulate(
        const BatchEventPropInput* input, SpikeOutput* fwd_batch, const double* post_errors)
    MaximaOutput* eventprop_compute_maxima_batch(const BatchEventPropInput* input)
    BackwardOutput* eventprop_backward_maxima_batch(
        const BatchEventPropInput* input, MaximaOutput* fwd_batch, const double* maxima_errors)
    BackwardOutput* eventprop_backward_maxima_batch_accumulate(
        const BatchEventPropInput* input, MaximaOutput* fwd_batch, const double* maxima_errors)

    # Pointer-based batch functions (no flattening)
    SpikeOutput* eventprop_compute_spikes_batch_ptrs(const BatchEventPropInputPtrs* input)
    BackwardOutput* eventprop_backward_spikes_batch_accumulate_ptrs(
        const BatchEventPropInputPtrs* input, SpikeOutput* fwd_batch,
        const double** post_errors, const int* n_post_errors)
    MaximaOutput* eventprop_compute_maxima_batch_ptrs(const BatchEventPropInputPtrs* input)
    BackwardOutput* eventprop_backward_maxima_batch_accumulate_ptrs(
        const BatchEventPropInputPtrs* input, MaximaOutput* fwd_batch,
        const double** maxima_errors, int n_post)
    DropoutOutput* eventprop_dropout_spikes_batch(
        const BatchEventPropInput* input, double p_drop, int seed)

    # Direct batch functions (array of EventPropInput structs)
    SpikeOutput* eventprop_compute_spikes_batch_direct(const BatchEventPropInputDirect* input)
    MaximaOutput* eventprop_compute_maxima_batch_direct(const BatchEventPropInputDirect* input)

    void eventprop_free_batch_spike_output(SpikeOutput* output, int batch_size)
    void eventprop_free_backward_batch_output(BackwardOutput* output, int batch_size)
    void eventprop_free_batch_maxima_output(MaximaOutput* output, int batch_size)
    void eventprop_free_dropout_output(DropoutOutput* output, int batch_size)
