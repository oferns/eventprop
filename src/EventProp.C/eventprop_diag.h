/**
 * @file eventprop_diag.h
 * @brief Diagnostic / debugging helpers for EventProp.
 *
 * Voltage traces, lambda-I traces — useful for visualising membrane dynamics
 * and verifying gradient computations. Not part of the core training loop.
 */

#ifndef EVENTPROP_DIAG_H
#define EVENTPROP_DIAG_H

#include "eventprop.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Input for voltage-trace computation. */
typedef struct VoltageTraceInput {
    EventPropInput input;              /* Forward pass parameters (weights, spikes, v_th, tau_mem, tau_syn) */
    int target_nrn_idx;                /* Post-synaptic neuron to trace */
    const double* fwd_times;           /* Forward-pass spike times (borrowed) */
    const int* fwd_sources;            /* Forward-pass spike sources (borrowed) */
    int n_post_spikes;                 /* Number of forward-pass spikes */
    double t_max;                      /* Simulation end time (seconds) */
    double dt;                         /* Time step (seconds) */
} VoltageTraceInput;

/** Voltage trace output */
typedef struct VoltageTraceOutput {
    int n_t;                           /* Number of time points (= t_max / dt) */
    double* v_trace;                   /* Voltage values (malloc'd, length n_t) */
    double* ts;                        /* Time array (malloc'd, length n_t) */
    EventPropError error_no;           /* Error code (EVENTPROP_OK on success) */
} VoltageTraceOutput;

/** Input for the synaptic adjoint (lambda-I) trace.
 *
 *  Mirrors VoltageTraceInput so one forward-pass spike list serves both diagnostics, and
 *  adds the error signal the adjoint is driven by.
 *
 *  fwd_times / fwd_sources must be the spikes of EVERY post-synaptic neuron, not just the
 *  target's. The adjoint decays lambda_v for all neurons at every spike of any neuron, and
 *  measures spike positions against the last spike in the whole list, so a pre-filtered
 *  single-neuron list yields a trace of the right shape with subtly wrong values. The
 *  target index selects which neuron's jumps are summed, nothing else.
 */
typedef struct LambdaITraceInput {
    EventPropInput input;              /* Forward pass parameters; the synaptic current at
                                        * each post spike is rebuilt from these rather than
                                        * handed over, as in the backward pass */
    int target_nrn_idx;                /* Post-synaptic neuron to trace */
    const double* fwd_times;           /* Forward-pass spike times, all neurons (borrowed) */
    const int* fwd_sources;            /* Forward-pass spike sources, all neurons (borrowed) */
    const double* post_errors;         /* Error signal at each post spike (borrowed) */
    int n_post_spikes;                 /* Number of forward-pass spikes */
    double t_max;                      /* Simulation end time (seconds) */
    double dt;                         /* Time step (seconds) */
} LambdaITraceInput;

/** Lambda-I trace output. */
typedef struct LambdaITraceOutput {
    int n_t;                           /* Number of time points (= t_max / dt) */
    double* lambda_trace;              /* Lambda-I values (malloc'd, length n_t) */
    double* ts;                        /* Time array (malloc'd, length n_t) */
    EventPropError error_no;           /* Error code (EVENTPROP_OK on success) */
} LambdaITraceOutput;

/** Output of the pointwise adjoint query. Holds no heap memory, so it has no free function. */
typedef struct LambdaIAtTimeOutput {
    double lambda_i;                 /* lambda_i at the queried time (0.0 on error) */
    EventPropError error_no;         /* Error code (EVENTPROP_OK on success) */
} LambdaIAtTimeOutput;

void eventprop_free_voltage_trace_output(VoltageTraceOutput output);
void eventprop_free_lambda_i_trace_output(LambdaITraceOutput output);

/**
 * Compute the membrane potential trace for a single post-synaptic neuron.
 * Includes both pre-synaptic drive and post-synaptic spike-reset effects.
 */
VoltageTraceOutput eventprop_compute_voltage_trace(VoltageTraceInput input);

/**
 * Compute the synaptic adjoint (lambda_i) trace for a single neuron.
 *
 * Follows the reference jump condition: at each post-synaptic spike the adjoint jumps by
 * 1/(current - v_th) * (v_th * lambda_v + error), where current is the synaptic current
 * rebuilt at that spike (see eventprop_rebuild_post_currents). The error term makes this a
 * loss-driven quantity: with post_errors all zero the trace is identically zero.
 *
 * Returns an all-zero trace with EVENTPROP_OK when there are no post spikes, which is the
 * correct limit (no jumps to sum).
 */
LambdaITraceOutput eventprop_compute_lambda_i_trace(LambdaITraceInput input);

/**
 * Compute the synaptic adjoint (lambda_i) for one neuron at one time.
 *
 * The same sweep eventprop_compute_lambda_i_trace samples on a grid, evaluated at a single t,
 * so the two entry points agree by construction rather than by review. Use it when a handful
 * of values is wanted: the reference's figure-1 style queries ask for lambda_i at chosen times
 * rather than on a uniform grid.
 *
 * input.t_max and input.dt are ignored — the sweep and t fully determine the answer. The
 * zero-post-spikes case is lambda_i = 0 with EVENTPROP_OK.
 */
LambdaIAtTimeOutput eventprop_compute_lambda_i_at_time(LambdaITraceInput input, double t);

#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_DIAG_H */
