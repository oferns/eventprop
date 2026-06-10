/**
 * @file eventprop_diag.h
 * @brief Diagnostic / debugging helpers for EventProp.
 *
 * Voltage traces, lambda-I traces — useful for visualising membrane dynamics
 * and verifying gradient computations. Not part of the core training loop.
 */

#ifndef EVENTPROP_DIAG_H
#define EVENTPROP_DIAG_H

#include <stdint.h>
#include "eventprop.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Voltage trace input (borrowed data) ─────────────────────────────── */

/** Input for voltage-trace computation. */
typedef struct {
    const ForwardInputSpikes* input;   /* Forward pass parameters (weights, spikes, v_th, tau_mem, tau_syn) */
    int target_nrn_idx;                /* Post-synaptic neuron to trace */
    const double* fwd_times;           /* Forward-pass spike times (borrowed) */
    const int32_t* fwd_sources;        /* Forward-pass spike sources (borrowed) */
    int n_post_spikes;                 /* Number of forward-pass spikes */
    double t_max;                      /* Simulation end time (seconds) */
    double dt;                         /* Time step (seconds) */
} VoltageTraceInput;

/** Voltage trace output (owned memory). */
typedef struct {
    int n_t;                           /* Number of time points (= t_max / dt) */
    double* v_trace;                   /* Voltage values (malloc'd, length n_t) */
    double* ts;                        /* Time array (malloc'd, length n_t) */
} VoltageTraceOutput;

/**
 * Compute the membrane potential trace for a single post-synaptic neuron.
 * Includes both pre-synaptic drive and post-synaptic spike-reset effects.
 */
VoltageTraceOutput eventprop_compute_voltage_trace(const VoltageTraceInput* input);

/** Free memory owned by a VoltageTraceOutput. */
void eventprop_free_voltage_trace_output(VoltageTraceOutput* output);


/* ── Lambda-I trace input (borrowed data) ────────────────────────────── */

/** Input for lambda-I trace computation. */
typedef struct {
    const double* fwd_times;           /* Forward-pass spike times (borrowed) */
    int n_post_spikes;                 /* Number of forward-pass spikes */
    int target_nrn_idx;                /* Post-synaptic neuron to trace */
    double v_th;                       /* Spike threshold voltage */
    double tau_mem;                    /* Membrane time constant (seconds) */
    double tau_syn;                    /* Synaptic time constant (seconds) */
    double t_max;                      /* Simulation end time (seconds) */
    double dt;                         /* Time step (seconds) */
} LambdaITraceInput;

/** Lambda-I trace output (owned memory). */
typedef struct {
    int n_t;                           /* Number of time points (= t_max / dt) */
    double* lambda_trace;              /* Lambda-I values (malloc'd, length n_t) */
    double* ts;                        /* Time array (malloc'd, length n_t) */
} LambdaITraceOutput;

/**
 * Compute the synaptic adjoint (lambda_i) trace for a single neuron.
 * Useful for debugging the backward pass.
 */
LambdaITraceOutput eventprop_compute_lambda_i_trace(const LambdaITraceInput* input);

/** Free memory owned by a LambdaITraceOutput. */
void eventprop_free_lambda_i_trace_output(LambdaITraceOutput* output);


#ifdef __cplusplus
}
#endif

#endif /* EVENTPROP_DIAG_H */
