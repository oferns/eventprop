"""View 3 — the exact gradient, drawn.

This is the page the repository exists to justify. EventProp's claim is that a spike
time is a *differentiable* quantity: the threshold crossing is root-found rather than
picked off a grid, and a time-reversed adjoint sweep differentiates that root exactly.

The demonstration is deliberately small — 5 inputs, 1 hidden neuron, 1 output neuron —
and the objective is the sum of the output spike times, so the gradient literally means
"how far does this spike move if I move this synapse". A central finite difference is
plotted next to it, and the two agree to about a part in a billion.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import state
from eventweb import analysis, dataset, figures as F


def render() -> None:
    st.header("Exact gradients through a spike time")

    te = state.split("test")
    c1, _ = st.columns([1, 3], gap="large")
    with c1:
        idx = st.selectbox("Input sample", range(te.n), index=42,
                           format_func=lambda i: f"{i} · {_name(te.labels[i])}")
        seed = st.slider("Weight draw", 0, 9, 0,
                         help="The toy 5-1-1 network's random hidden weights. Agreement "
                              "should not depend on the draw, so try a few.")
    coords = tuple(float(v) for v in te.coords[idx])
    demo = state.gradient_demo("eigen", coords, seed)

    if not demo.fired:
        st.warning(
            "This combination left the output neuron silent, so there is no spike to "
            "differentiate and both methods would trivially agree on zero. Try another "
            "sample or weight draw."
        )
        return

    st.markdown(
        f"The output neuron fired **{demo.n_output_spikes}** time"
        f"{'s' if demo.n_output_spikes != 1 else ''} at "
        f"{np.round(demo.output_spike_times * 1e3, 2).tolist()} ms."
    )

    st.plotly_chart(
        F.spike_raster(demo.input_spike_times, demo.input_spike_sources, dataset.N_CHANNELS,
                       "Input spikes — the sample, encoded as first-spike latencies",
                       height=170, color="#94a3b8"),
        width="stretch",
    )

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.plotly_chart(
            F.voltage_trace(demo.hidden_voltage_t, [("hidden neuron", demo.hidden_voltage)],
                            "Hidden membrane potential",
                            spikes=[("hidden", demo.hidden_spikes)]),
            width="stretch",
        )
        st.caption("Each crossing is root-found on the interval between input spikes, not "
                   "taken as the first grid point that happens to exceed `v_th`. That is "
                   "what makes a crossing time differentiable at all.")
    with g2:
        st.plotly_chart(
            F.voltage_trace(demo.output_voltage_t, [("output neuron", demo.output_voltage)],
                            "Output membrane potential",
                            spikes=[("output", demo.output_spike_times)]),
            width="stretch",
        )
        st.caption("This neuron is driven only by the hidden spike above. Move a hidden "
                   "synapse and both crossings move together — which is the chain the "
                   "adjoint has to walk.")

    st.subheader("The adjoint")
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.plotly_chart(F.adjoint_trace(demo.lambda_t, demo.lambda_trace,
                                        "λ(t) on the output neuron"), width="stretch")
    with a2:
        if demo.hidden_lambda_trace.size:
            st.plotly_chart(F.adjoint_trace(demo.hidden_lambda_t, demo.hidden_lambda_trace,
                                            "λ(t) on the hidden neuron"),
                            width="stretch")
        else:
            st.info("The hidden neuron is silent here, so it has no adjoint to plot.")
    st.markdown(
        "λ satisfies the adjoint equation, integrated **backwards** from the last spike, "
        "jumping at each postsynaptic spike in proportion to the error signal sitting on "
        "it. The backward sweep is what converts those jumps into error impulses on the "
        "presynaptic spikes: the hidden neuron's λ above exists only because the output "
        "neuron spiked."
    )

    st.divider()
    st.subheader("Exact vs finite differences")
    names = [f"w{i}" for i in range(demo.hidden_w.shape[0])]
    h1, h2 = st.columns([2, 1], gap="large")
    with h1:
        st.plotly_chart(F.gradient_comparison(demo.analytic, demo.fd, demo.fd_valid, names,
                                              "d(sum of output spike times) / dw"),
                        width="stretch")
    with h2:
        valid = demo.fd_valid.ravel()
        rel = demo.rel_error.ravel()
        st.metric("Max relative error", _fmt(_max_rel(rel, valid)), "exact vs central diff")
        st.metric("Synapses compared", f"{int(valid.sum())} / {valid.size}")
        if (~valid).any():
            st.metric("Not comparable", int((~valid).sum()),
                      "perturbation changed the spike count", delta_color="off")
        st.metric(
            "Output synapse",
            _fmt(_rel(demo.output_analytic, demo.output_fd)) if demo.output_fd_valid else "n/a",
            f"exact {demo.output_analytic:.4g} vs FD {demo.output_fd:.4g}",
            delta_color="off",
        )

    st.markdown(
        f"""
**Why some synapses are marked instead of compared.** A perturbation of {analysis.WEPS:g}
can push a neuron over or under threshold and change the *number* of spikes. The objective
is then discontinuous in that weight: a central difference measures a chord across the
jump, and there is no derivative for the exact method to agree with. Calling that
"disagreement" would measure the wrong thing; hiding it would hide the discontinuity that
makes this problem hard.

**Why the agreement is not luck.** The finite-difference noise floor here is around
`machine_eps / t_eps`, i.e. `1e-16 / 1e-3`. Anything near a part in a billion is the exact
gradient rather than a lucky step size, and `selfcheck.py` re-runs the comparison from a
terminal.
"""
    )
    st.caption(
        "Traces and adjoints use the Eigen reference binding. The C library implements the "
        "same diagnostics in `eventprop_diag.h` — and guards the silent-neuron case the "
        "Eigen core does not — but no Python entry point reaches them yet."
    )


def _name(label) -> str:
    return dataset.CLASS_NAMES[int(label)]


def _max_rel(rel: np.ndarray, valid: np.ndarray) -> float:
    picked = rel[valid & np.isfinite(rel)]
    return float(picked.max()) if picked.size else float("nan")


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("nan")


def _fmt(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.1e}"
