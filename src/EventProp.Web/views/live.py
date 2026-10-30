"""View 1 — the live Yin-Yang classifier.

The point of this page is that a single click runs the real pipeline: TTFS encode, 200
LIF neurons in the C library, threshold crossings root-found, class read off the FIRST
output spike. Nothing here is a simulation of the library; it is the library.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import state
from eventweb import dataset, figures as F
from eventweb.dataset import CLASS_NAMES


def render() -> None:
    st.caption(
        "Chapter 2 left the gradient living in the instant of a crossing. This is that "
        "instant under a microscope, and the signal that reaches it: pick any point, and "
        "read the membrane voltage, the crossings, and the adjoint that runs backwards "
        "through them."
    )

    ctl, main = st.columns([1, 2.6], gap="large")

    seed = ctl.selectbox("Trained run (seed)", range(10), index=0,
                         help="Ten independent published runs, shipped as checkpoints.")
    snap_names = [f"{e} epochs" for e in state.checkpoint(seed).snapshot_epochs]
    snap = ctl.selectbox("Snapshot", range(len(snap_names)), index=len(snap_names) - 1,
                         format_func=lambda i: snap_names[i],
                         help="Weight snapshots saved during training. See the note on "
                              "epoch alignment in eventweb/network.py.")
    backend_key = ctl.radio("Backend", ["c", "eigen"], format_func=lambda k: {
        "c": "C library (Cython)", "eigen": "Eigen reference (pybind11)"}[k])

    # Opens on the stimulus the story has followed since chapter 1; the sliders stay for
    # readers who want to hunt for their own point.
    te = state.split("test")
    hero = state.story()
    mode = ctl.radio("Choose a point", ["the story's point", "sliders"])
    if mode == "sliders":
        x = ctl.slider("x", 0.0, 1.0, 0.68, 0.005)
        y = ctl.slider("y", 0.0, 1.0, 0.45, 0.005)
        coords = dataset.mirror(np.array([[x, y]]))
        truth = int(dataset.which_class(np.array([x]), np.array([y]))[0])
    else:
        coords = te.coords[hero["index"] : hero["index"] + 1]
        truth = int(hero["truth"])
        ctl.caption(
            f"test index {hero['index']} · true class `{CLASS_NAMES[truth]}` · first-spike "
            f"margin {hero['margin'] * 1e3:.3f} ms — the point chapter 1 picked, and the one "
            "chapter 2 differentiates."
        )

    be = state.backend(backend_key)
    model = state.net(backend_key, seed, snap)

    # --- The forward pass, on the real library -----------------------------------------
    times, sources = dataset.encode(coords)
    pred, margin, first = model.predict(coords)
    p = int(pred[0])
    m = float(margin[0])
    hidden_vec = model.hidden.post_batch[0]
    n_hidden_fired = len({int(s) for s in np.asarray(hidden_vec.sources)})

    with main:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            grid, axis = state.boundary(backend_key, seed, snap, 81)
            truth_grid, _ = state.true_grid(81)
            st.plotly_chart(
                F.decision_boundary(
                    grid, axis, "What the network believes",
                    true_labels=truth_grid,
                    points=[(float(coords[0, 0]), float(coords[0, 1]))],
                ),
                width="stretch",
            )
            inside = bool(dataset.in_support(np.array([coords[0, 0]]), np.array([coords[0, 1]]))[0])
            if not inside:
                st.caption("This point is outside the disc the data is drawn from — "
                           "anything the network says here is extrapolation.")
        with right:
            verdict = (
                f"**neuron {p} · {CLASS_NAMES[p]}**" if p >= 0 else "**no output spike**"
            )
            correct = p == truth
            st.metric(
                "Prediction",
                verdict.replace("**", ""),
                f"{'correct' if correct else 'wrong'} vs {CLASS_NAMES[truth]}",
                delta_color="normal" if correct else "inverse",
            )
            st.caption(
                "First-spike margin: "
                + ("a neuron stayed silent, so nothing to compare." if p < 0 or not np.isfinite(m)
                   else f"{m * 1e3:.2f} ms ahead of the runner-up.")
            )
            st.plotly_chart(F.first_spike_readout(first[0], truth, "Read-out"),
                            width="stretch")

    # --- Where the spikes went ---------------------------------------------------------
    st.subheader("The same point, as spikes")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(
            F.spike_raster(times[0], sources[0], dataset.N_CHANNELS,
                           "Input — 4 coordinates + bias, one spike each", height=200),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            F.spike_raster(np.asarray(hidden_vec.times), np.asarray(hidden_vec.sources),
                           200, f"Hidden layer — {n_hidden_fired} of 200 neurons fired",
                           height=200, color="#6366f1"),
            width="stretch",
        )

    st.subheader("Why that neuron fired when it did")
    fired_ids = sorted({int(s) for s in np.asarray(hidden_vec.sources)})
    if not be.supports_traces():
        st.info(
            "Voltage traces come from the Eigen reference binding. The C library exports "
            "`eventprop_compute_voltage_trace` (see `src/EventProp.C/eventprop_diag.h`) but "
            "the Cython layer wraps no Python entry point for it yet — switch the backend "
            "to see the membrane potential."
        )
    else:
        default = fired_ids[0] if fired_ids else 0
        nrn = st.slider("Hidden neuron", 0, 199, default,
                        help="Pick one of the neurons that fired to see its voltage climb "
                             "to threshold and reset.")
        t_i, v_i = model.hidden.get_voltage_trace_for_neuron(0, int(nrn), t_max=dataset.T_MAX,
                                                            dt=2.5e-4)
        my_spikes = np.asarray(hidden_vec.times)[np.asarray(hidden_vec.sources).astype(int) == nrn]
        st.plotly_chart(
            F.voltage_trace(t_i, [(f"neuron {nrn}", v_i)], "Membrane potential",
                            spikes=[(f"neuron {nrn}", my_spikes)]),
            width="stretch",
        )
        if nrn in fired_ids:
            st.caption(f"Crossed `v_th` at {np.round(np.asarray(my_spikes) * 1e3, 2).tolist()} ms. "
                       "Those crossing times are root-found, not sampled — which is why they "
                       "can be differentiated at all.")

    # --- Accuracy of this exact network -------------------------------------------------
    st.divider()
    pred_te, _margins, first_te = state.evaluated(backend_key, seed, snap, "test")
    from eventweb import analysis

    stats = analysis.silence_stats(first_te, te.labels)
    a, b, c, d = st.columns(4)
    a.metric("Test accuracy", f"{(pred_te == te.labels).mean():.1%}",
             f"{int((pred_te == te.labels).sum())} / {te.n}")
    b.metric("Published reference", f"{state.PUBLISHED['deep_ann']:.1%}",
             "deep ANN on this dataset", delta_color="off")
    c.metric("Shallow ceiling", f"{state.PUBLISHED['shallow']:.1%}",
             "no hidden layer can do better", delta_color="off")
    d.metric("Samples with no output spike", f"{stats['overall']:.1%}",
             f"{stats['n_silent']} of {stats['n']}", delta_color="off")
    st.caption("Every no-spike sample is scored wrong by construction, so the accuracy above "
               "already pays that penalty.")
