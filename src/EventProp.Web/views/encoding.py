"""Chapter 1 — a spike train is a number.

The claim is deliberately small and completely checkable: the network's entire visual field
is five arrival times, four of which are the coordinates in milliseconds. Everything later
in the app is downstream of that restatement, so this page spends its time on the encoder
and the read-out race, and stops exactly where the interesting difficulty starts.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import state
from eventweb import dataset, figures as F

SNAPSHOT = -1
SEED = 0
BACKEND = "c"


def render() -> None:
    te = state.split("test")
    default = state.story_default(BACKEND, SEED, SNAPSHOT)
    hero = state.story(BACKEND, SEED, SNAPSHOT)
    x, y = (float(v) for v in hero["coords"][:2])

    with st.expander("The point this story follows", expanded=False):
        st.caption(
            "One stimulus runs through all seven chapters: its spike train here, its "
            "gradient in chapter 2, its adjoint in chapter 3. Move it and every chapter "
            "follows it. This one is chosen automatically as the correctly-classified point "
            "whose first-spike race is **closest** — a decision that nearly went the other "
            "way, which is the only kind worth three chapters."
        )
        chosen = st.number_input("test index", 0, te.n - 1, value=int(default), step=1,
                                 key="story_index")
        st.caption(
            f"index {int(chosen)} · true class "
            f"`{dataset.CLASS_NAMES[int(te.labels[int(chosen)])]}`"
        )

    # --- Beat 1: the code ---------------------------------------------------------------
    st.markdown(
        f"Everything the network will ever know about this point is **five spikes**. Four of "
        f"them are the coordinates themselves: input neuron *i* fires at "
        f"`xᵢ × {dataset.T_MAX * 1e3:.0f} ms`, and the dataset stores the 4-vector as "
        f"`[x, y, 1−x, 1−y]`, so the encoder is handed a point and its mirror. The fifth is a "
        f"constant bias spike at `t = 0`, identical for every stimulus in the dataset."
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(
            F.encoder_map(hero["coords"], dataset.T_MAX,
                          "The code: a coordinate becomes a latency"),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            F.spike_raster(hero["times"], hero["sources"], dataset.N_CHANNELS,
                           "Five spikes, once each",
                           t_max_ms=dataset.T_MAX * 1e3, height=290),
            width="stretch",
        )
    st.caption(
        "Nothing is sampled over time and nothing is a firing rate. Because the map is a "
        "straight line through the origin, the spike train loses no information — it restates "
        f"`({x:.3f}, {y:.3f})` in milliseconds."
    )

    # --- Beat 2: the read-out is a race -------------------------------------------------
    st.subheader("The answer is a race")
    truth_name = dataset.CLASS_NAMES[hero["truth"]]
    pred_name = dataset.CLASS_NAMES[hero["pred"]]
    b1, b2 = st.columns([1.15, 1], gap="large")
    with b1:
        st.plotly_chart(
            F.first_spike_readout(hero["first"], hero["truth"],
                                  "Which output neuron fires first"),
            width="stretch",
        )
    with b2:
        st.markdown(
            f"Whichever output neuron fires **first** wins; the gap to the runner-up is how "
            f"confident the timing is.\n\n"
            f"- predicted `{pred_name}`, true `{truth_name}`\n"
            f"- first spikes (ms): "
            + ", ".join(
                f"{dataset.CLASS_NAMES[i]} "
                + (f"{t * 1e3:.3f}" if np.isfinite(t) else "silent")
                for i, t in enumerate(hero["first"])
            )
            + f"\n- margin to the runner-up: **{hero['margin'] * 1e3:.3f} ms**"
        )
    st.caption(
        "A tiny margin is not a weak network — it is a point that sits close to a boundary, "
        "which is why chapter 4 will show this one again as a mistake waiting to happen."
    )

    # --- Beat 3: why the dataset exists -------------------------------------------------
    st.subheader("Why anyone would build a dataset like this")
    pred, _, _ = state.evaluated(BACKEND, SEED, SNAPSHOT, "test")
    measured = float((pred == te.labels).mean())
    m1, m2, m3 = st.columns(3)
    m1.metric("Shallow read-out, published", f"{state.PUBLISHED['shallow']:.1%}",
              help="Dataset README, “Reference results”: 64.3 ± 0.2 % test accuracy, same "
                   "training settings as the ANN, no hidden layer.")
    m2.metric("Deep ANN, published", f"{state.PUBLISHED['deep_ann']:.1%}",
              help="Same README: 98.7 ± 0.3 % for a 4-120-3 artificial network.")
    m3.metric("This network, measured here", f"{measured:.2%}",
              delta=f"{(measured - state.PUBLISHED['deep_ann']) * 100:+.1f} pts vs the ANN",
              delta_color="off",
              help=f"C library, seed {SEED}, final snapshot, all {te.n} held-out points. "
                   "selfcheck.py step 1 recomputes it on both backends.")
    st.caption(
        "The two published numbers come from the dataset's own README "
        "(`eventprop/yin_yang_data_set/README.md`), quoted as published — nothing on this "
        "page fits a baseline. The gap between them is the whole reason the shape exists: no "
        "read-out that looks at the inputs once can carve it, so something has to be hidden "
        "in between."
    )

    # --- Beat 4: what the hidden layer buys ---------------------------------------------
    st.subheader("What the hidden layer buys")
    grid, axis = state.boundary(BACKEND, SEED, SNAPSHOT, 81)
    labels, _ = state.true_grid(81)
    st.plotly_chart(
        F.decision_boundary(grid, axis, "Three regions, one 200-neuron hidden layer",
                            true_labels=labels,
                            points=[(x, y)], height=460),
        width="stretch",
    )
    st.caption(
        "The white contour is the true shape; the fill is what this network decided. The "
        f"cross is the point from the first figure — inside the region it was asked to claim, "
        f"and {hero['margin'] * 1e3:.3f} ms from giving it up."
    )
