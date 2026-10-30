"""View 2 — what the network learns, and how fast.

Built entirely from the shipped checkpoints: 10 published runs, their per-epoch test
accuracy, and 11 weight snapshots each. Nothing is retrained to make these pictures,
which is why they render instantly and why they show the published result rather than a
partial run that ended when the visitor got bored.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import state
from eventweb import analysis, figures as F
from eventweb.dataset import CLASS_NAMES


def render() -> None:
    te = state.split("test")
    tr = state.split("train")

    a, b = st.columns([1.25, 1], gap="large")
    with a:
        st.plotly_chart(
            F.dataset_scatter([tr.xy(), te.xy()], [tr.labels, te.labels],
                              "3 classes from 4 numbers"),
            width="stretch",
        )
    with b:
        st.markdown(
            "A point `(x, y)` inside a disc with two dots cut out of it, plus a mirrored "
            "copy `(1-x, 1-y)` — so a sample really is `[x, y, 1-x, 1-y]`, and each of the "
            "four becomes one spike.\n\n"
            "**The reason EventProp demos on this and not MNIST:** the task is unlearnable "
            "for a shallow classifier and easy for a deep one. The dataset's own paper "
            "reports 64.3% for a single layer against 98.7% for a deep ANN, so a two-layer "
            "spiking network landing near 98% is evidence the hidden layer is doing real "
            "work — not evidence the task was already linear."
        )
        st.metric("Shallow ceiling", f"{state.PUBLISHED['shallow']:.1%}", delta_color="off")
        st.metric(
            "Deep reference",
            f"{state.PUBLISHED['deep_ann']:.1%}",
            f"+{(state.PUBLISHED['deep_ann'] - state.PUBLISHED['shallow']) * 100:.1f} pts",
            delta_color="off",
        )
        st.caption("Both are published results for this dataset — measured by neither this "
                   "app nor EventProp.")

    st.divider()
    st.subheader("Ten independent runs")
    accs = [state.checkpoint(s).accuracies for s in range(10)]
    final = np.array(accs)[:, -1]

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(F.learning_curves(accs, "#3b82f6", "Test accuracy per seed",
                                          mean_color="#0ea5e9", log_y=False,
                                          ylabel="test accuracy"), width="stretch")
    with c2:
        st.plotly_chart(F.learning_curves(accs, "#ef4444", "Test error per seed (log)",
                                          mean_color="#f59e0b", log_y=True,
                                          ylabel="test error"), width="stretch")

    d1, d2, d3 = st.columns(3)
    d1.metric("Mean final test accuracy", f"{final.mean():.1%}",
              f"±{final.std():.1%} over {len(final)} seeds", delta_color="off")
    d2.metric("Spread", f"{final.min():.1%} … {final.max():.1%}", "seed to seed",
              delta_color="off")
    d3.metric("Epochs to 95%", _epochs_to_median(accs, 0.95), "median over the 10 seeds",
              delta_color="off")
    st.caption("Accuracy is recorded *before* each epoch, so entry i describes the model "
               "after i epochs. That off-by-one is worked through in `eventweb/network.py`.")

    st.divider()
    st.subheader("Watch the boundary appear")
    e1, e2 = st.columns([1, 2], gap="large")
    seed = e1.selectbox("seed", range(10), index=0, key="evo_seed")
    ckpt = state.checkpoint(seed)
    epochs = list(ckpt.snapshot_epochs)
    snap = e1.select_slider("snapshot", options=list(range(len(epochs))), value=len(epochs) - 1,
                            format_func=lambda i: f"{epochs[i]} epochs")
    resolution = e1.select_slider("grid", [21, 41, 81], value=41,
                                  help="Each cell is one forward pass through the library.")
    with e1:
        st.markdown(f"**after {epochs[snap]} epochs** — the network has been shown "
                    f"{epochs[snap] * 5000:,} samples by this point.")
        st.caption(f"Redrawn from {resolution}×{resolution} = {resolution * resolution} "
                   "forward passes through the C library.")
    grid, axis = state.boundary("c", seed, snap, resolution)
    truth_grid, _ = state.true_grid(resolution)
    with e2:
        st.plotly_chart(
            F.decision_boundary(grid, axis, f"Decision boundary · seed {seed}",
                                true_labels=truth_grid, height=520),
            width="stretch",
        )
        st.caption("White contours are the true class shapes. Where the coloured regions "
                   "bulge past them is what the network got wrong.")

    st.divider()
    st.subheader("Where the mistakes are")
    seed2 = st.selectbox("seed for the confusion matrix", range(10), index=0, key="cm_seed")
    pred, margin, first = state.evaluated("c", seed2, len(epochs) - 1, "test")
    cm = analysis.confusion(pred, te.labels)
    stats = analysis.silence_stats(first, te.labels)
    fired = analysis.fired_neuron_stats(first)
    silent_by_class = ", ".join(
        f"{CLASS_NAMES[int(k)]} {v:.0%}" for k, v in stats["per_class"].items()
    )
    f1, f2 = st.columns([1, 1], gap="large")
    with f1:
        st.plotly_chart(F.confusion_matrix(cm, f"Seed {seed2} · 1000 test samples"),
                        width="stretch")
    with f2:
        st.markdown(
            f"""
| | |
|---|---|
| Samples with **no** output spike at all | {stats['n_silent']} / {stats['n']} ({stats['overall']:.1%}) |
| Mean output neurons firing per sample | {fired['mean']:.2f} of 3 |
| Silent rate by class | {silent_by_class} |
| Median timing margin, correct calls | {_median_margin(pred, margin, te.labels)} |

Roughly a third of samples finish the 30 ms window with no spike anywhere in the output
layer. They are scored as errors — the read-out has nothing to rank — which is a fact
worth knowing before quoting any accuracy figure from this app.
"""
        )


def _epochs_to_median(curves: list[np.ndarray], target: float) -> str:
    """Median epoch at which the seeds first reached `target` accuracy."""
    reached = []
    for c in curves:
        hit = np.flatnonzero(np.asarray(c) >= target)
        if len(hit):
            reached.append(int(hit[0]))
    return f"{int(np.median(reached))}" if reached else "not reached"


def _median_margin(pred: np.ndarray, margin: np.ndarray, labels: np.ndarray) -> str:
    ok = (pred == labels) & np.isfinite(margin)
    return f"{np.median(margin[ok]) * 1e3:.2f} ms" if ok.any() else "n/a"
