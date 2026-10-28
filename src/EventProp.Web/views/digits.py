"""View 5 — real data: the published MNIST runs, reproduced in the browser.

Everything else in this app runs the Yin-Yang task, which is synthetic by construction: a
2-D plane, a closed-form label, a picture the visitor can stand inside. That is why the
boundary page works, and it is also its limit — it cannot answer "does this do anything on
data nobody designed?".

So this page runs the other half of the paper. `paper/mnist_vmax_{0..9}.pkl` are ten
published 784 → 350 → 10 networks trained with the voltage-maxima loss, and the numbers
below are those weights run now, through the library in this checkout, on all 10 000 MNIST
test digits.

Two things are worth being precise about:

* **The read-out is different.** Yin-Yang reads the first output spike; VMax reads the
  tallest voltage *maximum* (`argmax` of `Maxima.values`, as
  `eventprop/loss_layer.py::get_predictions` does). Same two-layer library, different
  entry point: `eventprop_compute_maxima_batch` instead of `eventprop_compute_spikes_batch`,
  and that path never looks at `v_th`.
* **The residual against the recorded accuracy is real and unexplained.** Reproduced and
  recorded agree to within 8 samples in 10 000 (|Δ| ≤ 0.0008) across the ten runs, in both
  directions, with no systematic sign — and that gap is smaller than the run's own
  epoch-to-epoch swing over its last 50 epochs. What was measured and ruled out as the
  cause: dropout at evaluation time (`_training_data` applies it to training batches
  only), batch layout (batches of 250 and 7 give bit-identical predictions), and near-tie
  float fragility (across all ten runs on all 10 000 digits no maxima margin is ever zero,
  none falls below 1e-9, and the smallest is 5.3e-05 — four orders of magnitude above the
  noise a re-run could move it by). The C library and the Eigen core agree with each other
  on every one of the 10 000 predictions; what is not reproducible is the historical run
  that produced the file — which is the general point AGENTS.md makes about committed
  artefacts.

The pixel files are not vendored: they are downloaded on first use into
`.cache/mnist/`, because redistribution terms for the corpus should be checked by whoever
ships it rather than assumed by this repository.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import state
from eventweb import figures as F, mnist
from eventweb.backends import BACKENDS, available


def render() -> None:
    st.header("Handwritten digits — the published MNIST runs")

    if not state.mnist_ready():
        _first_run()
        return

    images, labels = state.mnist_test()
    ready = [k for k in available() if BACKENDS[k].supports_maxima()]
    if not ready:
        st.error("Neither backend exposes voltage maxima from Python, so the VMax "
                 "read-out cannot run. `eventprop_c.li_layer` and `eventprop.li_layer` "
                 "are the modules this page needs.")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        seed = st.slider("Published run", 0, 9, 0,
                         help="Each of the ten is an independently trained network from "
                              "the paper's own artefacts.")
        backend_key = st.radio(
            "Backend", ready,
            format_func=lambda k: {"c": "C library (Cython)", "eigen": "Eigen reference"}.get(k, k),
            help="Same weights, same encoder, two independent implementations of the "
                 "maxima pass.",
        )
    ck = state.mnist_run(seed)
    pred, margin, truth = state.mnist_evaluated(backend_key, seed)
    measured = float((pred == truth).mean())
    recorded = float(ck.accuracies[-1])
    delta_samples = round((measured - recorded) * len(truth))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy, measured here", f"{measured:.2%}",
              help=f"{BACKENDS[backend_key].short} backend, all {len(truth)} test digits, "
                   f"the final snapshot's weights.")
    m2.metric("Recorded in the file", f"{recorded:.2%}",
              delta=f"{measured - recorded:+.2%}", help="Field 2 of the checkpoint — the "
              "test curve's last epoch. Field 4 is the validation curve; they differ.")
    m3.metric("Difference", f"{delta_samples:+d} digits",
              help="Expressed in digits, because a percentage hides how small this is: "
                   "the run's own accuracy swings by more than this between adjacent "
                   "epochs near the end of training.")
    m4.metric("Smallest decision margin", f"{float(np.min(margin)):.2g}",
              help="Top-1 minus top-2 voltage maximum, minimum over the test set. Two "
                   "significant figures because the interesting range is 1e-5, not 1e-3: "
                   "across the ten published runs no margin is ever zero or below 1e-9, so "
                   "no prediction here is a tie that argmax happened to break.")

    st.caption(
        f"Run {seed}: weights {list(ck.weights_at(-1)[0].shape)} → "
        f"{list(ck.weights_at(-1)[1].shape)}, {ck.origin}. Encoding is the reference's: "
        f"one spike per pixel above intensity {mnist.PIXEL_MIN}, latency "
        f"{mnist.T_MIN:.0e} s for the brightest pixel up to {mnist.T_MAX * 1e3:.0f} ms for "
        f"the dimmest, so a digit is a pattern of arrival times rather than of rates."
    )

    with st.expander("How do the backends compare on this?"):
        if len(available()) < 2 or len(ready) < 2:
            st.info("Only one backend with maxima support is importable here, so there is "
                    "nothing to compare against.")
        elif st.button("Run all 10 000 digits through both backends"):
            other = [k for k in ready if k != backend_key][0]
            p2, _, _ = state.mnist_evaluated(other, seed)
            agree = float((pred == p2).mean())
            st.write(
                f"**{BACKENDS[backend_key].short} vs {BACKENDS[other].short}: "
                f"{agree:.4%} of {len(truth)} predictions identical** "
                f"({int((pred != p2).sum())} differ). Accuracy "
                f"{measured:.4f} vs {float((p2 == truth).mean()):.4f}. Two independent "
                f"implementations of the maxima pass, on real trained weights at 784 "
                f"inputs."
            )
        else:
            st.caption("Roughly ten seconds per backend, and the result is cached.")

    st.divider()
    st.subheader("One digit, end to end")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        idx = st.number_input("Test digit", 0, len(images) - 1, 0, step=1)
        n_spikes = int(np.count_nonzero(images[idx] > mnist.PIXEL_MIN))
        st.caption(f"True label **{int(truth[idx])}** · predicted "
                   f"**{int(pred[idx])}** · {n_spikes} input spikes")
        st.plotly_chart(F.digit_image(images[idx], "The input", height=220),
                        width="stretch")
    with c2:
        values = state.mnist_maxima(backend_key, seed, idx)
        st.plotly_chart(F.maxima_bars(values, int(truth[idx]),
                                      "Output maxima — the tallest bar is the prediction",
                                      height=220), width="stretch")
        neurons, times = state.mnist_hidden(backend_key, seed, idx)
        st.plotly_chart(F.hidden_latencies(neurons, times, ck.weights_at(-1)[0].shape[1],
                                           f"Hidden layer: {len(neurons)} of "
                                           f"{ck.weights_at(-1)[0].shape[1]} neurons fired",
                                           height=240), width="stretch")

    with st.expander("All ten published runs, and what this machine makes of them"):
        accs = [state.mnist_run(s).accuracies for s in range(10)]
        st.plotly_chart(F.learning_curves(accs, BACKENDS[backend_key].color,
                                          "Recorded test accuracy per run",
                                          log_y=False, ylabel="test accuracy"),
                        width="stretch")
        # Behind a button, not merely inside the expander: `st.expander` does NOT defer
        # execution, so anything written directly in its body runs on every repaint — and
        # this block is ten full test-set passes, roughly eighty seconds.
        if st.button("Evaluate all ten on this machine",
                     help="Ten runs x 10 000 digits. Slow once, cached per run after that."):
            rows = []
            for s in range(10):
                p, _, t = state.mnist_evaluated(backend_key, s)
                rows.append((s, float((p == t).mean()),
                             float(state.mnist_run(s).accuracies[-1])))
            st.dataframe(
                [{"run": s, "measured": round(m, 4), "recorded": round(r, 4),
                  "delta": round(m - r, 4)} for s, m, r in rows],
                width="stretch", hide_index=True,
            )
            worst = max(abs(r[1] - r[2]) for r in rows)
            st.caption(f"Largest |measured − recorded| across the ten: {worst:.4f} "
                       f"({round(worst * len(truth))} digits). Recorded values come from "
                       f"the files; measured values were produced on this machine.")
        else:
            st.caption("The curves above are read straight from the ten checkpoint files "
                       "and cost nothing. The measured-vs-recorded table is the slow part, "
                       "so it waits for the button.")


def _first_run() -> None:
    st.info("The MNIST pixel files are not in this repository, so this page needs one "
            "download (~1.6 MB) before it can show anything. They are cached in "
            f"`{mnist.cache_dir()}` and never committed.")
    col, _ = st.columns([1, 2])
    with col:
        if st.button("Download the test set"):
            try:
                with st.spinner("Fetching t10k images and labels…"):
                    state.mnist_test()
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.rerun()
    st.caption(f"Served from `{mnist.URL_BASE}` — the same mirror the reference "
               f"library's own loader uses, so the pixels behind these numbers match the "
               f"ones the published runs were evaluated on.")
