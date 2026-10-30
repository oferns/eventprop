"""View 6 — a continuous target: fitting the regression read-out, live.

Every other page in this app answers a classification question, because that is the question
the paper asks: which output neuron fires first (TTFS), or which has the tallest voltage
maximum (VMax). A continuous target needs a different read-out, and the C library has one:

    y_hat = bias + Σ a[n] · maxima[n]

with least-squares or smooth Huber error, `dL/dmaxima` written into the buffers the maxima
backward pass reads, and `dL/da` / `dL/db` handed back. So a training step is *forward maxima →
regression → backward*, and the arithmetic below never leaves the library.

What the fit is asked to predict is deliberately physical: one coordinate of the stimulus. The
question "can a single weighted sum over 200 hidden voltage maxima say where the stimulus was?"
has a measurable answer, and the answer is embarrassingly good — held-out R² ≈ 0.995 after a few
hundred steps, against 0.997-0.999 for the best closed-form linear fit on the same features. That
matters for what EventProp is *for*: it shows the maxima carry a continuous variable the
threshold-crossing machinery can be trained on, which is the version of this you would want for
regressing a movement target or a time-series step rather than a class.

The reference numbers come with a trap in them, and the page shows the trap rather than hiding
it. Exact least squares on 400 points and 201 parameters *interpolates* — train R² = 1.0000, and
then it generalises worse than the regularised fits (0.994 for x, but only 0.961 for y). Ridge
with the penalty chosen on the dataset's validation split reaches 0.9986 / 0.9972. So "the
optimiser beat the closed-form answer" is a real result here, not a bug in the plot.

Three things this page will not let you forget:

* **It is the C library only, and that is not laziness.** The reference library ships no
  regression loss, so there is no Eigen implementation to compare against and no backend picker
  to offer. Parity for this loss is managed ↔ native in .NET
  (`src/EventProp.Tests/RegressionLayerParityTests.cs`), and the page says so rather than
  implying a cross-backend agreement it never measured.
* **The gradients come back summed over the batch**, by design, and this page divides by the
  batch size before touching them. A binding that quietly averaged is how a learning rate ends
  up coupled to a batch size.
* **The dashed line is a baseline, not a ceiling.** It is ridge on the same maxima with the
  penalty chosen on the validation split. Exact least squares is worse here, not better — it has
  more parameters than points and interpolates — so a fit above the exact answer is correct, and
  the expander below shows the λ path that proves it.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import state
from eventweb import figures as F, regression
from eventweb.backends import BACKENDS, available


def render() -> None:
    if "c" not in available():
        st.error("The C backend is not importable in this interpreter, and the regression "
                 "read-out exists only there. `eventweb/runtime.py` names the interpreter the "
                 "extension was built for.")
        return

    st.caption(
        "One weighted sum over the hidden layer's voltage maxima, `ŷ = b + Σ aₙ·maximaₙ`, "
        "asked to predict a coordinate of the stimulus. Frozen published weights, "
        f"{state.READOUT_TRAIN_N} training points, {state.READOUT_EVAL_N} held out — the "
        "numbers below are measured on those held-out points, never on the fit's own batch."
    )

    left, right = st.columns([1, 2], gap="large")
    with left:
        seed = st.slider("Published run", 0, 9, 0,
                         help="The hidden weights come from one of the ten published "
                              "Yin-Yang networks, same as every other page.")
        target_label = st.radio(
            "Target", list(regression.TARGETS),
            help="Which coordinate the read-out has to recover. Both are in dataset units, "
                 "so the prediction is directly comparable with the plane the other pages draw.",
        )
        column = regression.TARGETS[target_label]

        what = st.radio(
            "What trains",
            ["the read-out only", "read-out + weights (adjoint)"],
            help="Read-out only asks whether the maxima already carry the variable — the "
                 "network is untouched. The second option also pushes dL/dmaxima through the "
                 "time-reversed adjoint sweep to get dL/dW, which is the part that is "
                 "EventProp rather than a linear model.",
        )
        train_weights = what.startswith("read-out +")

        steps = st.slider("Steps", 50, 1000, 600, 50)
        batch = st.select_slider("Batch", options=[8, 16, 32, 64], value=64,
                                 help="Gradients arrive summed over the batch; the page "
                                      "divides by this number so the learning rate means what "
                                      "it says.")
        lr_readout = st.select_slider(
            "Learning rate (read-out)", options=[1e-3, 2e-3, 5e-3, 1e-2, 2e-2], value=1e-2,
            format_func=lambda v: f"{v:g}",
            help="Adam. Plain SGD does not survive this problem: 200 near-collinear features, "
                 "condition number ~3e6, and every plain-SGD step size tried either crawled or "
                 "diverged.")
        lr_weights = 1e-5
        if train_weights:
            lr_weights = st.select_slider(
                "Learning rate (weights)", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-5,
                format_func=lambda v: f"{v:g}",
                help="Deliberately two to three orders of magnitude below the read-out's. These "
                     "are published trained weights: move them at the read-out's step size and "
                     "you destroy the network that makes the maxima informative before the "
                     "read-out has learned anything.",
            )

        loss_kind = st.radio(
            "Loss", ["least squares", "Huber δ = 1.0", "Huber δ = 0.1", "Huber δ = 0.01"],
            help="Non-finite δ selects least squares; a finite positive δ selects the smooth "
                 "Huber loss, whose derivative approaches ±δ from below and never reaches it. "
                 "That saturation is why a small δ needs more steps for the same fit, not "
                 "fewer: the gradient is capped.",
        )
        huber_delta = {"least squares": float("inf"), "Huber δ = 1.0": 1.0,
                       "Huber δ = 0.1": 0.1, "Huber δ = 0.01": 0.01}[loss_kind]

        standardise = st.toggle("Standardise the maxima per neuron", value=True, help=_STD_HELP)

    fit = state.readout_fit(seed, column, steps=steps, batch=batch, lr_readout=lr_readout,
                            lr_weights=lr_weights, train_weights=train_weights,
                            standardise=standardise, huber_delta=huber_delta)
    base = state.readout_baseline(seed, column)
    te_labels = state.split("test").labels[:fit.n_eval]

    if fit.divergence:
        st.error("The loss went non-finite — the fit stopped where it diverged. Lower the "
                 "learning rate or turn standardisation on.", icon="💥")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Held-out R²", f"{fit.after['r2']:.4f}", delta=f"{fit.after['r2'] - fit.before['r2']:+.3f}",
              help=f"Measured on {fit.n_eval} points the fit never saw. Before is the random "
                   f"initial read-out, which is usually worse than predicting the mean — "
                   f"a negative R² says exactly that.")
    m2.metric("Correlation", f"{fit.after['corr']:.4f}",
              help="Pearson r on the held-out points. High r with a compressed spread is "
                   "possible, which is why the scatter next to it is not optional.")
    m3.metric("RMSE", f"{fit.after['rmse']:.4f}",
              help=f"Same units as the target, which for a stimulus coordinate is dataset "
                   f"units over the unit square.")
    m4.metric("Best linear baseline (ridge)", f"{base['ridge_r2']:.4f}",
              help=f"Ridge on the same maxima, penalty λ = {base['ridge_lambda']:g} chosen on "
                   f"the dataset's **validation** split — never on the test points these metrics "
                   f"come from. Exact least squares scores {base['exact_r2']:.4f} here despite "
                   f"scoring {base['exact_train_r2']:.4f} on its own training data: it "
                   f"interpolates, and interpolation is not the same as generalising.")

    st.progress(1.0, text=(
        f"{fit.steps} steps · batch {fit.batch} · {fit.seconds:.2f} s · "
        f"{'read-out and weights' if fit.train_weights else 'read-out only'} · "
        f"{'standardised' if standardise else 'raw maxima'} · {loss_kind}"))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(F.readout_r2_curve(fit.eval_steps, fit.heldout_r2, base["ridge_r2"],
                                           BACKENDS["c"].color,
                                           "Held-out R², against the best linear baseline"),
                        width="stretch")
    with c2:
        st.plotly_chart(F.readout_loss_curve(fit.losses, BACKENDS["c"].color,
                                             "Mean loss per step"),
                        width="stretch")

    st.plotly_chart(F.readout_fit_scatter(fit.eval_target, fit.eval_pred_before,
                                          fit.eval_pred_after, BACKENDS["c"].color,
                                          f"One read-out against the {target_label.split(' —')[0]} "
                                          f"it was asked to predict",
                                          axis_label=f"true {target_label.split(' —')[0]}"),
                    width="stretch")
    st.caption("Grey is the random initial read-out: correlated with nothing, and scattered far "
               "wider than the target, which is why its R² is negative. Points smeared into a "
               "horizontal band instead of along the dotted line would mean a read-out that "
               "learned the mean and nothing else.")

    st.plotly_chart(F.readout_plane(fit.eval_xy[:, 0], fit.eval_xy[:, 1], fit.eval_target,
                                    fit.eval_pred_after, te_labels,
                                    "The stimulus, and what one weighted sum recovers of it"),
                    width="stretch")

    st.divider()
    st.subheader("Which neurons produced this number")
    pcol, pfig = st.columns([1, 2], gap="large")
    with pcol:
        index = st.number_input("Held-out point", 0, max(0, fit.n_eval - 1), 0)
        # Not cached, and deliberately: one forward pass over one stimulus point costs well
        # under a millisecond, and caching it would mean keying the cache on the fitted weights
        # — a pickle of a 5×200 array to save a rounding error.
        w_probe = (fit.w_final if (fit.train_weights and fit.w_final is not None)
                   else state.checkpoint(seed).weights_at(-1)[0])
        one_point = state.split("test").coords[int(index):int(index) + 1]
        pr = regression.probe("c", w_probe, fit.readout, one_point)
        pr["target"] = float(regression.target_values(one_point, column)[0])
        st.caption(f"True **{pr['target']:.3f}** · predicted **{pr['prediction']:.3f}** · "
                   f"{int(np.count_nonzero(np.abs(pr['terms']) > 1e-12))} of "
                   f"{pr['terms'].size} neurons contribute · "
                   f"{pr['skipped']} silent neuron(s) excluded")
    with pfig:
        st.plotly_chart(F.readout_contributions(pr["terms"], pr["bias"], pr["prediction"],
                                                pr["target"],
                                                "The largest contributions to that prediction"),
                        width="stretch")
    st.caption("A read-out is a weighted sum, so this *is* the explanation: green pushes the "
               "prediction up, red pulls it down, and the bars sum (with the bias) to the number "
               "shown. A neuron that never peaked contributes exactly nothing — the same rule "
               "the C loss applies when it counts a maximum as skipped. Note the asymmetry the "
               "library is careful about: a neuron **at rest** has a finite maximum of 0.0 and "
               "*does* count, it just contributes 0; only a non-finite maximum means 'no peak'.")

    with st.expander("Why the fitted line can sit above the exact fit"):
        st.markdown(
            f"Closed-form least squares on these features is the best *in-sample* fit there is — "
            f"literally: {base['n_params']} parameters for {base['n_train']} points, full rank, "
            f"train R² {base['exact_train_r2']:.4f}. Held out it is worth "
            f"{base['exact_r2']:.4f}. Ridge with λ = {base['ridge_lambda']:g} — picked on the "
            f"validation split, so this is not tuning on the test set — is worth "
            f"{base['ridge_r2']:.4f}.\n\n"
            f"So the dashed line is a *baseline*, not a ceiling, and a fit that ends above the "
            f"exact solution has found the same thing ridge finds: on "
            f"{base['n_params'] - 1} near-collinear features (condition number "
            f"{base['condition']:.2g}) the interpolating solution is not the one that extrapolates. "
            f"Early stopping is a regulariser with a name, and here it is doing its job.\n\n"
            "The λ path, so the trade is visible rather than asserted:\n\n"
            + "\n".join(f"- λ = {lam:g}: validation R² {r2:+.4f}" for lam, r2 in base["ridge_path"])
        )

    with st.expander("Why the standardisation toggle exists"):
        st.markdown(_STD_HELP)

    with st.expander("What this page is not"):
        st.markdown(
            "**Not a cross-backend comparison.** The regression read-out lives in the C library "
            "and the read-only reference library has no regression loss, so there is no Eigen "
            "implementation to agree with. Where the parity guarantee lives instead: "
            "`src/EventProp.C.Tests/test/maxima_regression_loss.cpp` (finite differences against "
            "the C implementation), `src/EventProp.Tests/RegressionLossParityTests.cs` and "
            "`RegressionLayerParityTests.cs` (managed ↔ native, the latter asserted through "
            "`Backward()` so it covers the whole chain), and "
            "`src/EventProp.Native.Python/test/test_maxima_regression.py` (the Python binding, "
            "including that the errors land where the backward pass reads them).\n\n"
            "**Not a benchmark.** The elapsed time above is one fit inside a request handler. "
            "`bench-reports/report_<tag>.md` holds the measured rows for the read-out kernel "
            "(`MaximaRegression`, `Batch_MaximaRegression`) with provenance; and note what those "
            "rows say — at read-out sizes the P/Invoke path is *slower* than managed, because "
            "the arithmetic is ~200 multiply-adds and the boundary crossing is not.\n\n"
            "**Not trained through dropout or augmentation**, and the target is a coordinate of a "
            "synthetic stimulus. It demonstrates the mechanism on a quantity with a ground truth "
            "you can check, which is the point of the Yin-Yang task in the first place."
        )

    if fit.train_weights:
        moved = float(np.abs(fit.w_final - state.checkpoint(seed).weights_at(-1)[0]).max()) \
            if fit.w_final is not None else 0.0
        st.info(
            f"The adjoint path did work here: max |dL/dW| during the run was "
            f"{fit.gradient_scale:.3g}, and the largest weight moved by {moved:.4f} from the "
            f"published value. That is the loss reaching the network rather than stopping at the "
            f"read-out — `eventprop_backward_maxima` consuming the `dL/dmaxima` that "
            f"`eventprop_maxima_regression` wrote."
        )


_STD_HELP = (
    "The maxima of a trained population all sit near the same value: measured here, mean 0.84 "
    "with a median per-neuron spread of 0.07 across samples. So the feature matrix is dominated "
    "by one common direction — condition number ≈ 3e6 — and a gradient step that is the right "
    "size along the informative directions is the wrong size along everything else.\n\n"
    "Writing the read-out on z[n] = (maxima[n] − μ[n]) / s[n] does not change the model — "
    "`b + Σ aₙz[n]` is `b′ + Σ aₙ/sₙ · maxima[n]`, which is the read-out the C entry point "
    "computes, with `bias′ = b − Σ aₙμₙ/sₙ`. It changes the coordinates the optimiser moves in. "
    "The conversion is done inside `eventweb/regression.py` and checked against finite "
    "differences of the C loss by step 15 of `selfcheck.py`, because the chain rule picks up an "
    "extra term here (`dL/daₙ = (dL/ddₙ − (dL/dβ)·μₙ)/sₙ`) and dropping that term biases the fit "
    "without raising a single error.\n\n"
    "**What the toggle actually costs**, measured at the page's own settings (Adam, 600 steps, "
    "batch 64, target x): standardised reaches held-out R² 0.9968, raw maxima 0.9353. So Adam "
    "absorbs most of the conditioning problem and the toggle is a quality knob here, not a "
    "correctness one. Without Adam it is a correctness one: on plain gradient descent the raw "
    "parameterisation ran to a non-finite loss at every step size that made progress, which is "
    "why the optimiser arrived before the toggle did."
)
