"""A continuous readout over voltage maxima, fitted through the C loss.

The two shipped losses (TTFS order, VMax tallest-maximum) are classification losses. A
continuous target needs a different readout, and this repository ships one:
`eventprop_maxima_regression` computes

    y_hat = bias + Σ a[n] · maxima[n]

with least-squares or smooth Huber error, writes `dL/dmaxima` into the buffers
`eventprop_backward_maxima` reads, and returns `dL/da`, `dL/db`. Reached from Python as
`LILayer.regression(...)`, so a step is *forward maxima → regression → backward*, with no loss
arithmetic in Python at all. The C implementation is the single source of truth: there is no
Eigen or reference-library counterpart, because the reference library has no regression loss, so
parity for this loss is managed↔native in .NET (`RegressionLayerParityTests`) and nothing here
may be phrased as "agrees with the reference".

Two things this module exists to get right, because both are silent when wrong:

**Summed gradients.** The library returns gradients *summed* over the batch and deliberately
never divides — a binding that silently averaged half the time is how a learning rate ends up
coupled to a batch size. Everything here is a mean, and the division happens in exactly one
place (`_mean`), so the learning rates on the page mean what they say.

**The readout is badly conditioned as written.** Maxima of a trained population sit near the
same value for every neuron (measured on the published Yin-Yang net: mean 0.84, median
per-neuron spread across samples 0.07), so the feature matrix has condition number ~3e6 and
plain gradient descent on `a` either crawls or diverges — while the *model* is fine: exact least
squares recovers the stimulus coordinate to held-out R² = 0.999. The fix is a change of
parameterisation, not of model. Write z[n] = (maxima[n] − μ[n]) / s[n] and

    y_hat = b + Σ a[n]·z[n] = (b − Σ a[n]·μ[n]/s[n]) + Σ (a[n]/s[n])·maxima[n]

so the C entry point is handed `decoder = a/s` and `bias = b − Σ a·μ/s` and sees an ordinary
readout, while the fit is carried in the well-scaled coordinates. Both readings describe the
same function, which is what makes this legitimate rather than a trick — and `test_agrees_with_
the_closed_form` in `selfcheck.py` is what makes it checkable: a converged fit must land on the
least-squares solution.

Careful with the chain rule, because the bias is entangled: with d[n] = a[n]/s[n] and
β = b − Σ a·μ/s,

    dL/da[n] = ( dL/dd[n] − (dL/dβ)·μ[n] ) / s[n]        dL/db = dL/dβ

and `dL/dmaxima` — the thing the adjoint consumes — is `(dL/dy)·d[n]`, which the C entry point
already writes correctly once it is given `d`. Getting the `−(dL/dβ)·μ[n]` term wrong does not
crash anything; the fit just converges to a slightly wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends import get as get_backend

# The LI path never reads v_th, so this is the LIF time constants without the threshold —
# see `Backend.li_layer`: the C validator deliberately does not check v_th on entry points
# that cannot use it.
LI_PARAMS = dict(tau_mem=20e-3, tau_syn=5e-3)

#: Continuous quantities this page can regress onto, and which column of the stimulus
#: coordinates carries each. Both are in dataset units, so predictions are directly readable
#: against the plane every other page of this app draws.
TARGETS = {"x — horizontal position": 0, "y — vertical position": 1}


#: Ridge scales the page's baseline sweeps, coarse on purpose: the point is the shape of the
#: curve (interpolating fit → regularised fit), not a tuned penalty.
RIDGE_LAMBDAS = (0.0, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0)


def target_values(coords: np.ndarray, column: int) -> np.ndarray:
    """The regression target for a batch of 4-vectors, as float64 (the bindings demand it)."""
    return np.ascontiguousarray(np.asarray(coords, dtype=np.float64)[:, column])


@dataclass(frozen=True)
class Readout:
    """A fitted readout, carried in the well-scaled coordinates when standardised.

    `a`/`b` are the numbers the page displays and optimises. `effective()` is the pair handed to
    the C entry point; `predict_from_maxima` is the same function evaluated on raw maxima, with
    the non-finite ones treated the way the library treats them (excluded from the readout, so
    their contribution is zero).
    """

    a: np.ndarray
    b: float
    mu: np.ndarray
    sd: np.ndarray
    standardise: bool = True

    def effective(self) -> tuple[np.ndarray, float]:
        if not self.standardise:
            return self.a, float(self.b)
        scale = self.a / self.sd
        return scale, float(self.b) - float((self.a * self.mu / self.sd).sum())

    def predict_from_maxima(self, maxima: np.ndarray) -> np.ndarray:
        """The readout evaluated on raw maxima — the same function the C entry point computes.

        Non-finite maxima contribute zero, which is the library's rule for a neuron that never
        peaked; a NaN reaching a dot product would turn one silent neuron into a NaN prediction
        for the whole sample.
        """
        values = _readout_features(maxima)
        if self.standardise:
            values = (values - self.mu) / self.sd
        return self.b + values @ self.a

    def gradient_in_own_coordinates(self, dL_dd: np.ndarray, dL_dbias: float,
                                    batch: int) -> tuple[np.ndarray, float]:
        """Summed gradients for the *effective* parameters → mean gradients for `a` and `b`.

        The library hands back `dL/dd` and `dL/dβ` for the parameters it was given (`d = a/s`,
        `β = b − Σ a·μ/s`), summed over the batch. Converting them is the one place the chain
        rule lives:

            dL/da[n] = ( dL/dd[n] − (dL/dβ)·μ[n] ) / s[n]      dL/db = dL/dβ

        The `μ` term is not decoration: `β` depends on every `a[n]`, so leaving it out biases the
        fit toward the feature means. It is invisible when `μ = 0` (unstandardised), which is
        precisely how such a bug survives a test that only checks the raw case.
        """
        n = max(1, int(batch))
        mean_db = float(dL_dbias) / n
        if not self.standardise:
            return np.asarray(dL_dd, dtype=np.float64) / n, mean_db
        mean_da = (np.asarray(dL_dd, dtype=np.float64) / n - mean_db * self.mu) / self.sd
        return mean_da, mean_db

    def sgd_step(self, dL_dd: np.ndarray, dL_dbias: float, batch: int,
                 lr: float) -> "Readout":
        """Plain SGD on the mean gradient — kept because it is the readable form of the update.

        The page uses Adam (see `Adam`), but this is what the returned gradients *mean*, and it
        is the version a reader can check by hand.
        """
        mean_da, mean_db = self.gradient_in_own_coordinates(dL_dd, dL_dbias, batch)
        return Readout(a=self.a - lr * mean_da, b=float(self.b) - lr * mean_db,
                       mu=self.mu, sd=self.sd, standardise=self.standardise)


@dataclass
class FitResult:
    """Everything the page needs, and the numbers `selfcheck.py` asserts on."""

    readout: Readout
    losses: np.ndarray                       # mean loss per step
    eval_steps: np.ndarray                   # step index of each held-out evaluation
    heldout_r2: np.ndarray                   # held-out R² at those steps
    before: dict                             # held-out metrics before any step
    after: dict                              # held-out metrics after the last step
    eval_xy: np.ndarray                      # (n, 2) the held-out stimulus coordinates
    eval_target: np.ndarray                  # (n,) what the read-out was asked to predict
    eval_pred_before: np.ndarray             # (n,) predictions from the random initial read-out
    eval_pred_after: np.ndarray              # (n,) predictions from the fitted one
    skipped: dict                            # neurons excluded per sample: mean / max / total
    train_weights: bool
    w_final: np.ndarray | None               # only when the weights were trained
    gradient_scale: float                    # max |dL/dW| seen, so "nothing trained" is visible
    seconds: float
    steps: int
    batch: int
    lr_readout: float
    lr_weights: float
    huber_delta: float
    n_train: int
    n_eval: int
    divergence: bool = False
    notes: list = field(default_factory=list)


class Adam:
    """Per-coordinate adaptive steps, for the readout and the weights separately.

    Plain SGD does not work here and the reason is measurable rather than stylistic: the readout
    has 200 near-collinear features (condition number ~3e6), so a step size that makes progress
    along the informative directions overshoots the rest and the loss goes to infinity — on every
    initialisation tried, least-squares being the only exception because it has no step size.
    Adam normalises each coordinate by its own gradient history, which is the same fix as the
    standardisation applied at the optimiser instead of the parameterisation, and it is what the
    reference library's own trainers use. Keeping both is deliberate: standardisation makes the
    problem well-scaled, Adam makes it convergable at a learning rate a visitor can type.
    """

    def __init__(self, shape, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        g = np.asarray(grad, dtype=np.float64)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return np.asarray(x, dtype=np.float64) - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _readout_features(maxima: np.ndarray) -> np.ndarray:
    """Non-finite maxima contribute nothing.

    That is the library's own rule (a non-finite maximum means the neuron never peaked, so it is
    excluded and counted in `n_skipped`), and a NaN leaking into a numpy dot product would turn
    one silent neuron into a NaN prediction for the sample. Note the contrast the library draws
    and this page must not blur: a neuron *at rest* has a finite maximum of 0.0 and does count.
    """
    return np.nan_to_num(np.asarray(maxima, dtype=np.float64), nan=0.0,
                         posinf=0.0, neginf=0.0)


def layer_for(backend_key: str, w_in: np.ndarray):
    """An LI layer over the given input weights — the one whose maxima the readout reads."""
    be = get_backend(backend_key)
    return be.li_layer(n=np.asarray(w_in).shape[1], n_in=np.asarray(w_in).shape[0],
                       w_in=w_in, **LI_PARAMS)


def maxima_of(backend_key: str, w_in: np.ndarray, coords: np.ndarray,
              layer=None) -> tuple[np.ndarray, np.ndarray]:
    """Raw maxima `(n, N)` plus per-sample skip counts, from one forward pass.

    Returns `(maxima, skipped)`. `skipped` is counted here as non-finite maxima, which is the
    same condition the C loss uses internally — and it is stated rather than read back from the
    loss because the page wants it for *evaluation* too, where no loss is computed.
    """
    be = get_backend(backend_key)
    layer = layer or layer_for(backend_key, w_in)
    times, sources = _encode(coords)
    vector = be.make_vector(times, sources)
    batch = layer.forward(vector)
    n = len(batch)
    n_neurons = int(np.asarray(w_in).shape[1])
    out = np.empty((n, n_neurons), dtype=np.float64)
    for i in range(n):
        out[i] = np.asarray(batch[i].values, dtype=np.float64)
    return out, np.count_nonzero(~np.isfinite(out), axis=1)


def _encode(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TTFS encoding, shared with every other page so the spikes are the same spikes."""
    from .dataset import encode
    return encode(np.atleast_2d(np.asarray(coords, dtype=np.float64)))


def feature_stats(maxima: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron mean and spread over the training set.

    Standard deviation, not standard error, and neurons that never vary (s ≈ 0) are given s = 1
    so they divide by nothing rather than amplifying noise: a constant feature is absorbed by the
    bias, and the standardisation must not invent a direction with a huge coefficient in it.
    """
    values = _readout_features(maxima)
    mu = values.mean(axis=0)
    sd = values.std(axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    return mu, sd


def ridge_fit(train_maxima: np.ndarray, targets: np.ndarray, valid_maxima: np.ndarray,
              valid_targets: np.ndarray, lambdas: tuple[float, ...] = RIDGE_LAMBDAS
              ) -> tuple[Readout, dict]:
    """Ridge read-out with the penalty chosen on a **validation** split, never on the test set.

    Exists because "the best linear read-out" is not well defined until you say best *at what*.
    The exact solution is best in-sample and demonstrably worse held-out; this one is the best
    honest linear baseline, and the λ path is returned so the page can show the trade rather than
    assert a winner. Choosing λ on the split the metrics are reported from would be tuning on the
    test set, which is why `valid_*` is a separate argument and why the caller passes the dataset's
    own validation split rather than the test one.
    """
    values = _readout_features(train_maxima)
    mu, sd = feature_stats(values)
    design = (values - mu) / sd
    design1 = np.c_[design, np.ones(len(design))]
    y = np.asarray(targets, dtype=np.float64)
    pen = np.eye(design1.shape[1])
    pen[-1, -1] = 0.0                      # never penalise the intercept
    vdesign1 = np.c_[(_readout_features(valid_maxima) - mu) / sd, np.ones(len(valid_maxima))]
    vy = np.asarray(valid_targets, dtype=np.float64)

    path, best = [], None
    for lam in lambdas:
        coef = np.linalg.solve(design1.T @ design1 + lam * pen, design1.T @ y)
        pred = vdesign1 @ coef
        ss = float(((pred - vy) ** 2).sum())
        r2_valid = (1.0 - ss / float(((vy - vy.mean()) ** 2).sum())) if vy.size > 1 else float("nan")
        path.append((float(lam), r2_valid))
        if best is None or r2_valid > best[1]:
            best = (float(lam), r2_valid, coef)
    coef = best[2]
    readout = Readout(a=np.ascontiguousarray(coef[:-1]), b=float(coef[-1]), mu=mu, sd=sd,
                      standardise=True)
    return readout, {"lambda": best[0], "valid_r2": best[1],
                     "path": [(lam, r2) for lam, r2 in path]}


def probe(backend_key: str, w_in: np.ndarray, readout: Readout,
          coords: np.ndarray) -> dict:
    """One stimulus point, broken into the terms that make its prediction.

    `terms[n] = a[n] · z[n]`, so `bias + terms.sum()` is the prediction — the read-out is a
    weighted sum, which makes this the explanation rather than a decoration. Silent neurons
    contribute exactly 0, the same rule the C loss uses, so a bar that is absent is a neuron
    that never peaked rather than one with a small weight.
    """
    maxima, skipped = maxima_of(backend_key, w_in, np.atleast_2d(coords))
    values = _readout_features(maxima)
    if readout.standardise:
        values = (values - readout.mu) / readout.sd
    terms = np.asarray(readout.a * values[0], dtype=np.float64)
    return {"terms": terms, "prediction": float(readout.b + terms.sum()),
            "bias": float(readout.b), "skipped": int(skipped[0]),
            "maxima": np.asarray(maxima[0], dtype=np.float64)}


def metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """R², Pearson correlation and RMSE — computed here so before/after use one definition."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    out = {
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(((pred - target) ** 2).mean())),
        "corr": float("nan"),
    }
    if np.std(pred) > 0 and np.std(target) > 0:
        out["corr"] = float(np.corrcoef(pred, target)[0, 1])
    return out


def least_squares_fit(train_maxima: np.ndarray, targets: np.ndarray,
                      standardise: bool = True) -> Readout:
    """The exact least-squares read-out — which is *not* a generalisation ceiling.

    That sentence cost a wrong page to learn. At 400 training points and 201 parameters the
    design is full rank, so this fit interpolates: train R² = 1.0000, held-out 0.994 for x but
    0.961 for y. Ridge-regularised descent generalises better than it does, so a fitted read-out
    sitting *above* this line is behaving correctly rather than having beaten a bound.

    Still worth having: it has no learning rate and no optimiser, so it is the reference for
    "is the model class capable of this at all".
    """
    values = _readout_features(train_maxima)
    if standardise:
        mu, sd = feature_stats(values)
        design = (values - mu) / sd
    else:
        design, mu, sd = values, np.zeros(values.shape[1]), np.ones(values.shape[1])
    design1 = np.c_[design, np.ones(len(design))]
    coef, _, _, _ = np.linalg.lstsq(design1, np.asarray(targets, dtype=np.float64), rcond=None)
    return Readout(a=np.ascontiguousarray(coef[:-1]), b=float(coef[-1]), mu=mu, sd=sd,
                   standardise=standardise)


def fit(backend_key: str, w_in: np.ndarray, train_coords: np.ndarray, targets: np.ndarray,
        eval_coords: np.ndarray, eval_targets: np.ndarray, *, steps: int = 200,
        batch: int = 32, lr_readout: float = 2e-3, lr_weights: float = 1e-5,
        train_weights: bool = False, standardise: bool = True,
        huber_delta: float = np.inf, seed: int = 0, eval_every: int = 10,
        progress=None) -> FitResult:
    """Fit the readout (and optionally the layer's own weights) with plain SGD.

    `train_weights` is the part that exercises EventProp rather than a linear model: after
    `regression()` has written `dL/dmaxima`, `backward()` runs the time-reversed adjoint sweep
    and produces `dL/dW` — exact gradients through the *threshold-crossing-free* maxima path, so
    a continuous target trains the network and not only the readout.

    The initial readout is small and random, never zero: `dL/dmaxima = dL/dy · a`, so a zero
    readout gives the weights no gradient at all and the loss still falls through the bias, which
    looks exactly like training while the network learns nothing. Both bindings pin that property
    (see `Regression_AZeroReadoutLeavesTheWeightsNoGradient`); a demo that started from zeros
    would be demonstrating the bug rather than the fix.
    """
    import time

    be = get_backend(backend_key)
    w0 = np.ascontiguousarray(np.asarray(w_in, dtype=np.float64))
    n_neurons = w0.shape[1]
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    eval_targets = np.ascontiguousarray(eval_targets, dtype=np.float64)
    n_train = len(targets)
    batch = max(1, min(int(batch), n_train))

    layer = layer_for(backend_key, w0)
    train_maxima, _ = maxima_of(backend_key, w0, train_coords, layer=layer)
    mu, sd = feature_stats(train_maxima)
    rng = np.random.default_rng(seed)
    # 1/sqrt(N) on purpose. In standardised coordinates the prediction's spread is roughly
    # |a|·sqrt(N), so a "small" 0.05 per coordinate is actually ~0.7 of the target's own range:
    # a big first residual, big gradients, and a step size that has no stable range to live in.
    # Measured: 0.05 diverges from least-squares-quality features within 200 steps.
    readout = Readout(a=rng.normal(0.0, 0.05 / np.sqrt(n_neurons), size=n_neurons), b=0.0,
                      mu=mu, sd=sd, standardise=standardise)
    opt_a = Adam(n_neurons, lr=lr_readout)
    opt_b = Adam((), lr=lr_readout)
    opt_w = Adam(w0.shape, lr=lr_weights) if train_weights else None

    eval_maxima, eval_skipped = maxima_of(backend_key, w0, eval_coords)
    eval_xy = np.atleast_2d(np.asarray(eval_coords, dtype=np.float64))[:, :2].copy()
    pred_before = readout.predict_from_maxima(eval_maxima)
    before = metrics(pred_before, eval_targets)

    losses = np.empty(steps, dtype=np.float64)
    eval_steps: list[int] = []
    heldout: list[float] = []
    skip_total = 0
    skip_max = 0
    grad_scale = 0.0
    w_final: np.ndarray | None = None
    diverged = False
    t0 = time.perf_counter()

    for step in range(steps):
        lo = int(rng.integers(0, n_train)) if n_train > batch else 0
        idx = (lo + np.arange(batch)) % n_train          # wraps, so every step is a full batch
        bc, bt = np.atleast_2d(train_coords)[idx], targets[idx]

        times, sources = _encode(bc)
        layer.forward(be.make_vector(times, sources))

        d_eff, bias_eff = readout.effective()
        dg = np.zeros(n_neurons, dtype=np.float64)
        bg = np.zeros(1, dtype=np.float64)
        out = layer.regression(d_eff, bias_eff, bt, huber_delta,
                               decoder_grad=dg, bias_grad=bg)
        losses[step] = float(np.mean(np.asarray(out["losses"], dtype=np.float64)))
        skipped = np.asarray(out["n_skipped"], dtype=np.int64)
        skip_total += int(skipped.sum())
        skip_max = max(skip_max, int(skipped.max()) if skipped.size else 0)

        # The library returns SUMS. Everything downstream — Adam's moments, a learning rate a
        # visitor reads, the loss curve's scale — is meaningless until that has been undone.
        mean_da, mean_db = readout.gradient_in_own_coordinates(dg, float(bg[0]), len(bt))
        a_new = opt_a.step(readout.a, mean_da)
        b_new = float(opt_b.step(np.float64(readout.b), mean_db))
        readout = Readout(a=a_new, b=b_new, mu=readout.mu, sd=readout.sd,
                          standardise=readout.standardise)

        if train_weights:
            layer.backward()
            grad = np.asarray(layer.gradient, dtype=np.float64).reshape(w0.shape) / max(1, len(bt))
            grad_scale = max(grad_scale, float(np.abs(grad).max()))
            w0 = np.ascontiguousarray(opt_w.step(w0, grad))
            layer.w_in = w0
            w_final = w0.copy()

        if not np.isfinite(losses[step]):
            diverged = True
            losses[step:] = np.nan
            break

        if (step + 1) % eval_every == 0 or step == steps - 1:
            em, _ = maxima_of(backend_key, w0, eval_coords,
                              layer=(layer if train_weights else None))
            eval_maxima = em
            heldout.append(metrics(readout.predict_from_maxima(eval_maxima), eval_targets)["r2"])
            eval_steps.append(step + 1)
            if progress is not None:
                progress((step + 1) / steps)

    if train_weights:
        eval_maxima, eval_skipped = maxima_of(backend_key, w0, eval_coords)
    pred_after = readout.predict_from_maxima(eval_maxima)
    after = metrics(pred_after, eval_targets)

    if train_weights and w_final is not None:
        # Report skips from the final weights, not from the last minibatch of the last step:
        # "how much of this population carries no signal" is a property of the fitted network.
        _, skip_final = maxima_of(backend_key, w0, train_coords)
        skip_total = int(skip_final.sum())
        skip_max = int(skip_final.max()) if skip_final.size else 0

    return FitResult(
        readout=readout, losses=losses, eval_steps=np.asarray(eval_steps, dtype=np.int64),
        heldout_r2=np.asarray(heldout, dtype=np.float64), before=before, after=after,
        skipped={"mean_per_sample": skip_total / max(1, steps * batch), "max": skip_max,
                 "total": skip_total, "eval_mean_per_sample":
                     float(eval_skipped.mean()) if eval_skipped.size else 0.0},
        eval_xy=eval_xy, eval_target=eval_targets, eval_pred_before=pred_before,
        eval_pred_after=pred_after,
        train_weights=bool(train_weights), w_final=w_final, gradient_scale=grad_scale,
        seconds=time.perf_counter() - t0, steps=steps, batch=batch, lr_readout=lr_readout,
        lr_weights=lr_weights, huber_delta=huber_delta, n_train=n_train, n_eval=len(eval_targets),
        divergence=diverged,
    )
