"""Streamlit-cached access to data, checkpoints and built networks.

Every loader that touches disk or builds a native object is cached, and this is not a
performance nicety: the previous version of this app retrained a two-layer network from
scratch inside a view on *every* widget interaction, which made the page unusable and
hid the fact that ten finished, published networks were sitting one directory away.

`Net` holds pybind11/Cython objects, so it is cached with `cache_resource` (identity
semantics, never pickled). Arrays go through `cache_data`.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from eventweb import analysis, dataset, mnist, network, regression, runtime
from eventweb.backends import BACKENDS

ROOT = runtime.ROOT
DATA_DIR = ROOT / "src/EventProp.Python/eventprop/yin_yang_data_set/publication_data"
CKPT_DIR = ROOT / "src/EventProp.Python/paper"

# Published reference accuracies for the SAME task, from the dataset README
# (lkriener/yin_yang_data_set) — the shallow-vs-deep gap the dataset was designed to
# expose. Quoted as published, not measured here.
PUBLISHED = {"shallow": 0.643, "deep_ann": 0.987}


@st.cache_resource(show_spinner=False)
def backend(key: str):
    return BACKENDS[key]


@st.cache_resource(show_spinner=False)
def split(name: str) -> dataset.Split:
    return dataset.load_split(DATA_DIR, name)


@st.cache_resource(show_spinner=False)
def checkpoint(seed: int) -> network.Checkpoint:
    return network.load_checkpoint(CKPT_DIR, seed)


@st.cache_resource(show_spinner=False)
def net(backend_key: str, seed: int, snapshot: int) -> network.Net:
    return network.Net(backend(backend_key), *checkpoint(seed).weights_at(snapshot))


@st.cache_data(show_spinner=False)
def true_grid(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth class field on the same grid, straight from the geometry."""
    axis = np.linspace(0.0, 1.0, resolution)
    xg, yg = np.meshgrid(axis, axis)
    return dataset.which_class(xg, yg), axis


@st.cache_data(show_spinner="Evaluating the network over the plane…")
def boundary(backend_key: str, seed: int, snapshot: int, resolution: int):
    return analysis.boundary_grid(net(backend_key, seed, snapshot), resolution=resolution)


@st.cache_data(show_spinner="Running the test set…")
def evaluated(backend_key: str, seed: int, snapshot: int, split_name: str):
    return analysis.predict_split(net(backend_key, seed, snapshot), split(split_name))


@st.cache_data(show_spinner="Measuring both backends…")
def timings(split_name: str) -> dict:
    return {k: analysis.backend_timings(BACKENDS[k], split(split_name)) for k in BACKENDS}


@st.cache_data(show_spinner="Comparing the backends…")
def backend_agreement(seed: int, snapshot: int, split_name: str, n: int = 300) -> dict:
    """Predictions from both bindings on the same samples — parity, measured here."""
    te = split(split_name).coords[:n]
    a = net("c", seed, snapshot).predict(te)[0]
    b = net("eigen", seed, snapshot).predict(te)[0]
    return {"n": int(n), "agree": float((a == b).mean()), "disagreements": int((a != b).sum())}


@st.cache_data(show_spinner="Computing the adjoint…")
def gradient_demo(backend_key: str, coords_key: tuple[float, ...], seed: int):
    return analysis.gradient_demo(backend(backend_key), np.asarray(coords_key), seed=seed)


# --- Digits (MNIST / VMax) ------------------------------------------------------------


def mnist_ready() -> bool:
    """Whether the test set is on disk. Not cached: it is a stat call, and the answer
    changes the moment the visitor clicks the download button."""
    return mnist.have_test_set()


@st.cache_data(show_spinner="Fetching the MNIST test set (~1.6 MB)…")
def mnist_test() -> tuple[np.ndarray, np.ndarray]:
    mnist.fetch_test_set()
    return mnist.load_test()


@st.cache_resource(show_spinner=False)
def mnist_run(seed: int) -> network.Checkpoint:
    """A published VMax run. Final weights only — these files hold one snapshot."""
    return network.load_checkpoint(CKPT_DIR, seed, prefix="mnist_vmax")


@st.cache_data(show_spinner="Running 10 000 test digits…")
def mnist_evaluated(backend_key: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(predictions, margins, labels)` for the whole test set, from the given backend."""
    images, labels = mnist_test()
    wh, wo = mnist_run(seed).weights_at(-1)
    pred, margin = mnist.predict(backend_key, wh, wo, images)
    return pred, margin, labels


@st.cache_data(show_spinner="Reading out the maxima…")
def mnist_maxima(backend_key: str, seed: int, index: int) -> np.ndarray:
    """The ten output maxima behind one digit's prediction."""
    images, _ = mnist_test()
    wh, wo = mnist_run(seed).weights_at(-1)
    values, _ = mnist.forward_maxima(backend_key, wh, wo, images[index:index + 1])
    return values[0]


@st.cache_data(show_spinner="Watching the hidden layer…")
def mnist_hidden(backend_key: str, seed: int, index: int) -> tuple[np.ndarray, np.ndarray]:
    images, _ = mnist_test()
    wh, _ = mnist_run(seed).weights_at(-1)
    return mnist.hidden_spikes(backend_key, wh, images[index])


@st.cache_data(show_spinner="Timing both backends…")
def mnist_timings(seed: int, n_digits: int = 2000, batch: int = 250,
                  repeats: int = 3) -> dict[str, dict]:
    """Median maxima-forward time per batch, for every backend that can do maxima.

    Cached on the parameters that define the measurement — sample count, batch size,
    repetitions — so changing any of them re-measures rather than serving a stale median.
    A timing cache keyed on nothing is how a page ends up quoting a figure from a run the
    visitor never asked for.
    """
    images, _ = mnist_test()
    wh, wo = mnist_run(seed).weights_at(-1)
    return {k: mnist.time_forward(k, wh, wo, images[:n_digits], batch=batch, repeats=repeats)
            for k in BACKENDS if BACKENDS[k].supports_maxima()}


# --- Continuous read-out (regression) --------------------------------------------------
#
# Pinned to the C backend on purpose. The regression read-out exists only in the C library —
# the read-only reference library ships no regression loss — so a backend picker here would
# offer one option and imply a cross-backend agreement nobody can measure. The parity guarantee
# for this loss is managed ↔ native in .NET; see `eventweb/regression.py`.

#: Training/held-out point counts for the read-out page. Fixed rather than widgets because the
#: closed-form ceiling is computed on the same training points, and the two figures are only
#: comparable when the split is the same.
READOUT_TRAIN_N = 400
READOUT_EVAL_N = 256


@st.cache_data(show_spinner="Fitting the read-out…")
def readout_fit(seed: int, column: int, *, steps: int = 600, batch: int = 64,
                lr_readout: float = 1e-2, lr_weights: float = 1e-5,
                train_weights: bool = False, standardise: bool = True,
                huber_delta: float = float("inf"),
                n_train: int = READOUT_TRAIN_N, n_eval: int = READOUT_EVAL_N
                ) -> regression.FitResult:
    """Fit the continuous read-out on one published network, from the C library.

    Cached on every parameter that changes the answer — including the learning rates, which a
    "just the seed" cache would happily ignore while serving a fit from a different run.
    """
    wh, _ = checkpoint(seed).weights_at(-1)
    tr = split("train").coords[:n_train]
    te = split("test").coords[:n_eval]
    return regression.fit("c", wh, tr, regression.target_values(tr, column),
                          te, regression.target_values(te, column),
                          steps=steps, batch=batch, lr_readout=lr_readout,
                          lr_weights=lr_weights, train_weights=train_weights,
                          standardise=standardise, huber_delta=huber_delta, seed=seed)


@st.cache_data(show_spinner="Fitting the linear baselines…")
def readout_baseline(seed: int, column: int, n_train: int = READOUT_TRAIN_N,
                     n_valid: int = READOUT_EVAL_N, n_eval: int = READOUT_EVAL_N) -> dict:
    """The two closed-form reference read-outs: exact least squares, and ridge with λ chosen
    on the dataset's own validation split.

    Both are reported because the difference between them is the lesson. Exact least squares
    interpolates (400 points, 201 parameters, train R² = 1.0) and generalises worse than a
    regularised fit; ridge with λ picked on validation is the honest best linear baseline. A
    fitted read-out above the exact line is behaving correctly, and without both numbers here the
    page would have called that a bug.
    """
    wh, _ = checkpoint(seed).weights_at(-1)
    tr = split("train").coords[:n_train]
    va = split("valid").coords[:n_valid]
    te = split("test").coords[:n_eval]
    ytr = regression.target_values(tr, column)
    yva = regression.target_values(va, column)
    yte = regression.target_values(te, column)
    m_tr, _ = regression.maxima_of("c", wh, tr)
    m_va, _ = regression.maxima_of("c", wh, va)
    m_te, _ = regression.maxima_of("c", wh, te)

    exact = regression.least_squares_fit(m_tr, ytr, standardise=True)
    ridge, info = regression.ridge_fit(m_tr, ytr, m_va, yva)
    exact_test = regression.metrics(exact.predict_from_maxima(m_te), yte)
    ridge_test = regression.metrics(ridge.predict_from_maxima(m_te), yte)
    exact_train = regression.metrics(exact.predict_from_maxima(m_tr), ytr)
    return {
        "exact_r2": exact_test["r2"], "exact_rmse": exact_test["rmse"],
        "exact_train_r2": exact_train["r2"],
        "ridge_r2": ridge_test["r2"], "ridge_rmse": ridge_test["rmse"],
        "ridge_lambda": info["lambda"], "ridge_valid_r2": info["valid_r2"],
        "ridge_path": info["path"],
        "condition": float(np.linalg.cond(m_tr - np.nan_to_num(m_tr).mean(0))),
        "n_params": m_tr.shape[1] + 1, "n_train": n_train,
    }
