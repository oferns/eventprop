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

from eventweb import analysis, dataset, network, runtime
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
