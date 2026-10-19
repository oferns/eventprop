"""Benchmark runner for Cython vs Pybind11 backends."""
import time
import numpy as np
from config import (CFG, LIF, TRAIN, CySpikes, CySpikesVector, CyLIFParams, CyLIFLayer, CyTTFS, CyTTFSParams,
                    PySpikes, PySpikesVector, PyLIFParams, PyLIFLayer, PyTTFS, PyTTFSParams)


def _enc_cy_batch(samples):
    sl = []
    for x in samples:
        t = np.array([v*30e-3 for v in list(x)+[0.0]], dtype=np.float64)
        s = np.arange(len(x)+1, dtype=np.int32); i = np.argsort(t)
        sl.append(CySpikes(t[i], s[i]))
    return CySpikesVector(sl)


def _enc_py_batch(samples):
    sl = []
    for x in samples:
        t = np.array([v*30e-3 for v in list(x)+[0.0]], dtype=np.float64).reshape(-1, 1)
        s = np.arange(len(x)+1, dtype=np.int32).reshape(-1, 1); i = np.argsort(t.ravel())
        sl.append(PySpikes(t[i], s[i]))
    return PySpikesVector(sl)


def run_benchmark(n_iters, batch_size):
    """Run forward/backward/full-step benchmarks.

    Returns dict with median times in ms for each phase/backend.
    """
    tr_s = np.load(CFG["data_dir"] / "train_samples.npy")[:batch_size]
    tr_l = np.load(CFG["data_dir"] / "train_labels.npy")[:batch_size]

    inp_cy_b = _enc_cy_batch(tr_s)
    inp_py_b = _enc_py_batch(tr_s)

    np.random.seed(42)
    wh_b = np.random.normal(TRAIN["h_mean"], TRAIN["h_std"], (CFG["n_in"], CFG["n_hidden"]))
    wo_b = np.random.normal(TRAIN["o_mean"], TRAIN["o_std"], (CFG["n_hidden"], CFG["n_out"]))

    cy_fwd, py_fwd = [], []
    for _ in range(n_iters):
        lp_cy = CyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
        lo_cy = CyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
        hl_cy = CyLIFLayer(lp_cy, w_in=wh_b.copy())
        ol_cy = CyTTFS(CyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_cy), w_in=wo_b.copy())
        t0 = time.perf_counter()
        ho_cy = hl_cy.forward(inp_cy_b); ol_cy.forward(ho_cy)
        cy_fwd.append(time.perf_counter()-t0)

        lp_py = PyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
        lo_py = PyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
        hl_py = PyLIFLayer(lp_py, w_in=wh_b.copy())
        ol_py = PyTTFS(PyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_py), w_in=wo_b.copy())
        t0 = time.perf_counter()
        ho_py = hl_py.forward(inp_py_b); ol_py.forward(ho_py)
        py_fwd.append(time.perf_counter()-t0)

    cy_bwd, py_bwd = [], []
    for _ in range(n_iters):
        lp_cy = CyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
        lo_cy = CyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
        hl_cy = CyLIFLayer(lp_cy, w_in=wh_b.copy())
        ol_cy = CyTTFS(CyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_cy), w_in=wo_b.copy())
        ho_cy = hl_cy.forward(inp_cy_b); ol_cy.forward(ho_cy)
        t0 = time.perf_counter()
        ol_cy.backward(tr_l); hl_cy.backward()
        cy_bwd.append(time.perf_counter()-t0)

        lp_py = PyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
        lo_py = PyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
        hl_py = PyLIFLayer(lp_py, w_in=wh_b.copy())
        ol_py = PyTTFS(PyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_py), w_in=wo_b.copy())
        ho_py = hl_py.forward(inp_py_b); ol_py.forward(ho_py)
        t0 = time.perf_counter()
        ol_py.backward(tr_l); hl_py.backward()
        py_bwd.append(time.perf_counter()-t0)

    cy_full, py_full = [], []
    for _ in range(n_iters):
        wh_c, wo_c = wh_b.copy(), wo_b.copy()
        lp_cy = CyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
        lo_cy = CyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
        hl_cy = CyLIFLayer(lp_cy, w_in=wh_c)
        ol_cy = CyTTFS(CyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_cy), w_in=wo_c)
        t0 = time.perf_counter()
        ho_cy = hl_cy.forward(inp_cy_b); ol_cy.forward(ho_cy)
        ol_cy.backward(tr_l); hl_cy.backward()
        wh_c -= TRAIN["lr"]*hl_cy.gradient; wo_c -= TRAIN["lr"]*ol_cy.gradient
        cy_full.append(time.perf_counter()-t0)

        wh_p, wo_p = wh_b.copy(), wo_b.copy()
        lp_py = PyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
        lo_py = PyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
        hl_py = PyLIFLayer(lp_py, w_in=wh_p)
        ol_py = PyTTFS(PyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_py), w_in=wo_p)
        t0 = time.perf_counter()
        ho_py = hl_py.forward(inp_py_b); ol_py.forward(ho_py)
        ol_py.backward(tr_l); hl_py.backward()
        wh_p -= TRAIN["lr"]*hl_py.gradient; wo_p -= TRAIN["lr"]*ol_py.gradient
        py_full.append(time.perf_counter()-t0)

    return dict(
        cy_fwd=np.median(cy_fwd)*1e3, py_fwd=np.median(py_fwd)*1e3,
        cy_bwd=np.median(cy_bwd)*1e3, py_bwd=np.median(py_bwd)*1e3,
        cy_full=np.median(cy_full)*1e3, py_full=np.median(py_full)*1e3,
    )

