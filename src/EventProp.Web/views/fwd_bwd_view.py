"""Forward / Backward inspection view UI."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config import CFG, LIF, TRAIN, CyLIFParams, CyLIFLayer, CyTTFS, CyTTFSParams, PyLIFParams, PyLIFLayer, PyTTFS, PyTTFSParams, CySpikes, CySpikesVector, PySpikes, PySpikesVector
from plots import raster, raster_compare, fst_compare, grad_scatter, grad_divergence


def _enc_one_cy(samp):
    t = np.array([v*30e-3 for v in list(samp)+[0.0]], dtype=np.float64)
    s = np.arange(len(samp)+1, dtype=np.int32); i = np.argsort(t)
    return CySpikesVector([CySpikes(t[i],s[i])])


def _enc_one_py(samp):
    t = np.array([v*30e-3 for v in list(samp)+[0.0]], dtype=np.float64).reshape(-1, 1)
    s = np.arange(len(samp)+1, dtype=np.int32).reshape(-1, 1); i = np.argsort(t.ravel())
    return PySpikesVector([PySpikes(t[i],s[i])])


def _run_fwd(inp_cy, inp_py, wh, wo):
    lp_cy = CyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
    lo_cy = CyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
    hl_cy = CyLIFLayer(lp_cy, w_in=wh.copy())
    ol_cy = CyTTFS(CyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_cy), w_in=wo.copy())
    ho_cy = hl_cy.forward(inp_cy); ol_cy.forward(ho_cy)

    lp_py = PyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF)
    lo_py = PyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF)
    hl_py = PyLIFLayer(lp_py, w_in=wh.copy())
    ol_py = PyTTFS(PyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_py), w_in=wo.copy())
    ho_py = hl_py.forward(inp_py); ol_py.forward(ho_py)

    return hl_cy, ol_cy, ho_cy, hl_py, ol_py, ho_py


def _run_bwd(hl_cy, ol_cy, hl_py, ol_py, labels):
    ol_cy.backward(labels); hl_cy.backward()
    ol_py.backward(labels); hl_py.backward()
    return hl_cy.gradient, ol_cy.gradient, hl_py.gradient, ol_py.gradient


def _ttfs_pred(oo_spikes, n_out):
    pt = np.full(n_out, 1e9)
    for t,s in zip(oo_spikes.times, oo_spikes.sources): pt[int(s)] = float(t)
    return int(np.argmin(pt))


def render(checkpoint_idx, sample_idx):
    import pickle as pkl
    ckpt_path = CFG["ckpt_dir"] / f"yinyang_{checkpoint_idx}.pkl"
    with open(ckpt_path, "rb") as f: ckpt = pkl.load(f)
    wh, wo = np.array(ckpt[8][-1][0]), np.array(ckpt[8][-1][1])

    te_s = np.load(CFG["data_dir"]/"test_samples.npy")[:200]
    te_l = np.load(CFG["data_dir"]/"test_labels.npy")[:200]
    samp, lbl = te_s[sample_idx], te_l[sample_idx]

    inp_cy_v = _enc_one_cy(samp); inp_py_v = _enc_one_py(samp)
    hl_cy, ol_cy, ho_cy, hl_py, ol_py, ho_py = _run_fwd(inp_cy_v, inp_py_v, wh, wo)

    pred_cy = _ttfs_pred(ol_cy.post_batch[0], CFG["n_out"])
    pred_py = _ttfs_pred(ol_py.post_batch[0], CFG["n_out"])

    st.subheader("Forward Pass")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(raster(inp_cy_v[0], f"Input Spikes (label={lbl})", CFG["t_max_ms"]), use_container_width=True)
    with c2:
        n_fc = len(set(map(int,ho_cy[0].sources))); n_fp = len(set(map(int,ho_py[0].sources)))
        st.plotly_chart(raster_compare(ho_cy[0], ho_py[0], f"Hidden (Cy:{n_fc}, Py:{n_fp})", CFG["t_max_ms"]), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        ok = "OK" if pred_cy==lbl else "WRONG"
        st.plotly_chart(raster(ol_cy.post_batch[0], f"Cython Output (pred={pred_cy} {ok})", CFG["t_max_ms"]), use_container_width=True)
    with c2:
        ok = "OK" if pred_py==lbl else "WRONG"
        st.plotly_chart(raster(ol_py.post_batch[0], f"Pybind11 Output (pred={pred_py} {ok})", CFG["t_max_ms"]), use_container_width=True)

    st.subheader("First Spike Times")
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(fst_compare(hl_cy, hl_py, "Hidden Layer First Spike Times"), use_container_width=True)
    with c2: st.plotly_chart(fst_compare(ol_cy, ol_py, "Output Layer First Spike Times"), use_container_width=True)

    # Backward pass
    st.subheader("Backward Pass (Gradients)")
    g_hl_cy, g_ol_cy, g_hl_py, g_ol_py = _run_bwd(hl_cy, ol_cy, hl_py, ol_py, [lbl])

    cy_o_max = np.max(np.abs(g_ol_cy), axis=0)
    py_o_max = np.max(np.abs(g_ol_py), axis=0)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(grad_scatter(cy_o_max, py_o_max, "Output Layer Gradients"), use_container_width=True)
    with c2: st.plotly_chart(grad_divergence(cy_o_max, py_o_max, "Output Gradient Divergence (Top 30)"), use_container_width=True)

    cy_h_max = np.max(np.abs(g_hl_cy), axis=0)
    py_h_max = np.max(np.abs(g_hl_py), axis=0)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(grad_scatter(cy_h_max, py_h_max, "Hidden Layer Gradients"), use_container_width=True)
    with c2: st.plotly_chart(grad_divergence(cy_h_max, py_h_max, "Hidden Gradient Divergence (Top 30)"), use_container_width=True)

    diff_o = np.abs(g_ol_cy - g_ol_py); diff_h = np.abs(g_hl_cy - g_hl_py)
    c1, c2 = st.columns(2)
    with c1: st.metric("Output max |grad diff|", f"{diff_o.max():.6f}")
    with c2: st.metric("Hidden max |grad diff|", f"{diff_h.max():.6f}")
