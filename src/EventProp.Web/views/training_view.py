"""Training Comparison view UI."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config import CFG, TRAIN, CyLIFParams, CyLIFLayer, CyTTFS, CyTTFSParams, PyLIFParams, PyLIFLayer, PyTTFS, PyTTFSParams, CySpikes, CySpikesVector, PySpikes, PySpikesVector
from plots import heatmap, raster, raster_compare

LIF_KW = dict(tau_mem=0.02, tau_syn=0.005, v_th=1.0)


def render(seed, n_epochs):
    from training import run_training
    cy_a, py_a, cy_l, py_l, wh_cy_f, wo_cy_f, wh_py_f, wo_py_f = run_training(seed, n_epochs)

    st.subheader("Cython vs Pybind11 - Side-by-Side Training")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1,len(cy_a)+1),y=cy_a,mode="lines",line=dict(color="#22c55e",width=3),name="Cython (C)"))
        fig.add_trace(go.Scatter(x=np.arange(1,len(py_a)+1),y=py_a,mode="lines",line=dict(color="#ef4444",width=3),name="Pybind11 (Eigen)"))
        title = f"Train Accuracy (seed={seed})<br>Final: Cy={cy_a[-1]:.2%} vs Py={py_a[-1]:.2%}"
        fig.update_layout(title=title,xaxis_title="Epoch",yaxis_title="Accuracy",height=350,margin=dict(l=65,r=20,t=70,b=50),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1,len(cy_l)+1),y=cy_l,mode="lines",line=dict(color="#22c55e",width=3),name="Cython (C)"))
        fig.add_trace(go.Scatter(x=np.arange(1,len(py_l)+1),y=py_l,mode="lines",line=dict(color="#ef4444",width=3),name="Pybind11 (Eigen)"))
        title = f"Train Loss (seed={seed})<br>Final: Cy={cy_l[-1]:.4f} vs Py={py_l[-1]:.4f}"
        fig.update_layout(title=title,xaxis_title="Epoch",yaxis_title="Loss",height=350,margin=dict(l=65,r=20,t=70,b=50),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    acc_diff = np.abs(cy_a - py_a)*100; loss_diff = np.abs(cy_l - py_l)
    st.subheader("Divergence Between Backends")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Max accuracy gap", f"{acc_diff.max():.2f}%")
    with c2: st.metric("Final accuracy gap", f"{acc_diff[-1]:.2f}%")
    with c3: st.metric("Final loss gap", f"{loss_diff[-1]:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(1,len(acc_diff)+1),y=acc_diff,marker_color="#f59e0b"))
    fig.update_layout(title="Accuracy Divergence Per Epoch (%)",xaxis_title="Epoch",yaxis_title="|Difference|%",height=250,margin=dict(l=65,r=20,t=45,b=50),showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Weight Comparison")
    w_diff_h = np.abs(wh_cy_f - wh_py_f); w_diff_o = np.abs(wo_cy_f - wo_py_f)
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(heatmap(wh_cy_f, "Hidden Weights (Cython)"), use_container_width=True)
    with c2: st.plotly_chart(heatmap(wh_py_f, "Hidden Weights (Pybind11)"), use_container_width=True)
    with c3: st.plotly_chart(heatmap(w_diff_h, f"|Weight Diff| hidden\nmax={w_diff_h.max():.4f}"), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(heatmap(wo_cy_f, "Output Weights (Cython)"), use_container_width=True)
    with c2: st.plotly_chart(heatmap(wo_py_f, "Output Weights (Pybind11)"), use_container_width=True)
    with c3: st.plotly_chart(heatmap(w_diff_o, f"|Weight Diff| output\nmax={w_diff_o.max():.4f}"), use_container_width=True)

    _show_trained_model_io(wh_cy_f, wo_cy_f, wh_py_f, wo_py_f)


def _enc_one_cy(samp):
    t = np.array([v*30e-3 for v in list(samp)+[0.0]], dtype=np.float64)
    s = np.arange(len(samp)+1, dtype=np.int32); i = np.argsort(t)
    return CySpikesVector([CySpikes(t[i],s[i])])

def _enc_one_py(samp):
    t = np.array([v*30e-3 for v in list(samp)+[0.0]], dtype=np.float64).reshape(-1, 1)
    s = np.arange(len(samp)+1, dtype=np.int32).reshape(-1, 1); i = np.argsort(t.ravel())
    return PySpikesVector([PySpikes(t[i],s[i])])

def _ttfs_pred(oo_spikes, n_out):
    pt = np.full(n_out, 1e9)
    for t,s in zip(oo_spikes.times, oo_spikes.sources): pt[int(s)] = float(t)
    return int(np.argmin(pt))

def _show_trained_model_io(wh_cy_f, wo_cy_f, wh_py_f, wo_py_f):
    st.subheader("Trained Model Input / Output")
    te_s = np.load(CFG["data_dir"]/"test_samples.npy")[:200]
    te_l = np.load(CFG["data_dir"]/"test_labels.npy")[:200]
    test_idx = st.slider("Test sample", 0, len(te_s)-1, 0)
    samp = te_s[test_idx]; lbl = te_l[test_idx]
    inp_cy_v = _enc_one_cy(samp); inp_py_v = _enc_one_py(samp)

    lp_cy2 = CyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF_KW)
    lo_cy2 = CyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF_KW)
    hl_cy2 = CyLIFLayer(lp_cy2, w_in=wh_cy_f)
    ol_cy2 = CyTTFS(CyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_cy2), w_in=wo_cy_f)
    ho_cy_v = hl_cy2.forward(inp_cy_v); ol_cy2.forward(ho_cy_v)

    lp_py2 = PyLIFParams(n=CFG["n_hidden"], n_in=CFG["n_in"], **LIF_KW)
    lo_py2 = PyLIFParams(n=CFG["n_out"], n_in=CFG["n_hidden"], **LIF_KW)
    hl_py2 = PyLIFLayer(lp_py2, w_in=wh_py_f)
    ol_py2 = PyTTFS(PyTTFSParams(alpha=TRAIN["alpha"], tau0=TRAIN["tau0"], tau1=TRAIN["tau1"], lif_parameters=lo_py2), w_in=wo_py_f)
    ho_py_v = hl_py2.forward(inp_py_v); ol_py2.forward(ho_py_v)

    pred_cy = _ttfs_pred(ol_cy2.post_batch[0], CFG["n_out"])
    pred_py = _ttfs_pred(ol_py2.post_batch[0], CFG["n_out"])

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(raster(inp_cy_v[0], f"Input Spikes (label={lbl})", CFG["t_max_ms"]), use_container_width=True)
    with c2:
        n_fc = len(set(map(int,ho_cy_v[0].sources))); n_fp = len(set(map(int,ho_py_v[0].sources)))
        st.plotly_chart(raster_compare(ho_cy_v[0], ho_py_v[0], f"Hidden (Cy:{n_fc}, Py:{n_fp})", CFG["t_max_ms"]), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        ts = np.array(ol_cy2.post_batch[0].times)*1e3; sr = np.array(ol_cy2.post_batch[0].sources)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts,y=sr,mode="markers",marker=dict(size=8,color="#22c55e")))
        ok = "OK" if pred_cy==lbl else "WRONG"
        fig.update_layout(title=f"Output (Cython) pred={pred_cy} {ok}",xaxis_title="Time (ms)",yaxis_title="Neuron",height=180,margin=dict(l=55,r=15,t=35,b=35),xaxis=dict(range=[0,CFG["t_max_ms"]]),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ts = np.array(ol_py2.post_batch[0].times)*1e3; sr = np.array(ol_py2.post_batch[0].sources)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts,y=sr,mode="markers",marker=dict(size=8,color="#ef4444")))
        ok = "OK" if pred_py==lbl else "WRONG"
        fig.update_layout(title=f"Output (Pybind11) pred={pred_py} {ok}",xaxis_title="Time (ms)",yaxis_title="Neuron",height=180,margin=dict(l=55,r=15,t=35,b=35),xaxis=dict(range=[0,CFG["t_max_ms"]]),yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    def fst_arr(spikes, n):
        ft = np.full(n, np.nan)
        for t,s in zip(spikes.times, spikes.sources):
            i = int(s)
            if np.isnan(ft[i]) or float(t)<ft[i]: ft[i] = float(t)*1e3
        return ft

    fst_cy = fst_arr(ol_cy2.post_batch[0], CFG["n_out"])
    fst_py = fst_arr(ol_py2.post_batch[0], CFG["n_out"])
    fig = go.Figure()
    x = np.arange(CFG["n_out"])
    fig.add_trace(go.Bar(name="Cython",x=x,y=fst_cy,marker_color="#22c55e",opacity=0.7))
    fig.add_trace(go.Bar(name="Pybind11",x=x,y=fst_py,marker_color="#ef4444",opacity=0.7))
    fig.update_layout(title="Output First Spike Times (ms)",xaxis_title="Neuron",yaxis_title="Time (ms)",height=250,barmode="group")
    st.plotly_chart(fig, use_container_width=True)
