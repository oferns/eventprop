"""Benchmark view UI."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from benchmark import run_benchmark


def render(n_iters, batch_size):
    st.subheader("Cython (C) vs Pybind11 (Eigen) - Performance")

    with st.spinner(f"Running {n_iters} iterations x {batch_size} samples..."):
        results = run_benchmark(n_iters, batch_size)

    phases = ["Forward", "Backward", "Full Step"]
    cy_times = [results["cy_fwd"], results["cy_bwd"], results["cy_full"]]
    py_times = [results["py_fwd"], results["py_bwd"], results["py_full"]]

    fig = go.Figure()
    x = np.arange(len(phases))
    fig.add_trace(go.Bar(name="Cython (C)",x=x,y=cy_times,marker_color="#22c55e",opacity=0.8))
    fig.add_trace(go.Bar(name="Pybind11 (Eigen)",x=x,y=py_times,marker_color="#ef4444",opacity=0.8))
    fig.update_layout(title=f"Median Time (ms) - {n_iters} iters x {batch_size} samples",xaxis=dict(tickvals=x,ticktext=phases),yaxis_title="Time (ms)",height=350,barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    speedups = [py/cy if cy>0 else 0 for py,cy in zip(py_times, cy_times)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=phases,y=speedups,marker_color=["#22c55e" if s>=1 else "#ef4444" for s in speedups]))
    fig.update_layout(title="Speedup Ratio (Pybind11 / Cython) - >1 = Pybind11 faster",yaxis_title="Ratio",height=250)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Forward speedup", f"{speedups[0]:.2f}x")
    with c2: st.metric("Backward speedup", f"{speedups[1]:.2f}x")
    with c3: st.metric("Full Step speedup", f"{speedups[2]:.2f}x")
