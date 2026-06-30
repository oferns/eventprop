"""EventProp Visualizer - Streamlit entry point."""
import streamlit as st
from config import CFG, load_all_checkpoints

st.set_page_config(page_title="EventProp Visualizer", layout="wide")
st.title("EventProp: Cython (C) vs Pybind11 (Eigen)")

# Sidebar controls
st.sidebar.header("Controls")
tab = st.sidebar.radio("View", ["Training Comparison", "Benchmark", "Forward/Backward"])

checkpoint_idx = st.sidebar.slider("Checkpoint Seed", 0, CFG["n_seeds"]-1, 0)
sample_idx = st.sidebar.slider("Sample Index", 0, 199, 0)

if tab == "Training Comparison":
    n_epochs = st.sidebar.slider("Epochs", 1, 200, 50)
    from views.training_view import render as training_render
    training_render(checkpoint_idx, n_epochs)

elif tab == "Benchmark":
    n_iters = st.sidebar.slider("Iterations", 5, 50, 20)
    batch_size = st.sidebar.slider("Batch Size", 4, 64, value=32, step=4)
    from views.benchmark_view import render as benchmark_render
    benchmark_render(n_iters, batch_size)

elif tab == "Forward/Backward":
    from views.fwd_bwd_view import render as fwd_bwd_render
    fwd_bwd_render(checkpoint_idx, sample_idx)
