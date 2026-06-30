# EventProp Visualizer

Streamlit app for exploring SNN forward/backward passes with interactive visualizations.

## Setup

```bash
cd src/EventProp.Web
"C:/Program Files/Python311/python.exe" -m pip install -r requirements.txt
```

## Run

```bash
"C:/Program Files/Python311/python.exe" -m streamlit run streamlit_app.py
```

## Features

- **Raster plots**: Spike times vs neuron index for input, hidden, and output layers
- **First spike times**: Bar chart showing which neurons fire and when
- **Weight matrices**: Heatmap visualization of learned weights
- **Backward pass**: Gradient heatmaps for TTFS loss
- **Backend comparison**: Toggle between Cython (C library) and Pybind11 (Eigen)
