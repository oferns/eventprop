"""Plotly figure builders. Pure functions of arrays -> `go.Figure`.

Kept free of `streamlit` and of any data loading: a view gathers numbers from
`eventweb.analysis` and hands them here, which is what makes it possible to render the
exact figure a visitor sees from a script and diff it.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .dataset import CLASS_COLORS, CLASS_NAMES

# Class -> fill colour for the decision-boundary raster. -1 is "no output spike fired",
# which is a real outcome (see silence_stats) and gets a visible grey rather than being
# folded into the nearest class.
_CLASS_Z = {-1: 0, 0: 1, 1: 2, 2: 3}
_BOUNDARY_SCALE = [
    [0.0, "#6b7280"],
    [0.2857, "#6b7280"],
    [0.2857, CLASS_COLORS[0]],
    [0.5714, CLASS_COLORS[0]],
    [0.5714, CLASS_COLORS[1]],
    [0.8571, CLASS_COLORS[1]],
    [0.8571, CLASS_COLORS[2]],
    [1.0, CLASS_COLORS[2]],
]
_AXIS = dict(showgrid=False, zeroline=False, showticklabels=False)
_LAYOUT = dict(margin=dict(l=8, r=8, t=44, b=8), paper_bgcolor="rgba(0,0,0,0)",
               plot_bgcolor="rgba(0,0,0,0)")


def _fig(title: str, height: int = 380, **kw) -> go.Figure:
    """A figure with this app's house style; `kw` wins over the defaults."""
    fig = go.Figure()
    opts = {**_LAYOUT, "title": dict(text=title, x=0.02, xanchor="left"), "height": height}
    opts.update(kw)
    fig.update_layout(**opts)
    return fig


# --- The dataset itself ---------------------------------------------------------------


def _boundary_points(labels: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates where the true class changes — the ground-truth outline.

    Drawn as markers rather than a `go.Contour` constraint: the constraint-contour
    property path differs across plotly versions, and a mismatch there takes the whole
    page down to show a line.
    """
    j, i = np.gradient(labels.astype(np.int64))
    edge = (j != 0) | (i != 0)
    xg, yg = np.meshgrid(axis, axis)
    return xg[edge], yg[edge]


def dataset_scatter(xy: list[np.ndarray], labels: list[np.ndarray], title: str, alpha: float = 0.55) -> go.Figure:
    """The Yin-Yang point cloud, one trace per class so the legend can filter it."""
    fig = _fig(title, height=430)
    for cls, name in enumerate(CLASS_NAMES):
        for (x, y), lab in zip(xy, labels):
            x, y, lab = np.asarray(x), np.asarray(y), np.asarray(lab)
            m = lab == cls
            fig.add_trace(
                go.Scatter(
                    x=x[m], y=y[m], mode="markers", name=name,
                    marker=dict(color=CLASS_COLORS[cls], size=4, opacity=alpha),
                )
            )
    fig.update_xaxes(**_AXIS, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(**_AXIS)
    fig.update_layout(legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"))
    return fig


def decision_boundary(
    grid: np.ndarray,
    axis: np.ndarray,
    title: str,
    *,
    true_labels: np.ndarray | None = None,
    points: list[tuple[float, float]] | None = None,
    height: int = 470,
    showlegend: bool = False,
) -> go.Figure:
    """Predicted class as a raster, optionally with the ground-truth boundary overlaid.

    `true_labels` is drawn as thin contour lines: the difference between the smooth
    analytic shape and the lumpy learned region is the interesting part of the picture.
    """
    fig = _fig(title, height=height)
    z = np.vectorize(_CLASS_Z.get)(grid).astype(float)
    fig.add_trace(
        go.Heatmap(
            z=z, x=axis, y=axis, zmin=0, zmax=3, colorscale=_BOUNDARY_SCALE,
            showscale=False, hoverongaps=False,
            customdata=grid,
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>class %{customdata}<extra></extra>",
        )
    )
    if true_labels is not None:
        bx, by = _boundary_points(true_labels, axis)
        fig.add_trace(
            go.Scatter(
                x=bx, y=by, mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(size=2.5, color="rgba(255,255,255,0.85)"),
            )
        )
    for x, y in points or []:
        fig.add_trace(
            go.Scatter(x=[x], y=[y], mode="markers", showlegend=False,
                       marker=dict(symbol="cross", size=16, color="black",
                                   line=dict(width=2.5, color="white")), hoverinfo="skip")
        )
    fig.update_xaxes(**_AXIS, range=[0, 1])
    fig.update_yaxes(**_AXIS, range=[0, 1], scaleanchor="x", scaleratio=1)
    fig.update_layout(showlegend=showlegend)
    return fig


# --- Spike rasters --------------------------------------------------------------------


def spike_raster(
    times: np.ndarray, sources: np.ndarray, n_neurons: int, title: str,
    t_max_ms: float = 30.0, height: int = 300, marker_size: int = 7, color: str = "#6366f1",
) -> go.Figure:
    """Spike times vs neuron index — the raster an SNN person expects first."""
    fig = _fig(title, height=height)
    t = np.asarray(times, dtype=np.float64) * 1e3
    s = np.asarray(sources, dtype=np.int64)
    fig.add_trace(
        go.Scatter(
            x=t, y=s, mode="markers", showlegend=False,
            marker=dict(size=marker_size, color=color, symbol="circle"),
            hovertemplate="neuron %{y}<br>%{x:.2f} ms<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, t_max_ms], title="time [ms]", gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(range=[-0.5, max(n_neurons - 0.5, 1)], title="neuron")
    return fig


def first_spike_readout(
    first: np.ndarray, label: int | None, title: str, t_max_ms: float = 30.0, height: int = 250
) -> go.Figure:
    """The read-out: one row per output neuron, marked where (and whether) it fired.

    Silent neurons are drawn as an explicit "no spike" tick instead of being omitted, so
    the reader can see that absence is the third possible outcome of a 30 ms window.
    """
    fig = _fig(title, height=height)
    row = np.asarray(first, dtype=np.float64).ravel()
    for i, t in enumerate(row):
        fired = np.isfinite(t)
        fig.add_trace(
            go.Scatter(
                x=[t if fired else t_max_ms], y=[i], mode="markers+text",
                text=["" if fired else "no spike"], textposition="middle right",
                marker=dict(size=17, color=CLASS_COLORS[i] if fired else "#6b7280",
                            symbol="circle" if fired else "x"),
                showlegend=False, hoverinfo="skip",
            )
        )
    if label is not None and 0 <= int(label) < len(row):
        fig.add_hline(y=int(label), line_width=2, line_color="rgba(255,255,255,0.5)")
    fig.update_xaxes(range=[0, t_max_ms], title="first spike time [ms]",
                     gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(tickvals=list(range(len(row))),
                     ticktext=[f"{i} · {CLASS_NAMES[i]}" for i in range(len(row))],
                     title="output neuron")
    return fig


# --- Membrane dynamics ----------------------------------------------------------------


def voltage_trace(
    ts: np.ndarray, traces: list[tuple[str, np.ndarray]], title: str,
    v_th: float = 1.0, spikes: list[tuple[str, np.ndarray]] | None = None, height: int = 300,
) -> go.Figure:
    """Membrane potential vs time, with the threshold the neuron keeps hitting."""
    fig = _fig(title, height=height)
    for name, y in traces:
        fig.add_trace(
            go.Scatter(x=np.asarray(ts) * 1e3, y=y, mode="lines", name=name,
                       line=dict(width=2))
        )
    fig.add_hline(y=v_th, line_dash="dash", line_color="rgba(248,113,113,0.9)",
                 annotation_text="v_th", annotation_position="right")
    fig.add_hline(y=0.0, line_width=1, line_color="rgba(128,128,128,0.4)")
    for name, st in spikes or []:
        st = np.asarray(st, dtype=np.float64)
        if st.size:
            fig.add_trace(
                go.Scatter(x=st * 1e3, y=np.full(st.size, v_th), mode="markers",
                           name=f"{name} spikes", showlegend=False,
                           marker=dict(symbol="triangle-up", size=9, color="#f87171"))
            )
    fig.update_xaxes(title="time [ms]", gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title="V [V_th]")
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0, xanchor="left"))
    return fig


def adjoint_trace(ts: np.ndarray, y: np.ndarray, title: str, height: int = 260) -> go.Figure:
    """The adjoint variable lambda(t): time-reversed, so it runs backwards in time."""
    fig = _fig(title, height=height)
    fig.add_trace(
        go.Scatter(x=np.asarray(ts) * 1e3, y=np.asarray(y), mode="lines", fill="tozeroy",
                   line=dict(color="#f59e0b", width=2), fillcolor="rgba(245,158,11,0.18)")
    )
    fig.update_xaxes(title="time [ms]", gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title="lambda")
    return fig


def gradient_comparison(
    analytic: np.ndarray, fd: np.ndarray, valid: np.ndarray, names: list[str],
    title: str, height: int = 330,
) -> go.Figure:
    """Exact adjoint gradient vs central finite difference, per synapse.

    Synapses where perturbing the weight changed the spike count are marked and NOT
    plotted as a difference: there the finite difference is a chord across a jump, and
    averaging it in would misreport the method rather than test it.
    """
    fig = _fig(title, height=height)
    a = np.asarray(analytic, dtype=np.float64).ravel()
    f = np.asarray(fd, dtype=np.float64).ravel()
    v = np.asarray(valid, dtype=bool).ravel()
    fig.add_trace(go.Bar(x=names, y=np.where(v, a, np.nan), name="EventProp (exact)",
                         marker_color="#22c55e"))
    fig.add_trace(go.Bar(x=names, y=np.where(v, f, np.nan), name="central difference",
                         marker_color="#94a3b8"))
    if (~v).any():
        fig.add_trace(
            go.Scatter(x=[n for n, ok in zip(names, v) if not ok],
                       y=np.full(int((~v).sum()), np.nanmax(np.abs(a)) if np.isfinite(a).any() else 0),
                       mode="markers", name="spike count changed — FD invalid",
                       marker=dict(symbol="x", size=13, color="#ef4444"))
        )
    fig.update_xaxes(title="input synapse")
    fig.update_yaxes(title="d(sum output spike times) / dw",
                     gridcolor="rgba(128,128,128,0.18)")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.14, x=0, xanchor="left"))
    return fig


# --- Accuracy / learning --------------------------------------------------------------


def learning_curves(
    accuracies: list[np.ndarray], color: str, title: str, mean_color: str | None = None,
    log_y: bool = True, height: int = 360, ylabel: str = "test accuracy",
) -> go.Figure:
    """Thin per-seed curves plus their mean — the shape a single run cannot show."""
    fig = _fig(title, height=height)
    for i, curve in enumerate(accuracies):
        y = np.asarray(curve, dtype=np.float64)
        if log_y:
            y = np.clip(1.0 - y, 1e-3, None)
        fig.add_trace(
            go.Scatter(x=np.arange(1, len(y) + 1), y=y, mode="lines",
                       line=dict(width=1, color=color), opacity=0.25,
                       showlegend=(i == 0), name=("each seed" if i == 0 else None),
                       hoverinfo="skip")
        )
    stacked = np.vstack([np.clip(1.0 - np.asarray(c), 1e-3, None) if log_y else np.asarray(c)
                         for c in accuracies])
    fig.add_trace(
        go.Scatter(x=np.arange(1, stacked.shape[1] + 1), y=stacked.mean(axis=0), mode="lines",
                   line=dict(width=3.5, color=mean_color or color), name="mean")
    )
    fig.update_xaxes(title="epoch", gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title=ylabel, type="log" if log_y else "linear")
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0, xanchor="left"))
    return fig


def confusion_matrix(cm: np.ndarray, title: str, height: int = 340) -> go.Figure:
    """Row = true class, column = predicted. The last column is 'no spike at all'."""
    fig = _fig(title, height=height)
    cols = [f"{c}" for c in CLASS_NAMES] + ["no spike"]
    fig.add_trace(
        go.Heatmap(z=cm, x=cols, y=list(CLASS_NAMES), colorscale="Blues",
                   text=cm, texttemplate="%{text}", showscale=False)
    )
    fig.update_xaxes(title="predicted", side="bottom")
    fig.update_yaxes(title="true")
    return fig


def timing_bars(rows: list[tuple[str, float, float]], title: str, height: int = 300) -> go.Figure:
    """Forward and forward+backward medians for each backend. Crude by design."""
    fig = _fig(title, height=height)
    names = [r[0] for r in rows]
    fig.add_trace(go.Bar(x=names, y=[r[1] for r in rows], name="forward", marker_color="#6366f1"))
    fig.add_trace(go.Bar(x=names, y=[r[2] for r in rows], name="forward + backward",
                         marker_color="#f59e0b"))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="median ms per batch", gridcolor="rgba(128,128,128,0.18)")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12, x=0, xanchor="left"))
    return fig


def forward_time_bars(rows: list[tuple[str, float]], title: str,
                      y_title: str = "median ms per batch", height: int = 260) -> go.Figure:
    """One timing per backend. Crude by design, like `timing_bars`.

    A separate builder rather than a third series on `timing_bars`: that chart's series are
    forward and forward+backward, and the maxima pass has no backward — folding them into
    one legend would imply a comparison that does not exist.
    """
    fig = _fig(title, height=height)
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    fig.add_trace(go.Bar(x=names, y=values, marker_color="#6366f1",
                         text=[f"{v:.2f} ms" for v in values], textposition="outside",
                         hovertemplate="%{x}<br>%{y:.3f} ms<extra></extra>"))
    fig.update_xaxes(title="")
    fig.update_yaxes(title=y_title, gridcolor="rgba(128,128,128,0.18)", rangemode="tozero",
                     range=[0, max(values) * 1.22 if values else 1])
    return fig


# --- The digits page (MNIST / VMax) ---------------------------------------------------


def digit_image(image: np.ndarray, title: str, height: int = 250) -> go.Figure:
    """One 28x28 digit, as the network sees it: intensity becomes spike latency.

    A `Heatmap`, not `go.Image`: with a float 2-D `z`, `go.Image` treats `zmin`/`zmax` as
    four-element info arrays and raises, which the smoketest caught before this page could
    throw on a visitor's first paint. Reversing `Greys` gives the MNIST look — black
    background, white stroke — and pinning 0..255 keeps the contrast comparable between
    digits instead of rescaling each one to its own brightest pixel.
    """
    fig = _fig(title, height=height)
    fig.add_trace(go.Heatmap(z=np.asarray(image, dtype=np.float64).reshape(28, 28),
                             colorscale="Greys", reversescale=True, zmin=0.0, zmax=255.0,
                             showscale=False, xgap=0, ygap=0, hoverongaps=False,
                             hovertemplate="pixel %{x},%{y}<br>intensity %{z:.0f}"
                                           "<extra></extra>"))
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS, autorange="reversed")
    return fig


def maxima_bars(values: np.ndarray, true_class: int | None, title: str,
                height: int = 280) -> go.Figure:
    """Voltage maxima per output neuron. The tallest bar IS the prediction (VMax read-out).

    Amber = what the network chose, red = the class that should have won (only when it
    differs), indigo = the rest. No threshold line and no vline on the true class: the
    maxima path does not read `v_th`, so a threshold guide would misdescribe what decides
    the class here, and Plotly's axis-spanning-shape annotation cannot be placed on a
    categorical axis at all.
    """
    fig = _fig(title, height=height)
    values = np.asarray(values, dtype=np.float64)
    pred = int(np.argmax(values))
    colors = []
    for i in range(values.size):
        if i == pred:
            colors.append("#f59e0b")
        elif true_class is not None and i == int(true_class):
            colors.append("#ef4444")
        else:
            colors.append("#6366f1")
    fig.add_trace(go.Bar(x=[str(i) for i in range(values.size)], y=values, marker_color=colors))
    fig.update_xaxes(title="output neuron (digit)")
    fig.update_yaxes(title="voltage maximum", gridcolor="rgba(128,128,128,0.18)")
    return fig


def hidden_latencies(neurons: np.ndarray, times: np.ndarray, n_hidden: int, title: str,
                     height: int = 300) -> go.Figure:
    """First spike of every hidden neuron that fired — the 350-wide middle layer, in time."""
    fig = _fig(title, height=height)
    fig.add_trace(go.Scatter(
        x=np.asarray(times, dtype=np.float64) * 1e3,
        y=np.asarray(neurons, dtype=np.int32),
        mode="markers",
        marker=dict(size=3, color="#22c55e", opacity=0.7),
        hovertemplate="neuron %{y}<br>%{x:.2f} ms<extra></extra>",
        showlegend=False,
    ))
    fig.update_xaxes(title="first spike (ms)", gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title="hidden neuron", range=[n_hidden - 0.5, -0.5])
    return fig
