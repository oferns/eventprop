"""The computations the Visualizer displays. No plotly, no streamlit.

Everything here returns plain arrays, so `selfcheck.py` — or a visitor with a terminal —
can verify the numbers a view is drawing rather than trusting the picture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .backends import Backend
from .dataset import N_CHANNELS, T_MAX, Split, encode, mirror
from .network import LIF, N_HIDDEN, N_OUT, Net


# --- Shared helpers -------------------------------------------------------------------


def first_spike_matrix(layer, n_out: int = N_OUT) -> np.ndarray:
    """First spike time per neuron from a layer's `post_batch`: `(n, n_out)`, NaN if silent.

    Seeded with +inf because `np.minimum` propagates NaN — a NaN seed makes every neuron
    look silent and quietly scores a working network at 0%.
    """
    vec = layer.post_batch
    n = len(vec)
    out = np.full((n, n_out), np.inf)
    for i in range(n):
        src = np.asarray(vec[i].sources, dtype=np.int64).ravel()
        tst = np.asarray(vec[i].times, dtype=np.float64).ravel()
        if src.size:
            keep = (src >= 0) & (src < n_out)
            if keep.any():
                np.minimum.at(out[i], src[keep], tst[keep])
    out[~np.isfinite(out)] = np.nan
    return out


# --- Dataset / accuracy statistics -----------------------------------------------------


def confusion(pred: np.ndarray, labels: np.ndarray, n_classes: int = N_OUT) -> np.ndarray:
    """Row = true class, column = predicted class. -1 (no spike) gets its own column."""
    out = np.zeros((n_classes, n_classes + 1), dtype=np.int64)
    for t, p in zip(labels, pred):
        out[int(t), n_classes if p < 0 else int(p)] += 1
    return out


def silence_stats(first: np.ndarray, labels: np.ndarray) -> dict:
    """How often the output layer stays silent, overall and per class.

    Roughly a third of the published run's output entries are NaN — no neuron fired
    inside the 30 ms window. Those samples are scored wrong by construction, so a
    headline accuracy quietly includes them; the number is worth showing next to it.
    """
    silent = ~np.isfinite(first).any(axis=1)
    return {
        "overall": float(silent.mean()),
        "per_class": {
            int(c): float(silent[labels == c].mean())
            for c in range(N_OUT)
            if (labels == c).any()
        },
        "n_silent": int(silent.sum()),
        "n": int(len(silent)),
    }


def fired_neuron_stats(first: np.ndarray) -> dict:
    """How many of the 3 output neurons fire per sample."""
    counts = np.isfinite(first).sum(axis=1)
    return {
        "mean": float(counts.mean()),
        "histogram": [int((counts == k).sum()) for k in range(first.shape[1] + 1)],
    }


# --- Decision boundary -----------------------------------------------------------------


def _grid_coords(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, resolution)
    xg, yg = np.meshgrid(axis, axis)
    return mirror(np.stack([xg.ravel(), yg.ravel()], axis=1)), axis


def _predict_chunked(net: Net, coords: np.ndarray, chunk: int) -> np.ndarray:
    pred = np.empty(len(coords), dtype=np.int64)
    for lo in range(0, len(coords), chunk):
        pred[lo : lo + chunk], _, _ = net.predict(coords[lo : lo + chunk])
    return pred


def boundary_grid(
    net: Net, resolution: int = 81, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class over the unit square; `grid[j, i]` at `(axis[i], axis[j])`."""
    coords, axis = _grid_coords(resolution)
    return _predict_chunked(net, coords, chunk).reshape(resolution, resolution), axis


def boundary_series(
    nets: list[Net], resolution: int = 41, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Boundaries for several snapshots stacked on axis 0 — the 'watch it learn' reel."""
    coords, axis = _grid_coords(resolution)
    out = np.empty((len(nets), resolution, resolution), dtype=np.int64)
    for k, net in enumerate(nets):
        out[k] = _predict_chunked(net, coords, chunk).reshape(resolution, resolution)
    return out, axis


# --- The exact-gradient demonstration --------------------------------------------------
#
# EventProp's claim is an *exact* derivative through spike times: the crossing is
# root-found, and the time-reversed adjoint sweep differentiates that root. The cleanest
# demonstration is a network small enough to inspect, an objective whose value IS a spike
# time, and a finite-difference estimate held up against it.
#
# Two hazards this code is written around:
#   * A scenario that never spikes makes both legs zero, which looks like agreement. So
#     every comparison below is gated on the output neuron actually having fired.
#   * Perturbing a weight can change the SPIKE COUNT. The objective is then discontinuous
#     in that weight and a central difference measures a chord across the jump, not a
#     derivative. Such entries are flagged rather than averaged in — and they are the
#     single best argument on the page for why an exact gradient is needed at all.

WEPS = 1e-6


@dataclass
class GradientDemo:
    fired: bool
    n_output_spikes: int
    output_spike_times: np.ndarray
    input_spike_times: np.ndarray
    input_spike_sources: np.ndarray
    hidden_voltage_t: np.ndarray
    hidden_voltage: np.ndarray
    hidden_spikes: np.ndarray
    hidden_lambda_t: np.ndarray
    hidden_lambda_trace: np.ndarray
    output_voltage_t: np.ndarray
    output_voltage: np.ndarray
    lambda_t: np.ndarray
    lambda_trace: np.ndarray
    hidden_w: np.ndarray
    output_w: np.ndarray
    analytic: np.ndarray
    fd: np.ndarray
    fd_valid: np.ndarray
    output_analytic: float
    output_fd: float
    output_fd_valid: bool
    rel_error: np.ndarray


def _chain(backend: Backend, vec, wh: np.ndarray, wo: np.ndarray):
    """Build the toy 5 -> 1 -> 1 pair wired the way the library expects.

    `Layer.__call__` records `ancestor_layer` only when handed the `(SpikesVector, Layer)`
    tuple form. Calling `forward()` twice produces an identical *forward* pass but leaves
    the pair unchained, so the hidden layer's error signal — which the output backward
    pass is what writes — never arrives and its gradient comes back silently zero.
    """
    h = backend.layer(n=1, n_in=wh.shape[0], w_in=wh, **LIF)
    o = backend.layer(n=1, n_in=1, w_in=wo, **LIF)
    hs, _ = h(vec)
    o((hs, h))
    return h, o


def _sum_of_spike_times(backend: Backend, vec, wh: np.ndarray, wo: np.ndarray):
    """Objective value and spike count for one weight setting (no error signals)."""
    _, o = _chain(backend, vec, wh, wo)
    times = np.asarray(o.post_batch[0].times, dtype=np.float64)
    return float(times.sum()), int(times.size)


def gradient_demo(
    backend: Backend,
    coords: np.ndarray,
    seed: int = 0,
    hidden_scale: float = 0.6,
    output_w: float = 4.0,
    t_max: float = T_MAX,
    dt: float = 2.5e-4,
) -> GradientDemo:
    """Analytic vs central-difference `d(sum of output spike times)/dw` on a 5-1-1 net.

    Works on either binding — it needs voltage and adjoint traces, which both now expose
    (see `Backend.supports_traces`). The guard below is what keeps a binding that lacks them
    from producing a half-populated figure instead of an error.
    """
    if not backend.supports_traces():
        raise ValueError(f"{backend.short} backend cannot produce the traces this view needs")

    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    if coords.shape[1] == 2:
        coords = mirror(coords)
    if coords.shape[1] != 4:
        raise ValueError("expected a test point as [x, y] or the stored 4-vector")
    times, sources = encode(coords)
    vec = backend.make_vector(times, sources)

    rng = np.random.default_rng(seed)
    # N_CHANNELS, not coords.shape[1]: the encoder emits a fifth bias spike on channel 4,
    # and the Eigen core does not validate spike sources against the weight matrix, so a
    # 4-row weight here is an out-of-bounds access rather than an error message.
    wh = np.abs(rng.normal(1.0, 0.25, (N_CHANNELS, 1))) * hidden_scale
    wo = np.array([[output_w]], dtype=np.float64)

    # Grow the hidden weights until the output neuron fires: below threshold there is
    # nothing to differentiate and both legs would agree at zero.
    _, n_spikes = _sum_of_spike_times(backend, vec, wh, wo)
    for _ in range(12):
        if n_spikes:
            break
        wh *= 1.4
        _, n_spikes = _sum_of_spike_times(backend, vec, wh, wo)
    fired = n_spikes > 0

    analytic = np.zeros_like(wh)
    fd = np.zeros_like(wh)
    fd_valid = np.zeros_like(wh, dtype=bool)
    out_analytic = out_fd = 0.0
    out_valid = False

    hidden_lambda_t = hidden_lambda = np.array([])
    if fired:
        # The objective is the sum of output spike times, so unit error on every output
        # spike makes the backward pass return exactly dF/dw.
        h, o = _chain(backend, vec, wh, wo)
        for i in range(n_spikes):
            o.post_batch[0].set_error(i, 1.0)
        o.backward()  # cascades to the hidden layer through ancestor_layer
        analytic = np.array(h.gradient, dtype=np.float64)
        out_analytic = float(np.array(o.gradient, dtype=np.float64).ravel()[0])

        for i in range(wh.shape[0]):
            wp, n_plus = _sum_of_spike_times(backend, vec, _bump(wh, i, +WEPS), wo)
            wm, n_minus = _sum_of_spike_times(backend, vec, _bump(wh, i, -WEPS), wo)
            if n_plus == n_minus == n_spikes:
                fd[i, 0] = (wp - wm) / (2 * WEPS)
                fd_valid[i, 0] = True

        wp, n_plus = _sum_of_spike_times(backend, vec, wh, _bump(wo, 0, +WEPS))
        wm, n_minus = _sum_of_spike_times(backend, vec, wh, _bump(wo, 0, -WEPS))
        if n_plus == n_minus == n_spikes:
            out_fd = (wp - wm) / (2 * WEPS)
            out_valid = True

        # Traces come from these same instances. A fresh forward pass would draw a flat
        # adjoint, because lambda is driven by the error signal the backward pass writes
        # — the plot would then contradict the explanation next to it.
        hv_t, hv = h.get_voltage_trace_for_neuron(0, 0, t_max=t_max, dt=dt)
        ov_t, ov = o.get_voltage_trace_for_neuron(0, 0, t_max=t_max, dt=dt)
        hs_src = np.asarray(h.post_batch[0].sources, dtype=np.int64)
        hs_tim = np.asarray(h.post_batch[0].times, dtype=np.float64)
        hidden_spike_times = hs_tim[hs_src == 0]
        out_spike_times = np.asarray(o.post_batch[0].times, dtype=np.float64)
        # Guarded because the Eigen core reads `times[n_spikes - 1]` unguarded in
        # compute_lambda_i, so asking for the adjoint of a silent neuron is an
        # out-of-bounds read rather than a zero. The C library returns a zero trace, but the
        # plot would then differ between backends for a reason that has nothing to do with
        # the maths, so neither backend is asked.
        lam_t, lam = o.get_lambda_i_trace_for_neuron(0, 0, t_max=t_max, dt=dt)
        if hidden_spike_times.size:
            hidden_lambda_t, hidden_lambda = h.get_lambda_i_trace_for_neuron(
                0, 0, t_max=t_max, dt=dt
            )
    else:
        h2, o2 = _chain(backend, vec, wh, wo)
        hv_t, hv = h2.get_voltage_trace_for_neuron(0, 0, t_max=t_max, dt=dt)
        ov_t, ov = o2.get_voltage_trace_for_neuron(0, 0, t_max=t_max, dt=dt)
        hs_src = np.asarray(h2.post_batch[0].sources, dtype=np.int64)
        hs_tim = np.asarray(h2.post_batch[0].times, dtype=np.float64)
        hidden_spike_times = hs_tim[hs_src == 0]
        out_spike_times = np.asarray(o2.post_batch[0].times, dtype=np.float64)

    denom = np.where(np.abs(fd) > 1e-9, np.abs(fd), np.nan)
    return GradientDemo(
        fired=fired,
        n_output_spikes=n_spikes,
        output_spike_times=out_spike_times,
        input_spike_times=times.ravel(),
        input_spike_sources=sources.ravel(),
        hidden_voltage_t=np.asarray(hv_t, dtype=np.float64),
        hidden_voltage=np.asarray(hv, dtype=np.float64),
        hidden_spikes=np.asarray(hidden_spike_times, dtype=np.float64),
        hidden_lambda_t=np.asarray(hidden_lambda_t, dtype=np.float64),
        hidden_lambda_trace=np.asarray(hidden_lambda, dtype=np.float64),
        output_voltage_t=np.asarray(ov_t, dtype=np.float64),
        output_voltage=np.asarray(ov, dtype=np.float64),
        lambda_t=np.asarray(lam_t, dtype=np.float64),
        lambda_trace=np.asarray(lam, dtype=np.float64),
        hidden_w=wh,
        output_w=wo,
        analytic=analytic,
        fd=fd,
        fd_valid=fd_valid,
        output_analytic=out_analytic,
        output_fd=out_fd,
        output_fd_valid=out_valid,
        rel_error=np.abs(analytic - fd) / denom,
    )


def _bump(w: np.ndarray, row: int, delta: float) -> np.ndarray:
    out = w.copy()
    out[row, 0] += delta
    return out


# --- Small backend A/B ("under the hood") ----------------------------------------------


def backend_timings(
    backend: Backend, split: Split, n_batches: int = 12, batch: int = 64
) -> dict:
    """Median forward and forward+backward wall time for one backend on Yin-Yang batches.

    Deliberately crude. It exists to show both bindings do the same work in the same
    ballpark; the authoritative numbers are in `bench-reports/`, which carry provenance
    and repeat statistics and refuse to merge across hosts.
    """
    coords = split.coords[: n_batches * batch]
    ys = split.labels[: n_batches * batch]
    fwd: list[float] = []
    step: list[float] = []
    rng = np.random.default_rng(0)
    wh = rng.normal(1.5, 0.78, (5, N_HIDDEN))
    wo = rng.normal(0.93, 0.1, (N_HIDDEN, N_OUT))
    for lo in range(0, len(coords) - batch + 1, batch):
        times, sources = encode(coords[lo : lo + batch])
        vec = backend.make_vector(times, sources)
        h = backend.layer(n=N_HIDDEN, n_in=5, w_in=wh, **LIF)
        out = backend.layer(n=N_OUT, n_in=N_HIDDEN, w_in=wo, **LIF)
        t0 = time.perf_counter()
        hs, _ = h(vec)
        out((hs, h))
        t1 = time.perf_counter()
        _set_ttfs_errors(out, ys[lo : lo + batch])
        out.backward()
        t2 = time.perf_counter()
        fwd.append(t1 - t0)
        step.append(t2 - t1)
    return {
        "forward_ms": float(np.median(fwd) * 1e3) if fwd else float("nan"),
        "backward_ms": float(np.median(step) * 1e3) if step else float("nan"),
        "batch": batch,
        "n_batches": len(fwd),
    }


def _set_ttfs_errors(layer, labels: np.ndarray) -> None:
    """A stand-in TTFS error signal: -1 on the correct class's spikes, +1 elsewhere.

    Enough of the real loss to drive the backward pass end to end. The authoritative
    version is the library's own `TTFSCrossEntropyLoss`.
    """
    for i in range(len(layer.post_batch)):
        correct = int(labels[i])
        spikes = layer.post_batch[i]
        src = np.asarray(spikes.sources, dtype=np.int64)
        for k in range(src.size):
            spikes.set_error(k, -1.0 if int(src[k]) == correct else 1.0)


def predict_split(net: Net, split: Split, chunk: int = 256):
    """`(pred, margin, first)` for every sample in a split."""
    pred = np.empty(split.n, dtype=np.int64)
    margin = np.empty(split.n)
    first = np.empty((split.n, N_OUT))
    for lo in range(0, split.n, chunk):
        hi = min(lo + chunk, split.n)
        pred[lo:hi], margin[lo:hi], first[lo:hi] = net.predict(split.coords[lo:hi])
    return pred, margin, first
