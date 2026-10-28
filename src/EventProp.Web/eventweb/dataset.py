"""The Yin-Yang dataset: loading, geometry, and TTFS encoding.

Source of the data
------------------
`eventprop/yin_yang_data_set/` is a nested git submodule inside the read-only reference
library — it is *the* published dataset (Kriener, Göltz, Petrovici, arXiv:2102.08211,
MIT), and `publication_data/*.npy` is the frozen train/validation/test split used by
"Fast and deep" (arXiv:1912.11443). The split is 5000/1000/1000 with per-class counts
that are balanced by construction (rejection sampling).

This module reads those `.npy` files. It never regenerates them: the dataset's own
README is explicit that train/val/test differ *only* by RNG seed, so re-drawing with a
different seed would silently overlap the test set.

Geometry
--------
A sample is a point `(x, y)` inside the disc of radius `r_big = 0.5` centred at
`(0.5, 0.5)`, plus two "dots" of radius `r_small = 0.1` at `(0.25, 0.5)` and
`(0.75, 0.5)`. Three classes: `yang`, `yin`, `dot`.

Every stored sample is the 4-vector `[x, y, 1-x, 1-y]` — the *mirrored axis*. Reflecting
through the centre puts each class centroid at the origin of the encoded frame, which is
what lets a bias-free LIF network separate the classes at all. It is the reason this
dataset (not MNIST) is the reference demo.

Encoding
--------
TTFS ("time to first spike"): coordinate `c` becomes a neuron firing at `c * t_max`, so
the *first* spike time carries the value. A fifth neuron fires at `t = 0` and acts as the
bias current, since the layers here carry no explicit bias term. Times are sorted
ascending because the library requires a non-decreasing spike list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Frozen publication split (do not regenerate; see module docstring) ---------------
N_TRAIN, N_VALID, N_TEST = 5000, 1000, 1000

# --- Yin-Yang geometry, mirroring dataset.py in the submodule -------------------------
R_SMALL = 0.1
R_BIG = 0.5

CLASS_NAMES = ("yang", "yin", "dot")
CLASS_COLORS = {0: "#ef4444", 1: "#3b82f6", 2: "#22c55e"}


def which_class(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ground-truth class of a point, from the geometry alone.

    Mirrors `YinYangDataset.which_class` in the submodule. Same criterion order, same
    short-circuit: the dots win over the yin/yang halves.
    """
    d_right = np.sqrt((x - 1.5 * R_BIG) ** 2 + (y - R_BIG) ** 2)
    d_left = np.sqrt((x - 0.5 * R_BIG) ** 2 + (y - R_BIG) ** 2)
    c1 = d_right <= R_SMALL
    c2 = (d_left > R_SMALL) & (d_left <= 0.5 * R_BIG)
    c3 = (y > R_BIG) & (d_right > 0.5 * R_BIG)
    is_yin = c1 | c2 | c3
    in_a_dot = (d_right < R_SMALL) | (d_left < R_SMALL)
    return np.where(in_a_dot, 2, is_yin.astype(int))


def in_support(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where a point can actually occur in the data (inside the big disc).

    Outside the disc no sample exists, so a classifier's answer there is extrapolation
    and should not be drawn as if it were a measured decision boundary.
    """
    return (x - R_BIG) ** 2 + (y - R_BIG) ** 2 <= R_BIG**2


def mirror(coords2: np.ndarray) -> np.ndarray:
    """`[x, y] -> [x, y, 1-x, 1-y]`, the 4-vector the dataset stores."""
    coords2 = np.atleast_2d(np.asarray(coords2, dtype=np.float64))
    return np.concatenate([coords2, 1.0 - coords2], axis=1)


@dataclass(frozen=True)
class Split:
    """One split of the frozen publication data."""

    name: str
    coords: np.ndarray  # (n, 4) — [x, y, 1-x, 1-y], in dataset units
    labels: np.ndarray  # (n,) int in {0, 1, 2}

    @property
    def n(self) -> int:
        return len(self.labels)

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.coords[:, 0], self.coords[:, 1]


def load_split(data_dir: Path, name: str) -> Split:
    """Load `train` / `valid` / `test` from the frozen `.npy` files.

    The submodule spells the validation split `validation_*`; the argument is `valid`
    because every call site is shorter for it.
    """
    stem = {"train": "train", "valid": "validation", "test": "test"}[name]
    coords = np.load(data_dir / f"{stem}_samples.npy")
    labels = np.load(data_dir / f"{stem}_labels.npy")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{stem}_samples.npy: expected (n, 4), got {coords.shape}")
    return Split(name=name, coords=coords, labels=labels.astype(np.int64))


# --- TTFS encoding --------------------------------------------------------------------
T_MAX = 30e-3  # seconds; the whole stimulus lives in one 30 ms window
N_CHANNELS = 5  # 4 mirrored coordinates + 1 bias spike at t = 0


def encode(coords: np.ndarray, t_max: float = T_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Flat TTFS encoding of a batch of 4-vectors.

    Returns `(times, sources, ...)`: for each of the 4 coordinates a spike at
    `c * t_max`, plus a bias spike at `t = 0` on channel 4.

    Times come back sorted ascending per sample. That is a hard requirement, not
    tidiness: `eventprop_validate_input_no_v_th` rejects a non-decreasing list (error 60)
    and the Eigen reference silently returns plausible garbage instead.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    n = coords.shape[0]
    # (n, 5): the 4 coordinate latencies and the constant bias spike.
    times = np.empty((n, N_CHANNELS), dtype=np.float64)
    times[:, :4] = coords * t_max
    times[:, 4] = 0.0
    sources = np.tile(np.arange(N_CHANNELS, dtype=np.int32), (n, 1))

    order = np.argsort(times, axis=1, kind="stable")
    times = np.take_along_axis(times, order, axis=1)
    sources = np.take_along_axis(sources, order, axis=1)
    return times, sources


def decode(first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TTFS read-out: the predicted class is whichever output neuron fires FIRST.

    Returns `(pred, margin)`; `pred` is -1 when no output neuron fired at all, and
    `margin` is the gap (in seconds) to the runner-up, i.e. how confident the timing is.
    A row with a single spike gets `margin = inf`.
    """
    finite = np.isfinite(first)
    n = first.shape[0]
    pred = np.full(n, -1, dtype=np.int64)
    margin = np.full(n, np.nan)
    any_spike = finite.any(axis=1)
    if not any_spike.any():
        return pred, margin
    # Push silences to +inf so argmin only ever selects a real spike.
    big = np.where(finite, first, np.inf)
    order = np.argsort(big, axis=1)
    best = order[:, 0]
    rows = np.flatnonzero(any_spike)
    pred[rows] = best[rows]
    second = np.take_along_axis(big, order[:, 1:], axis=1)
    margin[rows] = second[rows, 0] - big[rows, best[rows]]
    return pred, margin


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    """Fraction correct. A no-spike prediction (-1) can never match a label."""
    return float((pred == labels).mean())
