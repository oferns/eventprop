"""Running the shipped Yin-Yang networks, and reading their checkpoints.

The ten training runs in `src/EventProp.Python/paper/yinyang_{0..9}.pkl` are the
published EventProp result (~98% test accuracy against a ~64% ceiling for a shallow
classifier). They are *artefacts of the reference library*, produced by
`paper/figure2.py`, and this app reads them rather than retraining: a visitor should see
a network that actually works in milliseconds, not wait for a training run to finish.

Checkpoint layout (a 9-tuple; the empty slots are the validation/train results that
`figure2.py` did not ask the trainer to collect):

    [2] test_accuracies, one per epoch          (100 floats)
    [3] test_losses, one per epoch              (100 floats)
    [4] test first-spike times per epoch        (100 x (1000, 3))
    [8] weight snapshots                        (11 x (wh (5,200), wo (200,3)))

Epoch bookkeeping — the part that is easy to get wrong and impossible to notice.
`eventprop/training.py` evaluates the test set *before* each epoch and appends weight
snapshots on `epoch % save_every == 0` *after* each epoch. With `epochs=100`,
`save_every=10`, 0-based `epoch`:

    accuracy[i]  is the accuracy of the weights after  i      epochs
    snapshot k   holds  the weights after  [1,11,...,91,100][k] epochs

so the two series are *not* index-aligned, and the final snapshot has no accuracy entry
at all (the last recorded accuracy is the one measured after 99 epochs). Labelling a
snapshot "epoch 10" would misattribute ten epochs of learning to the wrong picture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pickle import load as pickle_load

import numpy as np

from .backends import Backend
from .dataset import N_CHANNELS, T_MAX, Split, decode, encode, mirror

# LIF parameters the published runs used (paper/figure2.py). Inference reads them, so a
# mismatch here silently changes the network's behaviour rather than failing loudly.
LIF = dict(tau_mem=20e-3, tau_syn=5e-3, v_th=1.0, v_leak=0.0)
N_HIDDEN = 200
N_OUT = 3

# Snapshots, in epochs-of-training-completed. Derived above; 11 entries, matching the
# 11 snapshots actually present in each file.
SNAPSHOT_EPOCHS = (1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100)

# ── Checkpoint file formats ─────────────────────────────────────────────────
#
# `paper/yinyang_*.pkl` are plain pickled tuples written before the fields had names, so
# every field is positional and the file itself says which layout it holds. Two layouts
# are live in this tree:
#
#   published / TTFS — 9 fields. What `paper/*.pkl` contain. The order is not a guess:
#     it is exactly what `TTFSMixin.get_data_for_pickling()` returns in this tree, and the
#     same method in the read-only reference submodule — `paper/figure2.py` unpacks it as
#     `_, _, accs, losses, first_spikes, _, _, _, _`, and the shipped files match: field 4
#     is `(100, 1000, 3)` first-spike times, field 8 the 11 weight snapshots.
#
#   base trainer — 7 fields. What the non-TTFS `AbstractTraining` writes in both trees, the
#     reference's included. Weights land at index 6 and there are no first-spike times at
#     all. Read with the published indices, field 4 is `valid_accuracies` — empty, so every
#     neuron looks merely silent — and field 8 then raises IndexError. Two failure modes
#     from one cause, the first of them silent.
#
# Our trainer writes named keys now (`eventprop_c.training.checkpoint_payload`). The
# reference submodule cannot be edited, so its 7-tuple remains a thing that can turn up:
# the loader identifies what it was handed instead of trusting a length.

PUBLISHED_TUPLE_LEN = 9
#: The 9-field order, named. Identical in our `ttfs_training.py` and the reference
#: submodule's, which is what makes the published files legible at all.
PUBLISHED_FIELDS = ("losses", "accuracies", "test_accuracies", "test_losses",
                    "test_first_spikes", "valid_accuracies", "valid_losses",
                    "valid_first_spikes", "weights")
TRAINERS_TUPLE_LEN = 7

#: Keys a named-dict checkpoint must carry. `test_first_spikes` is optional: neither
#: trainer in this tree records first-spike times, so a fresh run legitimately has none.
REQUIRED_KEYS = ("test_accuracies", "test_losses", "weights")


def _weight_pairs(weights, source: str) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        return [(np.asarray(a, np.float64), np.asarray(b, np.float64)) for a, b in weights]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source}: weights is not a sequence of (wh, wo) pairs: {exc}") from exc


def parse_checkpoint(data, *, seed: int, source: str) -> Checkpoint:
    """Build a `Checkpoint` from an unpickled payload, identifying which layout it is.

    Anything unrecognised raises `ValueError` naming what was found and what is wanted.
    Guessing here is what turns a version skew into a confidently plotted wrong answer.
    """
    if isinstance(data, dict):
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"{source}: checkpoint dict is missing {missing}; "
                f"found keys {sorted(data)}")
        acc, los = data["test_accuracies"], data["test_losses"]
        fs = data.get("test_first_spikes", data.get("first_spikes", []))
        weights = _weight_pairs(data["weights"], source)
        origin = (f"named-dict checkpoint "
                  f"(format={data.get('format', 'unspecified')}, "
                  f"version={data.get('format_version', '?')})")
    elif isinstance(data, tuple) and len(data) == PUBLISHED_TUPLE_LEN:
        acc, los, fs = data[2], data[3], data[4]
        weights = _weight_pairs(data[8], source)
        origin = f"published {PUBLISHED_TUPLE_LEN}-field tuple"
    elif isinstance(data, tuple) and len(data) == TRAINERS_TUPLE_LEN:
        raise ValueError(
            f"{source}: this is the {TRAINERS_TUPLE_LEN}-field tuple the base (non-TTFS) "
            f"trainer writes — (losses, accuracies, test_accuracies, test_losses, "
            f"valid_accuracies, valid_losses, weights), with weights at index 6 and no "
            f"first-spike times at all. The Visualizer cannot read it as a published run. "
            f"Train with eventprop_c.ttfs_training (which records first spikes and saves "
            f"named keys), or point ckpt_dir at the published paper/*.pkl.")
    else:
        shape = f" of length {len(data)}" if isinstance(data, (tuple, list)) else ""
        raise ValueError(
            f"{source}: unrecognised checkpoint payload {type(data).__name__}{shape}; "
            f"expected a {PUBLISHED_TUPLE_LEN}-field published tuple, or a dict with "
            f"{list(REQUIRED_KEYS)}")

    accuracies = np.asarray(acc, dtype=np.float64)
    losses = np.asarray(los, dtype=np.float64)
    first_spikes = np.asarray(fs, dtype=np.float64)

    if not weights:
        raise ValueError(f"{source}: checkpoint carries no weight snapshots")
    for name, series in (("test accuracies", accuracies), ("test losses", losses)):
        if series.ndim != 1:
            raise ValueError(f"{source}: {name} is {series.ndim}-D, expected a 1-D series")
    if accuracies.size and losses.size and accuracies.shape != losses.shape:
        raise ValueError(
            f"{source}: accuracy/loss series differ: "
            f"{accuracies.shape} vs {losses.shape}")
    if first_spikes.size and first_spikes.ndim != 3:
        raise ValueError(
            f"{source}: first-spike times are {first_spikes.ndim}-D, expected "
            f"(epochs, samples, neurons) — reading the wrong field of a positional "
            f"tuple is the usual cause")
    if (first_spikes.size and accuracies.size
            and first_spikes.shape[0] != accuracies.shape[0]):
        raise ValueError(
            f"{source}: {first_spikes.shape[0]} first-spike epochs vs "
            f"{accuracies.shape[0]} recorded accuracies — the payload is inconsistent")

    return Checkpoint(
        seed=seed,
        accuracies=accuracies,
        losses=losses,
        first_spikes=first_spikes,
        weights=weights,
        has_first_spikes=bool(first_spikes.size),
        origin=origin,
    )


@dataclass(frozen=True)
class Checkpoint:
    """One published run."""

    seed: int
    accuracies: np.ndarray  # (100,) test accuracy after epoch i
    losses: np.ndarray  # (100,) test loss after epoch i
    first_spikes: np.ndarray  # (100, 1000, 3) first-spike times per epoch, empty if not recorded
    weights: list  # 11 x (wh, wo)
    snapshot_epochs: tuple = SNAPSHOT_EPOCHS
    has_first_spikes: bool = True
    origin: str = ""  # which on-disk layout this came from; see parse_checkpoint

    def weights_at(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        wh, wo = self.weights[k]
        return np.asarray(wh, dtype=np.float64), np.asarray(wo, dtype=np.float64)


def load_checkpoint(ckpt_dir: Path, seed: int) -> Checkpoint:
    path = ckpt_dir / f"yinyang_{seed}.pkl"
    with open(path, "rb") as f:
        data = pickle_load(f)
    return parse_checkpoint(data, seed=seed, source=path.name)


class Net:
    """A trained Yin-Yang net bound to one backend.

    Inference needs only the two LIF layers: TTFS in, TTFS out, class = first output
    spike. The loss layer exists only to produce gradients, so the app's forward pass
    does not depend on the TTFS loss parameters at all.
    """

    def __init__(self, backend: Backend, wh: np.ndarray, wo: np.ndarray, t_max: float = T_MAX):
        self.backend = backend
        self.wh = np.ascontiguousarray(wh, dtype=np.float64)
        self.wo = np.ascontiguousarray(wo, dtype=np.float64)
        self.t_max = t_max
        n_in, n_hidden = self.wh.shape
        n_hidden_o, n_out = self.wo.shape
        if (n_in, n_hidden) != (N_CHANNELS, N_HIDDEN) or n_hidden_o != n_hidden or n_out != N_OUT:
            raise ValueError(
                f"weight shapes {self.wh.shape}/{self.wo.shape} do not match the "
                f"Yin-Yang topology ({N_CHANNELS},{N_HIDDEN})/({N_HIDDEN},{N_OUT})"
            )
        self.hidden = backend.layer(n=n_hidden, n_in=n_in, w_in=self.wh, **LIF)
        self.output = backend.layer(n=n_out, n_in=n_hidden, w_in=self.wo, **LIF)

    def spikes_for(self, coords: np.ndarray):
        """Forward pass on a batch of 4-vectors; returns the output `SpikesVector`."""
        times, sources = encode(coords, t_max=self.t_max)
        inp = self.backend.make_vector(times, sources)
        self.hidden.forward(inp)
        return self.output.forward(self.hidden.post_batch)

    def first_spikes(self, coords: np.ndarray) -> np.ndarray:
        """First output-spike time per neuron, `(n, 3)`, NaN where a neuron stayed silent.

        `np.minimum.at` rather than "first occurrence wins": it does not care whether a
        binding happens to hand back spikes in time order, so both backends are read the
        same way.
        """
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        vec = self.spikes_for(coords)
        # Seed with +inf, not NaN: `np.minimum` propagates NaN, so a NaN seed makes every
        # neuron look silent and the whole network scores 0% while spiking normally.
        out = np.full((len(coords), N_OUT), np.inf)
        for i in range(len(coords)):
            s = vec[i]
            src = np.asarray(s.sources).ravel().astype(np.int64)
            tst = np.asarray(s.times).ravel().astype(np.float64)
            keep = (src >= 0) & (src < N_OUT)
            if not keep.any():
                continue
            np.minimum.at(out[i], src[keep], tst[keep])
        out[~np.isfinite(out)] = np.nan
        return out

    def predict(self, coords: np.ndarray):
        """Returns `(pred, margin, first)`; see `dataset.decode` for conventions."""
        first = self.first_spikes(coords)
        pred, margin = decode(first)
        return pred, margin, first

    def accuracy(self, split: Split) -> float:
        pred, _, _ = self.predict(split.coords)
        return float((pred == split.labels).mean())

    def decision_boundary(
        self, resolution: int = 41, chunk: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicted class on a resolution x resolution grid over the unit square.

        Returns `(grid, x_edges, y_edges)` with `grid[j, i]` the class predicted at
        `(x_edges[i], y_edges[j])`, or -1 where no output neuron fired. Chunked because
        the hidden layer materialises a per-sample spike buffer.
        """
        xs = np.linspace(0.0, 1.0, resolution)
        xg, yg = np.meshgrid(xs, xs)
        coords = mirror(np.stack([xg.ravel(), yg.ravel()], axis=1))
        preds = np.empty(len(coords), dtype=np.int64)
        for lo in range(0, len(coords), chunk):
            block = coords[lo : lo + chunk]
            preds[lo : lo + chunk], _, _ = self.predict(block)
        return preds.reshape(resolution, resolution), xs, xs
