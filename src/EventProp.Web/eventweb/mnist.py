"""MNIST: data, the reference's spike encoding, and the voltage-maxima read-out.

Why this module exists
----------------------
`paper/mnist_vmax_{0..9}.pkl` are ten published runs the Visualizer could not show: the
weights were readable but nothing could produce the inputs they were trained on. The
reference library does ship a pipeline (`eventprop.mnist`), and it is deliberately *not*
reused here, for two measurable reasons:

* it imports `tqdm` at module scope, and neither this app's interpreter nor its venv has
  tqdm — `import eventprop.mnist` raises `ModuleNotFoundError` before any of it runs;
* its reader consumes the IDX files a byte at a time (`ord(image.read(1))` per pixel,
  60 000 x 784 of them). This module reads the same files with `np.frombuffer` and got
  the test set in under a second.

What *is* copied, exactly, is the encoding. `encode()` is line-for-line equivalent to the
reference's `get_batch` (asserted against a literal transcription of that loop in
`selfcheck.py` step 12), because the shipped weights are only meaningful to the network
they were trained on: change the latency rule and the published accuracy stops being the
number under test.

The read-out is `argmax` over the output layer's voltage **maxima**, which is what
`eventprop/loss_layer.py`'s `get_predictions()` does for VMax. That is a different
read-out from Yin-Yang's first-spike rule — same library, different half of it: maxima
go through `eventprop_compute_maxima_batch`, and that path does not read `v_th` (see
`src/EventProp.C/eventprop.h`), which is why the layer parameters below take none.
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

from .backends import BACKENDS

# The reference's window for MNIST: `MNISTMixin.t_min/t_max`. Note it is 20 ms, not the
# 30 ms the Yin-Yang pages use — the artefacts and the encoder have to agree.
T_MIN = 0.0
T_MAX = 20e-3

#: Pixels at or below this intensity emit nothing. Mirrors `if pix > 1` in the reference,
#: which is what keeps a black background from contributing 784 spikes per image.
PIXEL_MIN = 1

VALID_NUM = 5000  # reference split: train_images[:5000] is validation, [5000:] trains

#: The mirror the reference library's own loader uses, so the pixels behind the
#: published numbers are the same pixels this page shows.
URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
_TEST_LABELS = "t10k-labels-idx1-ubyte.gz"


def cache_dir() -> Path:
    """App-owned cache, deliberately NOT inside the read-only reference submodule.

    `eventprop.mnist` writes its derived pickles into `eventprop/mnist_dataset/`, i.e.
    into a submodule this repository treats as a specification. The cache lives with the
    app that created it.
    """
    return Path(__file__).resolve().parent.parent / ".cache" / "mnist"


def have_test_set() -> bool:
    d = cache_dir()
    return (d / _TEST_IMAGES).exists() and (d / _TEST_LABELS).exists()


def fetch_test_set() -> Path:
    """Download the two t10k files (~1.6 MB total). Raises on any network failure.

    Explicit, not implicit: a page that silently reaches the network on first paint is a
    page that hangs on an air-gapped machine, and this repository ships to three
    platforms, at least one of which will be behind a proxy.
    """
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name in (_TEST_IMAGES, _TEST_LABELS):
        dest = d / name
        if not dest.exists():
            try:
                urllib.request.urlretrieve(URL_BASE + name, dest)
            except OSError as exc:
                raise RuntimeError(
                    f"could not download {URL_BASE + name}: {exc}. The MNIST files are "
                    f"not vendored in this repository; place them in {d} manually to use "
                    f"the digits page.") from None
    return d


def _read_idx(path: Path, header: int, item: str, shape: tuple[int, ...]) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    count = int(np.frombuffer(raw[:8], dtype=">u4")[1])
    n = count * int(np.prod(shape))
    flat = np.frombuffer(raw[header:header + n], dtype=item)
    if flat.size != n:
        raise ValueError(f"{path.name}: expected {n} values, found {flat.size} — truncated?")
    return flat.reshape((count, *shape))


def load_test() -> tuple[np.ndarray, np.ndarray]:
    """`(images, labels)`: images as `(10000, 784)` uint8, labels as int32 0..9."""
    d = cache_dir()
    images = _read_idx(d / _TEST_IMAGES, 16, np.uint8, (784,))
    labels = _read_idx(d / _TEST_LABELS, 8, np.uint8, ()).astype(np.int32).ravel()
    return images, labels


def encode(images: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Intensity -> first-spike latency: a bright pixel fires early.

    `t = t_min + (1 - pixel/255) * (t_max - t_min)`, one spike per lit pixel, source =
    pixel index, sorted ascending by time. Returns per-sample `(times, sources)` pairs.

    Sortedness is a hard requirement, not tidiness:
    `eventprop_validate_input_no_v_th` rejects a non-decreasing list with error 60, and
    the Eigen reference returns plausible garbage for the same input. Ties (two pixels of
    equal intensity) are legitimate — the contract is non-decreasing — and `np.argsort`'s
    default kind matches the reference's, so the two encoders agree element for element.
    """
    out = []
    for row in images:
        live = np.flatnonzero(row > PIXEL_MIN)
        t = T_MIN + (1.0 - row[live].astype(np.float64) / 255.0) * (T_MAX - T_MIN)
        order = np.argsort(t)
        out.append((np.ascontiguousarray(t[order]), live[order].astype(np.int32)))
    return out


def encode_reference_literal(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A transcription of the reference loop, kept as the oracle for `encode()`.

    Slow on purpose: it is the thing we compare the vectorised version against, so that
    'we mirrored the reference' is a measurement rather than a claim about reading it.
    """
    times, sources = [], []
    for idx, pix in enumerate(image):
        if pix > PIXEL_MIN:
            times.append(T_MIN + (1 - pix / 255) * (T_MAX - T_MIN))
            sources.append(idx)
    times = np.array(times, dtype=np.float64)
    sources = np.array(sources, dtype=np.int32)
    order = np.argsort(times)
    return times[order], sources[order]


def forward_maxima(backend_key: str, wh: np.ndarray, wo: np.ndarray,
                   images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run two layers and return `(maxima_values, maxima_times)`, shaped (n, n_out).

    Hidden LIF layer -> its spikes -> an LI (integr-and-fire read-out) layer whose
    `forward` returns voltage maxima. `argmax` over those values is the VMax prediction.
    """
    be = BACKENDS[backend_key]
    hidden = be.layer(n=wh.shape[1], n_in=wh.shape[0], w_in=wh,
                      tau_mem=20e-3, tau_syn=5e-3, v_th=1.0, v_leak=0.0)
    readout = be.li_layer(n=wo.shape[1], n_in=wo.shape[0], w_in=wo,
                          tau_mem=20e-3, tau_syn=5e-3, v_leak=0.0)
    drive = encode(images)
    # Plain lists, not `np.array(..., dtype=object)`: with one sample that idiom yields a
    # 2-D object array, and `make_vector` documents why that matters.
    vector = be.make_vector([t for t, _ in drive], [s for _, s in drive])
    hidden.forward(vector)
    maxima = readout.forward(hidden.post_batch)
    values = np.array([np.asarray(m.values, dtype=np.float64) for m in maxima])
    times = np.array([np.asarray(m.times, dtype=np.float64) for m in maxima])
    return values, times


def hidden_spikes(backend_key: str, wh: np.ndarray, image: np.ndarray):
    """`(neurons, first_spike_times)` for one digit's pass through the hidden layer.

    The first spike per neuron is taken with an order-independent minimum because
    `post_batch` carries spikes in time order, not neuron order — and because the arrays
    are zero-copy views into the layer, they are copied out here while it is still
    referenced. Reading them after the layer is gone yields uninitialised memory, not an
    error.
    """
    be = BACKENDS[backend_key]
    hidden = be.layer(n=wh.shape[1], n_in=wh.shape[0], w_in=wh,
                      tau_mem=20e-3, tau_syn=5e-3, v_th=1.0, v_leak=0.0)
    (times, sources), = encode(np.asarray([image]))
    hidden.forward(be.make_vector([times], [sources]))
    n_hidden = wh.shape[1]
    first = np.full(n_hidden, np.inf)
    for s in hidden.post_batch:
        t = np.array(s.times, dtype=np.float64, copy=True)
        src = np.array(s.sources, dtype=np.int32, copy=True)
        if t.size:
            np.minimum.at(first, src, t)
    fired = np.flatnonzero(np.isfinite(first))
    return fired, first[fired]


def predict(backend_key: str, wh: np.ndarray, wo: np.ndarray, images: np.ndarray,
            batch: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """`(predictions, margins)` for `images`, in minibatches.

    The margin is top-1 minus top-2 maxima, returned because it is the number that says how
    much a prediction is worth. Measured over all 10 000 test digits for each of the ten
    published runs: no margin is ever exactly zero, none falls below 1e-9, and the smallest
    seen is 5.3e-05 (run 3). So no prediction here rests on a tie broken by `argmax`'s
    lowest-index rule — which is what rules out near-tie float noise as the explanation for
    the gap between the reproduced and recorded accuracies.

    Batching changes nothing arithmetically (measured: batch 250 and batch 7 give
    bit-identical predictions on all 10 000 test images); it exists so a 10 000-sample
    batch does not spike memory.
    """
    preds, margins = [], []
    for start in range(0, len(images), batch):
        values, _ = forward_maxima(backend_key, wh, wo, images[start:start + batch])
        preds.append(np.argmax(values, axis=1).astype(np.int32))
        top2 = np.sort(values, axis=1)[:, -2:]
        margins.append(top2[:, 1] - top2[:, 0])
    return np.concatenate(preds), np.concatenate(margins)
