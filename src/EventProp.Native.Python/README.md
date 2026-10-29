# EventProp Python Bindings (Cython)

Compiled Cython wrappers over the EventProp C library. The wrappers hand out numpy arrays that
point at C-allocated memory — no copy across the FFI — and the reference library's layer and
training API is reimplemented on top of them.

numpy is required (every layer uses it), and so is a built `libeventprop_c_api`: **this package
does not build the C library**, it links against one.

## Build

The extension is compiled from `_native.c`, which Cython generates from `_native.pyx`.
`_native.c` is deliberately untracked, so a fresh clone has none and an edited `.pyx` does not
rebuild by itself — the build scripts below regenerate it when the `.pyx` is newer.

From the repository root, which also builds the C library the extension links against:

```bash
./build.sh --python python3.14              # Windows: .\build.ps1 -PythonExe <venv python>
```

`--python` is not cosmetic. The extension is tagged for one interpreter
(`_native.cpython-314-darwin.so`), the Visualizer is pinned to CPython 3.14, and a build against
a different one produces an extension that the app cannot import.

Standalone, with the C library already built:

```bash
python3.14 -m cython eventprop_c/_native.pyx
python3.14 -m pip install -e ".[dev]" --no-build-isolation
```

Cython is a **build prerequisite**, not a dependency the build fetches; the build step probes for
it and names the interpreter that lacks it.

## Finding the shared library

Three operating systems, three mechanisms, all handled by
[`eventprop_c/_library_path.py`](eventprop_c/_library_path.py) at import:

| OS | Mechanism |
|---|---|
| macOS | rpath linked at build time; the module is a no-op |
| Linux | preloads `libeventprop_c_api.so` with `RTLD_GLOBAL`, which is what makes `EVENTPROP_C_LIB_DIR` work and survives a moved checkout where an absolute rpath cannot |
| Windows | `os.add_dll_directory` — `PATH` is excluded from extension dependency resolution since Python 3.8 |

The import also runs `abi_check()`, which compares this extension's compiled `sizeof` for every
public struct against the library's and **refuses to import on a mismatch**. A stale extension is
memory corruption rather than staleness; `docs/POSTMORTEMS.md` has the incident. Current ABI
version: **3** (`eventprop.h`, `eventprop_abi.c`).

## Two API layers

### Batch surface — `eventprop_c._native` (re-exported from `eventprop_c`)

Flat arrays in, batch wrapper out. Everything here is batch-first; a batch of one is a batch.

| Function | Returns |
|---|---|
| `compute_spikes_batch(times, sources, n_per_sample, w, n_pre, n_post, tau_mem, tau_syn, v_th)` | `times`, `sources` per sample |
| `compute_maxima_batch(times, sources, n_per_sample, w, n_pre, n_post, tau_mem, tau_syn, v_th)` | maxima `times`, `values`, plus a writable `errors` buffer |
| `backward_spikes_batch` / `backward_maxima_batch` (aliases for `…_accumulate`) | summed gradient + per-sample spike errors |
| `dropout_spikes_batch(times, sources, n_per_sample, p_drop, seed)` | surviving spikes |
| `compute_voltage_trace`, `compute_lambda_i_trace`, `compute_lambda_i_at_time` | diagnostics, below |
| `maxima_regression(...)` | continuous readout, below |
| `abi_check`, `abi_struct_sizes` | the import-time guard above |

Note that `SpikeOutput` carries **only** `times` and `sources`. The synaptic current at a spike
is rebuilt by the backward pass rather than stored — the header comment at that struct explains
why, and it is not a field to go looking for.

### Layer surface — `eventprop_c`

`LIFLayer` / `LIFLayerParameters`, `LILayer` / `LILayerParameters` (the integrate-and-fire
read-out whose `forward` returns voltage maxima), `Spikes` / `SpikesVector` / `Maxima` /
`MaximaVector`, `Layer` and its dataset/weight-distribution helpers,
`TTFSCrossEntropyLoss` and `VMaxCrossEntropyLoss`, the optimizers, and the training helpers
(`TwoLayerTTFS`, `OneLayerVMax`, `TwoLayerVMax`, `dropout_spikes`).

It mirrors the reference library's API and must reproduce its numbers; parity is asserted against
real trained checkpoints in `tests/test_paper_parity.py`, which is the only suite that runs the
shipped `paper/*.pkl` weights through both backends.

## Continuous readout and regression

EventProp's two shipped losses are classification losses over spike order and voltage maxima. A
**continuous** target needs a readout whose gradient the adjoint can propagate, so the C library
provides one:

```
y_hat = bias + Σₙ a[n] · maxima[n]
```

with least squares or smooth Huber on top, and `dL/dmaxima` written into exactly the buffer
`backward_maxima` reads. A regression step is therefore *forward maxima → regress → backward*,
with no loss arithmetic in Python at all.

It lives in the C library (`eventprop_loss.h`) rather than in either binding so that P/Invoke and
Cython compute the same gradient from the same code. The reference library has no regression loss,
so there is no C-vs-Eigen parity test for it; the guarantee is **managed ↔ native** instead,
between the C implementation and `CW.EventProp.RegressionLoss` in C#
(`src/EventProp.Tests/RegressionLossParityTests.cs`).

### Five ways to ask for it

| Call | Allocation | Readout gradients | Use when |
|---|---|---|---|
| `layer.regression(decoder, bias, targets, huber_delta=np.inf, *, decoder_grad=, bias_grad=)` | none on the C side | yes, accumulated over the batch | the training loop |
| `maxima.regress(decoder, bias, targets, huber_delta, *, decoder_grad=, bias_grad=)` | none on the C side | yes | you hold the maxima from a forward pass, not the layer |
| `batch.regress(decoder, bias, targets, huber_delta, *, decoder_grad=, bias_grad=)` | none on the C side | yes | driving the C batch directly |
| `maxima.regress_sample(idx, decoder, bias, target, huber_delta)` | none at all | no | fixed readout, or one sample needs its own target or loss scale |
| `maxima_regression(values, decoder, bias, target, huber_delta)` | C mallocs, arrays are copied out | yes, for one sample | notebooks, one-offs, debugging |

`layer` is `LILayer`, `maxima` is the `MaximaVector` its `forward()` returns, `batch` is the
underlying `_MaximaBatchOutput`. The layer form is the one to write by hand, because the step is
two calls and no loss arithmetic appears in Python at all:

```python
step = layer.regression(decoder, bias, targets)   # writes dL/dmaxima
layer.backward()                                  # dL/dw_in, summed over the batch
```

That is the same two-step the shipped `VMaxCrossEntropyLoss` performs internally; the difference
is that here you own the readout and the target. It requires a prior `forward()`, since the errors
have to land in the buffers forward allocated — and a `MaximaVector` built in Python rather than
returned by the C backend **refuses**, because it has no native error buffers to write into.

The batch forms write into the `errors` arrays `compute_maxima_batch` already allocated and return
`{"predictions", "losses", "n_skipped", "decoder_grad", "bias_grad"}`. Two details of that
contract that read as bugs if you guess:

- **The raw entry points require `huber_delta` positionally.** Only `layer.regression` defaults it
  (to least squares). Low-level code never picks a loss implicitly.
- **`decoder_grad`/`bias_grad` in the result are views of the buffers you passed**, not snapshots
  and not the same objects. Writing one is visible in the other; `is` is the wrong test, the data
  pointer is the right one.

### Rules that are not guessable

- **Gradients are sums.** Nothing divides by a batch size. A library that silently averaged half
  the time is how a learning rate ends up coupled to a batch size.
- **Gradients accumulate; nothing zeroes.** Clearing your own buffers is `ZeroGrad()`'s job, on
  both bindings.
- **A non-finite maxima value means that neuron never peaked.** It is excluded from the readout,
  given zero error, and counted in `n_skipped`. That is a valid datum, not a failure; a sample
  whose whole population is non-finite is valid too, the readout collapsing to the bias with
  `decoder_grad` legitimately zero.
- **"At rest" is not "never peaked".** A neuron whose voltage never rises still has a *finite*
  maximum of 0.0, which enters the readout like any other value — zero weights produce **no**
  skips. Measured rather than assumed, because the two phrases read as one sentence and this
  document got it wrong until a test disagreed.
- **`huber_delta` selects the loss.** Non-finite (pass `math.inf`) is least squares; a finite
  positive value is smooth Huber, `r² / (sqrt(1 + q²) + 1)` with `q = r / delta`. It is *smooth*:
  `dL/dy_hat` approaches `±delta` from below and never reaches it. The textbook spelling
  `delta²(sqrt(1+q²) − 1)` is exactly zero in floating point for small residuals, which is why the
  stable form is used and why a C test asserts the curvature survives.
- **Refusals are exceptions, never plausible numbers**: non-finite decoder weight (even for a
  neuron the readout excluded, so a poisoned weight cannot hide behind one), non-finite target or
  bias, `huber_delta ≤ 0` or NaN, and length mismatches — error codes 69–74 in `eventprop.h`.

### A whole step

```python
import numpy as np
from eventprop_c import compute_maxima_batch, backward_maxima_batch

batch = compute_maxima_batch(flat_times, flat_sources, n_per_sample, w,
                             n_pre, n_out, tau_mem, tau_syn, v_th)
step = batch.regress(decoder, bias, targets, np.inf,
                     decoder_grad=np.zeros(n_out), bias_grad=np.zeros(1))
grad_w = backward_maxima_batch(batch).grad          # dL/dw, summed over the batch

w      -= lr * np.asarray(grad_w).reshape(n_pre, n_out)
decoder -= lr * step["decoder_grad"]
bias   -= lr * float(step["bias_grad"][0])
```

The shape above was measured on a synthetic timing regression (6 inputs → 4 read-out neurons,
target a linear function of the first-spike time, plain SGD, lr 0.02, held-out set fixed).
Held-out correlation improved from **−0.51** at initialisation to **+0.54 / +0.58 / +0.58** after
200 / 600 / 1500 steps, with R² reaching **+0.34** and held-out loss falling 0.092 → 0.052. The
initial value being *negative* is the point: the fit reverses a wrong readout rather than merely
polishing a lucky one. Those figures are from that seed and setup — they are a demonstration that
the gradients train, not a benchmark, and `bench-reports/` stays the authority on cost.

One caution from the same run: plain SGD at a too-large step size can drive the readout
non-finite. When that happened it surfaced as **error 73** — *a readout weight or bias is NaN or
±Inf* — rather than as NaN gradients propagating quietly for two hundred steps. Its neighbour 74
(`EVENTPROP_NONFINITE_PREDICTION`) is the same protection one level earlier, for a readout that
summed past the finite range.

### Not included

The maxima readout only. A first-spike-latency readout is expressible the same way — push errors
at each neuron's first spike — but needs an explicit policy for silent neurons, whose first-spike
time is NaN, and that policy is a decision rather than an implementation detail.

## Diagnostics

`compute_voltage_trace` and `compute_lambda_i_trace` return `(ts, values)` as **views over the
library's buffers, not copies**: their length is your `dt`, so a copy would scale cost with a
slider. `compute_lambda_i_at_time` evaluates the same sweep at one `t` and agrees with the grid
sample it replaces.

λ is loss-driven: with no error signal on the output spikes the correct adjoint is zero
everywhere, so a flat λ trace is not necessarily a bug. The full caveats, including the reference
core's zero-spike defect that the C library does *not* inherit, are in
[`../EventProp.Web/README.md`](../EventProp.Web/README.md).

`LIFLayer` also exposes reference-style helpers: `get_voltage_trace_for_neuron`,
`get_lambda_i_trace_for_neuron`, `get_lambda_i_for_neuron` — they require a prior `forward()` on
that layer rather than recomputing one.

## Array ownership — read before holding one

Every view returned here keeps the object that owns its memory alive, so a view cannot outlive
its buffers; a new view helper must pass an owner, and that argument is required so that
forgetting is a compile error. Pinned by
[`test/test_view_ownership.py`](test/test_view_ownership.py).

Two consequences for code that switches backends: the views here are **writeable** (writing the
error buffer is how you drive the backward pass), while the pybind11 reference binding returns
**read-only** arrays — so the same stray write either raises or silently corrupts, depending on
the backend. And `ndarray.base` is not evidence about ownership: pybind11 retains its parent
through a capsule that never appears there. The rules live in the *Python bindings*
invariants in `../../AGENTS.md`.

## Tests

```bash
cd src/EventProp.Native.Python
PYTHONPATH="$PWD/../EventProp.Python:$PWD" python3.14 -m pytest test tests test_backward.py
```

`test/` covers the binding itself (ABI, layers, chains, the regression readout, trace views,
view ownership, library discovery); `tests/` is the paper-parity suite against real checkpoints.
The skips you will see are the platform-conditional loader tests for the other two operating
systems.

## Not in this package

Managed C# parity (`src/EventProp`, `src/EventProp.Native`, tested in `src/EventProp.Tests`), and
performance measurements (`src/EventProp.Native.Benchmarks`, `bench-reports/`).
