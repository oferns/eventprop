# EventProp Visualizer

A Streamlit app that shows what EventProp actually does: exact gradients through spike
times, on the task EventProp was published with, running against the real library.

It does not train anything while you wait. Ten finished, published Yin-Yang networks ship
in this checkout (`src/EventProp.Python/paper/yinyang_{0..9}.pkl`), so every page renders
in milliseconds and shows a result that was actually measured rather than a partial run.

The same directory holds ten published **MNIST** runs (`mnist_vmax_{0..9}.pkl` — VMax loss,
784→350→10, written in a different on-disk layout). The **Digits** page runs those weights
over the real 10 000-digit test set, which makes it the one page that is not synthetic —
and the one that needs a download, because the pixel files are not vendored here (see
Verify below).

**Reading a number** is the one page that does compute while you wait: it fits a continuous
read-out over the hidden layer's voltage maxima in about a second. Frozen published weights are
still the point — the fit moves the read-out (and optionally the weights) from a published start,
and every number it prints is measured on held-out points.

## Pages

| Page | What it shows |
|---|---|
| **Yin-Yang, live** | A point in the plane → four spikes → 200 LIF neurons → the class is whichever output neuron fires first. Decision boundary computed by the C library, plus the membrane potential of any hidden neuron. |
| **How it learns** | Why this dataset exists (64% shallow ceiling vs 98% deep), ten published learning curves, the boundary at each saved weight snapshot, and where the mistakes land. |
| **Exact gradients** | The adjoint: membrane potential and its root-found crossings, λ(t) on both neurons, and the exact gradient beside a central finite difference. They agree to ~1e-7 relative. |
| **Digits (MNIST)** | The published VMax networks on the real 10 000-digit test set — the maxima read-out instead of first spikes, one digit followed end to end, the reproduced accuracy beside the one recorded in the file, and what each backend costs per digit. Needs a one-time ~1.6 MB download. |
| **Reading a number** | A *continuous* target instead of a class: one weighted sum over the 200 hidden voltage maxima, fitted live against the C library's regression loss, with the closed-form linear baselines drawn alongside. C backend only — the reference library has no regression loss to compare with. |
| **Under the hood** | Interpreter/ABI status, what each backend can do from Python, C-vs-Eigen agreement on identical inputs, crude wall-clock, and one live defect report. |

## Setup

The app loads two **pre-built** extension modules from sibling projects, and CPython
extension filenames carry their ABI tag. The interpreter is therefore not a preference:
it must match the built extensions, or the app cannot see them at all.

macOS / Linux:

```bash
cd src/EventProp.Web
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell) — a Windows venv keeps its interpreter in `Scripts\`, not `bin\`:

```powershell
cd src\EventProp.Web
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the extensions were not built for that interpreter, build them from the repository
root, naming the interpreter that will run the app:

```bash
./build.sh --python python3.14
```

```powershell
.\build.ps1 -PythonExe (Resolve-Path .\src\EventProp.Web\.venv\Scripts\python.exe).Path
```

Steps 7–8 pip-install the two Python layers **for the interpreter you named only**, so
the flag is not cosmetic: build with the default `python` and the app's venv still cannot
import them. On Windows the C library lands in `src\EventProp.C\build\Release\` because
MSVC's generator is multi-config; `eventprop_c/_library_path.py` probes `build\Release`,
`build\Debug` and `build`, and registers the directory with `os.add_dll_directory`, so
there is nothing to add to `PATH` (the loader ignores it for extension modules anyway).

`.python-version` pins 3.14 for pyenv/uv users. `eventweb/runtime.py` performs the ABI
check at startup and fails with a message naming the interpreter, the tag it wants and
the tags it found — rather than a `ModuleNotFoundError` from three frames deep.

## Run

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

```powershell
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Below, `.venv/bin/python` stands for `.venv\Scripts\python.exe` on Windows.

## Verify

Two headless checks. Both exit non-zero on failure and neither needs a browser.

```bash
.venv/bin/python selfcheck.py    # the numbers: dataset integrity, C library vs the
                                 # accuracy the reference recorded for the same weights,
                                 # snapshot/epoch alignment, C-vs-Eigen agreement,
                                 # exact gradient vs finite differences, checkpoint-layout
                                 # identification, the ten published MNIST runs, the
                                 # continuous read-out and its chain rule
.venv/bin/python smoketest.py    # the pages: renders every view and several widget
                                 # interactions through streamlit.testing.v1.AppTest
```

`selfcheck.py` is the one that matters scientifically. It runs the shipped weights through
the C library and requires the accuracy to match what the reference training run recorded
for those exact weights, so a change that breaks encoding, the TTFS read-out or the weight
plumbing fails there instead of quietly drawing a wrong picture.

The MNIST steps **skip rather than fail** when `.cache/mnist/` is empty, so the script stays
runnable on a machine with no network. Open the Digits page once, or drop the two t10k files
into that directory, and steps 12 and 13 run: step 12 asserts the encoder is byte-identical
to the reference's, that both backends produce identical predictions, and that the recorded
accuracy is reproduced over all 10 000 digits to within 0.002. Step 13 checks the page's
timing panel against a stopwatch pass over the same test set.

Step 15 covers the continuous read-out, which is the one part of the app with **no second
implementation to disagree with**: the loss lives in the C library and the reference library has
none, so the usual check ("both backends agree") is unavailable. What is checked instead is the
plumbing this app adds on top of the C entry point — the standardised parameterisation agrees with
`eventprop_maxima_regression` on every prediction (3e-16), its chain rule matches finite
differences of the C loss for both loss shapes (7e-12), a non-finite maximum leaves the read-out
instead of poisoning it, the fit reaches held-out R² > 0.98 while the random initial read-out sits
below zero, the adjoint sweep moves the network's own weights, and exact least squares really does
interpolate (train R² > 0.999) while the validation-tuned ridge holds out at least as well on both
coordinates.

## Layout

```
streamlit_app.py      entry point: ABI guard, then navigation
selfcheck.py          headless verification of displayed numbers
smoketest.py          headless verification that pages render
state.py              streamlit-cached loaders (data, checkpoints, built nets)
views/                live.py · learning.py · gradient.py · digits.py · regression.py
                      · internals.py
eventweb/             streamlit-free core
  runtime.py            repository root + ABI/interpreter guard
  dataset.py            Yin-Yang geometry, frozen split, TTFS encoding + read-out
  network.py            checkpoint loading, inference, decision boundary
  backends.py           the two native bindings, capabilities probed not asserted
  analysis.py           boundary grids, confusion, silence stats, gradient demo
  figures.py            plotly builders
  regression.py         the continuous read-out: C fit loop, ridge baseline, per-neuron probe
```

`eventweb/` imports neither streamlit nor plotly-by-way-of-ui, so anything the app draws
can be recomputed and checked from a plain Python prompt.

## What it reads

| Path | Owner |
|---|---|
| `src/EventProp.Python/eventprop/yin_yang_data_set/publication_data/` | nested submodule — the published dataset (Kriener, Göltz, Petrovici, arXiv:2102.08211), MIT. Frozen 5000/1000/1000 split. |
| `src/EventProp.Python/paper/yinyang_{0..9}.pkl` | reference library's training artefacts: per-epoch accuracy/loss, per-epoch first-spike times, 11 weight snapshots. |
| `src/EventProp.Python/paper/mnist_vmax_{0..9}.pkl` | ten published MNIST runs (VMax loss, 784→350→10), 7-field layout, final weights only. Loaded and verified by `selfcheck.py`; rendered by the Digits page. |
| `.cache/mnist/` (created on demand, gitignored) | the t10k pixel files, fetched from the same mirror the reference library's loader uses. Not vendored: redistribution terms for the corpus should be checked by whoever ships it, not assumed by this repository. |
| `src/EventProp.C/` | the C library, via the Cython binding. |
| `src/EventProp.Python/` | read-only reference library, via pybind11. |

Both submodule paths are read-only; their local state is owned by
`scripts/submodule-state.sh`, not by this app. Never regenerate the split — the train and
test sets differ only by RNG seed, so re-drawing them overlaps the test set.

## Five things worth knowing before you trust a plot

**λ(t) is loss-driven, so an all-zero λ trace is not necessarily a bug.** The adjoint jumps
by `1/(i - v_th) · (v_th · λ_v + error)` at each output spike: with no error signal on those
spikes the correct answer is zero everywhere. The trace pages set an error before asking, and
`selfcheck.py` asserts the traced adjoint is non-trivial before it compares anything —
otherwise a comparison of two zero traces looks like agreement.

Both bindings produce both traces, and `selfcheck.py` compares them (voltage to machine
precision, adjoint to ~1e-13 on a shipped network). Availability is still *probed*
(`Backend.supports_traces()`) rather than hard-coded, because the C library exported these
functions long before the Cython layer wrapped them — a table that read the header would
promise a plot the binding could not deliver.

**`compute_lambda_i` segfaults on a neuron that never spiked.**
`post_spikes.times[post_spikes.n_spikes - 1]` is unguarded, so `n_spikes == 0` reads
`times[-1]`. Present in the reference core and inherited by `src/EventProp.Cpp.Wrapper/`;
the C library handles the case correctly by returning an all-zero trace. The app never
requests λ for a silent neuron, so the page does not crash — but the defect is still
there, and the Under-the-hood page says so out loud.

**Every timing in the app is an existence proof, not a result.** The Digits page can time
the maxima pass and Under the hood can time forward/backward on Yin-Yang; both are medians
taken inside a Streamlit handler, where the tail belongs to the garbage collector and the
request loop rather than to the kernel. The measured numbers are in
`../bench-reports/report_<tag>.md`, with provenance (`vcs_rev`, `cpu`, `tool_version`, dirty
flags) and repeat statistics, and `merge.py` refuses to combine them across hosts. Per
`BENCHMARK_SCHEMA.md`, compare per-pair geomeans across hosts, never the µs columns. The
digits panel keeps encoding, marshalling and layer construction *outside* its timer and
prints them separately — leave them inside and you have measured this page instead of the
library — and `selfcheck.py` step 13 checks its per-digit figure and its 10 000-digit
projection against a stopwatch pass over the real test set. That cross-check is load-bearing:
a derived number has nothing forcing it to be in the right units, and the one version of it
that did not survive the check was wrong by a factor of 250.

**Spike and maxima arrays handed to a plot are views, and only one backend lets you write them.**
Both bindings hand out views into their own buffers — reading `spikes.times` twice returns two
arrays that share memory on *both* backends — so a view is not scratch space: writing into a
`times` or `values` view changes what the backend reports. They differ on what happens when you
write anyway. The reference pybind11 binding returns **read-only** arrays, so a stray write
raises `ValueError: assignment destination is read-only`; the Cython binding's are **writeable**,
because writing error signals is the supported way to drive the backward pass — and the same
stray write therefore corrupts the library's buffer instead of raising. That asymmetry is the
reason `eventweb` copies out wherever it wants to hold or mutate something
(`eventweb/mnist.py::hidden_spikes`), and it is a convention rather than a workaround: the
reference library is read-only, so there is nothing to fix upstream, only something to code
against.

Both are safe to *read* for as long as you hold the array, which is worth stating because the
obvious probe lies. Measured with only the array kept and every owner name dropped — layer,
spike vector, maxima object — through 200 rounds of allocator churn plus `gc.collect()`: values
identical on both backends. The reference binding keeps its parent object alive through a capsule
pybind11 attaches, which does **not** appear as `ndarray.base`; so `base is None` does not mean
"this may dangle". Our own binding keeps the wrapper as the array's base object, pinned by
`../EventProp.Native.Python/test/test_view_ownership.py`. Measured on darwin-arm64, CPython 3.14;
two reads sharing memory was true on both, as was read-only versus writeable.

**The dashed line on the read-out page is a baseline, not a ceiling — and the read-out is C-only.**
Both halves of that sentence were learned by getting it wrong. *C-only*: the regression read-out
(`ŷ = b + Σ aₙ·maximaₙ`, least squares or smooth Huber, `dL/dmaxima` written into the buffers the
maxima backward pass reads) exists in the C library and the read-only reference library ships no
regression loss at all, so that page has no backend picker to offer and no cross-backend agreement
to show; the parity guarantee for this loss is managed ↔ native in .NET
(`RegressionLossParityTests`, and `RegressionLayerParityTests`, which asserts the whole
`Regression → Backward` chain rather than just the numbers). *Baseline*: the page draws the
closed-form linear reference beside the live fit, and it is tempting to call that line a ceiling.
It is not. With 400 training points and 201 parameters, exact least squares interpolates — train
R² = 1.0000 — and then generalises worse than a regularised fit (held-out 0.9956 for `x`, 0.9435
for `y`; at 200 points it collapses to −0.26). Ridge with the penalty chosen on the dataset's own
validation split — never on the test points the metrics come from — holds out at 0.9986 / 0.9972,
and a few hundred steps of Adam on the read-out reach 0.9957. So a fit finishing above the exact
answer has found regularisation by early stopping, which is a real effect and now written down
where it confused us first.

Two more things about that fit are load-bearing and easy to dismiss as tuning noise. The maxima of
a trained population sit on a common offset (mean ≈0.84, median per-neuron spread ≈0.07), so the
feature matrix is near-collinear (condition number ≈3e6). Adam absorbs most of that — measured at
the page's settings, standardised reaches held-out R² 0.9968 and raw maxima 0.9353, so the toggle
is a quality knob once Adam is in place — but on plain gradient descent the raw parameterisation
ran to a non-finite loss at every step size that made progress, which is how Adam got there first.
The page writes the read-out on
`z = (maxima − μ)/s`, which is the *same model* re-parameterised — `b′ + Σ aₙ/sₙ · maximaₙ` is what
the C entry point is handed — and Adam handles the rest. Converting the loss's gradients back is
where a bug would hide: `dL/daₙ = (dL/ddₙ − (dL/dβ)·μₙ)/sₙ`, and dropping the `μ` term biases every
fit while raising nothing. It is worth O(10) here, so step 15 checks it against finite differences
of the C loss rather than against the algebra that produced it (max deviation 7e-12, both loss
shapes). Gradients come back **summed** over the batch, by the C contract, and the page divides by
the batch size before touching them, so the learning rate does not silently depend on the batch.
The read-out is also a working regression demo: training the weights as well as the read-out moves
the published weights by up to 6e-4 with max |dL/dW| ≈ 0.05 in 120 steps, which is the loss
reaching the network through the adjoint sweep rather than stopping at the read-out.

## Platform status

Built and verified on **macOS (darwin-arm64)**, CPython 3.14.10, streamlit 1.66.1,
plotly 7.1.0, numpy 2.5.4.

Confirmed on **Linux (x86_64, GCC)** with CPython 3.12.3, streamlit 1.66.1, plotly 7.1.0,
numpy 2.5.4: the server answers, `smoketest.py` renders every page, and `selfcheck.py`
passes end to end — exact gradient vs finite differences and C-vs-Eigen trace agreement
included. Note it ran on **3.12, not the pinned 3.14**: `eventweb/runtime.py` compares the
running interpreter's tag against the extension files actually present, so `.python-version`
records the interpreter this app was *verified* on rather than one the guard enforces. Any
interpreter runs the app once the extensions carry its tag — but `./build.sh --python <that
interpreter>` re-tags **both** modules, and the interpreter it replaces can then no longer
import them.

On Linux the Cython layer resolves `libeventprop_c_api.so` through a baked **absolute**
`RUNPATH` pointing at `src/EventProp.C/build` (verify with `readelf -d`), so nothing goes in
`LD_LIBRARY_PATH` — and `clean.sh c` takes the app down with it, because the directory in
that RUNPATH is what disappears.

**Windows is unverified by this work.** Nothing here is platform-specific except the
interpreter/ABI requirement above, which is a property of the built extensions and applies
everywhere; the app is wired into neither `build.sh` nor `build.ps1`, so no build step and no
CI checks that it still starts.
