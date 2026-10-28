# EventProp Visualizer

A Streamlit app that shows what EventProp actually does: exact gradients through spike
times, on the task EventProp was published with, running against the real library.

It does not train anything while you wait. Ten finished, published Yin-Yang networks ship
in this checkout (`src/EventProp.Python/paper/yinyang_{0..9}.pkl`), so every page renders
in milliseconds and shows a result that was actually measured rather than a partial run.

## Pages

| Page | What it shows |
|---|---|
| **Yin-Yang, live** | A point in the plane → four spikes → 200 LIF neurons → the class is whichever output neuron fires first. Decision boundary computed by the C library, plus the membrane potential of any hidden neuron. |
| **How it learns** | Why this dataset exists (64% shallow ceiling vs 98% deep), ten published learning curves, the boundary at each saved weight snapshot, and where the mistakes land. |
| **Exact gradients** | The adjoint: membrane potential and its root-found crossings, λ(t) on both neurons, and the exact gradient beside a central finite difference. They agree to ~1e-7 relative. |
| **Under the hood** | Interpreter/ABI status, what each backend can do from Python, C-vs-Eigen agreement on identical inputs, crude wall-clock, and one live defect report. |

## Setup

The app loads two **pre-built** extension modules from sibling projects, and CPython
extension filenames carry their ABI tag. The interpreter is therefore not a preference:
it must match the built extensions, or the app cannot see them at all.

```bash
cd src/EventProp.Web
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

If the extensions were not built for that interpreter, build them:

```bash
./build.sh --python python3.14     # from the repository root
```

`.python-version` pins 3.14 for pyenv/uv users. `eventweb/runtime.py` performs the ABI
check at startup and fails with a message naming the interpreter, the tag it wants and
the tags it found — rather than a `ModuleNotFoundError` from three frames deep.

## Run

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

## Verify

Two headless checks. Both exit non-zero on failure and neither needs a browser.

```bash
.venv/bin/python selfcheck.py    # the numbers: dataset integrity, C library vs the
                                 # accuracy the reference recorded for the same weights,
                                 # snapshot/epoch alignment, C-vs-Eigen agreement,
                                 # exact gradient vs finite differences
.venv/bin/python smoketest.py    # the pages: renders every view and several widget
                                 # interactions through streamlit.testing.v1.AppTest
```

`selfcheck.py` is the one that matters scientifically. It runs the shipped weights through
the C library and requires the accuracy to match what the reference training run recorded
for those exact weights, so a change that breaks encoding, the TTFS read-out or the weight
plumbing fails there instead of quietly drawing a wrong picture.

## Layout

```
streamlit_app.py      entry point: ABI guard, then navigation
selfcheck.py          headless verification of displayed numbers
smoketest.py          headless verification that pages render
state.py              streamlit-cached loaders (data, checkpoints, built nets)
views/                live.py · learning.py · gradient.py · internals.py
eventweb/             streamlit-free core
  runtime.py            repository root + ABI/interpreter guard
  dataset.py            Yin-Yang geometry, frozen split, TTFS encoding + read-out
  network.py            checkpoint loading, inference, decision boundary
  backends.py           the two native bindings, capabilities probed not asserted
  analysis.py           boundary grids, confusion, silence stats, gradient demo
  figures.py            plotly builders
```

`eventweb/` imports neither streamlit nor plotly-by-way-of-ui, so anything the app draws
can be recomputed and checked from a plain Python prompt.

## What it reads

| Path | Owner |
|---|---|
| `src/EventProp.Python/eventprop/yin_yang_data_set/publication_data/` | nested submodule — the published dataset (Kriener, Göltz, Petrovici, arXiv:2102.08211), MIT. Frozen 5000/1000/1000 split. |
| `src/EventProp.Python/paper/yinyang_{0..9}.pkl` | reference library's training artefacts: per-epoch accuracy/loss, per-epoch first-spike times, 11 weight snapshots. |
| `src/EventProp.C/` | the C library, via the Cython binding. |
| `src/EventProp.Python/` | read-only reference library, via pybind11. |

Both submodule paths are read-only; their local state is owned by
`scripts/submodule-state.sh`, not by this app. Never regenerate the split — the train and
test sets differ only by RNG seed, so re-drawing them overlaps the test set.

## Two things worth knowing before you trust a plot

**Voltage and adjoint traces only exist on the Eigen binding.** The C library implements
both (`src/EventProp.C/eventprop_diag.h`) and `_native.pyx` declares them, but the Cython
layer wraps no Python entry point, so the capability is real in the library and
unreachable from here. The app probes rather than assumes
(`Backend.supports_traces()`) and says so in the UI.

**`compute_lambda_i` segfaults on a neuron that never spiked.**
`post_spikes.times[post_spikes.n_spikes - 1]` is unguarded, so `n_spikes == 0` reads
`times[-1]`. Present in the reference core and inherited by `src/EventProp.Cpp.Wrapper/`;
the C library handles the case correctly by returning an all-zero trace. The app never
requests λ for a silent neuron, so the page does not crash — but the defect is still
there, and the Under-the-hood page says so out loud.

## Platform status

Built and verified on **macOS (darwin-arm64)**, CPython 3.14.10, streamlit 1.66.1,
plotly 7.1.0, numpy 2.5.4.

**Linux and Windows are unverified by this work.** Nothing here is platform-specific
except the interpreter-version requirement, which is a property of the built extensions
and applies everywhere; but the instructions above are only known-good on macOS, and the
app is wired into neither `build.sh` nor `build.ps1`, so no build step and no CI checks
that it still starts.
