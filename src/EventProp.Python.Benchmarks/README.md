# EventProp Python Benchmarks

Compares **pybind11** (`EventProp.Python`) vs **ctypes** (`eventprop`) across the Python public API surface.

## What's Benchmarked

| Category | Operations | Notes |
|----------|-----------|-------|
| **LIF Layer** | `forward()`, `backward()` | Core forward/backward pass (spike prediction) |
| **LI Layer** | `forward()`, `backward()` | Membrane potential maxima computation |
| **TTFS Loss** | `forward()`, `backward()` | First-spike-time cross-entropy (includes first spike extraction) |
| **VMax Loss** | `forward()`, `backward()` | Voltage-maxima cross-entropy (includes sum0 computation) |

These measure the **Python layer overhead** — FFI marshalling, array conversions, tuple packing/unpacking, and Python-side logic (error setup, first spike extraction). The raw C/C++ algorithm is already benchmarked separately in `EventProp.Native.Benchmarks` and `.NET EventProp.Benchmarks`.

## Problem Sizes (match .NET/C benchmarks)

| Size | Pre-synaptic | Post-synaptic | Spikes/neuron |
|------|-------------|---------------|---------------|
| Tiny | 10 | 5 | 20 |
| Small | 50 | 20 | 100 |
| Medium | 200 | 80 | 500 |
| Large | 500 | 200 | 1000 |

## Spike Patterns (match .NET/C benchmarks)

| Pattern | Generation | Notes |
|---------|-----------|-------|
| Uniform | Evenly spaced in [0, 1) | Baseline |
| Poisson | Rate-based @200Hz, strictly increasing | Realistic spiking |
| Burst | Clusters of 5 spikes, 10ms gaps between bursts | Correlated input |

All use RNG seed 42 for reproducibility.

## Running

### Prerequisites
Both libraries must be built and importable:
```bash
# Build C library (if not already done)
./build.sh c

# Install EventProp.Python (pybind11) in editable mode
cd src/EventProp.Python && pip install -e .

# Install eventprop (ctypes) in editable mode
cd src/EventProp.Native.Python && pip install -e .

# Verify both import simultaneously
PYTHONPATH="../EventProp.Python:../EventProp.Native.Python" python3 -c \
  "from eventprop.eventprop_cpp import compute_spikes_batch_cpp; from eventprop_c import NativeApi; print('OK')"
```

### Run all benchmarks
```bash
cd src/EventProp.Python.Benchmarks
PYTHONPATH="../EventProp.Python:../EventProp.Native.Python" pytest test_benchmarks.py -v --benchmark-only
```

### Run subset (e.g., Tiny/Small only)
```bash
PYTHONPATH="../EventProp.Python:../EventProp.Native.Python" pytest test_benchmarks.py -v --benchmark-only -k "Tiny or Small"
```

### Run specific operation
```bash
# LIF forward only
PYTHONPATH="../EventProp.Python:../EventProp.Native.Python" pytest test_benchmarks.py -v --benchmark-only -k "LIFForward"

# ctypes backward only
PYTHONPATH="../EventProp.Python:../EventProp.Native.Python" pytest test_benchmarks.py -v --benchmark-only -k "Backward and ctypes"
```

### Compare with .NET benchmarks
After running Python benchmarks, compare results against:
- **C/C++ native**: `src/EventProp.Native.Benchmarks` (raw algorithm, no Python overhead)
- **.NET**: `src/EventProp.Benchmarks` (cross-language comparison including Python bridge)

## Architecture

```
EventProp.Python.Benchmarks/
├── conftest.py               # Shared fixtures: sizes, patterns, helpers
├── test_benchmarks.py        # All benchmark classes (pybind11 vs ctypes)
├── test_kernel_benchmarks.py # Raw C kernels, no layer — isolates binding overhead and
│                             # publishes the impl: "C" records that close the chain
├── generate_report.py        # Standalone HTML over existing JSON (local only)
└── README.md                 # This file
```

Both libraries are imported simultaneously via `sys.path` manipulation — no reinstall needed.

## Running these benchmarks

The supported entry point is the cross-suite pipeline at the **repo root**. It runs both
Python suites (including `test_kernel_benchmarks.py`), adapts the records, merges them with
the native and .NET data, and writes `bench-reports/report_<tag>.md` plus `BENCHMARKS.md`:

```bash
cd src/EventProp.Python.Benchmarks

../../run_benchmarks.sh                # quick (Tiny/Small, skips XL)
../../run_benchmarks.sh --full         # includes XL
../../run_benchmarks.sh --suites python --no-reports --no-benchmarks-md
                                       # a scratch subset: the two flags stop it
                                       # rewriting the committed reports in place
```

This suite alone, no pipeline and no merge — pytest-benchmark JSON only:

```bash
cd src/EventProp.Python.Benchmarks
PYTHONPATH=../EventProp.Python:../EventProp.Native.Python \
  python3.14 -m pytest test_benchmarks.py test_kernel_benchmarks.py \
    --benchmark-only --benchmark-json=results_py.json
```

Name the interpreter that actually has the built `eventprop_c` extension. A bare `python`
or `python3` on macOS is frequently Xcode's (3.9.6 on the reference machine), which cannot
import the `cpython-314` extension — the suite then fails at import, not at the measurement.

> **`run_benchmarks.sh` no longer lives in this directory.** It predated
> `scripts/bench/` entirely, invoked bare `python`, ran only `test_benchmarks.py` (never
> `test_kernel_benchmarks.py`, whose `impl: "C"` kernel records close the cross-suite
> chain), emitted no `records_*.json` for the pipeline to merge, and shared its name with
> two other scripts of different scope. See the `run_benchmarks.sh` name-collision note in
> the repo-root `AGENTS.md`.

## Standalone HTML report

`generate_report.py` renders pytest-benchmark JSON plus BenchmarkDotNet JSON into one local
HTML file. **Nothing in the pipeline reads it, and it is not the committed report** —
`BENCHMARKS.md` and `bench-reports/report_<tag>.md` come from `scripts/bench/run_all.py`:

```bash
cd src/EventProp.Python.Benchmarks
python3.14 generate_report.py \
    .benchmarks/*/0002_baseline.json \
    ../EventProp.Benchmarks/BenchmarkDotNet.Artifacts/results/*.json
```

The HTML report shows:
- **Comparison tables** grouped by operation (LIF Forward, ComputeSums, etc.)
- **Backend columns**: pybind11 (Eigen), Cython (C lib), Managed (.NET), Native (C lib)
- **Ratio column**: slowdown vs fastest backend for each row
- **Summary cards**: total benchmarks, categories, backends compared

### What's Compared

| Backend | Engine | FFI Layer |
|---------|--------|-----------|
| pybind11 (Eigen) | C++ Eigen core | Python ↔ C++ via pybind11 |
| Cython (C lib) | Pure C library | Python ↔ C via Cython memory views |
| Managed (.NET) | Fully managed C# implementation | None (pure .NET) |
| Native (C lib) | Same pure C library as Cython | .NET P/Invoke via LibraryImport |

This isolates:
- **Engine comparison**: Eigen vs C algorithm efficiency
- **FFI overhead**: pybind11 vs Cython memory views vs P/Invoke
- **Managed code cost**: .NET implementation vs native paths
