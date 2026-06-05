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
├── conftest.py          # Shared fixtures: sizes, patterns, helpers
├── test_benchmarks.py   # All benchmark classes (pybind11 vs ctypes)
└── README.md            # This file
```

Both libraries are imported simultaneously via `sys.path` manipulation — no reinstall needed.
