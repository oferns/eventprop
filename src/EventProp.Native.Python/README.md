# EventProp Python Bindings (ctypes)

Zero-copy access to the EventProp C library using Python ctypes. No numpy required.

## Features

- **Zero-copy memory access** — Direct pointer access to C-allocated arrays
- **No external dependencies** — Only uses Python standard library (ctypes)
- **Automatic cleanup** — C memory freed on garbage collection
- **Memoryview support** — Compatible with Python's built-in memoryview

## Installation

```bash
pip install .
```

Or use directly without installation:

```python
import sys
sys.path.insert(0, "/path/to/eventprop")

from eventprop import NativeApi
```

## Usage

```python
from eventprop import NativeApi

# Load library (provide path to libeventprop_c_api.so/dylib)
lib = NativeApi("/path/to/libeventprop_c_api.dylib")

# Forward pass with zero-copy access
output = lib.compute_spikes(w, times, sources)

# Zero-copy iteration over spike data (no allocation!)
for t in output.times:  # Direct pointer access, no copy
    print(f"Spike at {t:.6f}")

# Access individual elements (no copy)
first_time = output.times[0]
first_source = output.sources[0]

# Convert to Python list (copy) if needed
times_list = output.times.to_list()

# Cleanup is automatic on garbage collection
```

## API Reference

### NativeApi (EventPropCtypes)

Main library wrapper providing access to all alloc functions:

- `compute_spikes(w, times, sources, v_th=1.0, tau_mem=0.02, tau_syn=0.005)`
- `compute_sums(w, times, sources, tau_mem=0.02, tau_syn=0.005)`
- `compute_maxima(w, times, sources, tau_mem=0.02, tau_syn=0.005)`
- `backward_spikes(...)` — Backward pass for spikes loss
- `backward_maxima(...)` — Backward pass for maxima loss

### Output Classes

All output classes provide zero-copy array views:

- **SpikeOutput**: `n_spikes`, `times`, `sources`, `currents`
- **SumsOutput**: `n_rows`, `n_post`, `sum0`, `sum1`
- **MaximaOutput**: `n_post`, `times`, `values`
- **BackwardSpikesOutput**: `n_pre`, `n_post`, `grad`, `spike_errors`
- **BackwardMaximaOutput**: `n_pre`, `n_post`, `gradient`, `spike_errors`

## Performance Characteristics

- **No allocation** for array views — just pointer manipulation
- **Zero memory copying** when accessing C data directly
- **Minimal overhead** for element access (single pointer dereference)

## Testing

```bash
python -m pytest test/
```

Or run the zero-copy tests directly:

```bash
python eventprop/test_zerocopy.py
```

## License

MIT License — see LICENSE file for details.
