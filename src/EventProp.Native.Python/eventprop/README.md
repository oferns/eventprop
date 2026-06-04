# EventProp Ctypes Bindings

Zero-copy access to the EventProp C library using Python ctypes. No numpy required.

## Features

- **Zero-copy memory access** — Direct pointer access to C-allocated arrays
- **No external dependencies** — Only uses Python standard library (ctypes)
- **Automatic cleanup** — C memory freed on garbage collection
- **Memoryview support** — Compatible with Python's built-in memoryview

## Usage

```python
from eventprop_ctypes import EventPropCtypes

# Load library (provide path to libeventprop_c_api.so/dylib)
lib = EventPropCtypes("/path/to/libeventprop_c_api.dylib")

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

### EventPropCtypes

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

### Array Views

Zero-copy array access classes:

- **DoubleArrayView**: For double arrays (times, currents, sums)
  - `__len__()`, `__getitem__(index)`, `__iter__()`
  - `to_list()` — Convert to Python list (copy)
  
- **Int32ArrayView**: For int32 arrays (sources, indices)
  - Same interface as DoubleArrayView

## Performance Characteristics

- **No allocation** for array views — just pointer manipulation
- **Zero memory copying** when accessing C data directly
- **Minimal overhead** for element access (single pointer dereference)

## Example

```python
from eventprop_ctypes import load_library

# Quick access to library functions
lib = load_library("/path/to/libeventprop_c_api.dylib")

# Compute spikes with zero-copy access
output = lib.compute_spikes(w, times, sources)

# Process spike data without copying
for i in range(output.n_spikes):
    t = output.times[i]  # Zero-copy access
    s = output.sources[i]  # Zero-copy access
    
    if t > threshold:
        print(f"Late spike at {t:.6f} from neuron {s}")

# Cleanup happens automatically
```

## Installation

No installation required — just import the module:

```python
import sys
sys.path.insert(0, "/path/to/eventprop_ctypes")

from eventprop_ctypes import EventPropCtypes
```
