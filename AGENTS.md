# AGENTS.md — EventProp

## Project Overview

This repository implements the **EventProp** algorithm — an event-based backpropagation method for training Spiking Neural Networks (SNNs). It computes exact gradients through spike times using the Leaky Integrate-and-Fire (LIF) neuron model with synaptic filtering, leveraging the adjoint method from control theory.

### Reference Library

**`EventProp.Python/`** is a git submodule of the original Python + C++ (Eigen/pybind11) EventProp library. It is the **reference implementation** — everything else in this repo should mimic its API and produce matching input/output.

### Project Map

| Project | Language | Role |
|---------|----------|------|
| **EventProp.Python/** | Python + C++ (Eigen/pybind11) | Reference library (git submodule, read-only) |
| **EventProp.C/** | C | Pure C reimplementation with batch API and flat-array FFI variants |
| **EventProp.Cpp.Wrapper/** | C++ (Eigen) | Convenience copy of the C++ core. Used for native benchmarks and parity tests |
| **EventProp.Native.Python/** | Python + Cython | Python layer using C library via Cython bindings (`eventprop_c/_native.pyx`) |
| **EventProp.Native/** | C# (.NET 10) | P/Invoke bindings wrapping EventProp.C via `LibraryImport` |
| **EventProp/** | C# (.NET 10) | Self-contained, fully managed .NET implementation of EventProp.Python |
| **EventProp.Tests/** | C# (.NET 10) | TUnit test suite for .NET projects (correctness + parity) |
| **EventProp.Benchmarks/** | C# (.NET 10) | BenchmarkDotNet project comparing native vs managed .NET libraries |
| **EventProp.Native.Benchmarks/** | C++ (Google Benchmark) | Benchmarks comparing EventProp.C vs EventProp.Python C++ code |
| **EventProp.Native.ParityTests/** | C++ (CMake) | Standalone parity tests comparing C vs Eigen implementations |
| **EventProp.Python.Benchmarks/** | Python (pytest-benchmark) | Benchmarks comparing EventProp.Python vs EventProp.Native.Python |

## Solution Structure

```
eventprop.net/
├── build.sh                    # Build all targets (C, C++, Python, .NET)
├── clean.sh                    # Clean all or individual targets
├── src/
│   ├── EventProp.Python/       # Reference library (git submodule, read-only)
│   ├── EventProp.C/            # Pure C implementation (libeventprop_c_api)
│   │   ├── eventprop.h/.c      # Core algorithm + single-sample functions
│   │   ├── eventprop_batch.h/.c# Batch API with flat-array FFI variants
│   │   └── tests.c             # Unit tests (being rewritten to Google Test)
│   ├── EventProp.Cpp.Wrapper/  # C++ Eigen core (convenience copy, no pybind11)
│   ├── EventProp.Native/       # .NET P/Invoke bindings (C library)
│   ├── EventProp/              # Managed .NET implementation
│   ├── EventProp.Tests/        # TUnit test suite (correctness + parity)
│   ├── EventProp.Benchmarks/   # BenchmarkDotNet (native vs managed .NET)
│   ├── EventProp.Native.Python/  # Python + Cython bindings (C library)
│   │   └── eventprop_c/_native.pyx  # Thin Cython wrappers, zero-copy numpy views
│   ├── EventProp.Native.Benchmarks/  # Google Benchmark (C vs C++)
│   ├── EventProp.Native.ParityTests/ # Standalone parity tests (C vs Eigen)
│   └── EventProp.Python.Benchmarks/  # pytest-benchmark (pybind11 vs Cython)
├── coverage-report/            # Generated HTML coverage reports (gitignored)
└── AGENTS.md                   # This file
```

## Conventions

- **Target framework**: .NET 10.0 (`net10.0`)
- **Root namespace**: `CW.EventProp` (library), `CW.EventProp.Tests` (tests)
- **Nullable**: enabled across all projects
- **Implicit usings**: enabled
- **Test framework**: TUnit 1.45+ (Microsoft Testing Platform) for .NET; Google Test planned for C/C++
- **Coverage**: TUnit `--coverage --coverage-output-format=cobertura` + ReportGenerator

## Building

```bash
# Build everything — Release mode
./build.sh release

# Build everything — Debug mode
./build.sh debug

# Clean all build outputs
clean.sh              # removes everything (all targets)
clean.sh c            # only C library
clean.sh cpp          # only C++ wrapper
clean.sh python       # only Python package
clean.sh dotnet       # only .NET projects
```

### Submodule Patches

Platform-specific patches are applied automatically by `build.sh`. They live in the repo root:

| Patch | Platform | Fixes |
|-------|----------|-------|
| `submodule-macos.patch` | macOS (AppleClang + Homebrew) | Homebrew include paths, `-Xpreprocessor -fopenmp`, link args, NumPy deprecation |
| `submodule-linux.patch` | Linux (GCC/Clang) | OpenMP link flag, remove C++17 `<execution>` parallel policies, NumPy deprecation |

The submodule is read-only — patches must be reapplied after pulling from upstream.

## Running Tests & Coverage

### .NET Tests (TUnit on .NET 10)

```bash
# All tests — Release mode (required for parity tests)
cd src/EventProp.Tests && dotnet run -c Release --

# Filtered
cd src/EventProp.Tests && dotnet run -c Release -- --treenode-filter "/*/*/BackwardSpikes/*"
```

> **Note**: TUnit uses Microsoft Testing Platform (MTP), not VSTest. `dotnet test` does NOT work. The `-c` flag goes BEFORE `--`, TUnit flags go AFTER `--`. Use `--treenode-filter` instead of `--filter`.

### C/C++ Parity Tests (C vs Eigen)

```bash
cd src/EventProp.Native.ParityTests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./parity_test /path/to/libeventprop_c_api.dylib
```

### Native Benchmarks (C vs C++)

```bash
cd src/EventProp.Native.Benchmarks && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./bench_c --benchmark_min_time=0.1s
./bench_cpp --benchmark_min_time=0.1s
```

### Python Benchmarks (pytest-benchmark)

```bash
cd src/EventProp.Python.Benchmarks && pytest test_benchmarks.py -v
```

### Coverage Report Generation

```bash
# Run with coverage
cd src/EventProp.Tests && dotnet run -c Release -- --coverage --coverage-output-format=cobertura

# Generate HTML report
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Release/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html
```

## C Library API Design

### Batch-First Architecture

The C library (`eventprop_batch.h`) provides batch operations as the primary interface. All batch functions accept per-sample input arrays and return typed result structs. Callers free via corresponding `_free_` functions.

**Core batch types:**

| Type | Role | Free Function |
|------|------|---------------|
| `SpikesBatch` | Forward spike output (borrowed inputs) | `eventprop_free_spikes_batch` |
| `SpikesBatchWithInputs` | Forward spike output (owned inputs, flat-FFI variant) | `eventprop_free_spikes_with_inputs_batch` |
| `MaximaBatch` / `MaximaBatchWithInputs` | Maxima output | `eventprop_free_maxima_batch` / `_with_inputs` |
| `BackwardsBatch` | Per-sample backward results | `eventprop_free_backwards_batch` |
| `AccumulatedBackwardResult` | Combined backward + summed gradient | `eventprop_free_accumulated_backward_result` |
| `DropoutBatch` | Dropout output (per-sample surviving spikes) | `eventprop_free_dropout_batch` |

### Flat-Array FFI Variants

Functions suffixed `_flat` accept flat numpy-compatible arrays and per-sample offset counts. They are designed for Cython/P/Invoke callers that have contiguous buffers:

```c
SpikesBatchWithInputs eventprop_compute_spikes_batch_flat(
    const double* w, int n_pre, int n_post,
    const double* times, const int* sources, const int* n_per_sample,
    int batch_size, double tau_mem, double tau_syn, double v_th);

MaximaBatchWithInputs eventprop_compute_maxima_batch_flat(...);
DropoutBatch eventprop_dropout_spikes_batch(
    const double* times, const int* sources, const int* n_per_sample,
    int batch_size, double p_drop, int seed);
```

### Combined Backward + Accumulate

The primary FFI entry point for backward pass is the combined function — one call, zero intermediate allocations in Python:

```c
AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(
    const EventPropInput* inputs, const SpikesBatch spikes_batch);

AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
    const EventPropInput* inputs, const SpikesBatch spikes_batch);
```

Returns `GradientOutput summed_grads` (ready for weight update) + per-sample `.spike_errors` preserved. Per-sample `.grad` buffers are freed after summation.

### Memory Ownership Rules

- Batch free functions take pointers (`Type*`) to safely null out internal fields after freeing
- Individual output free functions exist for all types: `eventprop_free_spike_output`, `eventprop_free_maxima_output`, `eventprop_free_backward_output`, `eventprop_free_dropout_output`
- Dimension fields are removed from output structs — dimensions are inferred from input parameters

## Python Bindings (`_native.pyx`)

### Architecture

Thin Cython wrappers with managed heap-backed wrapper classes that own C memory:

| Wrapper Class | Owns | Provides |
|---------------|------|----------|
| `_SpikesBatchOutput` | `SpikesBatchWithInputs*` | Zero-copy views via `.get_spike_times(i)`, `.get_sources(i)` |
| `_MaximaBatchOutput` | `MaximaBatchWithInputs*` | Zero-copy views for maxima times/values |
| `_AccumulatedBackwardResult` | `AccumulatedBackwardResult*` | `.grad` (flat), `.rows`, `.cols`, `.get_spike_errors(i)` |
| `_DropoutBatchOutput` | `DropoutBatch*` | Zero-copy views via `[i]` → `(times, sources)` tuple |

All wrappers use `__dealloc__` to call the appropriate C free function. Numpy views are created via `PyArray_SimpleNewFromData` (zero-copy).

### Layer Chain Flow

1. Python constructs input once via flat variant (`compute_spikes_batch_flat`)
2. All subsequent batch calls receive direct pointers to C-owned memory from previous returns
3. No concatenation, no array rebuilding in Python after initial input
4. Backward pass uses combined accumulate function — one FFI call per layer

## Architecture Notes (.NET)

### Zero-Allocation Hot Paths

- **`MatrixView`** — `ref struct` providing zero-copy row-major matrix views over flat `double[]` arrays
- **`StridedSpan`** — column access without allocation (strided read-only span)
- **`ScratchBuffers`** — pre-allocated reusable buffers; callers allocate once and pass in

### Biophysical Parameters

All functions share these parameters:
- `tauMem` — membrane time constant (seconds)
- `tauSyn` — synaptic time constant (seconds)
- `vTh` — spike threshold voltage

### Weight Matrix Layout

Row-major: `w[row * cols + col]` where rows = pre-synaptic neurons, cols = post-synaptic neurons.

## Key Algorithms

- **`ComputeSums`**: Pre-computes cumulative exponential-weighted sums for fast membrane potential evaluation
- **`ComputeSpikes`**: Forward pass — finds threshold-crossing times via root-finding (Toms748/Brent)
- **`Backward`**: Adjoint method — time-reversed sweep propagating λ_v (membrane) and λ_i (synaptic) adjoints
- **`ComputeMaxima`**: Finds membrane potential maxima per neuron (for voltage-based loss)

## Platform Support

- **macOS** — developed and tested
- **Linux** — intended support
- **Windows** — supported via MSVC + CMake (VS Code workspace configs provided)
