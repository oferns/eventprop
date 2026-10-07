# AGENTS.md — EventProp

## Project Overview

This repository implements the **EventProp** algorithm — an event-based backpropagation method for training Spiking Neural Networks (SNNs). It computes exact gradients through spike times using the Leaky Integrate-and-Fire (LIF) neuron model with synaptic filtering, leveraging the adjoint method from control theory.

### Reference Library

**`EventProp.Python/`** is a git submodule of the original Python + C++ (Eigen/pybind11) EventProp library. It is the **reference implementation** — everything else in this repo should mimic its API and produce matching input/output.

### Project Map

| Project | Language | Role |
|---------|----------|------|
| **EventProp.Python/** | Python + C++ (Eigen/pybind11) | Reference library (git submodule, read-only) |
| **EventProp.C/** | C | Pure C reimplementation with a batch-first API (per-sample input arrays) |
| **EventProp.Cpp.Wrapper/** | C++ (Eigen) | Convenience copy of the C++ core. Used for native benchmarks and parity tests |
| **EventProp.Native.Python/** | Python + Cython | Python layer using C library via Cython bindings (`eventprop_c/_native.pyx`) |
| **EventProp.Native/** | C# (.NET 10) | P/Invoke bindings wrapping EventProp.C via `LibraryImport` |
| **EventProp/** | C# (.NET 10) | Self-contained, fully managed .NET implementation of EventProp.Python |
| **EventProp.C.Tests/** | C++ (Google Test v1.15.2) | Unit tests for EventProp.C library (201 tests, ASan-compatible) |
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
│   │   ├── eventprop.h         # Public API + single-sample functions
│   │   ├── eventprop_validation.c  # Input validation + error strings
│   │   ├── eventprop_simd.c/h      # SIMD vector helpers (AVX2/NEON, exported non-static)
│   │   ├── eventprop_memory.c      # Public output-freeing functions
│   │   ├── eventprop_sums.c        # Cumulative exponential sums
│   │   ├── eventprop_maxima.c      # Maxima + voltage-based backward pass
│   │   ├── eventprop_spikes.c      # Forward pass (TOMS748/Brent root finding)
│   │   ├── eventprop_backward.c    # Adjoint (lambda) sweep backward pass
│   │   ├── eventprop_internal.h    # Shared constants, VOLTAGE macro, context types
│   │   ├── eventprop_batch.h/.c    # Batch-first API (per-sample input arrays)
│   │   ├── eventprop_diag.h/.c     # Diagnostic functions (voltage traces, lambda sweeps)
│   │   └── CMakeLists.txt          # Build config: strict warnings, SIMD, OpenMP, ASan
│   ├── EventProp.Cpp.Wrapper/  # C++ Eigen core (convenience copy, no pybind11)
│   ├── EventProp.Native/       # .NET P/Invoke bindings (C library)
│   ├── EventProp.C.Tests/      # Google Test v1.15.2 suite for C library (201 tests)
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
- **Test framework**: TUnit 1.45+ (Microsoft Testing Platform) for .NET; Google Test v1.15.2 for C/C++ (`EventProp.C.Tests/`)
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

### C Library Unit Tests (Google Test)

```bash
cd src/EventProp.C.Tests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./eventprop_tests
```

**With AddressSanitizer** (Debug builds only, detects use-after-free, buffer overflows):
```bash
cd src/EventProp.C/build && rm -rf * && cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON && cmake --build .
cd ../EventProp.C.Tests/build && rm -rf * && cmake .. -DENABLE_ASAN=ON && cmake --build .
./eventprop_tests
```
> **Note**: On macOS, ASan automatically builds a static library (`.a`) instead of shared (`.dylib`) due to AppleClang linker limitations.

### C/C++ Parity Tests (C vs Eigen)

```bash
cd src/EventProp.Native.ParityTests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./parity_test /path/to/libeventprop_c_api.dylib
```

### Native Benchmarks (C vs C++) — Standard Workflow

**Use the benchmark scripts** (`run_benchmarks.bat` on Windows, `run_benchmarks.sh` on Linux/macOS). These handle building, running both benchmarks, saving JSON to `baselines/`, and invoking Google Benchmark's `compare.py` for human-readable comparison reports.

```bash
# Build first (if not already built)
cd src/EventProp.Native.Benchmarks && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .

# Run full suite and compare (Windows)
cd src/EventProp.Native.Benchmarks
cmd /c "run_benchmarks.bat"

# Run full suite and compare (Linux/macOS)
cd src/EventProp.Native.Benchmarks
./run_benchmarks.sh
```

**Script modes:**

| Command | Action |
|---------|--------|
| `run_benchmarks` (no args) | Run C + C++, save JSON, auto-compare latest results |
| `run_benchmarks c` | Run only C benchmark |
| `run_benchmarks cpp` | Run only C++ benchmark |
| `run_benchmarks compare` | Compare last saved results (no re-run) |
| `run_benchmarks compare A.json B.json` | Compare two specific JSON files |

**Environment variables** (set before calling):

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_TIME` | `0.1s` | Minimum benchmark time per test (`--benchmark_min_time`) |
| `FILTER` | *(none)* | Regex filter for benchmarks (`--benchmark_filter`) |

**Example — quick run of a single benchmark family:**
```bash
# Windows
cmd /c "set MIN_TIME=0.05s & set FILTER=ComputeSpikes/Tiny & run_benchmarks.bat"

# Linux/macOS
MIN_TIME=0.05s FILTER=ComputeSpikes/Tiny ./run_benchmarks.sh
```

**Output:**
- JSON results saved to `src/EventProp.Native.Benchmarks/baselines/bench_c_<id>.json` and `bench_cpp_<id>.json`
- Comparison report printed to stdout via Google Benchmark's `compare.py`
- Results are gitignored (`baselines/*.json`) but scripts are tracked

> **Note**: Full suite runs can take several minutes. Use `FILTER` for quick validation of specific benchmark families.

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

## C Library Coding Style

### Error Handling: Single-Return Pattern

All functions use a single `goto cleanup_and_return` at the end. On error, set `result.error_no` and jump to cleanup. Output free functions are only called when `result.error_no != EVENTPROP_OK`, preserving result data for successful returns.

```c
ResultType eventprop_foo(const Input input) {
    ResultType result = {0};
    SomeCtx ctx = {0};  // context groups heap allocations

    if (eventprop_validate_input(input)) return result;

    ctx.data = malloc(...);
    if (!ctx.data) {
        result.error_no = EVENTPROP_SPECIFIC_ALLOC_CODE;
        goto cleanup_and_return;
    }

    /* ... main logic ... */

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) free_ctx(&ctx);
    return result;
}
```

### Error Codes: Specific, Not Generic

Each allocation site gets its own error code. No generic `EVENTPROP_ALLOCATION_FAILED` — callers should know exactly which allocation failed.

### Context Structs for Cleanup

Group related heap allocations into a context struct with a single free function. This eliminates copy-pasted cleanup across multiple error paths and prevents forgotten frees.

### Sequential Allocation Checks

Never do simultaneous `realloc` calls (e.g., `realloc(a)` then `realloc(b)`). If the first succeeds but the second fails, you lose the original pointer to `a`. Always check each allocation before moving to the next:

```c
double* new_a = realloc(old_a, ...);
if (!new_a) { result.error_no = CODE_A; goto cleanup_and_return; }
a = new_a;

double* new_b = realloc(old_b, ...);
if (!new_b) { free(new_a); result.error_no = CODE_B; goto cleanup_and_return; }  // undo a
b = new_b;
```

### Designated Initializers

Use `{ .field = value }` syntax for struct initialization — remaining fields are zero-initialized per C99 §6.7.8/20.

### Magic Numbers as Named Constants

Extract hardcoded thresholds to named constants at the top of files (e.g., `TOMS748_NEAR_ZERO`, `SPIKE_THRESHOLD_EPS`).

### Compiler Warnings

The CMake build enables strict warnings: `-Wall -Wextra -Wpedantic -Wshadow -Wimplicit-fallthrough -Wformat=2 -Wnull-dereference` (GCC/Clang), `/W4` (MSVC). GCC additionally gets `-Wduplicated-cond`, `-Wmisleading-indentation`, `-Wlogical-op`. Code must compile with zero warnings.

### Memory Testing: AddressSanitizer

Both library and test CMakeLists support `-DENABLE_ASAN=ON` for memory error detection (use-after-free, buffer overflow). On macOS this builds a static library instead of shared (AppleClang limitation).

```bash
cd src/EventProp.C/build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build .
cd ../EventProp.C.Tests/build && cmake .. -DENABLE_ASAN=ON
cmake --build .
./eventprop_tests
```

## C Library API Design

### Batch-First Architecture

The C library (`eventprop_batch.h`) provides batch operations as the primary interface. All batch functions accept per-sample input arrays and return typed result structs. Callers free via corresponding `_free_` functions.

**Core batch types:**

| Type | Role | Free Function |
|------|------|---------------|
| `SpikesBatch` | Forward spike output (per-sample, borrowed inputs) | `eventprop_free_spikes_batch` |
| `MaximaBatch` | Maxima output (per-sample) | `eventprop_free_maxima_batch` |
| `BackwardsBatch` | Per-sample backward results | `eventprop_free_backwards_batch` |
| `AccumulatedBackwardResult` | Combined backward + summed gradient (`GradientOutput*`) | `eventprop_free_accumulated_backward_result` |
| `DropoutBatch` | Dropout output (per-sample surviving spikes) | `eventprop_free_dropout_batch` |

> **Note:** Batch structs expose an atomic `has_errors` flag (`atomic_int`, set via `atomic_store_explicit(..., memory_order_relaxed)` from OpenMP loops — safe because all threads store the same value and the parallel region's exit barrier provides visibility). Error *details* are in the top-level `error_no` field and per-sample `error_no` values.

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

- All 13 public free functions are consolidated in `eventprop_memory.c` (core, batch, and diagnostic outputs)
- All free functions (batch and single-sample) take their struct **by value**
- Free functions exist for every heap-backed type: `eventprop_free_spike_output`, `eventprop_free_sums_output`, `eventprop_free_maxima_output`, `eventprop_free_backward_output`, `eventprop_free_dropout_output`, `eventprop_free_gradient_output`
- Dimension fields are removed from output structs — dimensions are inferred from input parameters

## Python Bindings (`_native.pyx`)

### Architecture

Thin Cython wrappers with managed heap-backed wrapper classes that own C memory:

| Wrapper Class | Owns | Provides |
|---------------|------|----------|
| `_SpikesBatchOutput` | heap `EventPropInput[]` + `SpikesBatch*` | Zero-copy views via `.get_spike_times(i)`, `.get_sources(i)`; inputs reused by backward/maxima calls |
| `_MaximaBatchOutput` | heap `EventPropInput[]` + `MaximaBatch*` | Zero-copy views for maxima times/values; inputs reused by backward calls |
| `_AccumulatedBackwardResult` | `AccumulatedBackwardResult*` | `.grad` (flat), `.rows`, `.cols`, `.get_spike_errors(i)` |
| `_DropoutBatchOutput` | `DropoutBatch*` | Zero-copy views via `[i]` → `(times, sources)` tuple |

All wrappers use `__dealloc__` to call the appropriate C free function. Numpy views are created via `PyArray_SimpleNewFromData` (zero-copy).

### Layer Chain Flow

1. Python constructs `EventPropInput[]` once and passes it to batch functions
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
