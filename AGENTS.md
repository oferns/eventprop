# AGENTS.md — EventProp

## Project Overview

This repository implements the **EventProp** algorithm — an event-based backpropagation method for training Spiking Neural Networks (SNNs). It computes exact gradients through spike times using the Leaky Integrate-and-Fire (LIF) neuron model with synaptic filtering, leveraging the adjoint method from control theory.

### Reference Library

**`EventProp.Python/`** is a git submodule of the original Python + C++ (Eigen/pybind11) EventProp library. It is the **reference implementation** — everything else in this repo should mimic its API and produce matching input/output.

### Project Map

| Project | Language | Role |
|---------|----------|------|
| **EventProp.Python/** | Python + C++ (Eigen/pybind11) | Reference library (git submodule, read-only) |
| **EventProp.C/** | C | Pure C reimplementation of the Eigen core. Has 2 sets of batch functions — one general-purpose, one designed specifically for the Cython bindings in EventProp.Native.Python |
| **EventProp.Cpp.Wrapper/** | C++ (Eigen) | Convenience copy of the C++ code from EventProp.Python minus pybind11 bindings. Used for native benchmarks and parity tests |
| **EventProp.Native.Python/** | Python + Cython | The Python layer from EventProp.Python rewritten to use the C library via Cython bindings (`eventprop_c/_native.pyx`) |
| **EventProp.Native/** | C# (.NET 10) | P/Invoke bindings wrapping EventProp.C via `LibraryImport` + `[UnmanagedCallConv]` |
| **EventProp/** | C# (.NET 10) | Self-contained, fully managed .NET implementation of EventProp.Python |
| **EventProp.Tests/** | C# (.NET 10) | TUnit test suite for .NET projects (correctness + parity) |
| **EventProp.Benchmarks/** | C# (.NET 10) | BenchmarkDotNet project comparing native vs managed .NET libraries |
| **EventProp.Native.Benchmarks/** | C++ (Google Benchmark) | Benchmarks comparing EventProp.C vs EventProp.Python C++ code |
| **EventProp.Native.ParityTests/** | C++ (CMake) | Standalone parity tests comparing C vs Eigen implementations |
| **EventProp.Python.Benchmarks/** | Python (pytest-benchmark) | Benchmarks comparing EventProp.Python vs EventProp.Native.Python |

## Solution Structure

```
eventprop.net/
├── EventProp.slnx              # .NET 10 solution (XML-based)
├── EventProp.sln               # Traditional solution (CLI compatibility)
├── build.sh                    # Build all targets (C, C++, Python, .NET)
├── clean.sh                    # Clean all or individual targets
├── src/
│   ├── EventProp.Python/       # Reference library (git submodule, read-only)
│   ├── EventProp.C/            # Pure C implementation (libeventprop_c_api)
│   ├── EventProp.Cpp.Wrapper/  # C++ Eigen core (convenience copy, no pybind11)
│   │   ├── tests.cpp           # Eigen direct unit tests (23 tests)
│   │   └── tests_capi.cpp      # C API wrapper unit tests (14 tests)
│   ├── EventProp.Native/       # .NET P/Invoke bindings (C library)
│   │   └── NativeApi.cs        # LibraryImport + UnmanagedCallConv
│   ├── EventProp/              # Managed .NET implementation
│   │   ├── SpikesData.cs       # Spike event container
│   │   ├── MaximaData.cs       # Membrane potential maxima
│   │   ├── ScratchBuffers.cs   # Reusable pre-allocated buffers
│   │   ├── EventPropMath.cs    # Core algorithm implementations
│   │   ├── Layer.cs            # Abstract layer + chain composition
│   │   ├── LIFLayer.cs         # Leaky Integrate-and-Fire layer
│   │   ├── LILayer.cs          # Integrate-and-Fire layer
│   │   ├── WeightInitializer.cs# Diagonal weight initialization
│   │   └── BatchVectors.cs     # Per-batch state vectors
│   ├── EventProp.Tests/        # TUnit test suite (correctness + parity)
│   ├── EventProp.Benchmarks/   # BenchmarkDotNet (native vs managed .NET)
│   ├── EventProp.Native.Python/  # Python + Cython bindings (C library)
│   │   └── eventprop_c/        # Cython package (_native.pyx → _native.so)
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
- **Test framework**: TUnit 1.45+ (Microsoft Testing Platform)
- **Coverage**: TUnit `--coverage --coverage-output-format=cobertura` + ReportGenerator
- **Platform**: Developed and tested on macOS. Linux support intended; Windows if feasible.

## Building

```bash
# Build everything (C, C++, Python, .NET, parity tests, benchmarks) — Release mode
./build.sh release

# Build everything — Debug mode
./build.sh debug

# Clean all build outputs
clean.sh              # removes everything (all targets)
clean.sh c            # only C library
clean.sh cpp          # only C++ wrapper
clean.sh python       # only Python package
clean.sh dotnet       # only .NET projects
clean.sh parity       # only parity tests (C vs Eigen)
clean.sh benchmarks   # only native benchmarks (C vs C++)
clean.sh all          # same as clean.sh (no args)
```

### Submodule Patches

Platform-specific patches are applied automatically by `build.sh`. They live in the repo root:

| Patch | Platform | Fixes |
|-------|----------|-------|
| `submodule-macos.patch` | macOS (AppleClang + Homebrew) | Homebrew include paths, `-Xpreprocessor -fopenmp`, link args, NumPy deprecation |
| `submodule-linux.patch` | Linux (GCC/Clang) | OpenMP link flag, remove C++17 `<execution>` parallel policies, NumPy deprecation |

To apply manually:
```bash
cd src/EventProp.Python
patch -p1 < ../../submodule-macos.patch   # or submodule-linux.patch
```

**Note**: The `src/pybind11` nested submodule may need updating separately (`git submodule update --init`). The patches don't include submodule pointer changes since `patch` can't handle them.

The submodule is read-only — patches must be reapplied after pulling from upstream.

## Running Tests & Coverage

### Quick Reference — Copy-Paste Ready Commands

```bash
# ── .NET Tests (TUnit on .NET 10) ────────────────────────────────
# Debug (fast, different FP ordering — NOT for parity tests)
cd src/EventProp.Tests && dotnet run --

# Release (required for parity tests — matches C/C++ FP ordering)
cd src/EventProp.Tests && dotnet run -c Release --

# With filter (Release mode)
cd src/EventProp.Tests && dotnet run -c Release -- --treenode-filter "/*/*/ComputeSumsTests/*"

# ── C/C++ Parity Tests (C vs Eigen, standalone) ─────────────────
cd src/EventProp.Native.ParityTests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . && ./parity_test ../../EventProp.C/build/libeventprop_c_api.dylib

# ── C API Wrapper Tests (C++ flat-array → Eigen) ───────────────
cd src/EventProp.Cpp.Wrapper && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . && ./eventprop_capi_tests

# ── C/C++ Benchmarks (C vs Eigen performance) ───────────────────
cd src/EventProp.Native.Benchmarks && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . && ./bench_c --benchmark_min_time=0.1s

# ── Python Benchmarks (pybind11 vs Cython) ─────────────────────
cd src/EventProp.Python.Benchmarks && pytest test_benchmarks.py -v

# ── .NET Benchmarks (native vs managed) ────────────────────────
cd src/EventProp.Benchmarks && dotnet run -c Release -- --iteration-time 250
```

### ⚠️ How to Run .NET Tests on .NET 10

TUnit uses the Microsoft Testing Platform — **not VSTest**. `dotnet test` does NOT work.

**The `-c` flag goes BEFORE `--`, TUnit flags go AFTER `--`.** The `dotnet run` CLI splits on `--`: everything before is a dotnet build option, everything after goes to the test runner.

```bash
# ── CORRECT: -c before --, TUnit flags after -- ─────────────────
cd src/EventProp.Tests && dotnet run -c Release --                          # all tests, release
cd src/EventProp.Tests && dotnet run -c Release -- --treenode-filter "/*/*/BackwardSpikes/*"  # filtered

# ── WRONG: -c after -- (TUnit rejects it as unknown flag): ─────────
# cd src/EventProp.Tests && dotnet run -- -c Release   ← BROKEN
# cd src/EventProp.Tests && dotnet test                ← BROKEN (VSTest unsupported)
```

### ⚠️ TUnit vs VSTest — Critical Difference

TUnit runs on Microsoft.Testing.Platform (MTP), **not VSTest**. If you come from xUnit/NUnit/MSTest:

- **`--filter` is silently rejected** — the flag is ignored, MTP help prints, and zero tests run
- This looks like a "test failure" when it's actually just an unrecognised flag
- **Always use `--treenode-filter` instead**

| VSTest (xUnit/NUnit/MSTest) | TUnit (MTP) |
|------------------------------|-------------|
| `--filter "Category=Integration"` | `--treenode-filter "/*/*/*/*[Category=Integration]"` |
| `--filter "FullyQualifiedName~LoginTests"` | `--treenode-filter "/*/*/LoginTests/*"` |
| `--filter "Name=AcceptCookiesTest"` | `--treenode-filter "/*/*/*/AcceptCookiesTest"` |

### TUnit Flags Reference (Passed after `--`, NOT before)

Everything **before** `--` is a dotnet CLI option (e.g. `-c Release`). Everything **after** `--` goes to TUnit.

| Flag | Effect | Example |
|------|--------|---------|
| `--no-progress` | Suppress the ASCII spinner, cleaner output for grep/parsing | `dotnet run -c Release -- --no-progress` |
| `--treenode-filter "/*/*/BackwardSpikes/*"` | Run only tests matching pattern | `dotnet run -c Release -- --treenode-filter "/*/*/BackwardSpikes/*"` |
| `--list-tests` | List all test names without running | `dotnet run -c Release -- --list-tests` |
| `--output Detailed` | Show full failure details in output | `dotnet run -c Release -- --output Detailed` |
| `--fail-fast` | Stop after first failure | `dotnet run -c Release -- --fail-fast` |
| `--coverage` | Enable coverage collection | `dotnet run -c Release -- --coverage` |
| `--coverage-output <file>` | Specify coverage output file path | `dotnet run -c Release -- --coverage --coverage-output coverage.cobertura.xml` |
| `--coverage-output-format cobertura` | Coverage format (cobertura, json, etc.) | `dotnet run -c Release -- --coverage-output-format cobertura` |

### Common TUnit Filter Patterns (treenode-filter)

TUnit uses a tree path format: `/*/[project]/[class]/[test]` (4 segments). Use wildcards (`*`) to match any segment:

```bash
# All BackwardSpikes tests
dotnet run -c Release -- --treenode-filter "/*/*/BackwardSpikes/*"

# All tests in NativeVsManagedTests
dotnet run -c Release -- --treenode-filter "/*/*/NativeVsManagedTests/*"

# All parity tests (NumericalParityTests + NativeLibraryParityTests)
dotnet run -c Release -- --treenode-filter "/*/*/*ParityTests/*"

# Specific test
dotnet run -c Release -- --treenode-filter "/*/*/BackwardSpikes_Uniform_Tiny"

# All tests matching a pattern in the name
dotnet run -c Release -- --treenode-filter "/*/*/*[Name~Poisson*]"
```

### Parsing TUnit Output (Common Patterns)

TUnit output is verbose and multi-line. Use these grep patterns to extract meaningful information:

```bash
# Count total failures
dotnet run -c Release -- 2>&1 | grep "failed " | wc -l

# List only failed test names
dotnet run -c Release -- 2>&1 | grep "failed BackwardSpikes"

# Get failure reason for a specific test (3 lines after the name)
dotnet run -c Release -- 2>&1 | grep "failed BackwardSpikes_Uniform_Tiny" -A3

# Show only C vs Eigen comparison failures (NativeLibraryParityTests)
dotnet run -c Release -- 2>&1 | grep "CompareNativeBackwardSpikes" -B3

# Show only .NET vs Eigen comparison failures (NumericalParityTests)
dotnet run -c Release -- 2>&1 | grep "CompareBackwardSpikes" -B3

# Show only C vs .NET comparison failures (NativeVsManagedTests)
dotnet run -c Release -- 2>&1 | grep "CompareNativeVsManaged" -B3

# Check for Toms748 root-finder exceptions
dotnet run -c Release -- 2>&1 | grep "f(a) and f(b)"

# Check for spike count mismatches
dotnet run -c Release -- 2>&1 | grep "SPIKE COUNT"

# Get unique failure categories (test name only)
dotnet run -c Release -- 2>&1 | grep "failed BackwardSpikes" | sort -u
```

### Quick Commands (Copy-Paste Ready)

```bash
# Run all tests — Release mode
cd src/EventProp.Tests && dotnet run -c Release --

# Run only backward pass tests
cd src/EventProp.Tests && dotnet run -c Release -- --treenode-filter "/*/*/BackwardSpikes/*"

# Run only C vs Eigen parity tests
cd src/EventProp.Tests && dotnet run -c Release -- --treenode-filter "/*/*/NativeLibraryParityTests/*"

# Count failures
cd src/EventProp.Tests && dotnet run -c Release -- 2>&1 | grep "failed " | wc -l

# Show first failure reason
cd src/EventProp.Tests && dotnet run -c Release -- --fail-fast 2>&1 | grep "failed" -A3
```

### C/C++ Parity Test (Standalone Native Library)

**Location**: `src/EventProp.Native.ParityTests/`

A standalone C++ test project that compares the pure C implementation (`libeventprop_c_api.dylib`) against the Eigen-based core (from EventProp.Cpp.Wrapper). Uses identical input data and asserts numerical equality. Built automatically by `./build.sh`.

**Build & Run**:
```bash
cd src/EventProp.Native.ParityTests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./parity_test /path/to/libeventprop_c_api.dylib
```

**What it tests**: ComputeSums, ComputeSpikes, BackwardSpikes, ComputeMaxima, BackwardMaxima across uniform/poisson/burst patterns.

**Current results**: 85/85 pass. All C outputs match Eigen within tolerance (absolute 1e-10, relative 1e-9).

**Test suite expanded** to cover: ComputeSums, ComputeSpikes, BackwardSpikes, ComputeMaxima, BackwardMaxima across uniform/poisson/simultaneous/regular/exponential patterns at multiple scales. Zero-spike edge case (ComputeMaxima zero) returns NaN consistently in both backends.

### C/C++ Benchmark Projects

**Location**: `src/EventProp.Native.Benchmarks/`

Compares C library vs Eigen core (C++ wrapper) performance across problem sizes and spike patterns. Built automatically by `./build.sh`.

**Build & Run**:
```bash
cd src/EventProp.Native.Benchmarks && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./bench_c --benchmark_min_time=0.1s
./bench_cpp --benchmark_min_time=0.1s
```

### Python Benchmarks (pytest-benchmark)

**Location**: `src/EventProp.Python.Benchmarks/`

Compares EventProp.Python (pybind11) vs EventProp.Native.Python (Cython → C library) at the Python level.

**Run**:
```bash
cd src/EventProp.Python.Benchmarks && pytest test_benchmarks.py -v
```

### Python Bindings (Cython)

**Location**: `src/EventProp.Native.Python/`

The Python layer from EventProp.Python rewritten to use the C library via Cython bindings. The `eventprop_c/_native.pyx` module provides typed FFI to `libeventprop_c_api`, and the Python layer classes (`LIFLayer`, `LILayer`, etc.) delegate to these bindings.

**Usage**: Install the package:
```bash
cd src/EventProp.Native.Python && pip install -e .
```

### Native Unit Tests

Three independent native test suites validate correctness across implementations:

| Suite | Location | Tests | Coverage |
|-------|----------|-------|----------|
| **C library** (`tests.c`) | `EventProp.C/` | 17 | All functions + edge cases (zero spikes, negative weights, large-scale) |
| **Eigen direct** (`tests.cpp`) | `EventProp.Cpp.Wrapper/` | 23 | All Eigen methods + degenerate shapes, parameter variations, temporal extremes |
| **C API wrapper** (`tests_capi.cpp`) | `EventProp.Cpp.Wrapper/` | 14 | Flat-array C interface mirroring pure C suite |
| **Parity tests** (C vs Eigen) | `EventProp.Native.ParityTests/` | 85 | Numerical equivalence at ≤50×20 scale |
| **.NET** (TUnit) | `EventProp.Tests/` | 408+ | Full managed API + three-way parity (C ≡ Eigen ≡ .NET) |

### Coverage Report Generation

```bash
# Run with coverage (output → bin/Release/net10.0/TestResults/coverage.cobertura.xml)
cd src/EventProp.Tests && dotnet run -c Release -- --coverage --coverage-output coverage.cobertura.xml --coverage-output-format cobertura

# Generate HTML coverage report (from test project directory)
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Release/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html

# Open report
open coverage-report/index.html
```

**Important**: `dotnet test` outputs "Testing with VSTest target is no longer supported" on .NET 10. Always use `dotnet run --`.

### Test Project Structure (EventProp.Tests)

- **NativeVsManagedTests.cs** — C# (EventPropMath) vs NativeApi (C library). Tests: ComputeSums, ComputeSpikes, ComputeMaxima, Backward. Uses uniform patterns only.
- **NumericalParityTests.cs** — Three-way: NativeApi (C) ≡ NativeCppApi (Eigen, source of truth) ≡ EventPropMath (.NET). All patterns/sizes. **This is the main parity test suite.**
- **NativeLibraryParityTests.cs** — C vs Eigen ONLY (no .NET). Tests: BackwardSpikes, BackwardMaxima. All patterns/sizes.
- **ComputeSpikesTests.cs** — Unit tests for ComputeSpikes forward pass.
- **ComputeSumsTests.cs** — Unit tests for ComputeSums.
- **BackwardSpikesTests.cs** — Numerical gradient verification for backward pass.

## Architecture Notes

### Zero-Allocation Hot Paths

The core algorithm (`EventPropMath`) is designed for allocation-free execution in training loops. Key patterns:

- **`MatrixView`** — `ref struct` providing zero-copy row-major matrix views over flat `double[]` arrays
- **`StridedSpan`** — column access without allocation (strided read-only span)
- **`ScratchBuffers`** — pre-allocated reusable buffers; callers allocate once and pass in
- **`Span<double>` / `ReadOnlySpan<double>`** — used throughout instead of arrays for zero-copy operations

### Data Flow

1. **Input**: `SpikesData` (times + source neuron indices per sample)
2. **Forward**: `EventPropMath.ComputeSpikes()` — predicts post-synaptic spike times
3. **Loss**: `MaximaData` (membrane potential maxima per neuron) or first-spike times
4. **Backward**: `EventPropMath.Backward()` — adjoint method sweep computing weight gradients

### Biophysical Parameters

All functions share these parameters:
- `tauMem` — membrane time constant (seconds)
- `tauSyn` — synaptic time constant (seconds)
- `vTh` — spike threshold voltage

### Weight Matrix Layout

Row-major: `w[row * cols + col]` where rows = pre-synaptic neurons, cols = post-synaptic neurons.

## TDD Workflow

Tests are written before implementation. Each new function should have:
1. Failing test(s) defining the expected behavior
2. Minimal implementation to pass those tests
3. Coverage verification

## Key Algorithms

- **`ComputeSums`**: Pre-computes cumulative exponential-weighted sums for fast membrane potential evaluation
- **`ComputeSpikes`**: Forward pass — finds threshold-crossing times via root-finding (Toms748/Brent)
- **`Backward`**: Adjoint method — time-reversed sweep propagating λ_v (membrane) and λ_i (synaptic) adjoints
- **`ComputeMaxima`**: Finds membrane potential maxima per neuron (for voltage-based loss)

---

## Historical Notes

### Parity Test Limitation — Large-Scale In-Process Testing

The C library and Eigen core **both use OpenMP** (`#pragma omp parallel for` in `compute_spikes`). When loaded in the same process (parity test binary links Eigen statically + loads C via `dlopen`), large-scale tests (200×80, 500 spikes) cause an **intermittent bus error** due to memory allocator contention between the two OpenMP thread pools.

- Both libraries link against the same `/opt/homebrew/opt/libomp/lib/libomp.dylib`
- `OMP_NUM_THREADS=1` reduces but doesn't eliminate the issue (still ~10% failure rate)
- Each library passes large-scale tests **in isolation** (C: 17/17, Eigen direct: 23/23, C API wrapper: 14/14)
- **Does NOT affect native benchmarks** (`bench_c` and `bench_cpp` are separate processes, each using only one library)

**Workaround**: Keep parity tests at ≤50×20 scale. Validate large-scale correctness via isolated unit test suites.

### C Library Optimization Campaign (2026-06-01)

The C library was optimised to match or exceed the C++ Eigen core. Key changes:

| Optimization | Impact |
|-------------|--------|
| Larger initial buffer capacity + triple growth (#2) | 2.4–2.9x improvement on ComputeSpikes at Large scale |
| Fix diff[256] stack buffer overflow risk (#5) | 2.4–3.1x improvement on BackwardSpikes at Large scale |
| Factor exp() in compute_v (#6) | Closes remaining ComputeSpikes gap entirely — C ≈ C++ at all scales |

### OOB Buffer Overflow Fix (2026-06-01)

Fixed buffer overflow in `eventprop_compute_spikes` where burst patterns caused more output spikes than the fixed-size buffer could hold. Replaced with per-neuron dynamic buffers that grow via `realloc`.

### C Library Batch Operations (2026-06-05)

Added `eventprop_compute_spikes_batch_alloc()` to C library for batch processing. EventProp.Native.Python uses a dedicated set of batch functions designed for the Cython interface.

### Benchmark Gaps Implemented (2026-06-09)

- **FullTrainingStep** composite benchmark (Sums → Spikes → Backward chain)
- **BatchTrainingStep** multi-sample benchmark (batch sizes 4–32)
- **BackwardSpikes XL** size variant

### SIMD Audit (2026-06-09)

Existing SIMD helpers: `fma_row_simd`, `scale_inplace_simd`, `vadd_scaled_simd`, `dot_product_simd`. Added `vsub_simd` (<2% impact). Remaining scalar loops are inherently sequential (root-finding) or lack cross-platform SIMD transcendental support.
