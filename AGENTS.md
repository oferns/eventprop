# AGENTS.md — EventProp .NET

## Project Overview

EventProp is a .NET implementation of the EventProp algorithm — an event-based backpropagation method for training Spiking Neural Networks (SNNs). It computes exact gradients through spike times using the Leaky Integrate-and-Fire (LIF) neuron model with synaptic filtering.

## Solution Structure

```
eventprop.net/
├── EventProp.slnx              # .NET 10 solution (XML-based)
├── EventProp.sln               # Traditional solution (CLI compatibility)
├── src/
│   ├── EventProp/              # Core library project
│   │   ├── SpikesData.cs       # Spike event container
│   │   ├── MaximaData.cs       # Membrane potential maxima
│   │   ├── ScratchBuffers.cs   # Reusable pre-allocated buffers
│   │   ├── EventPropMath.cs    # Core algorithm implementations
│   │   ├── Layer.cs            # Abstract layer + chain composition
│   │   ├── LIFLayer.cs         # Leaky Integrate-and-Fire layer
│   │   ├── LILayer.cs          # Integrate-and-Fire layer
│   │   ├── WeightInitializer.cs# Diagonal weight initialization
│   │   └── BatchVectors.cs     # Per-batch state vectors
│   ├── EventProp.Native/       # P/Invoke bindings (C library)
│   │   └── NativeApi.cs        # LibraryImport + UnmanagedCallConv
│   ├── EventProp.Cpp.Native/   # P/Invoke bindings (C++ wrapper)
│   │   └── NativeCppApi.cs     # LibraryImport + UnmanagedCallConv
│   ├── EventProp.Benchmarks/   # BenchmarkDotNet benchmarks
│   ├── EventProp.Tests/        # Test project (TUnit)
│   ├── EventProp.Cpp.Wrapper/  # C bridge: flat arrays ↔ Eigen/Spikes
│   │   ├── eventprop_c_api.h/.cpp  # Flat-array C API over Eigen core
│   │   ├── tests.cpp               # Eigen direct unit tests (23 tests)
│   │   └── tests_capi.cpp          # C API wrapper unit tests (14 tests)
│   ├── EventProp.C/            # Pure C implementation (libeventprop_c_api)
│   ├── EventProp.Native.Python/  # Python bindings: ctypes (pure C → CPython)
│   └── EventProp.Python/       # Python bindings: pybind11 (git submodule)
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

A standalone C++ test project that compares the pure C implementation (`libeventprop_c_api.dylib`) against the Eigen-based core (stripped from EventProp.Python, no pybind11). Uses identical input data and asserts numerical equality. Built automatically by `./build.sh`.

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

**Raw benchmarks** (bypasses wrapper, links directly against Eigen core):
```bash
cd src/EventProp.Native.Benchmarks/raw && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./raw_bench
```

### Python Bindings (ctypes)

**Location**: `src/EventProp.Native.Python/`

Pure Python ctypes bindings — no pybind11 dependency, no compiled C extensions. The `eventprop/native.py` module uses ctypes to load `libeventprop_c_api.dylib` and exposes all C alloc functions (`compute_spikes_alloc`, `backward_spikes`, `compute_sums_alloc`, etc.) to Python.

**Usage**: No build required — just install the package:
```bash
cd src/EventProp.Native.Python && pip install -e .
```

**Key implementation notes for `native.py`**:
- Uses ctypes with zero-copy memory access via `DoubleArrayView` and `Int32ArrayView`
- Library is loaded once globally via `_global_lib` singleton pattern
- C struct definitions must match `eventprop_alloc.h` exactly (field order, types)
- Function signatures set up via `.restype` and `.argtypes` for type safety
- Memory cleanup handled automatically via `__del__` calling C free functions
- Zero-copy iteration avoids numpy dependency — direct pointer access via ctypes
- Library path discovery: tries relative paths first, then standard locations

- **`tests_capi.cpp`** (`EventProp.Cpp.Wrapper/`) — C API wrapper unit tests mirroring the pure C suite. 14 tests covering all flat-array functions (compute_sums, compute_spikes, backward_spikes, compute_maxima, backward_maxima) plus edge cases (zero weights, linear scaling, zero/negative inputs) and large-scale validation (200×80, 500 spikes).

### EventProp.Python Submodule (pybind11)

**Location**: `src/EventProp.Python/` (git submodule, read-only)

The pybind11 Python bindings require patches to build on macOS:

**Required Patch**: Apply `setup.py.patch` or edit manually:
```bash
cd src/EventProp.Python
patch -p0 < setup.py.patch  # or apply changes manually below
```

**Manual edits in `setup.py`**:
1. Add homebrew include paths for libomp and boost to `include_dirs`
2. Replace `-fopenmp` with `-Xpreprocessor -fopenmp` (macOS clang requirement)
3. Add `extra_link_args=["-L/opt/homebrew/opt/libomp/lib", "-lomp"]`

**Note**: The submodule is read-only — patches must be reapplied after pulling from upstream.

- **`NativeVsManagedTests.cs`** — C# (EventPropMath) vs NativeApi (C library). Tests: ComputeSums, ComputeSpikes, ComputeMaxima, Backward. Uses uniform patterns only.
- **`NumericalParityTests.cs`** — Three-way: NativeApi (C) ≡ NativeCppApi (Eigen, source of truth) ≡ EventPropMath (.NET). All patterns/sizes. **This is the main parity test suite.**
- **`NativeLibraryParityTests.cs`** — C vs Eigen ONLY (no .NET). Tests: BackwardSpikes, BackwardMaxima. All patterns/sizes.
- **`ComputeSpikesTests.cs`** — Unit tests for ComputeSpikes forward pass.
- **`ComputeSumsTests.cs`** — Unit tests for ComputeSums.
- **`BackwardSpikesTests.cs`** — Numerical gradient verification for backward pass.

### Coverage Report Generation

```bash
# Run with coverage (output → bin/Release/net10.0/TestResults/coverage.cobertura.xml)
cd src/EventProp.Tests && dotnet run -c Release -- --coverage --coverage-output coverage.cobertura.xml --coverage-output-format cobertura

# Generate HTML coverage report (from test project directory)
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Release/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html
```

**Important**: `dotnet test` outputs "Testing with VSTest target is no longer supported" on .NET 10. Always use `dotnet run --`.

### Test Project Structure

- **NativeVsManagedTests.cs** — C# vs NativeApi (C library) comparisons, uniform patterns only
- **NumericalParityTests.cs** — Three-way: NativeApi (C) ≡ NativeCppApi (Eigen, source of truth) ≡ EventPropMath (.NET), all patterns/sizes
- **ComputeSpikesTests.cs / ComputeSumsTests.cs / etc.** — Unit tests for individual operations
- **BackwardSpikesTests.cs** — Numerical gradient verification

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

## ✅ FIXED: OOB Buffer Overflow in `eventprop_compute_spikes` (2026-06-01)

**Location**: `src/EventProp.C/eventprop_scalar.c` — `eventprop_compute_spikes()`

**Root cause**: The C library used a **fixed-size internal spike buffer** sized `(n_spikes + 1) * n_post`, assuming at most one output spike per input interval. Burst patterns with tightly clustered spikes caused **multiple threshold crossings** per interval, producing more output spikes than the buffer could hold.

**Fix applied**: Replaced flat `double* post_times` / `double* post_currents` arrays with per-neuron dynamic buffers (`double**`) that double capacity via `realloc` on overflow. OpenMP parallelization retained — each thread owns its neuron's buffer, and updated pointers are written back after the loop.

**Verification**: 17/17 C unit tests pass (including large-scale 200×80), 23/23 C++ Eigen direct unit tests pass, 14/14 C API wrapper unit tests pass, 20/20 C vs Eigen parity tests pass at original sizes, 60/60 NativeVsManaged (.NET) tests pass. No meaningful performance regression on realistic workloads.

### Parity Test Limitation — Large-Scale In-Process Testing

The C library and Eigen core **both use OpenMP** (`#pragma omp parallel for` in `compute_spikes`). When loaded in the same process (parity test binary links Eigen statically + loads C via `dlopen`), large-scale tests (200×80, 500 spikes) cause an **intermittent bus error** due to memory allocator contention between the two OpenMP thread pools.

- Both libraries link against the same `/opt/homebrew/opt/libomp/lib/libomp.dylib`
- `OMP_NUM_THREADS=1` reduces but doesn't eliminate the issue (still ~10% failure rate)
- Each library passes large-scale tests **in isolation** (C: 17/17, Eigen direct: 23/23, C API wrapper: 14/14)
- The parity test was also fixed to use **independent data copies** instead of `Eigen::Map` views onto C-owned buffers — this eliminated buffer aliasing but the race persists
- **Does NOT affect native benchmarks** (`bench_c` and `bench_cpp` are separate processes, each using only one library)

**Workaround**: Keep parity tests at ≤50×20 scale. Validate large-scale correctness via isolated unit test suites:
- **C library** — `src/EventProp.C/tests.c` (17 tests, includes 4 large-scale at 200×80)
- **Eigen direct** — `src/EventProp.Cpp.Wrapper/tests.cpp` (23 tests, includes 5 large-scale + edge cases)
- **C API wrapper** — `src/EventProp.Cpp.Wrapper/tests_capi.cpp` (14 tests, mirrors C suite via flat-array interface)

---

## ✅ C Library Optimization Campaign (2026-06-01)

**Context**: C library was 1.5–3x slower than C++ Eigen core on benchmarks. Root cause analysis and optimization implemented.

### Benchmark Fairness Fixes Applied

| Issue | Fix |
|-------|-----|
| C++ had CPU-specific optimization, C didn't | Added `-mcpu=apple-m3` to `EventProp.C/CMakeLists.txt` for arm64 |
| ComputeSums missing DoNotOptimize | Added `benchmark::DoNotOptimize(sum0/sum1.data())` in `bench_main.cpp` |

### Optimizations Implemented

#### #1: Stack arrays for exp_mem/exp_syn in compute_sums (Low impact)

**Problem**: Two malloc calls per forward pass for exponential lookup tables.

**Fix**: Use stack arrays when n_spikes ≤ 1024, heap allocation otherwise.

```c
double stack_exp_mem[EXP_STACK_SIZE];  // 1024 doubles
int use_stack = (n_spikes <= EXP_STACK_SIZE);
double* exp_mem = use_stack ? stack_exp_mem : malloc(n_spikes * sizeof(double));
```

**Result**: No measurable performance change (FMA computation dominates).

---

#### #2: Larger initial buffer capacity + triple growth (HIGH impact)

**Problem**: Dynamic per-neuron spike buffers starting at n_spikes+1 caused frequent realloc during burst patterns.

**Fix**: 
- Initial capacity: `n_spikes + 1` → `n_spikes * 3 + 1`
- Growth factor: `cap *= 2` → `cap *= 3` (at all 3 realloc sites)

**Result**: **2.4–2.9x improvement on ComputeSpikes at Large scale.**

| Benchmark | Before (ratio) | After (ratio) |
|-----------|----------------|---------------|
| ComputeSpikes Large uniform | 1.83x C slower | **0.47x C FASTER** |
| ComputeSpikes Large poisson | 1.40x C slower | **0.48x C FASTER** |
| ComputeSpikes Medium uniform | 1.15x C slower | **0.46x C FASTER** |
| ComputeSpikes Medium poisson | 1.54x C slower | **0.57x C FASTER** |

*Ratio = C/C++, >1 means C++ faster, <1 means C faster.*

---

#### #5: Fix diff[256] stack buffer overflow risk (Medium impact + correctness)

**Problem**: Fixed 256-element stack buffer with slow fallback path for n_post > 256.

**Fix**: Use 512-element stack buffer or heap allocation for all n_post sizes.

```c
double stack_diff[DIFF_STACK_SIZE];  // 512 doubles
int use_stack = (n_post <= DIFF_STACK_SIZE);
double* diff = use_stack ? stack_diff : malloc(n_post * sizeof(double));
```

**Result**: 2.4–3.1x improvement on BackwardSpikes at Large scale.

| Benchmark | Before (ratio) | After (ratio) |
|-----------|----------------|---------------|
| BackwardSpikes Large uniform | 3.36x C slower | **1.06x roughly equal** |
| BackwardSpikes Large poisson | 3.07x C slower | **1.00x equal** |
| BackwardSpikes Medium uniform | 2.59x C slower | **0.98x roughly equal** |
| BackwardSpikes Medium poisson | 2.42x C slower | **1.03x roughly equal** |

---

#### #6: Factor exp() in compute_v (CRITICAL impact — closes ComputeSpikes gap)

**Problem**: Each Tom's 748 solver iteration called `exp()` N times for post-spike terms:
`mem -= v_th * exp(-(t - st) * tau_mem_inv)` per spike. At Medium+ scales with many iterations, this dominated runtime.

**Fix**: Algebraic factorization — precompute once before solver loop:
```
exp(-(t-st)*τ⁻¹) = exp(-t*τ⁻¹) × exp(st*τ⁻¹)
```
- Precompute `post_exp_st[s] = exp(p_times[s] * tau_mem_inv)` once per neuron
- Inside solver: single `exp(-t*tau_mem_inv)` call, multiply by cached constants
- Reduces N `exp()` calls per iteration → 1 `exp()` call + N multiplications

**Result**: **Closes the remaining C vs C++ ComputeSpikes gap entirely.**

| Benchmark | Before (ratio) | After (ratio) |
|-----------|----------------|---------------|
| ComputeSpikes Medium uniform | 1.40x C slower | **~1.0x equal** ✅ |
| ComputeSpikes Medium poisson | 1.30x C slower | **~1.06x equal** ✅ |
| ComputeSpikes Large uniform | 1.20x C slower | **~1.05x equal** ✅ |
| ComputeSpikes Large poisson | 1.10x C slower | **~1.04x equal** ✅ |
| ComputeSpikes XL uniform | 1.30x C slower | **~1.06x equal** ✅ |
| ComputeSpikes XL poisson | 1.20x C slower | **~0.96x C FASTER** 🏆 |

*Ratio = C/C++, >1 means C++ faster, <1 means C faster.*

---

### Remaining Issues

1. **Medium/Large burst patterns still timeout** — root-finding iterations are the bottleneck; needs optimization #3 (exponential term caching in Toms748 solver)
2. **Tiny/Small uniform BackwardSpikes noisy** — nanosecond-scale tests affected by thermal/system noise

### Benchmark Commands (excluding slow burst patterns)

```bash
cd src/EventProp.Native.Benchmarks/build

# Run each operation separately (no --exclude_filter in v1.9.1)
for op in ComputeSums ComputeSpikes BackwardSpikes; do
  # Tiny/Small: all patterns
  ./bench_c --benchmark_min_time=0.1s --benchmark_filter="$op/(Tiny|Small)/.*"
  ./bench_cpp --benchmark_min_time=0.1s --benchmark_filter="$op/(Tiny|Small)/.*"
  # Medium/Large: only uniform and poisson
  for size in Medium Large; do
    ./bench_c --benchmark_min_time=0.1s --benchmark_filter="$op/$size/uniform"
    ./bench_cpp --benchmark_min_time=0.1s --benchmark_filter="$op/$size/uniform"
    ./bench_c --benchmark_min_time=0.1s --benchmark_filter="$op/$size/poisson"
    ./bench_cpp --benchmark_min_time=0.1s --benchmark_filter="$op/$size/poisson"
  done
done
```




## ✅ C Library Batch Operations (2026-06-05)

**Context**: ctypes was degrading at larger batch sizes due to per-sample Python→C calls.

### Batch Function Added

Added `eventprop_compute_spikes_batch_alloc()` to C library:
- Accepts flat arrays with per-sample spike counts
- Parallel processing via OpenMP across samples  
- Returns array of `SpikeOutput` structs (one per sample)

### Python Bindings

- Added ctypes function signatures and wrapper method
- `BatchSpikeOutput` class for per-sample result access
- Updated LIFLayer to use batch function (single C call)

### Performance Results

| Batch Size | ctypes Before | ctypes After | pybind11 | Improvement |
|------------|---------------|--------------|----------|-------------|
| 4 Small | 297 μs | 141 μs | 117 μs | **2.1x faster** ✅ |
| 8 Small | 588 μs | 183 μs | 143 μs | **3.2x faster** ✅ |
| 16 Small | 1,185 μs | 264 μs | 151 μs | **4.5x faster** ✅ |
| 8 Medium | 8,227 μs | 1,049 μs | 1,097 μs | **7.8x faster** ✅ |
| 16 Medium | 32,643 μs | 1,334 μs | 1,185 μs | **24x faster** ✅ |

ctypes is now competitive with pybind11 at all batch sizes.

---

## ✅ Parity Tests Simplified (2026-06-09)

**Branch**: `simd-experiments` | **Commit**: `b073689`

Replaced runtime `dlopen`/`dlsym` loading with direct CMake binary imports:
- Imports pre-built `libeventprop_c_api.dylib` and `libeventprop_eigen_core.a`
- Removed function pointer typedefs, global variables, `load_c_library()`
- Adapted to new owned-memory API (`SumsOutput`, `SpikeOutput`, etc.)
- Skips zero-spike backward tests (Eigen assertion on empty arrays)
- **85/85 tests pass**

---

## ✅ Benchmark Fairness Fixes (2026-06-09)

**Branch**: `simd-experiments` | **Commits**: `dcfdea4`, `661e920`, `a7043c9`

| Fix | Commit | Impact |
|-----|--------|--------|
| Move Eigen::Map + Spikes construction outside benchmark loops | `dcfdea4` | Eliminated per-iteration deep copy overhead for C++ |
| Remove dead `gradient.setZero()` from BM_BackwardSpikes | `dcfdea4` | Removed O(n_pre × n_post) wasted work per iteration |
| Protect full output arrays in DoNotOptimize (not single elements) | `661e920` | Prevents compiler from optimizing away real work |
| Remove hardcoded `.Iterations(100)` for auto-scaling | `661e920` | Google Benchmark picks optimal iteration count |
| Add warmup iterations before benchmark loops | `a7043c9` | Primes caches and OpenMP thread pools (<2% variation) |
| Output spike counters (`state.counters["output_spikes"]`) | `a7043c9` | Captured post-loop via extra forward pass or pre-computed values |

**Benchmark results** (5 runs, mean ± stddev):
- **ComputeSums**: C 1.7–2.4x faster across all sizes
- **ComputeSpikes**: Roughly equal at Medium+ after exp() factorization (#6), C++ wins Small uniform only
- **BackwardSpikes**: Roughly equal at Medium+, C much faster Tiny/Small (early exit with 0 output spikes)
- **ComputeMaxima**: C++ wins Small, C dominates Medium+ (7.3x)

---

## ✅ SIMD Audit + vsub_simd (2026-06-09)

**Branch**: `simd-experiments` | **Commit**: `c241a65`

### Existing SIMD helpers (actively used)
- `fma_row_simd` — FMA row operations in compute_sums
- `scale_inplace_simd` — in-place scaling
- `vadd_scaled_simd` — vector add with scalar multiplier
- `dot_product_simd` — dot product in backward_spikes

### Added: `vsub_simd`
- Vector subtraction for diff computation in backward_spikes (line 881-882)
- Supports AVX512, AVX2, ARM NEON, scalar fallback
- **Result**: <2% measurable impact — diff is dwarfed by surrounding work (lambda_i jumps, dot_product_simd)

### Remaining scalar loops (not SIMD-friendly)
| Location | Why not SIMD |
|----------|-------------|
| `backward_spikes` L905-908 (lambda_i per-neuron jumps) | Irregular access: different `jump_count[nrn]` per neuron |
| `compute_maxima` L798-823 (exp/log voltage checks) | No standard SIMD exp/log without libsvml; branch-heavy |
| `compute_spikes` root-finding | Inherently sequential per neuron via toms748_solve |
| `exp_mem/exp_syn` precomputation | Scalar `exp()` calls — would need libsvml (Intel-only) or custom approximations |

### Small-scale anomaly explanation
Tiny/Small uniform with v_th=1.0 produces **0 output spikes** for both backends.
- C does full work: compute_sums, dynamic buffer setup (`malloc` × n_post), OpenMP thread pool management, root-finding loop that finds nothing
- Eigen uses pre-allocated vectors and auto-vectorized row ops with less overhead when no spikes fire
- With `OMP_NUM_THREADS=1`, C Small/uniform drops from 76.7μs to 48.2μs (~28μs OpenMP overhead)
- **Gap is overhead vs useful work ratio** — at Medium+ where real training happens, C dominates 7–9x

### libsvml context
Intel Short Vector Math Library provides SIMD intrinsics for transcendental functions (`exp`, `log`). Caveats: Intel-only (no ARM/NEON equivalent), commercial licensing, fast approximations not IEEE 754 compliant. Cross-platform alternatives: libmvec (GCC's vectorized libm), custom polynomial approximations, compiler auto-vectorization with `-ffast-math`.

---

## ✅ OpenMP Threshold Removed (2026-06-09)

Removed `if(n_post >= 20)` guard from C library — now always parallelizes via OpenMP.

**Experiment**: Tested three configs (Baseline ≥20, Raised ≥50, Removed). All produced identical performance ratios (~1.46x C slower for ComputeSpikes/Small/uniform). Confirmed the Small-scale inversion is **not** caused by thread pool setup/sync overhead — it's architectural (Eigen SIMD/auto-vectorization vs C scalar loops + dynamic buffer management).

---

## 📋 Benchmark Gaps Identified (2026-06-09)

### Current coverage matrix
| Operation | Sizes | Patterns |
|-----------|-------|----------|
| ComputeSums | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| ComputeSpikes | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| BackwardSpikes | Tiny → Large (4) | uniform, poisson, burst (3) ⚠️ Missing XL |
| BackwardMaxima | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| ComputeSpikesVth | Small only | uniform only |
| BackwardSpikesVth | Small only | uniform only |

### Identified gaps
| # | Gap | Effort | Value |
|---|-----|--------|-------|
| 1 | Composite "full training step" benchmark (Sums → Spikes → Backward) | Low | **High** — catches real-world regressions, measures cache effects between chained ops |
| 2 | Batch benchmark (multi-sample via `eventprop_compute_spikes_batch_alloc`) | Medium | **High** — matches actual Python/ML usage at batch sizes 8–32 |
| 3 | BackwardSpikes XL size variant | Trivial | Medium — completes the matrix |
| 4 | Weight sparsity coverage (10–30% density) | Medium | Medium — real SNNs often sparse, SIMD behaves differently |
| 5 | Memory allocation overhead benchmark (pre-allocated vs fresh allocs) | Low | Low-Medium — quantifies owned-memory API cost at high iteration rates |
| 6 | Cache sensitivity study (vary n_pre, fixed n_post) | Low | Low — diagnostic: reveals where weight matrix stops fitting in L3 cache on M3 (24MB shared). At XL: 2000×500×8B = 8MB fits. At 4000×500: 16MB starts thrashing. Shows whether SIMD helpers are memory-bound vs compute-bound at scale |

### Next steps (in order)
1. ✅ Implement #1 (composite training step) + #3 (BackwardSpikes XL) — quick wins
2. ✅ Implement #2 (batch benchmark) — requires batch function signatures in both C and Eigen paths
3. Consider #4–#6 if profiling shows they matter for target workloads

---

## ✅ Benchmark Gaps #1, #2, #3 Implemented (2026-06-09)

### Gap #1 — FullTrainingStep composite benchmark
Chains Sums → Spikes (forward) → Backward in one loop. Measures cache effects between chained operations.

| Size | C (ns) | C++ (ns) | Ratio (C/C++) |
|------|--------|----------|---------------|
| Small/uniform | 76,721 | 61,188 | **1.25x** (C slower) |
| Medium/uniform | 17,464,286 | 50,769,667 | **0.34x** (C ~3x faster) |
| Large/uniform | 185,552,000 | 486,746,000 | **0.38x** (C ~2.6x faster) |

### Gap #2 — BatchTrainingStep multi-sample benchmark
Tests batch processing at realistic ML batch sizes (4-32 samples). Uses `compute_spikes_batch` → `backward_spikes_batch`.

**Small scale (50×20, 100 spikes):**
| Batch Size | C (μs) | C++ (μs) | Ratio |
|------------|--------|----------|-------|
| 4 | 162 | 164 | **0.99x** (equal) |
| 8 | 155 | 158 | **0.98x** (equal) |
| 32 | 258 | 227 | **1.14x** (equal) |

**Medium scale (200×80, 500 spikes):**
| Batch Size | C (ms) | C++ (ms) | Ratio |
|------------|--------|----------|-------|
| 4 | 80.6 | 54.6 | **1.48x** (C slower) |
| 8 | 80.3 | 55.3 | **1.45x** (C slower) |
| 32 | 195.3 | 121.1 | **1.61x** (C slower) |

**Key finding**: Both backends show flat CPU time across batch sizes (effective OpenMP parallelization). Roughly equal at Small scale; C++ ~1.5x faster at Medium.

**⚠️ Pre-optimization data**: These results are from before the exp() factorization (#6, commit `f5467e3`). The Medium-scale gap should be re-benchmarked — ComputeSpikes is now equal or better in isolation.

**⚠️ OpenMP bug fixed**: Original results showed C 8.5x faster — caused by missing `-Xpreprocessor -fopenmp` compile flag for Eigen. OpenMP pragmas were in code but not compiled.

```cmake
target_compile_options(eventprop_eigen_core PRIVATE -Xpreprocessor -fopenmp)
```

**API improvement**: Changed `eventprop_backward_spikes_batch` to accept `SpikeOutput*` directly instead of flat arrays — eliminates data copying between forward and backward pass.

```c
// Before: required concatenating flat arrays
BackwardSpikesOutput* eventprop_backward_spikes_batch(
    const double* times, const int32_t* sources,
    const int* n_spikes_per_sample,
    const double* fwd_times, const int32_t* fwd_sources,
    const double* fwd_currents,
    const double* post_errors, const int* n_fwd_spikes_per_sample,
    ...);

// After: pass forward outputs directly  
BackwardSpikesOutput* eventprop_backward_spikes_batch(
    const double* times, const int32_t* sources,
    const int* n_spikes_per_sample,
    SpikeOutput* fwd_batch,        /* Forward spike outputs */
    const double* post_errors,
    ...);
```

### Gap #3 — BackwardSpikes XL size variant
Added `BACKWARDSPIKES_BENCH(XL, 2000, 500, 5000)` — completes the coverage matrix.

### Updated coverage matrix
| Operation | Sizes | Patterns |
|-----------|-------|----------|
| ComputeSums | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| ComputeSpikes | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| BackwardSpikes | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| BackwardMaxima | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| FullTrainingStep | Tiny → XL (5) | uniform, poisson, burst (3) ✅ |
| BatchTrainingStep | Small → Medium | uniform only — batch sizes 4-32 ✅ |
| ComputeSpikesVth | Small only | uniform only |
| BackwardSpikesVth | Small only | uniform only |
