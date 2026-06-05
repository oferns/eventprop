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

**Current results**: 20/20 pass. All C outputs match Eigen within tolerance (absolute 1e-10, relative 1e-9).

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

### Remaining Issues

1. **Medium/Large burst patterns still timeout** — root-finding iterations are the bottleneck; needs optimization #3 (exponential term caching in Toms748 solver)
2. **ComputeSums still ~1.3x slower** — likely SIMD implementation vs Eigen auto-vectorization difference
3. **Tiny/Small uniform BackwardSpikes noisy** — nanosecond-scale tests affected by thermal/system noise

### Optimization Priority (for future work)

| # | Optimization | Effort | Impact |
|---|-------------|--------|--------|
| 3 | Cache exponential computations in root-finding | Medium | High (fixes burst timeout) |
| 4 | Increase SIMD fallback threshold from 2 to vector width | Low | Low-Medium |
| 6 | Pre-allocate jump storage in backward_spikes (reduces realloc) | Low-Medium | Medium |
| 7 | Parallelize compute_maxima with OpenMP | Low | Low |

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

---

## C Library Import Issues (2026-06-04)

### Problem Identified
The Python benchmarks are having trouble importing the C library. The ctypes module is not being found when running tests.

### Current Status
- C library is at `src/EventProp.C/build/libeventprop_c_api.dylib`
- ctypes module should load it from this location
- Import issues need to be resolved before running benchmarks

### Next Steps
1. Fix import issues in test_benchmarks_ctypes.py
2. Test all benchmarks work correctly
3. Run full benchmark suite

---

## C Library Complementary API Design (2026-06-02)

### Problem Identified

The original `eventprop_compute_spikes` function has an OOB buffer overflow vulnerability:
- Caller must guess output buffer size beforehand
- Function writes past buffer bounds when producing more spikes than allocated
- No way to know if buffer was large enough after the call

Example failure: v_th=0.01 produced 240,986 spikes when buffer only held 101,000.

### Proposed Solution

Create a complementary API with automatic memory allocation that coexists with original code.

#### Pattern

```c
/* Output struct with owned memory */
typedef struct {
    int n_spikes;            // Count (replaces return value)
    double* times;           // malloc'd by function, freed by caller
    int32_t* sources;        // malloc'd by function, freed by caller  
    double* currents;        // malloc'd by function, freed by caller
} SpikeOutput;

/* Function allocates internally */
SpikeOutput eventprop_compute_spikes_alloc(const double* w, int n_pre, int n_post:
                                           const double* times, const int32_t* sources, int n_spikes,\n                                           Double v_th, double tau_mem, double tau_syn);\n\n/* Cleanup helper */\nvoid eventprop_free_spike_output(SpikeOutput* output);\n```

#### Functions to Implement
1. `eventprop_compute_spikes_alloc` - forward pass (high priority)
2. `eventprop_compute_sums_alloc` - cumulative sums
3. `eventprop_compute_maxima_alloc` - membrane potential maxima
4. `eventprop_backward_spikes_alloc` - backward pass (spikes)
5. `eventprop_backward_maxima_alloc` - backward pass (maxima)

#### Files Created
- `src/EventProp.C/eventprop_alloc.h` - header with struct definitions and function declarations
- `src/EventProp.C/eventprop_alloc.c` - C file with stub implementations

#### Implementation Notes
- Keep existing API unchanged - just add new functions alongside
- Function knows exact output size needed, so can allocate exactly that much
- Eliminates OOB risk by removing caller's guesswork about buffer sizes

#### Status (when session ended)
- Header and C file created with stubs for eventprop_compute_spikes_alloc only
- Build issues: syntax errors (colons vs commas in function signatures)
- Need to complete implementation and add unit tests

#### Next Steps
1. Fix syntax errors in existing stubs
2. Complete eventprop_compute_spikes_alloc implementation
3. Add unit tests for the new API
4. Implement remaining functions one at a time


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
