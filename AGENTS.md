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
│   ├── EventProp.Native/       # .NET P/Invoke bindings (C library) + native LIFLayer/LILayer
│   ├── EventProp.Abstractions/ # Shared Layer contract (SpikesData, *Parameters, WeightInitializer)
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
- **CMake**: floor is `3.21` in all five CMake projects; each ships a `CMakePresets.json` (see [CMake Presets](#cmake-presets))

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

### CMake Presets

All five CMake projects declare `cmake_minimum_required(VERSION 3.21)` and ship a `CMakePresets.json`. Presets are the supported way to configure and build; the hand-typed `mkdir build && cd build` sequences are kept only as a fallback.

Presets live **one per project**, not at the repo root: a configure preset has no `sourceDir` field, so its source directory is always the directory containing its own `CMakePresets.json`. Run preset commands from the project directory, and `cmake --list-presets` there to see what is available.

| Project | Configure / build presets |
|---|---|
| `EventProp.C` | `release`, `debug`, `asan`, `werror` |
| `EventProp.Cpp.Wrapper` | `release`, `debug` |
| `EventProp.C.Tests` | `release`, `asan` — plus matching `testPresets` for `ctest --preset` |
| `EventProp.Native.ParityTests` | `release`, `debug` |
| `EventProp.Native.Benchmarks` | `release` |

Every preset writes to `${sourceDir}/build`, which is what the cross-project imports hardcode: `EventProp.C.Tests`, `EventProp.Native.ParityTests` and `EventProp.Native.Benchmarks` all consume a **pre-built** library out of `EventProp.C/build` and `EventProp.Cpp.Wrapper/build`. Build order therefore matters:

```bash
# 1. C library, 2. Eigen core, 3. consumers
cd src/EventProp.C                 && cmake --preset release && cmake --build --preset release
cd ../EventProp.Cpp.Wrapper        && cmake --preset release && cmake --build --preset release
cd ../EventProp.C.Tests            && cmake --preset release && cmake --build --preset release && ctest --preset release
cd ../EventProp.Native.ParityTests && cmake --preset release && cmake --build --preset release
cd ../EventProp.Native.Benchmarks  && cmake --preset release && cmake --build --preset release
```

- `werror` (`EventProp.C` only) sets `CMAKE_COMPILE_WARNING_AS_ERROR=ON`, making the "code must compile with zero warnings" rule mechanical instead of advisory. It builds in `build-werror/` so it never clobbers the release build.
- `asan` presets set `ENABLE_ASAN=ON` with `CMAKE_BUILD_TYPE=Debug` and `cleanFirst: true`, because the library switches SHARED→STATIC inside the same `build/` directory the test project hardcodes.
- `CMakeUserPresets.json` is gitignored — use it for machine-local overrides.
- The floor is 3.21 for presets schema v3 (`condition`, `installDir`, `${hostSystemName}`) while staying inside stock distro packages (Ubuntu 22.04 ships 3.22, Debian 12 ships 3.25). Schema v4's `include` (3.23) would allow one root file sharing base presets across projects, but it costs Ubuntu 22.04 support, so it is not used.

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
# Presets (preferred)
cd src/EventProp.C.Tests
cmake --preset release && cmake --build --preset release
ctest --preset release            # 214 tests; gtest_discover_tests registers them with CTest
./build/eventprop_tests           # or run the binary directly
```

```bash
# Manual equivalent
cd src/EventProp.C.Tests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./eventprop_tests
```

**With AddressSanitizer** (Debug builds only, detects use-after-free, buffer overflows):
```bash
cd src/EventProp.C        && cmake --preset asan && cmake --build --preset asan
cd ../EventProp.C.Tests   && cmake --preset asan && cmake --build --preset asan
ctest --preset asan
```
> **Note**: On macOS, ASan automatically builds a static library (`.a`) instead of shared (`.dylib`) due to AppleClang linker limitations. The `asan` build presets use `cleanFirst` because the library flips SHARED→STATIC in the shared `build/` directory.

> **Note**: `ASAN_OPTIONS=detect_leaks=1` is deliberately **not** in the presets. `gtest_discover_tests` runs the test executable at *build* time and inherits the configure preset's `environment`; macOS ASan aborts on `detect_leaks` ("not supported on this platform"), which fails discovery and takes the whole build down. On Linux, supply it at run time: `ASAN_OPTIONS=detect_leaks=1 ctest --preset asan`.

### C/C++ Parity Tests (C vs Eigen)

```bash
cd src/EventProp.Native.ParityTests
cmake --preset release && cmake --build --preset release
./build/parity_test ../../EventProp.C/build/libeventprop_c_api.dylib
```
> **Note**: requires the `release` presets of `EventProp.C` (SHARED library) and `EventProp.Cpp.Wrapper` (Eigen core) — both are imported from their `./build` directories at configure time, so configuring before them fails with a `FATAL_ERROR` naming the missing library.

### Native Benchmarks (C vs C++) — Standard Workflow

**Use the benchmark scripts** (`run_benchmarks.bat` on Windows, `run_benchmarks.sh` on Linux/macOS). These handle building, running both benchmarks, saving JSON to `baselines/`, and invoking Google Benchmark's `compare.py` for human-readable comparison reports.

```bash
# Build first (if not already built) — presets preferred
# Requires the release presets of EventProp.C and EventProp.Cpp.Wrapper first
cd src/EventProp.Native.Benchmarks && cmake --preset release && cmake --build --preset release

# Manual equivalent
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

**Benchmark data contract** (`bench_main.cpp`): every generator ends with `SortTimesSources`, because the C library rejects non-decreasing input spike times (`EVENTPROP_UNSORTED_SPIKE_TIMES`, 60) while the Eigen reference does not validate. `GenerateBurstData` fills its arrays neuron by neuron, so its bursts interleave — without the sort the C side returns 0 spikes in ~0 ns and the C-vs-C++ comparison measures a validation short-circuit (it did: `ComputeSpikes/Small/burst` reported 36 ns against the Eigen side's 83 µs). The C benchmarks also publish `c_error` / `c_fwd_error` / `c_bwd_error` counters; a non-zero value means the row measures an error path, not the algorithm. Rows with `output_spikes == 0` are legitimate (high threshold, no neuron fires) but time an empty loop — `BackwardSpikes/Tiny|Small/uniform` are in that regime, so their ratios are overhead-on-noise and should be excluded from any geomean.

**Burst at Medium/Large/XL** is registered behind `#ifndef _MSC_VER` for `ComputeSpikes`, `BackwardSpikes` and `FullTrainingStep`, and unconditionally for `ComputeSums` (which has no root finding, so the Toms748 caveat never applied). Probed on macOS: both backends complete with matching spike counts (Medium 10669, Large 58287, XL 771544) and `c_err = 0`. Burst is the heaviest workload in the suite — XL produces ~770k output spikes against ~56k for XL poisson — so it is the best stress on the forward pass's per-neuron spike-row growth. XL costs ~16 s per iteration for `BackwardSpikes`, so keep it out of quick runs (`FILTER` excludes `XL`).

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
# Presets (preferred) — library first, then tests, both with ENABLE_ASAN=ON
cd src/EventProp.C        && cmake --preset asan && cmake --build --preset asan
cd ../EventProp.C.Tests   && cmake --preset asan && cmake --build --preset asan
ctest --preset asan
```

```bash
# Manual equivalent
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
    const EventPropInput* inputs, const SpikesBatch spikes_batch,
    const double* const* post_spike_errors);

AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
    const EventPropInput* inputs, const MaximaBatch maxima_batch);
```

Returns `GradientOutput summed_grads` (ready for weight update) + per-sample `.spike_errors` preserved. Per-sample `.grad` buffers are freed after summation.

### Spike Output Ownership

`SpikeOutput` carries only `times` + `sources` (+ `n_spikes`). Two former fields are gone:

- **`currents`** — the synaptic current at each output spike is *derived*, not stored. `eventprop_backward_spikes` rebuilds it with one ascending cursor over `input.times` into a row of `n_post` scratch doubles (`ctx.cur_sum`) plus one double per output spike (`ctx.currents`). The forward pass never needed it beyond that one line, and the backward pass already walks the input spikes, so storing it was pure overhead. The accumulation must run **ascending** and be scaled once at the end — the sweep runs time-reversed, and a row rebased at the sweep time loses six significant digits per post spike at benchmark sizes (gradient drifts ~1e-5 relative).
- **`errors`** — the caller-owned loss/error signal per output spike. It moved to an argument: `eventprop_backward_spikes(input, spike, post_spike_errors)` and `eventprop_backward_spikes_batch(inputs, spikes_batch, post_spike_errors)`, where `post_spike_errors` is an array of `batch_size` `const double*` (one per sample, may be `NULL` for empty samples). The library no longer allocates a buffer the caller only writes into.

Post-spike times must be non-decreasing; the backward pass validates this and returns `EVENTPROP_UNSORTED_SPIKE_TIMES` (60). New allocation codes: `EVENTPROP_ALLOC_BWD_CUR_SUM` (62), `EVENTPROP_ALLOC_BWD_CURRENTS` (63).

`MaximaOutput.errors` is unchanged (still library-allocated), as is `BackwardOutput.spike_errors`.

### Memory Ownership Rules

- All 13 public free functions are consolidated in `eventprop_memory.c` (core, batch, and diagnostic outputs)
- All free functions (batch and single-sample) take their struct **by value**
- Free functions exist for every heap-backed type: `eventprop_free_spike_output`, `eventprop_free_sums_output`, `eventprop_free_maxima_output`, `eventprop_free_backward_output`, `eventprop_free_dropout_output`, `eventprop_free_gradient_output`
- Dimension fields are removed from output structs — dimensions are inferred from input parameters

### Spike Time Ordering: A Shared Contract, Not a Parity Gap

Every spike list the library consumes must be **non-decreasing**. `eventprop_validate_input_no_v_th` enforces it on input times (`EVENTPROP_UNSORTED_SPIKE_TIMES`, 60) and `eventprop_backward_spikes` enforces it on post-spike times, because the currents there are rebuilt with one forward cursor over the post spikes.

The requirement is not a C-only strictness — it follows from the algorithm both implementations share. `compute_sums` builds prefix sums **in array order** (`sum0.row(i+1) = sum0.row(i) + exp(t_i/tau_syn) * w.row(sources[i])`), and the forward pass then indexes `sum0[k][nrn]` by position while treating `k` as "all input spikes up to `times[k]`". That is only true if `times` is sorted. An unsorted array is therefore not a valid alternative input; it describes a different network.

The Eigen reference has the same requirement and does not check it. Measured on a 4x2, 10-spike input with two interleaved bursts:

| Call | Result |
|---|---|
| C, unsorted | `error_no = 60`, 0 spikes |
| Eigen, unsorted | **3 spikes** — plausible-looking garbage |
| Eigen, sorted | 5 spikes |
| C, sorted | 5 spikes, identical times/sources, max delta **2.8e-15** |

So the C library is the stricter *diagnostic*, not the divergent *implementation*: on sorted input the two agree to root-finder tolerance. The Eigen `backward` does build a sorted index list internally, but that merges input and post spikes for the time-reversed sweep and does nothing for the prefix-sum ordering. Callers must sort; the C library tells you when they didn't, the reference silently returns wrong numbers.

Pinned by `ComputeSpikes.UnsortedTimesRejected` and `ComputeSpikes.EqualTimesAccepted` (`src/EventProp.C.Tests/test/compute_spikes.cpp`), alongside the existing validator and `BackwardSpikes` ordering tests. Non-decreasing (equal times) is valid and is the simultaneous-spike case the parity suite covers.

**Non-negative times are the same kind of shared contract.** `eventprop_validate_input_no_v_th` rejects `times[i] < 0` with `EVENTPROP_NEGATIVE_SPIKE_TIME` (7); the Eigen reference does not check. It is not C-only strictness either — the forward pass floors the first interval at `t_before = 0.0`, so it cannot bracket a spike before t = 0, and a negative input time is outside the model the prefix sums describe.

This bit two numerical-gradient tests in `src/EventProp.Native.Python/test/test_loss_layer.py`: both take a **central difference** of the loss with respect to a spike time, and the base spike train started at exactly `0.0`, so `spike_idx = 0` was perturbed to `-1e-8`. Only the Cython side failed, because only the C library validates. Fix: start the base trains at `0.001` (`(np.arange(n) + 1) * 0.001`) so the minus leg stays non-negative. The tests' intent — numerical vs analytical spike-time gradient — is unchanged.

While fixing them, a second test-side issue surfaced: `t_eps = 1e-8` is below the conditioning limit for a central difference on O(1) losses. Its absolute noise floor is `~machine_eps / t_eps ≈ 2.2e-8`, which swamps the analytical gradient where that gradient is legitimately **exactly 0** (the last input spike, which no first-spike time depends on), and the assertion used `rtol` with no `atol`. `t_eps = 1e-7` measured the best worst-case agreement (1.0e-9, against 8.1e-9 at 1e-8). When a finite-difference test fails on an entry that is analytically zero, check the noise floor before assuming a gradient bug.

### Cross-Suite Benchmark Data Contract

Three suites benchmark the same algorithms — native `bench_main.cpp` (C vs Eigen), .NET `EventProp.Benchmarks` (BenchmarkDotNet, managed vs P/Invoke), and `EventProp.Python.Benchmarks` (pytest-benchmark, pybind11 vs Cython). They share sizes (Tiny 10/5/20, Small 50/20/100, Medium 200/80/500, Large 500/200/1000; native and .NET add XL), taus (0.02 / 0.005), `v_th = 1.0`, and weights U(0.5, 2.0).

Spike patterns are generated by **one canonical definition implemented three times**, and the copies must stay in sync:

| Suite | Entry point |
|---|---|
| native | `GenerateBurstData` / `GeneratePoissonData` in `src/EventProp.Native.Benchmarks/bench_main.cpp` |
| .NET | `SpikePatterns.Burst` / `SpikePatterns.Poisson` in `src/EventProp.Benchmarks/SpikePatterns.cs` |
| Python | `generate_burst_spikes` / `generate_poisson_spikes` in `src/EventProp.Python.Benchmarks/conftest.py` |

Every generator upholds, and the Python suite asserts it at generation time (`validate_spike_data` in `conftest.py`):

- **times are non-decreasing** — the C library returns `EVENTPROP_UNSORTED_SPIKE_TIMES` (60) otherwise, and the Cython layer reports `error_no` without raising, so an unsorted array silently measures a validation short-circuit.
- **every spike lands inside [0, 1)** — `max_time / tau_syn` must stay under `EVENTPROP_EXP_OVERFLOW_THRESHOLD` (700). The old .NET burst reached 5.94 s → ratio 1188 → `EVENTPROP_TIMES_TAU_MISMATCH`, so that pattern could not be fed to the native side at all.
- **exactly `n_spikes` are produced, none left unwritten** — a fixed per-neuron burst offset saturates at a handful of bursting neurons, and the remaining slots keep their zero-initialised value: thousands of simultaneous spikes at t = 0. That inflated XL burst to 771k output spikes and made `BackwardSpikes/XL/burst` cost 15.6 s per iteration; the canonical generator uses a computed stagger and yields 757k spikes in 7.9 s.

Burst is deterministic (no RNG in times/sources), so Python and native produce **bit-identical** burst data. Uniform times are identical by construction (only `sources` differ, since `std::mt19937`, `System.Random` and PCG64 cannot agree). Poisson uses the same inverse-CDF form in all three (`-log(1-u)/rate` plus a strictly-increasing fix), so it is distributionally identical. Cross-suite numbers are therefore statistical, not exact — compare ratios, not absolute counts.

Two known caveats: the poisson form concentrates the whole spike train in ~40 ms rather than spreading it over [0, 1) ("Poisson @200 Hz" is a misnomer shared by all three suites), and the Python suite skips Medium+/burst as `SLOW_BURST` — a runtime-cost guard written against the old pathological data, worth re-measuring now that the pattern is sane.

**The poisson misnomer, quantified.** Each spike time is an independent exponential order statistic `-log(1-u)/200`, not a homogeneous Poisson process over [0, 1). Measured (seed 42): max_time ~11 ms / effective ~1.8 kHz at Tiny (20 spikes), ~30 ms / ~3.3 kHz at Small (100), ~46 ms / ~10.8 kHz at Medium (500), ~39 ms / ~25.6 kHz at Large (1000) — 9–128x the 200 Hz label. The pattern key stays `poisson` across all three suites (renaming would break the shared `kPatterns`/`SIZES`/parametrize contract); the generators' docstrings state the real distribution. Cross-suite ratios remain valid because all three use the identical form.

### Layer gradient semantics (C# == Python)

`Layer.Backward()` **accumulates** into `Gradient` and never zeroes it, matching the Python reference (`backward_spikes_batch_cpp(..., self.gradient)` adds; the training loop calls `optimizer.zero_grad()` separately). `ZeroGrad()` is the only zeroing. Call `ZeroGrad()` before `Backward()` for a single-step gradient. This is pinned by `LayerContractTests.Backward_AccumulatesAcrossCalls`, which asserts a second `Backward()` **doubles** the gradient (and guards against a vacuous all-zero gradient by requiring the layer to fire). The managed and native layers implement the same semantic; the native layers add the summed gradient out of the native buffer rather than overwriting.

### Chaining model (intentional C# divergence from Python)

Python threads layers through an `ancestor_layer` field and a `__call__(SpikesVector, Layer)` idiom; its base `backward()` recurses into the ancestor and raises `RuntimeError("Run backward to create errors first.")` when the child has not run. C# has no `ancestor_layer` and no `__call__`; chaining is the explicit `Layer.Chain<TLayerA,TLayerB>` whose `Backward()` calls `A.Backward(); B.Backward()` directly. The ancestor-recursion and its "run backward first" guard have no C# counterpart — this is a deliberate simplification, documented on `LayerChain.Backward`.

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

### Zero-Copy Output Lifetime

`get_spike_errors(i)` and `get_maxima_errors(i)` return numpy views into C-owned memory. The owner is the wrapper, not the view, so anything that hands a view to a caller must keep the wrapper alive for as long as the view can be read. `LIFLayer.backward` / `LILayer.backward` therefore store the `AccumulatedBackwardResult` on `self._backward_result`: the consumer is the ancestor layer's `backward()`, which runs after the local `result` would otherwise go out of scope. See *Fixed — Cython layers: use-after-free in spike-error propagation*.

### Paper Parity Suite

`src/EventProp.Native.Python/tests/test_paper_parity.py` runs the pre-trained checkpoints in `EventProp.Python/paper/` (YinYang TTFS, MNIST VMax) through **both** backends and compares forwards, predictions and gradients. It is the only suite that exercises real trained weights, so it catches divergences that the synthetic fixtures miss.

Two of its tests were themselves broken and are fixed here — when a parity test fails, check the test before blaming the library:

- `test_forward_predictions` used `np.nanargmin` over `first_spike_times`. A sample where no output neuron fires is an all-NaN row, which the **reference** produces (1 of 64 YinYang samples), and `nanargmin` raises `ValueError` on it. Predictions now use a NaN-safe argmin with `-1` for all-NaN rows, on both sides.
- `test_forward_vmax_values` read `py_loss.values_batch`, which does not exist on the pybind11 `VMaxCrossEntropyLoss`. The two backends expose the same numbers under different names: `py_loss.maxima_batch[i].values` vs `cy_loss.values_batch[i]`.

## Architecture Notes (.NET)

### Shared Layer Contract (`EventProp.Abstractions`)

`EventProp.Abstractions` (assembly `CW.EventProp.Abstractions`, namespace `CW.EventProp`, no dependencies) holds the types both backends share, mirroring the Python library's `ABC`-based `layer.py`:

- `Layer` — abstract base class (not an interface) with abstract `Forward`/`Backward`/`ZeroGrad`/`GetWeights`/`SetWeights`/`NIn`/`NOut` and the nested `LayerChain<TA,TB>`. An abstract class (not `IDisposable`-style interfaces) is used because the contract passes concrete reference types (`SpikesVector`, `SpikesData`) and shares state (`Gradient`, `_inputBatch`, `_ranForward`); the Python reference uses `ABC` + `@abstractmethod` the same way.
- `SpikesData`, `MaximaData`, `SpikesVector`, `MaximaVector` — heap-backed data carriers (ArrayPool-rentable), shared by both backends.
- `LIFLayerParameters`, `LILayerParameters`, `WeightInitializer` (+ Gaussian/Uniform/Diagonal).

`EventProp` (managed) and `EventProp.Native` (P/Invoke) each `ProjectReference` Abstractions and implement `Layer`. The compiler enforces the contract; benchmarks and parity tests run the same scenario against both backends.

### Native Layers (`EventProp.Native.LIFLayer` / `LILayer`)

Thin P/Invoke-backed `Layer` implementations so `EventProp.Benchmarks` compares managed vs native like-for-like. Key design points:

- **Friendly batch wrappers** (`BatchFriendly.cs`): `Batch.ComputeSpikesBatch` / `ComputeMaximaBatch` / `BackwardSpikesBatchAccumulate` / `BackwardMaximaBatchAccumulate` take flat arrays (`w`, concatenated `times`/`sources`, `nSpikesPerSample`) and marshal to an `EventPropInput[]` on the stack, so callers never hand-build per-sample structs. `SpikesBatch`/`MaximaBatch` expose per-sample indexing (`SpikeAt`/`MaximaAt`, plus a `this[int]` indexer on `SpikesBatch`).
- **`ref struct` outputs cannot be stored on a class field.** `SpikesBatch`/`MaximaBatch` are `ref struct`s, so the native layers cannot cache the forward batch across `Forward()`→`Backward()`. `Backward()` recomputes the forward batch (deterministic for the same weights + input) in the same scope and feeds the accumulate call.
- **Loss-derivative plumbing differs by loss type.** For the spike loss, the C backward reads `post_spike_errors` from a caller-supplied pointer array; the layer pins each `SpikesData.Errors` array (`GCHandle`) and passes that array. For the maxima loss, the C backward reads `errors` from the **native** `MaximaOutput` (library-allocated), so `LILayer.Backward` copies the managed `MaximaData.Errors` into the native batch's errors array before the accumulate call.
- Native layers are `IDisposable` (free pinned handles); the managed layers expose `Dispose()` for their scratch buffers.

Parity is pinned by `EventProp.Tests/NativeLayerParityTests.cs` (LIF spikes+gradient, LI maxima+gradient, native batch forward).

### Layer Contract Suite (`EventProp.Tests/LayerContractTests.cs`)

The `Layer` contract is tested against **every** backend, so a new backend must satisfy the same behaviour and the managed/native pair cannot drift. A `LayerBackend` record carries the factory (`CreateLif`/`CreateLi`) plus the backend-specific plumbing (`SetSpikeError`, `GetMaxima`); each scenario runs over `AllBackends()`.

Covered per backend: gradient shape, forward/backward determinism, `ZeroGrad`, backward accumulation (a second `Backward()` doubles the gradient — Python semantics, with a firing guard so the check is non-vacuous), weight span round-trip, weight-length validation (ctor + `SetWeights`), Backward/Output-before-Forward throws, forward-replaces-output, batch forward == per-sample, batch gradient == sum of per-sample gradients, LI maxima shape, LI backward determinism. Cross-backend: LIF/LI/batch gradients match between managed and native (with firing + non-zero-gradient guards), and a **mixed managed↔native `LayerChain`** produces the same output as a homogeneous chain.

`Layer` is `IDisposable` (managed layers own ArrayPool buffers, native layers own pinned handles + native batches); `LayerChain.Dispose()` disposes both children. `SyncAssert` gained a `bool` overload (`IsTrue`/`IsFalse`).

> TUnit 1.45 has no `PropertySource`; `MethodDataSource` on a non-static test needs `[CombinedDataSources]` (TUnit0070). The suite sidesteps that by looping `AllBackends()` inside each `[Test]` body.

### Zero-Allocation Hot Paths

- **`MatrixView`** — `ref struct` providing zero-copy row-major matrix views over flat `double[]` arrays
- **`StridedSpan`** — column access without allocation (strided read-only span)
- **`ScratchBuffers`** — pre-allocated reusable buffers; callers allocate once and pass in
- **`PostSpikeRow`** — one growable scratch row per post-synaptic neuron (`times` / `currents` / `exp(st/tau_mem)`), rented once from `ArrayPool` and grown ×3 in place. A neuron may cross the threshold any number of times, so its spikes cannot live in a fixed-stride slice of a shared slab; each `Parallel.For` iteration owns exactly one row, so growth needs no atomics

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

## Fixed Defects — Invariants and Failure Modes

Every entry below is **fixed**; there are no open defects in this list. They are kept because each records a constraint that is not visible in the current code and is easy to re-introduce: an invariant a refactor must preserve, a parity contract shared by both backends, a diagnostic signature for recognising the failure class, and the regression test that pins it. Treat this as a post-mortem log and a checklist for touching the areas named, not as a bug tracker entry.

### Fixed — C forward pass: spurious spikes in the final interval

`eventprop_compute_spikes` (`src/EventProp.C/eventprop_spikes.c`) special-cased the bracket floor for the last interval as `t_before = count > 0 ? p_times[count - 1] : 0.0`, dropping the `times[i - 1]` term that every other interval applies. When the neuron had not fired yet, the floor collapsed to **0.0**, the `t_before <= tmax` guard in `generate_bracket_candidates` passed for an analytical peak sitting *behind* the interval, and the solver bracketed a region where `sum0[i]`/`sum1[i]` are an extrapolation of a spike train that has not arrived yet — reporting roots at times where the real membrane potential is far below `v_th`.

The rule that was missing: `sum0[i]`/`sum1[i]` include every input spike up to `times[i - 1]`, so the voltage model of interval `i` is only valid for `t >= times[i - 1]`.

Measured on the trained YinYang checkpoint (`paper/yinyang_0.pkl`, hidden layer 5x200, 64 test samples), the C library emitted **strictly more** spikes than the Eigen reference — a superset, never a subset — on 35 of 64 samples (1888 vs 1833 total). Neuron 177, sample 0: C reported a spike at `t = 0.013792378751` while the true voltage there is **0.701925314967** against `v_th = 1.0`; the interval's analytical peak is `tmax = 0.017081371`, behind `times[4] = 0.022977101`, and the true peak of the whole window is 0.960201051.

**Fix**: the last-interval branch was deleted, not patched. One floor serves every interval — `max(times[i - 1], post spikes <= t_input)` — because `t_input` is `INFINITY` in the last interval, so the existing scan already takes the maximum over all post spikes, which is exactly what the Eigen reference does. After the fix the two backends agree exactly: 1833 vs 1833 spikes, zero differing samples, and the full `tests/test_paper_parity.py` suite goes from 6 failures to 0.

This is the reverse of the framing in *Spike Time Ordering* above: there the C library is the stricter *diagnostic* of a contract both implementations share. Here the C library was the **divergent implementation** — the Eigen reference and the managed .NET side (`EventPropMath.cs`, which already floors every interval with `timesArr[i - 1]`) were both correct. When a parity gap is reported, check which side is the outlier before deciding which to change.

Regression tests: `ComputeSpikes.LastIntervalPeakBehindLastInputTimeNoSpuriousSpike` (the neuron-177 fixture, asserts 0 spikes; fails pre-fix with the root at `t = 0.013792389158539134`) and `ComputeSpikes.LastIntervalGenuineSpikeStillFound` (a last interval that genuinely crosses `v_th` at 0.023057647, so the new floor cannot suppress real spikes) in `src/EventProp.C.Tests/test/compute_spikes.cpp`.

### Fixed — Cython layers: use-after-free in spike-error propagation

`LIFLayer.backward` and `LILayer.backward` (`eventprop_c/lif_layer.py`, `eventprop_c/li_layer.py`) propagated the loss signal to the previous layer with zero-copy views into the native `AccumulatedBackwardResult`:

```python
result = backward_spikes_batch_accumulate(...)
for i in range(result.batch_size):
    inp._errors = result.get_spike_errors(i)   # view into result's native buffer
# result is a local — it dies here, __dealloc__ frees the buffer, views dangle
```

The consumer is the *ancestor* layer's `backward()`, invoked by `super().backward()` at the end of the same method, and the views stay live on the input spikes after that. Dropping the last reference to `result` frees the buffers they point at, so the ancestor's backward reads freed heap memory.

Symptom on the YinYang fixture: the hidden layer's incoming spike errors were denormal garbage (~1e-158, 2 nonzero per sample) instead of 299.15 / 1.88 / 5.34 / 4.33, and the hidden gradient was ~1000x too small with 179 of 222 entries exactly zero (`max|g| = 9.4e-4` against the reference's `1.265454`). The *loss* layer's own gradient was correct to 8.5e-14 throughout, which is what localises this to propagation rather than to the adjoint sweep — the C `spike_errors` output is healthy (max 106.0) when read directly.

**Fix**: pin the result on the layer (`self._backward_result = result`) so the native buffer outlives the views handed to the ancestor. `Layer.__init__` initialises the attribute. After the fix the hidden gradient matches the reference exactly (`max|g| = 1.265453817356891` vs `py[1,119] = -1.265454e+00`).

Regression tests: `SpikeErrorPropagationLifetimeTest` in `src/EventProp.Native.Python/test/test_layer_chain.py`. The freed `spike_errors` blocks are ~11 KB, so the test recycles that size class with same-size numpy allocations immediately after `backward()` — that makes the dangling read deterministic (max|error| = 5.1e+77 pre-fix, ~8-11 post-fix) instead of leaving it to heap layout, which is not reliably detectable. A plain `gc.collect()` + `bytearray` churn passed against the unfixed code.

### Fixed — Managed .NET forward pass: per-neuron spike buffers overflow

`EventPropMath.ComputeSpikesParallel` (`src/EventProp/EventPropMath.cs`) used to size each post-synaptic neuron's scratch run at `maxSpikesPerNeuron = max(nSpikes + 1, buffers.MaxSpikesPerNeuron)` and index it as `neuronOffset = targetNrn * maxSpikesPerNeuron`. A neuron can emit **more** spikes than that bound, and the only guard was against the end of the whole buffer (`neuronOffset + neuronSpikeCount < buffers.PostSpikeTimes.Length`), so past the bound writes spilled into the neighbouring neuron's run. Because the loop is a `Parallel.For` over neurons, adjacent runs were written concurrently and the surviving spike set depended on thread interleaving.

Symptoms (input `w = 15.0 + i`, `tau_mem=0.02`, `tau_syn=0.005`, `v_th=1.0`, input times `[0.01, 0.02, 0.03]`, sources `[0, 1, 0]`, `n_post=3`, capacity hint 4):

- Spike count nondeterministic across runs: 21, 26, 26, 26, 27, 27 — the C library returns 31 every time.
- Per-neuron counts exceeded the bound (10 / 8 / 8), i.e. every run past slot 4 was outside its region.
- Post spikes sharing an identical time also shared one current (`t=0.036340173829084`, `src=0,1,2` all reporting `4.9835283556` instead of `4.9835283556 / 5.3081473409 / 5.6327663262`) — the wrong entries were spillover, not the formula.

**Fix** (mirrors the C refactor `54fa50a refactor(C): replace per-neuron spike slab with per-thread scratch rows`): `ScratchBuffers` now owns `PostSpikeRow[]`, one growable row per neuron, and `ComputeSpikesParallel` appends with `row.Grow(neuronSpikeCount, neuronSpikeCount + 1)` instead of bounding writes. `MaxSpikesPerNeuron` is an initial capacity hint, no longer a bound. Two invariants matter for any future change here:

- `Grow` must **copy the live spikes** into the new arrays — `ArrayPool.Rent` hands back dirty memory, so losing them silently corrupts the neuron's voltage model (observed as spike counts *dropping*, e.g. 4156 vs C's 6381 on `ComputeSpikes_Uniform_Medium`).
- `Grow` must **not return the superseded arrays to the pool** while the sweep still caches them in locals; handing them over can give the same memory to another thread. Growth is logarithmic, so the abandoned arrays are a handful per row and the GC reclaims them.

Regression tests: `ComputeSpikes_PerNeuronSpikeBurst_MatchesNativeSpikeSet` (`src/EventProp.Tests/NativeVsManagedEdgeTests.cs`, pins the 31-spike set against C) and `PostSpikeRow_GrowPreservesLiveSpikes` (`src/EventProp.Tests/ScratchBuffersTests.cs`).

### Fixed — C input validation rejected `v_th <= 0` on entry points that never read it

`eventprop_validate_input` (`src/EventProp.C/eventprop_validation.c`) required `v_th > 0` for every entry point, but `ComputeSums`, `ComputeMaxima`, and `backward_maxima` never read the threshold voltage — the cumulative sums and the maxima loss are defined for any `v_th`. The Eigen reference and the managed .NET implementations accept `v_th = 0`, and `src/EventProp.Native/Api.cs` builds `EventPropInput.Create(..., 0.0, tauMem, tauSyn)` because those .NET-facing APIs have no `v_th` parameter at all. Result: 40 parity tests failed with `EVENTPROP_INVALID_VTH` (11) on the C side.

**Fix**: the validator was split rather than loosened, so no check is duplicated and no call site lost coverage:

| Function | Checks | Used by |
|---|---|---|
| `eventprop_validate_input_no_v_th` | dimensions, spike count, taus, spike times/sources, pointers | `ComputeSums`, `ComputeMaxima`, `backward_maxima` |
| `eventprop_validate_input` | the above **plus** `!(v_th > 0.0)` (negated so NaN is rejected) | `ComputeSpikes`, `BackwardSpikes`, diagnostic traces |

The strict wrapper calls the structural one first, so the two can never drift apart. `eventprop_diag.c` keeps its own `v_th <= 0` guard at line ~108 — that one guards the division in the λ_I jump computation, not the input.

New C tests pinning the contract: `ValidateInputNoVth_*` (5 in `src/EventProp.C.Tests/test/validate_input.cpp` — zero/negative/NaN `v_th` accepted, structural errors still caught) and `ZeroVthAcceptedAndIgnored` in `compute_sums.cpp`, `compute_maxima.cpp`, `backward_maxima.cpp`, each running one input with `v_th = 1.0` and `v_th = 0.0` and asserting identical output.

## Platform Support

- **macOS** — developed and tested
- **Linux** — intended support
- **Windows** — supported via MSVC + CMake (VS Code workspace configs provided)
