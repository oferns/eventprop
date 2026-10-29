# EventProp

A multi-language implementation of the **EventProp** algorithm for training Spiking Neural Networks.

## AI Involvement

A local AI model (Qwen) was used to help with most aspects of these libraries. An iterative approach was take avoid AI "slop".  

## What is EventProp?

EventProp is an event-based backpropagation method that computes **exact gradients** through spike times in spiking neural networks. Unlike surrogate gradient approaches, EventProp analytically differentiates through the threshold-crossing event itself using the adjoint method from control theory — enabling precise weight updates without approximation.

This repository contains implementations in:
- **C** — native library with OpenMP parallelization and SIMD helpers
- **C++ (Eigen)** — reference core from the original EventProp.Python library
- **Python** — bindings via pybind11 (reference) and Cython → C library
- **C# (.NET 10)** — fully managed implementation + P/Invoke bindings to the C library

## Project Structure

```
eventprop.net/
├── build.sh                    # Build all targets (C, C++, Python, .NET)
├── clean.sh                    # Clean all or individual targets
├── src/
│   ├── EventProp.Python/       # Reference library (git submodule, read-only)
│   │                           # Python + C++ (Eigen/pybind11) — source of truth
│   ├── EventProp.C/            # Pure C implementation (libeventprop_c_api)
│   │                           # Batch API, flat-array FFI, strict warnings, ASan support
│   ├── EventProp.C.Tests/      # Google Test suite for the C library
│   ├── EventProp.Cpp.Wrapper/  # C++ Eigen core (convenience copy, no pybind11)
│   │                           # Used for native benchmarks and parity tests
│   ├── EventProp.Native/       # .NET P/Invoke bindings → C library
│   ├── EventProp/              # Fully managed .NET implementation
│   ├── EventProp.Tests/        # TUnit test suite (correctness + parity)
│   ├── EventProp.Benchmarks/   # BenchmarkDotNet (native vs managed .NET)
│   ├── EventProp.Native.Python/  # Python + Cython bindings → C library
│   ├── EventProp.Native.Benchmarks/  # Google Benchmark (C vs C++)
│   ├── EventProp.Native.ParityTests/ # Standalone parity tests (C vs Eigen)
│   └── EventProp.Python.Benchmarks/  # pytest-benchmark (pybind11 vs Cython)
└── coverage-report/            # Generated HTML coverage reports (gitignored)
```

## Building

```bash
# Build everything — Release mode
./build.sh release

# Build everything — Debug mode
./build.sh debug

# Clean build outputs (targets compose; nothing here touches bench-reports/ or BENCHMARKS.md)
./clean.sh                        # everything
./clean.sh c cpp parity           # only these targets
./clean.sh --dry-run -full        # show what -full would remove, remove nothing
```

Both scripts share one contract with their PowerShell twins (`build.ps1`, `clean.ps1`) and
print their full interface: `./build.sh --help`, `./clean.sh --help`. The Python layers are
built for one interpreter — pass `--python` / `-PythonExe` to pick the one that will run the
suites.

## Quick Start (.NET)

### Low-level API (EventPropMath)

```csharp
using CW.EventProp;

// Prepare input spikes (sorted by time)
var inputSpikes = new SpikesData(
    times: [0.01, 0.02, 0.03],
    sources: [0, 1, 0]
);

// Weight matrix (row-major): n_pre x n_post
double[] weights = new double[nPre * nPost];

// Allocate scratch buffers once, reuse across training steps
var buffers = new ScratchBuffers();
buffers.EnsureCapacity(nNeurons, nPre, inputSpikes.NSpikes);

// Forward pass — predict output spike times
var outputSpikes = EventPropMath.ComputeSpikes(
    weights, inputSpikes.Times, inputSpikes.Sources,
    nPost, vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, buffers
);

// Backward pass — compute gradients
var gradient = new double[nPre * nPost];
EventPropMath.Backward(
    inputSpikes, outputSpikes, weights,
    vTh: 1.0, tauMem: 0.02, tauSyn: 0.005, gradient, buffers
);

// Apply gradients to weights (e.g., via Adam optimizer)
```

### High-level API (Layer classes)

```csharp
using CW.EventProp;

// Create a layer with parameterized configuration
var layer = new LIFLayer(
    parameters: new LIFLayerParameters(
        tauMem: 0.02, tauSyn: 0.005, vTh: 1.0
    ),
    nPre: inputNeurons,
    nPost: outputNeurons,
    initializer: new DiagonalInitializer(scale: 0.5)
);

// Forward pass — returns output spike times
var outputSpikes = layer.Forward(inputTimes, inputSources);

// Backward pass — computes gradients in-place
layer.Backward(postTimes, postSources, postErrors);

// Access gradients and update weights
var gradient = layer.Gradient;
for (int i = 0; i < weights.Length; i++)
    weights[i] -= learningRate * gradient[i];

// Chain layers for multi-layer networks
var chain = layer1.Chain(layer2);
chain.Forward(inputTimes, inputSources);  // propagates through both
```

## Python Bindings

Two Python binding layers are available:

- **`EventProp.Python/`** — pybind11-based bindings (git submodule). The reference implementation with a complete Python layer.
- **`EventProp.Native.Python/`** — Cython bindings to the C library. Zero-copy numpy views, managed memory wrappers, and flat-array FFI variants for maximum efficiency. Also the home of the [continuous readout and regression loss](src/EventProp.Native.Python/README.md#continuous-readout-and-regression), which exists in C so that P/Invoke and Cython share one implementation.

**Install Cython bindings** — or let `./build.sh release` do it, including regenerating
`_native.c` from `_native.pyx` (Cython is a build prerequisite of that step, not something
the package fetches):
```bash
cd src/EventProp.Native.Python && pip install -e .
```

## Running Tests

### .NET Tests (TUnit)

```bash
cd src/EventProp.Tests && dotnet run -c Release -- --no-progress
```

> **Note**: TUnit uses Microsoft Testing Platform (MTP), not VSTest. `dotnet test` does NOT work. The `-c` flag goes BEFORE `--`, TUnit flags go AFTER `--`. Use `--treenode-filter` instead of `--filter`.

### C Library Unit Tests (Google Test)

```bash
cd src/EventProp.C.Tests
cmake --preset release && cmake --build --preset release
ctest --preset release            # or: ./build/eventprop_tests
```

**With AddressSanitizer** (Debug presets only — ASan needs a clean cache):

```bash
cd src/EventProp.C        && cmake --preset asan && cmake --build --preset asan
cd ../EventProp.C.Tests   && cmake --preset asan && cmake --build --preset asan
ctest --preset asan
```

> On macOS ASan builds a **static** library (an AppleClang linker limitation) — which is why the
> ASan presets `cleanFirst`: the library flips SHARED→STATIC inside the same `build/` directory
> the test project reads. On Linux, add leak detection at run time with
> `ASAN_OPTIONS=detect_leaks=1 ctest --preset asan`; it is not in the presets because test
> discovery runs the binary at build time and macOS ASan refuses `detect_leaks`.

### C/C++ Parity Tests (C vs Eigen)

This project imports a **pre-built** library from two siblings, so build those first:

```bash
cd src/EventProp.C                 && cmake --preset release && cmake --build --preset release
cd ../EventProp.Cpp.Wrapper        && cmake --preset release && cmake --build --preset release
cd ../EventProp.Native.ParityTests && cmake --preset release && cmake --build --preset release
./build/parity_test ../../EventProp.C/build/libeventprop_c_api.dylib
```

> The library-path argument is **decorative**: `parity_test` links the library and never reads
> `argv`, so which implementation it exercises is fixed at link time.

### Native Benchmarks (C vs C++)

```bash
cd src/EventProp.Native.Benchmarks
cmake --preset release && cmake --build --preset release
./build/bench_c   --benchmark_min_time=0.1s
./build/bench_cpp --benchmark_min_time=0.1s
```

On Windows the executables land in `build/Release/` (multi-config generator), not `build/`.

> For anything you intend to publish, use the cross-suite runner instead —
> `./run_benchmarks.sh` (macOS/Linux) or `./run_benchmarks.ps1` (Windows). It is slow by
> design, and **a subset run rewrites the committed `BENCHMARKS.md` and
> `bench-reports/report_<tag>.md` at the same path**: pass `--no-reports
> --no-benchmarks-md` for an exploratory run.

### Python Benchmarks (pytest-benchmark)

```bash
cd src/EventProp.Python.Benchmarks
PYTHONPATH=../EventProp.Python:../EventProp.Native.Python \
  python3 -m pytest test_benchmarks.py --benchmark-only -v
```

Two suites live here: `test_benchmarks.py` (layers, pybind11 vs Cython) and
`test_kernel_benchmarks.py` (raw C kernels with no layer, which isolates binding overhead).
Name the interpreter that has the built `eventprop_c` extension; the cross-suite workflow is
in that directory's `README.md`.

## Coverage

```bash
# Run with coverage
cd src/EventProp.Tests && dotnet run -c Release -- --coverage --coverage-output-format=cobertura

# Generate HTML report
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Release/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html
```

## Technical Details

### Memory Model (.NET)

All hot-path functions are **allocation-free** when `ScratchBuffers` is provided:

| Pattern | Implementation |
|---------|---------------|
| Matrix storage | Flat `double[]` with row-major indexing |
| Row access | `Span<double>` slice (zero-copy) |
| Column access | `StridedSpan` (strided view, no allocation) |
| Transpose | In-place into pre-allocated buffer |
| Vectors | `Span<double>` / `ReadOnlySpan<double>` |

### Biophysical Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Membrane time constant | `tauMem` | Decay rate of membrane potential (seconds) |
| Synaptic time constant | `tauSyn` | Decay rate of synaptic current (seconds) |
| Spike threshold | `vTh` | Membrane potential at which a spike fires (volts) |

### Key Algorithms

| Algorithm | Description |
|-----------|-------------|
| **ComputeSums** | Pre-computes cumulative exponential-weighted sums for fast membrane potential evaluation |
| **ComputeSpikes** | Forward pass — finds threshold-crossing times via root-finding (Toms748/Brent) |
| **Backward** | Adjoint method — time-reversed sweep propagating λ_v (membrane) and λ_i (synaptic) adjoints |
| **ComputeMaxima** | Finds membrane potential maxima per neuron (for voltage-based loss) |
| **MaximaRegression** | Continuous readout `b + Σ aₙ·maximaₙ` with least-squares or smooth Huber loss; writes `dL/dmaxima` for the adjoint, so a regression target needs no loss arithmetic in the binding — reached as `LILayer.Regression(...)` in .NET and `LILayer.regression(...)` in Python. C only — its parity partner is the managed `CW.EventProp.RegressionLoss`, not the Eigen core |

## Platform Support

- **macOS (arm64)** — AppleClang + Homebrew toolchain
- **Linux (x86_64)** — GCC/Clang; needs `libtbb-dev` for the Eigen core, and Cython in the
  interpreter that builds the Python extension
- **Windows (x86_64)** — MSVC + CMake; `build.ps1` imports the vcvars environment for you

All three are supported and driven by one contract shared between the shell and PowerShell
scripts. There is no CI, so platform status is established by running `./build.sh release`
(`.\build.ps1 release` on Windows) plus the suites **on that machine** — say which platform a
claim was measured on. `./scripts/submodule-state.sh status` reports the reference
submodule's local state, which is expected to be dirty on every platform.
