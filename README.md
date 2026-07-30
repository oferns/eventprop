# EventProp

A multi-language implementation of the **EventProp** algorithm for training Spiking Neural Networks.

## What is EventProp?

EventProp is an event-based backpropagation method that computes **exact gradients** through spike times in spiking neural networks. Unlike surrogate gradient approaches, EventProp analytically differentiates through the threshold-crossing event itself using the adjoint method from control theory — enabling precise weight updates without approximation.

This repository contains implementations in:
- **C** — high-performance native library with OpenMP parallelization and SIMD helpers
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
│   ├── EventProp.C.Tests/      # Google Test v1.15.2 suite for C library (176 tests)
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

# Clean all build outputs
clean.sh              # removes everything (all targets)
clean.sh c            # only C library
clean.sh cpp          # only C++ wrapper
clean.sh python       # only Python package
clean.sh dotnet       # only .NET projects
```

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
- **`EventProp.Native.Python/`** — Cython bindings to the C library. Zero-copy numpy views, managed memory wrappers, and flat-array FFI variants for maximum efficiency.

**Install Cython bindings**:
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
cd src/EventProp.C.Tests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./eventprop_tests  # 176 tests
```

**With AddressSanitizer** (Debug builds only):
```bash
cd src/EventProp.C/build && rm -rf * && cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON && cmake --build .
cd ../EventProp.C.Tests/build && rm -rf * && cmake .. -DENABLE_ASAN=ON && cmake --build .
./eventprop_tests  # detects use-after-free, buffer overflows
```

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

## Platform Support

- **macOS** — developed and tested
- **Linux** — intended support
- **Windows** — supported via MSVC + CMake
