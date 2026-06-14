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
│   │                           # 2 sets of batch functions: general + Cython interface
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

### Reference Library

**`EventProp.Python/`** is a git submodule of the original EventProp library. It is the **reference implementation** — all other implementations should mimic its API and produce matching results.

Before building, apply the macOS compatibility patch:
```bash
cd src/EventProp.Python && patch -p0 < submodule.patch
```

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
- **`EventProp.Native.Python/`** — Cython bindings to the C library. The Python layer from EventProp.Python rewritten to use `libeventprop_c_api` via Cython (`eventprop_c/_native.pyx`).

**Install pybind11 bindings**:
```bash
cd src/EventProp.Python && patch -p0 < submodule.patch && pip install -e .
```

**Install Cython bindings**:
```bash
cd src/EventProp.Native.Python && pip install -e .
```

## Running Tests

### .NET Tests (TUnit)

```bash
cd src/EventProp.Tests && dotnet run -c Release -- --no-progress
```

> **Note**: TUnit uses Microsoft Testing Platform (MTP), not VSTest. `dotnet test` does NOT work.
> The `-c` flag goes BEFORE `--`, TUnit flags go AFTER `--`.

### Native Unit Tests

Three independent native test suites validate correctness across implementations:

```bash
# C library unit tests (17 tests, includes large-scale 200×80)
cd src/EventProp.C && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . && ./test_runner

# Eigen direct unit tests (23 tests, edge cases + large-scale)
cd src/EventProp.Cpp.Wrapper && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . && ./eventprop_cpp_tests

# C API wrapper unit tests (14 tests, flat-array interface over Eigen)
cd src/EventProp.Cpp.Wrapper/build && ./eventprop_capi_tests
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| **C library** (`tests.c`) | 17 | All functions + edge cases (zero spikes, negative weights, large-scale) |
| **Eigen direct** (`tests.cpp`) | 23 | All Eigen methods + degenerate shapes, parameter variations, temporal extremes |
| **C API wrapper** (`tests_capi.cpp`) | 14 | Flat-array C interface mirroring pure C suite |
| **Parity tests** (C vs Eigen) | 85 | Numerical equivalence at ≤50×20 scale |
| **.NET** (TUnit) | 408+ | Full managed API + three-way parity (C ≡ Eigen ≡ .NET) |

### Python Benchmarks

```bash
cd src/EventProp.Python.Benchmarks && pytest test_benchmarks.py -v
```

### Native Benchmarks (C vs C++)

```bash
cd src/EventProp.Native.Benchmarks && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./bench_c --benchmark_min_time=0.1s
./bench_cpp --benchmark_min_time=0.1s
```

## Coverage

```bash
cd src/EventProp.Tests && dotnet run -c Release -- --coverage --coverage-output-format=cobertura
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Release/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html
open coverage-report/index.html
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

### Synaptic Kernel

The library uses an α-function synaptic kernel:

```
k(t) = τ_syn/(τ_mem - τ_syn) * (exp(-t/τ_mem) - exp(-t/τ_syn))
```

This produces a rise-and-decay current profile that is more biologically realistic than simple exponential decay.

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
- **Windows** — if feasible

## License

See [LICENSE](LICENSE) for details.
