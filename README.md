# EventProp .NET

A high-performance, allocation-free implementation of the **EventProp** algorithm for training Spiking Neural Networks in C#.

## What is EventProp?

EventProp is an event-based backpropagation method that computes **exact gradients** through spike times in spiking neural networks. Unlike surrogate gradient approaches, EventProp analytically differentiates through the threshold-crossing event itself, enabling precise weight updates without approximation.

### Key Features

- **Exact gradients** — no surrogate gradients or approximations
- **Zero-allocation hot paths** — `ref struct` matrix views, `Span<T>` operations, pre-allocated scratch buffers
- **LIF neurons** — Leaky Integrate-and-Fire model with dual time constants (membrane + synaptic)
- **Two loss modes** — Time-To-First-Spike (TTFS) cross-entropy and voltage-based (membrane potential maxima)
- **.NET 10 native** — modern C# with nullable reference types, implicit usings

## Quick Start

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

## Native Bindings

The project includes two P/Invoke binding layers for calling native code:

- **`EventProp.Native`** — Binds to the pure C library (`libeventprop_c_api`) via `LibraryImport` + `[UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]`
- **`EventProp.Cpp.Native`** — Binds to the C++ wrapper (`libeventprop_cpp_wrapper`) with Eigen/Boost support

Both use modern .NET 10 source-generated P/Invoke (no `DllImport`).

## Project Structure

```
src/
├── EventProp/              # Core library
│   ├── SpikesData.cs       # Spike event container
│   ├── MaximaData.cs       # Membrane potential maxima
│   ├── ScratchBuffers.cs   # Reusable pre-allocated buffers
│   ├── MatrixView.cs       # Zero-copy row-major matrix ref struct
│   ├── StridedSpan.cs      # Strided column access (no allocation)
│   ├── SpanAlgorithms.cs   # Element-wise math helpers
│   ├── SimdAlgorithms.cs   # SIMD-optimized algorithms (AVX2/AVX-512)
│   ├── EventPropMath.cs    # Core algorithm implementations
│   ├── Layer.cs            # Abstract layer + chain composition
│   ├── LIFLayer.cs         # Leaky Integrate-and-Fire layer
│   ├── LILayer.cs          # Integrate-and-Fire layer
│   ├── WeightInitializer.cs# Diagonal weight initialization
│   └── BatchVectors.cs     # Per-batch state vectors
├── EventProp.Native/       # P/Invoke bindings (C library)
├── EventProp.Cpp.Native/   # P/Invoke bindings (C++ wrapper)
├── EventProp.Benchmarks/   # BenchmarkDotNet benchmarks
└── EventProp.Tests/        # TUnit tests
```

## Running Tests

```bash
cd src/EventProp.Tests && dotnet run -- --report-trx
```

## Coverage

```bash
cd src/EventProp.Tests && dotnet run -- --coverage --coverage-output-format=cobertura
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Debug/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html
open coverage-report/index.html
```

## Technical Details

### Memory Model

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

## License

See [LICENSE](LICENSE) for details.
