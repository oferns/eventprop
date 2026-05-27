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
│   │   └── *.cs                # Algorithm implementations
│   └── EventProp.Tests/        # Test project (TUnit)
│       └── *.cs                # Unit tests
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

## Running Tests & Coverage

```bash
# Run tests
cd src/EventProp.Tests && dotnet run --

# Run with coverage (output → bin/Debug/net10.0/TestResults/coverage.cobertura.xml)
dotnet run --project src/EventProp.Tests -- --coverage --coverage-output coverage.cobertura.xml --coverage-output-format cobertura

# Generate HTML coverage report
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:src/EventProp.Tests/bin/Debug/net10.0/TestResults/*.cobertura.xml \
  -targetdir:coverage-report -reporttypes:Html

# Generate HTML coverage report
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Debug/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html
```

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
