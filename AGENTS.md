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
│   └── EventProp.Tests/        # Test project (TUnit)
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

### TUnit Test Running (Critical)

TUnit uses the Microsoft Testing Platform — **not** VSTest. `dotnet test` does NOT work on .NET 10.

**Always run from the solution root or the test project directory.**

```bash
# Run ALL tests (Debug mode — default)
cd src/EventProp.Tests && dotnet run --

# Run ALL tests (Release mode — required for numerical parity tests)
cd src/EventProp.Tests && dotnet run -- -c Release
```

### TUnit Flags Reference

| Flag | Effect | Example |
|------|--------|---------|
| `-c Release` | Build in Release mode (must use for parity tests — Debug has different FP ordering) | `dotnet run -- -c Release` |
| `--no-progress` | Suppress the ASCII spinner, cleaner output for grep/parsing | `dotnet run -- -c Release --no-progress` |
| `--filter "FullyQualifiedName~BackwardSpikes"` | Run only tests matching pattern | `dotnet run -- -c Release --filter "FullyQualifiedName~BackwardSpikes"` |
| `--list-tests` | List all test names without running | `dotnet run -- --list-tests` |
| `--output Detailed` | Show full failure details in output | `dotnet run -- -c Release --output Detailed` |
| `--fail-fast` | Stop after first failure | `dotnet run -- -c Release --fail-fast` |
| `--coverage` | Enable coverage collection | `dotnet run -- -c Release --coverage` |
| `--coverage-output <file>` | Specify coverage output file path | `dotnet run -- -c Release --coverage --coverage-output coverage.cobertura.xml` |
| `--coverage-output-format cobertura` | Coverage format (cobertura, json, etc.) | `dotnet run -- -c Release --coverage-output-format cobertura` |

### Parsing TUnit Output (Common Patterns)

TUnit output is verbose and multi-line. Use these grep patterns to extract meaningful information:

```bash
# Count total failures
dotnet run -- -c Release 2>&1 | grep "failed " | wc -l

# List only failed test names
dotnet run -- -c Release 2>&1 | grep "failed BackwardSpikes"

# Get failure reason for a specific test (3 lines after the name)
dotnet run -- -c Release 2>&1 | grep "failed BackwardSpikes_Uniform_Tiny" -A3

# Show only C vs Eigen comparison failures (NativeLibraryParityTests)
dotnet run -- -c Release 2>&1 | grep "CompareNativeBackwardSpikes" -B3

# Show only .NET vs Eigen comparison failures (NumericalParityTests)
dotnet run -- -c Release 2>&1 | grep "CompareBackwardSpikes" -B3

# Show only C vs .NET comparison failures (NativeVsManagedTests)
dotnet run -- -c Release 2>&1 | grep "CompareNativeVsManaged" -B3

# Check for Toms748 root-finder exceptions
dotnet run -- -c Release 2>&1 | grep "f(a) and f(b)"

# Check for spike count mismatches
dotnet run -- -c Release 2>&1 | grep "SPIKE COUNT"

# Get unique failure categories (test name only)
dotnet run -- -c Release 2>&1 | grep "failed BackwardSpikes" | sort -u
```

### Test Project Structure & What Each File Tests

- **`NativeVsManagedTests.cs`** — C# (EventPropMath) vs NativeApi (C library). Tests: ComputeSums, ComputeSpikes, ComputeMaxima, Backward. Uses uniform patterns only.
- **`NumericalParityTests.cs`** — Three-way: NativeApi (C) ≡ NativeCppApi (Eigen, source of truth) ≡ EventPropMath (.NET). All patterns/sizes. **This is the main parity test suite.**
- **`NativeLibraryParityTests.cs`** — C vs Eigen ONLY (no .NET). Tests: BackwardSpikes, BackwardMaxima. All patterns/sizes.
- **`ComputeSpikesTests.cs`** — Unit tests for ComputeSpikes forward pass.
- **`ComputeSumsTests.cs`** — Unit tests for ComputeSums.
- **`BackwardSpikesTests.cs`** — Numerical gradient verification for backward pass.

### Coverage Report Generation

```bash
# Run with coverage (output → bin/Debug/net10.0/TestResults/coverage.cobertura.xml)
cd src/EventProp.Tests && dotnet run -- -c Release --coverage --coverage-output coverage.cobertura.xml --coverage-output-format cobertura

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
