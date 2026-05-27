# EventProp C++ Benchmarks

C++ benchmarks for the EventProp algorithm (event-based backpropagation for SNNs), using [Google Benchmark](https://github.com/google/benchmark).

## Build

```bash
cd src/cpp/benchmarks && mkdir -p build && cd build
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)  # macOS/Linux
```

Two executables are built:
- `eventprop_benchmarks` — full algorithm benchmarks (ComputeSums, ComputeSpikes, BackwardSpikes, ComputeMaxima)
- `micro_benchmarks` — hot-path micro-benchmarks (exp precompute, Brent root finder, gradient accumulation, spike sorting)

## Run

```bash
# All benchmarks (default: ~1s per benchmark)
./eventprop_benchmarks

# Micro-benchmarks only
./micro_benchmarks

# Specific benchmark pattern
./eventprop_benchmarks --benchmark_filter='ComputeSpikes/Large'

# Longer runs for stable numbers
./eventprop_benchmarks --benchmark_min_time=2s

# List all available benchmarks
./eventprop_benchmarks --benchmark_list_tests
```

## Benchmark Coverage

### Algorithms × Sizes

| Algorithm | Tiny (10,5,20) | Small (50,20,100) | Medium (200,80,500) | Large (500,200,1000) | XL (2000,500,5000) |
|---|:-:|:-:|:-:|:-:|:-:|
| ComputeSums | ✓ | ✓ | ✓ | ✓ | ✓ |
| ComputeSpikes | ✓ | ✓ | ✓ | ✓ | ✓ |
| BackwardSpikes | ✓ | ✓ | ✓ | ✓ | ✓ |
| ComputeMaxima | ✓ | ✓ | ✓ | ✓ | ✓ |

### Spike Patterns (Small size)
- **uniform** — evenly spaced times `(i+1)/(n_spikes+1)` with random sources
- **poisson** — exponential inter-spike intervals (~200 Hz per neuron)
- **burst** — clusters of 5 spikes with 10ms gaps

### Training Loop
Full forward + backward iteration for each problem size. Measures the combined cost of one training step.

### Micro-benchmarks
| Benchmark | What it measures |
|---|---|
| ExpPrecompute/n=100,500 | `std::exp(t/τ)` calls for all spike times |
| BrentRootFinder/nrns=20,200 | Brent root-finding iterations (voltage threshold crossing) |
| GradientAccumulation/pre=50/20,200/80 | Backward pass inner loop (lambda computation + gradient update) |
| SpikeMergeSort/input=100+30,500+150 | Merge input + post spikes and sort descending by time |

## Parameters

All benchmarks use the same defaults as the .NET suite:
- **τ_mem** = 20ms (membrane time constant)
- **τ_syn** = 5ms (synaptic time constant)
- **V_th** = 1.0 (spike threshold voltage)
- **RNG seed** = 42 (reproducible weights and spike sources)
- **Post-synaptic errors** = 1.0 (uniform error signal for backward pass)

## Comparison with .NET Benchmarks

The C++ benchmarks call the same `eventprop_c_api` library that the .NET benchmarks use via P/Invoke. This means:

- **Same native code** — both .NET "Native" and C++ benchmarks exercise the identical Eigen-based implementation
- **Same data** — same RNG seed, same problem sizes, same parameters
- **Different measurement overhead** — Google Benchmark vs. BenchmarkDotNet

### Known Issue: Eigen Allocation Overhead

The C++ reference implementation (`eventprop_c_api.cpp`) uses `Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>` with row-copy loops (`make_matrix` / `copy_matrix_to_flat`). This introduces significant allocation overhead compared to the .NET managed code which uses zero-copy `Span<double>` views.

**Result**: The C++ native implementation is often **slower than the .NET managed code** for ComputeSums and ComputeSpikes. This is a **library implementation issue**, not a language limitation. A production C++ implementation with flat-array operations (matching the .NET `EventPropMath` pattern) would be significantly faster.

## CI Integration

Export results as CSV/JSON for regression tracking:

```bash
# CSV output (for spreadsheets)
./eventprop_benchmarks --benchmark_format=csv --benchmark_out=results.csv

# JSON output (for analysis)
./eventprop_benchmarks --benchmark_format=json --benchmark_out=results.json

# Both formats
./eventprop_benchmarks --benchmark_format=json --benchmark_out=results.json \
                       --benchmark_out_format=csv --benchmark_out_csv_file=results.csv
```

## Multi-threading

The C++ reference uses `#pragma omp parallel for` when OpenMP is available. On macOS, OpenMP is not bundled — you may need to install it:

```bash
brew install libomp
# Rebuild with OpenMP support
cd build && rm -rf * && cmake .. -DOpenMP_ROOT=$(brew --prefix libomp) && cmake --build .
```

## Directory Structure

```
src/cpp/benchmarks/
├── CMakeLists.txt              # Build config (FetchContent for Google Benchmark)
├── eventprop_benchmarks.cpp    # Full algorithm benchmarks (4 algorithms × 5 sizes)
├── micro_benchmarks.cpp        # Hot-path micro-benchmarks (4 sub-components)
├── README.md                   # This file
└── build/                      # Build output (gitignored)
```
