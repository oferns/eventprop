# Benchmark Comparison Report (Fixed Edition)

## Fixes Applied

1. **Removed unnecessary `memset`/`Array.Clear` on ComputeSpikes outputs** — native code overwrites these buffers, so clearing is wasted work
2. **Removed redundant `buffers.Clear()` from BackwardSpikes** — all scratch buffers are immediately re-initialized; clearing 3.5 MB was wasted (~340 μs saved)
3. **All benchmarks return `void`** — prevents JIT/compiler from optimizing away calls with unused return values
4. **Pre-filled error arrays in setup** — TrainingLoop no longer rewrites errors per iteration (always 1.0)

## Critical Discovery: NativeApi ≠ Pure C

The `.NET NativeApi` P/Invoke bindings load `libeventprop_c_api`, which is a **C++ Eigen wrapper**, NOT the pure scalar C library (`eventprop_scalar.c`). This means:

- "Native" column = P/Invoke overhead + **Eigen** implementation
- "Pure C" column = direct call to **scalar C** implementation (no P/Invoke)
- These are fundamentally different implementations!

To get a true "P/Invoke vs direct" comparison for scalar C, we'd need separate P/Invoke bindings for `eventprop_scalar.c`.

## Problem Size: Medium Uniform (NPre=200, NPost=80, NSpikes=500)

All timings in microseconds (μs). Lower is better.

| Operation | C++ Eigen | Pure C Scalar | .NET Managed | .NET NativeApi (Eigen+P/Invoke) |
|-----------|----------:|--------------:|-------------:|--------------------------------:|
| ComputeSums | 35 μs | **18 μs** | 23 μs | 4,709 μs |
| ComputeSpikes | **54 ms** | 55 ms | 63 ms | 202 ms |
| BackwardSpikes | **356 μs** | 471 μs | **457 μs** | 15,456 μs |
| ComputeMaxima | 773 μs | 791 μs | **783 μs** | 10,116 μs |
| TrainingLoop | **55 ms** | 55 ms | 65 ms | 217 ms |

## Speedup Ratios (relative to fastest per operation)

| Operation | C++ Eigen | Pure C Scalar | .NET Managed | NativeApi (Eigen+P/Invoke) |
|-----------|----------:|--------------:|-------------:|---------------------------:|
| ComputeSums | 1.9x | **1.0x** | 1.3x | 262x |
| ComputeSpikes | **1.0x** | 1.0x | 1.2x | 3.7x |
| BackwardSpikes | **1.0x** | 1.3x | 1.3x | 43.4x |
| ComputeMaxima | **1.0x** | 1.0x | 1.0x | 13.0x |
| TrainingLoop | **1.0x** | 1.0x | 1.2x | 3.9x |

## Key Findings

### .NET Managed vs Native C/C++ (ALGORITHM comparison)

After fixing the wasted `buffers.Clear()`, managed .NET is highly competitive:

- **ComputeSums**: 27% slower than Pure C scalar — excellent for pure scalar math with bounds checking
- **BackwardSpikes**: **457 μs vs 471 μs** — now FASTER than Pure C! (SIMD decay + no malloc overhead)
- **ComputeMaxima**: Essentially identical to Pure C (within noise)
- **ComputeSpikes**: ~16% slower than C++ Eigen (root-finding dominates, bounds checking adds small overhead)
- **TrainingLoop**: ~17% slower than C++ Eigen (dominated by ComputeSpikes ratio)

### Why BackwardSpikes is now faster in .NET

The Pure C implementation allocates 7 buffers per call via `malloc`/`calloc`:
```c
MergedSpike* merged = malloc(total * sizeof(MergedSpike));
double* all_errors = calloc(total, sizeof(double));
double* lambda_v = calloc(n_post, sizeof(double));
// ... 4 more allocations + 7 frees at end
```

The .NET implementation reuses pre-allocated `ScratchBuffers` and only allocates the merged array (~14 KB). After removing the redundant `buffers.Clear()`, the .NET path avoids both malloc overhead AND wasted zeroing.

### P/Invoke Overhead (NativeApi = Eigen + marshaling)

The NativeApi slowdown has TWO components:
1. **Eigen vs scalar C** — Eigen adds matrix copy overhead for small operations
2. **P/Invoke array pinning** — each `double[]` argument must be pinned for the native call

| Operation | Pure C Scalar | NativeApi (Eigen+P/Invoke) | Total slowdown |
|-----------|--------------:|--------------------------:|---------------:|
| ComputeSums | 18 μs | 4,709 μs | **262x** |
| BackwardSpikes | 471 μs | 15,456 μs | **33x** |
| ComputeSpikes | 55 ms | 202 ms | **3.7x** |

For compute-heavy operations (ComputeSpikes), P/Invoke overhead amortizes to ~3.7x. For lightweight ops (ComputeSums at 18 μs), it dominates completely.

### Recommendation

- **For production use**: Managed .NET is competitive — within 17% of C++ Eigen for full training loop
- **BackwardSpikes optimization**: Removing `buffers.Clear()` gave a 2x speedup (789 → 457 μs). Check other functions for similar waste.
- **NativeApi path**: Only justified if you need the specific Eigen implementation via P/Invoke. For scalar C, create separate bindings to `eventprop_scalar.c`.
- **Future optimization**: Pre-allocate merged spike arrays in ScratchBuffers to eliminate per-call GC pressure (~173 KB/call).

## Methodology Notes

- All benchmarks on Apple M3 (ARM64), .NET 10.0 Release mode
- C/C++ compiled with `-O3` via CMake Release configuration  
- Google Benchmark for C/C++, BenchmarkDotNet Short job for .NET (3 warmup + 3 iterations)
- Shared LCG PRNG seed ensures identical input data across all implementations
- Buffer clears removed from ComputeSpikes paths (native overwrites output)
- `buffers.Clear()` removed from BackwardSpikes (all buffers re-initialized explicitly)
