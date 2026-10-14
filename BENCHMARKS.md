# Library Benchmark Comparison

Generated 2026-10-14 on macOS (osx-arm64). Three suites, three **different** pairings —
they are not one geomean. Each suite's ratio column names the winner.

| Suite | Pairing | Tool | Ratio column |
|---|---|---|---|
| `EventProp.Native.Benchmarks` | C library vs Eigen C++ | Google Benchmark | `Eigen/C` (>1 = C wins) |
| `EventProp.Benchmarks` | managed .NET vs native .NET P/Invoke | BenchmarkDotNet | `Native/Managed` (<1 = native wins) |
| `EventProp.Python.Benchmarks` | pybind11 vs Cython | pytest-benchmark | `Cython/pybind11` (>1 = Cython slower) |

The suites share the **data contract** (canonical burst, inverse-CDF poisson, sorted
times, identical sizes/taus) so per-cell shapes are comparable, but the pairings differ:
suites 1 and 2 share the kernel families; suite 3 benchmarks layer ops, not kernels.

---

## Suite 1 — C vs Eigen (Eigen/C, >1 = C wins)

| Family | Geomean | Notes |
|---|---:|---|
| ComputeSums | **1.64x** | C wins; 1.9–2.0x at Medium/Large/XL (SIMD prefix sums) |
| ComputeSpikes | 0.95x | parity; root-finding dominates. C wins XL burst 1.07x, loses Medium 0.69x |
| BackwardSpikes | 1.52x* | *inflated by zero-spike noise (Tiny/Small uniform 16–20x = empty-loop regime); real ~1.02–1.08x |
| BackwardMaxima | 1.05x | parity |
| ComputeMaxima | 1.08x | parity |
| FullTrainingStep | 1.00x | parity |
| Dropout | 8.17x | C batch API vs C++ shim — API-shape artifact, not like-for-like |

**Headline**: C wins where it vectorizes (ComputeSums 1.6–2.0x); parity on root-finding-dominated
forward/backward; Dropout is not a fair comparison.

## Suite 2 — managed .NET vs native P/Invoke (Native/Managed, <1 = native wins)

| Family | Geomean | Notes |
|---|---:|---|
| Batch_Forward | **0.21x** | native batch API wins (4.8x) |
| LI_Forward | **0.30x** | native wins (3.3x) |
| Batch_TrainingStep | 0.35x | native wins |
| LIF_Forward | 0.42x | native wins (3.8x at Small) |
| LI_Backward | 0.52x | native wins |
| LIF_Backward | 0.69x | native wins Medium/Small; parity Large (1.03x) |
| BackwardSpikes | 0.74x | native wins Medium/Large/XL |
| TrainingLoop | 0.74x | native wins Medium/Large/XL |
| ComputeSpikes | 1.02x | parity (native XL 0.84x, managed Small 1.59x) |
| ComputeSums | 1.09x | managed ahead (P/Invoke marshalling overhead) |
| ComputeMaxima | 1.00x | parity |

**Headline**: native P/Invoke wins the layer/batch paths (0.2–0.7x — the new native layers
pay off); managed is competitive on single-kernel paths (parity, managed ahead at Tiny
where P/Invoke marshalling dominates).

## Suite 3 — pybind11 vs Cython (Cython/pybind11, >1 = Cython slower)

After removing per-call Python-side work from the Cython wrappers (see below):

| Cell | pybind11 (us) | Cython (us) | Cython/py |
|---|---:|---:|---:|
| batch[4] | 75.98 | 57.83 | 0.76 |
| batch[8] | 93.25 | 95.42 | 1.02 |
| batch[16] | 228.70 | 157.12 | 0.69 |
| dropout[4] | 3.66 | 1.91 | 0.52 |
| dropout[8] | 6.12 | 2.20 | 0.36 |
| dropout[16] | 11.69 | 2.85 | 0.24 |
| ttfs_step[4] | 136.45 | 176.36 | 1.29 |
| ttfs_step[8] | 194.38 | 255.86 | 1.32 |
| vmax_step[4] | 119.56 | 134.18 | 1.12 |
| vmax_step[8] | 177.70 | 177.46 | 1.00 |
| **GEOMEAN** | | | **0.74x** |

**Headline**: Cython is **0.74x** of pybind11 — faster overall. Wins came from removing
per-call Python-side work:
- dropout: reuse `SpikesVector._extract_flat_buffers` instead of re-`np.concatenate` +
  `.astype` every call (7.05us -> 1.93us, 3.6x).
- vmax loss forward/backward: vectorize (stack values, one exp/sum; one vector error write
  per sample instead of a per-neuron `set_error` loop).
- ttfs loss backward: one vector error write per sample instead of a per-neuron loop.

**Correctness fix found along the way**: the Cython maxima loss wrote errors to a detached
Python buffer that the C backward never read, so the vmax gradient was always **zero**. The
`_CythonMaximaView.errors` property now returns the C-owned buffer (via `get_maxima_errors`),
and the loss uses a numerically stable softmax. The vmax gradient is now correct (matches
pybind11 to 5e-15 on the MNIST fixture).

The remaining gap is the loss-layer training steps (`ttfs_step` 1.29-1.32x, `vmax_step[4]`
1.12x), now dominated by the C kernel + result wrapping (the same work pybind11 does), not
Python-side assembly.

**Build note**: the Cython extension must be generated with Cython 3.0.x. Cython 3.2.5
emits NumPy-2.x C-API (`PyDataType_TYPEOBJ`/`_KIND`/`_TYPE`) that does not compile against
NumPy 1.26.4. Regenerate with `python -m cython -3 eventprop_c/_native.pyx -o
eventprop_c/_native.c` then `python setup.py build_ext --inplace`.

---

## Cross-suite ratio table (suites 1 & 2, uniform pattern)

Both suites run the kernel families on identical uniform data. `C/E` = Eigen/C (C wins
if >1); `.NET` = Native/Managed (native wins if <1). Poisson cells exist only in suite 1
(the .NET single-kernel benchmarks use uniform only).

| cell | C/E | .NET |
|---|---:|---:|
| ComputeSums/Tiny/uniform | 1.23 | 1.26 |
| ComputeSums/Small/uniform | 1.32 | 1.19 |
| ComputeSums/Medium/uniform | 1.84 | 1.03 |
| ComputeSums/Large/uniform | 1.97 | 1.01 |
| ComputeSpikes/Small/uniform | 0.96 | 1.59 |
| ComputeSpikes/Medium/uniform | 0.76 | 0.90 |
| ComputeSpikes/Large/uniform | 0.96 | 0.89 |
| BackwardSpikes/Small/uniform | 19.63* | 3.11* |
| BackwardSpikes/Medium/uniform | 1.01 | 0.78 |
| BackwardSpikes/Large/uniform | 1.07 | 0.73 |

\* zero-spike / P/Invoke-overhead regime — excluded from geomeans.

**Reading**: the C library's ComputeSums advantage (1.2–2.0x over Eigen) does **not**
transfer to the .NET side (managed ≈ native, 1.0–1.3x) — the managed `ComputeSums` is
already SIMD-optimized, so the C win over Eigen is a C-vs-Eigen effect, not a
native-vs-managed one. On the root-finding paths both suites land near parity.

## Caveats (per AGENTS.md)
- Tiny/zero-spike rows measure P/Invoke overhead on an empty/short loop — excluded from geomeans.
- XL rows cost seconds/iteration; excluded from quick runs.
- Burst is the heaviest workload (XL ~770k output spikes); XL burst excluded from quick runs.
- "Poisson @200 Hz" is a misnomer (effective 1.8–25.6 kHz), shared by all three suites.
