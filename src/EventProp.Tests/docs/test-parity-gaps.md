# Test Parity Gaps — Edge Cases Not Yet Covered

## Summary

Current parity test coverage is strong for typical use cases but has gaps in edge cases that could mask numerical differences between C++ and C# implementations. Below are specific tests that should be added.

---

## 1. ComputeSpikes Edge Cases

### Gap: Single Pre-Neuron, Multiple Post-Neurons
**Current**: `ComputeSpikes_CppVsCs_MultipleNeurons` uses nPre=2, nPost=3
**Missing**: nPre=1, nPost=N (single input neuron driving multiple outputs)

```csharp
int nPre = 1, nPost = 5;
double[] w = [20.0, 0.1, 0.1, 0.1, 0.1]; // One strong, four weak
double[] times = [0.05];
int[] sources = [0];
```

**Why it matters**: Tests that a single input spike correctly propagates to multiple post-synaptic neurons with different thresholds. Could expose bugs in per-neuron spike detection loops.

### Gap: Single Neuron, Multiple Spikes
**Current**: No test with nPre=1, nPost=1 producing multiple spikes from one input spike
**Missing**: Strong weight causing rapid successive post-spikes

```csharp
int nPre = 1, nPost = 1;
double[] w = [50.0]; // Very strong — causes multiple spikes
double[] times = [0.1];
int[] sources = [0];
```

**Why it matters**: Tests the accumulation case where a single input spike causes multiple post-spike events. Could expose bugs in membrane potential reset logic or spike detection after reset.

### Gap: Very Close Tau Values
**Current**: No test with tauMem ≈ tauSyn
**Missing**: tauMem = 0.02, tauSyn = 0.019 (ratio very close to 1)

```csharp
double tauMem = 0.02, tauSyn = 0.019; // Difference of only 0.001
// Prefactors: k_prefactor = tau_syn / (tau_mem - tau_syn) = 0.019 / 0.001 = 19
// tmax_prefactor = 1.0 / (50 - 52.63) ≈ -380
```

**Why it matters**: The prefactors `tau_syn / (tau_mem - tau_syn)` and `1.0 / (1/tau_mem - 1/tau_syn)` blow up as tauMem → tauSyn. Tests numerical stability of these divisions in both implementations.

---

## 2. ComputeMaxima Edge Cases

### Gap: Maxima Before First Input Spike
**Current**: `BackwardMaxima_CppVsCs_MaximaAfterLastSpike` tests maxima AFTER last spike
**Missing**: Maxima time before the first input spike

```csharp
double[] maximaTimes = [0.005]; // Before first input spike at 0.01
```

**Why it matters**: The backward pass checks `if (t_vmax_bwd > t_bwd) continue;`. A maxima before the first spike should be skipped or handled correctly. Tests boundary condition in the time comparison logic.

### Gap: Maxima Exactly at Input Spike Time
**Current**: No test with maxima time exactly equal to an input spike time
**Missing**: maximaTimes[i] == inputTimes[j]

```csharp
double[] times = [0.01, 0.02];
double[] maximaTimes = [0.015, 0.02]; // Second maxima exactly at second spike
```

**Why it matters**: Tests floating-point equality comparison in the `t_before <= tmax && tmax <= t_input` check. Could expose off-by-one errors in boundary conditions.

---

## 3. BackwardSpikes Edge Cases

### Gap: Single Input Spike, Single Post-Spike
**Current**: `BackwardSpikes_CppVsCs_SingleInput_MultiplePosts` has single input but multiple posts
**Missing**: Simplest non-trivial case: 1 input, 1 post

```csharp
int nPre = 1, nPost = 1;
double[] w = [20.0];
double[] inputTimes = [0.1];
int[] inputSources = [0];
```

**Why it matters**: The simplest case that exercises the full backward pass. Should be a baseline test for all backward functions.

### Gap: No Input Spikes (nInputSpikes = 0)
**Current**: BackwardSpikes returns early if nPostSpikes == 0, but no test with zero input spikes
**Missing**: Post-spikes exist but no input spikes (edge case for the merged timeline)

```csharp
double[] inputTimes = []; // No input spikes
int[] inputSources = [];
// Post-spikes from a previous forward pass
```

**Why it matters**: Tests the merged timeline logic when there are only post-spikes and no input spikes. The gradient array should remain zero (no weights to update).

### Gap: All Input Spikes from Same Neuron
**Current**: `BackwardSpikes_MultiSpikeSameSource_GradientMatch` tests this but with nPre=1
**Missing**: nPre > 1 but all input spikes from one neuron

```csharp
int nPre = 5, nPost = 3;
double[] w = new double[15]; // Fill with some values
// All spikes from neuron 2
double[] inputTimes = [0.01, 0.02, 0.03, 0.04];
int[] inputSources = [2, 2, 2, 2];
```

**Why it matters**: Tests that only the weight row for neuron 2 contributes to gradients. All other rows should be zero.

### Gap: Post-Spike Sources Outside Input Spike Range
**Current**: All tests have post-spike sources within the range of input spike sources
**Missing**: Post-spike from neuron that had no input spikes

```csharp
int nPre = 3, nPost = 2;
// Input spikes from neurons 0 and 1 only
double[] inputTimes = [0.01, 0.02];
int[] inputSources = [0, 1];
// But post-spike from neuron 1 (which is also in input sources)
```

**Why it matters**: Tests that the backward pass correctly handles post-spike neurons that also appear as input spike sources. The adjoint arrays are sized for max(nPre, nPost).

---

## 4. BackwardMaxima Edge Cases

### Gap: Maxima Times Before All Input Spikes
**Current**: Tests maxima after last spike, but not before first
**Missing**: All maxima times precede all input spike times

```csharp
double[] maximaTimes = [0.005, 0.008]; // Before first input at 0.01
```

**Why it matters**: The backward pass iterates from last input spike backward. If all maxima are before the first spike, t_vmax_bwd > t_bwd for all iterations, and all gradients should be zero.

### Gap: Mixed Valid/NaN Maxima with Different Input Spikes
**Current**: `BackwardMaxima_CppVsCs_LargeNetwork` has mixed valid/NaN but with many spikes
**Missing**: Small network with specific valid/NaN pattern

```csharp
int nPre = 2, nPost = 3;
double[] maximaTimes = [0.015, double.NaN, 0.025]; // First and third valid
double[] maximaErrors = [0.1, 0.0, -0.2];
```

**Why it matters**: Tests that NaN maxima are correctly skipped and don't affect gradients for other neurons.

---

## 5. ComputeSums Edge Cases

### Gap: Single Input Spike
**Current**: `ComputeSums_CppVsCs_AreIdentical` uses 3 input spikes
**Missing**: Single input spike case

```csharp
double[] times = [0.05];
int[] sources = [1]; // Single spike from neuron 1
```

**Why it matters**: Tests the base case of sum computation with minimal data. Could expose off-by-one errors in the cumulative sum loop.

---

## Priority Ranking

| Priority | Test | Risk if Missing |
|----------|------|-----------------|
| **P0** | tauMem ≈ tauSyn | Numerical overflow/underflow could differ between C++ (double) and C# (double) |
| **P0** | Single input, single post | Baseline case — if this fails, everything is wrong |
| **P1** | Maxima before first input spike | Boundary condition in backward pass time comparison |
| **P1** | All weights zero for some neurons | Sparsity in gradient computation |
| **P2** | Maxima exactly at input spike time | Floating-point equality edge case |
| **P2** | Single neuron, multiple spikes | Spike reset logic verification |
| **P3** | No input spikes (nInputSpikes=0) | Edge case for merged timeline |
| **P3** | Single input spike (ComputeSums) | Minimal case verification |

## Implementation Notes

All new tests should follow the existing pattern:
1. Run both C# (`EventPropMath.*`) and C++ (`CxxNative.*`) implementations
2. Compare all output values within `1e-10` tolerance
3. Use `SyncAssert` for synchronous assertions in non-async tests
4. Include early return if no post-spike activity (to avoid division by zero or empty arrays)
