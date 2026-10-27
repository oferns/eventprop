# Post-mortem archive — EventProp

Every entry below is **fixed**; there are no open defects in this list. They are kept because each records a constraint that is not visible in the current code and is easy to re-introduce: an invariant a refactor must preserve, a parity contract shared by both backends, a diagnostic signature for recognising the failure class, and the regression test that pins it. Treat this as a post-mortem log and a checklist for touching the areas named, not as a bug tracker entry.

This is **reference material for diagnosing a recurrence**, not session-start instruction.
The rules that came out of these 20 investigations live in [`AGENTS.md`](../AGENTS.md); the
symptom signatures live here, because they are only findable by searching for the error
message you already have on screen. So: search this file for the string in front of you
before you read any source.

The value these entries carry that no code comment can hold is the **diagnostic signature** —
which misleading message the failure produces, and how far from the cause it appears. Most
of them point nowhere near the defect.

## Index — symptom → entry

| If you see this | Start at |
|---|---|
| C emits strictly **more** spikes than the Eigen reference (always a superset) | C forward pass: spurious spikes in the final interval |
| Gradient ~1000× too small; denormal spike errors (~1e-158); many exact zeros | Cython layers: use-after-free in spike-error propagation |
| .NET spike count **differs between runs** of identical input | Managed .NET forward pass: per-neuron spike buffers overflow |
| 40 parity failures with `EVENTPROP_INVALID_VTH` (11) on sums/maxima entry points | C input validation rejected `v_th <= 0` |
| `corrupt patch at …:24`; Linux submodule with only `loss_layer.py` patched | `submodule-linux.patch` was unparseable |
| `nan location mismatch` / `DESIRED: array(nan)` in a numerical-gradient test | the reference library's own TTFS gradient test compared NaN against 0.0 |
| Managed-vs-native parity fails **intermittently**; `Invalid source neuron index`; `MANAGED=0` | .NET: pointers taken from managed memory outlived their `fixed` pin |
| `ValueError: unparsable Mean: '169.8 ╬╝s'` from a suite that ran to completion | BenchmarkDotNet adapter read its CSV with the locale encoding |
| `STATUS_HEAP_CORRUPTION` (`0xC0000374`) in `RtlFreeHeap`, first test, zero program output | the Eigen core's arch flag was `PRIVATE` |
| A build script prints "Build complete" (or a report path) and nothing actually happened | the Windows scripts reported success they had not measured |
| A clean script exits 0 having removed nothing; an unknown target is accepted | the two clean scripts had drifted apart |
| `MISSING: …EventProp.Benchmarks.dll` immediately after "Build complete"; dll is really at `bin\x64\Release\` | the MSVC environment import leaked `Platform=X64` |
| `undefined reference to tbb::detail::r1::*`; a shared library that builds green and cannot `dlopen` | three C++ projects would not link on Linux |
| `cc1: fatal error: eventprop_c/_native.c: No such file or directory` | the Cython step reported two failures for one missing package |
| `ImportError: libeventprop_c_api.so: cannot open shared object file` after a green build | `import eventprop_c` failed on Linux: no rpath |
| `undefined symbol: omp_get_thread_num` from a step that just said "already built" | "already built" over an unloadable reference binary |
| Reference build dies inside pybind11 against NumPy 2 | Windows reference build: pybind11 had no required version |
| A clean script reports a count that does not match what vanished | the clean scripts counted paths they had not removed |
| Heap smash in the C-API shim's own test binary | the wrapper's C API had no write bound |
| `bench_rootfind` SIGSEGV (exit 139); ASan `heap-use-after-free` in `compute_v` | `bench_rootfind`: the bracket list outlived the spike rows |

## Two habits these entries keep recommending

1. **Ask the tool, don't test for the artifact.** Six of the twenty are "an existence check
   reported success about an unusable thing": a file that exists but does not load, a binary
   that exists but has undefined symbols, a patch that exists but does not parse, a count of
   paths that were never removed. Prefer `import`, `dlopen`, `git apply --check`, exit codes.
2. **When a parity test fails, ask which side is the outlier before changing either.** Twice
   the C library was the stricter *diagnostic* of a contract both implementations share; once
   it was the divergent *implementation* and the reference was right. The framing "the C
   library is stricter" is not evidence of innocence.

### Fixed — C forward pass: spurious spikes in the final interval

`eventprop_compute_spikes` (`src/EventProp.C/eventprop_spikes.c`) special-cased the bracket floor for the last interval as `t_before = count > 0 ? p_times[count - 1] : 0.0`, dropping the `times[i - 1]` term that every other interval applies. When the neuron had not fired yet, the floor collapsed to **0.0**, the `t_before <= tmax` guard in `generate_bracket_candidates` passed for an analytical peak sitting *behind* the interval, and the solver bracketed a region where `sum0[i]`/`sum1[i]` are an extrapolation of a spike train that has not arrived yet — reporting roots at times where the real membrane potential is far below `v_th`.

The rule that was missing: `sum0[i]`/`sum1[i]` include every input spike up to `times[i - 1]`, so the voltage model of interval `i` is only valid for `t >= times[i - 1]`.

Measured on the trained YinYang checkpoint (`paper/yinyang_0.pkl`, hidden layer 5x200, 64 test samples), the C library emitted **strictly more** spikes than the Eigen reference — a superset, never a subset — on 35 of 64 samples (1888 vs 1833 total). Neuron 177, sample 0: C reported a spike at `t = 0.013792378751` while the true voltage there is **0.701925314967** against `v_th = 1.0`; the interval's analytical peak is `tmax = 0.017081371`, behind `times[4] = 0.022977101`, and the true peak of the whole window is 0.960201051.

**Fix**: the last-interval branch was deleted, not patched. One floor serves every interval — `max(times[i - 1], post spikes <= t_input)` — because `t_input` is `INFINITY` in the last interval, so the existing scan already takes the maximum over all post spikes, which is exactly what the Eigen reference does. After the fix the two backends agree exactly: 1833 vs 1833 spikes, zero differing samples, and the full `tests/test_paper_parity.py` suite goes from 6 failures to 0.

This is the reverse of the framing in *Spike Time Ordering* above: there the C library is the stricter *diagnostic* of a contract both implementations share. Here the C library was the **divergent implementation** — the Eigen reference and the managed .NET side (`EventPropMath.cs`, which already floors every interval with `timesArr[i - 1]`) were both correct. When a parity gap is reported, check which side is the outlier before deciding which to change.

Regression tests: `ComputeSpikes.LastIntervalPeakBehindLastInputTimeNoSpuriousSpike` (the neuron-177 fixture, asserts 0 spikes; fails pre-fix with the root at `t = 0.013792389158539134`) and `ComputeSpikes.LastIntervalGenuineSpikeStillFound` (a last interval that genuinely crosses `v_th` at 0.023057647, so the new floor cannot suppress real spikes) in `src/EventProp.C.Tests/test/compute_spikes.cpp`.

### Fixed — Cython layers: use-after-free in spike-error propagation

`LIFLayer.backward` and `LILayer.backward` (`eventprop_c/lif_layer.py`, `eventprop_c/li_layer.py`) propagated the loss signal to the previous layer with zero-copy views into the native `AccumulatedBackwardResult`:

```python
result = backward_spikes_batch_accumulate(...)
for i in range(result.batch_size):
    inp._errors = result.get_spike_errors(i)   # view into result's native buffer
# result is a local — it dies here, __dealloc__ frees the buffer, views dangle
```

The consumer is the *ancestor* layer's `backward()`, invoked by `super().backward()` at the end of the same method, and the views stay live on the input spikes after that. Dropping the last reference to `result` frees the buffers they point at, so the ancestor's backward reads freed heap memory.

Symptom on the YinYang fixture: the hidden layer's incoming spike errors were denormal garbage (~1e-158, 2 nonzero per sample) instead of 299.15 / 1.88 / 5.34 / 4.33, and the hidden gradient was ~1000x too small with 179 of 222 entries exactly zero (`max|g| = 9.4e-4` against the reference's `1.265454`). The *loss* layer's own gradient was correct to 8.5e-14 throughout, which is what localises this to propagation rather than to the adjoint sweep — the C `spike_errors` output is healthy (max 106.0) when read directly.

**Fix**: pin the result on the layer (`self._backward_result = result`) so the native buffer outlives the views handed to the ancestor. `Layer.__init__` initialises the attribute. After the fix the hidden gradient matches the reference exactly (`max|g| = 1.265453817356891` vs `py[1,119] = -1.265454e+00`).

Regression tests: `SpikeErrorPropagationLifetimeTest` in `src/EventProp.Native.Python/test/test_layer_chain.py`. The freed `spike_errors` blocks are ~11 KB, so the test recycles that size class with same-size numpy allocations immediately after `backward()` — that makes the dangling read deterministic (max|error| = 5.1e+77 pre-fix, ~8-11 post-fix) instead of leaving it to heap layout, which is not reliably detectable. A plain `gc.collect()` + `bytearray` churn passed against the unfixed code.

### Fixed — Managed .NET forward pass: per-neuron spike buffers overflow

`EventPropMath.ComputeSpikesParallel` (`src/EventProp/EventPropMath.cs`) used to size each post-synaptic neuron's scratch run at `maxSpikesPerNeuron = max(nSpikes + 1, buffers.MaxSpikesPerNeuron)` and index it as `neuronOffset = targetNrn * maxSpikesPerNeuron`. A neuron can emit **more** spikes than that bound, and the only guard was against the end of the whole buffer (`neuronOffset + neuronSpikeCount < buffers.PostSpikeTimes.Length`), so past the bound writes spilled into the neighbouring neuron's run. Because the loop is a `Parallel.For` over neurons, adjacent runs were written concurrently and the surviving spike set depended on thread interleaving.

Symptoms (input `w = 15.0 + i`, `tau_mem=0.02`, `tau_syn=0.005`, `v_th=1.0`, input times `[0.01, 0.02, 0.03]`, sources `[0, 1, 0]`, `n_post=3`, capacity hint 4):

- Spike count nondeterministic across runs: 21, 26, 26, 26, 27, 27 — the C library returns 31 every time.
- Per-neuron counts exceeded the bound (10 / 8 / 8), i.e. every run past slot 4 was outside its region.
- Post spikes sharing an identical time also shared one current (`t=0.036340173829084`, `src=0,1,2` all reporting `4.9835283556` instead of `4.9835283556 / 5.3081473409 / 5.6327663262`) — the wrong entries were spillover, not the formula.

**Fix** (mirrors the C refactor `54fa50a refactor(C): replace per-neuron spike slab with per-thread scratch rows`): `ScratchBuffers` now owns `PostSpikeRow[]`, one growable row per neuron, and `ComputeSpikesParallel` appends with `row.Grow(neuronSpikeCount, neuronSpikeCount + 1)` instead of bounding writes. `MaxSpikesPerNeuron` is an initial capacity hint, no longer a bound. Two invariants matter for any future change here:

- `Grow` must **copy the live spikes** into the new arrays — `ArrayPool.Rent` hands back dirty memory, so losing them silently corrupts the neuron's voltage model (observed as spike counts *dropping*, e.g. 4156 vs C's 6381 on `ComputeSpikes_Uniform_Medium`).
- `Grow` must **not return the superseded arrays to the pool** while the sweep still caches them in locals; handing them over can give the same memory to another thread. Growth is logarithmic, so the abandoned arrays are a handful per row and the GC reclaims them.

Regression tests: `ComputeSpikes_PerNeuronSpikeBurst_MatchesNativeSpikeSet` (`src/EventProp.Tests/NativeVsManagedEdgeTests.cs`, pins the 31-spike set against C) and `PostSpikeRow_GrowPreservesLiveSpikes` (`src/EventProp.Tests/ScratchBuffersTests.cs`).

### Fixed — C input validation rejected `v_th <= 0` on entry points that never read it

`eventprop_validate_input` (`src/EventProp.C/eventprop_validation.c`) required `v_th > 0` for every entry point, but `ComputeSums`, `ComputeMaxima`, and `backward_maxima` never read the threshold voltage — the cumulative sums and the maxima loss are defined for any `v_th`. The Eigen reference and the managed .NET implementations accept `v_th = 0`, and `src/EventProp.Native/Api.cs` builds `EventPropInput.Create(..., 0.0, tauMem, tauSyn)` because those .NET-facing APIs have no `v_th` parameter at all. Result: 40 parity tests failed with `EVENTPROP_INVALID_VTH` (11) on the C side.

**Fix**: the validator was split rather than loosened, so no check is duplicated and no call site lost coverage:

| Function | Checks | Used by |
|---|---|---|
| `eventprop_validate_input_no_v_th` | dimensions, spike count, taus, spike times/sources, pointers | `ComputeSums`, `ComputeMaxima`, `backward_maxima` |
| `eventprop_validate_input` | the above **plus** `!(v_th > 0.0)` (negated so NaN is rejected) | `ComputeSpikes`, `BackwardSpikes`, diagnostic traces |

The strict wrapper calls the structural one first, so the two can never drift apart. `eventprop_diag.c` keeps its own `v_th <= 0` guard at line ~108 — that one guards the division in the λ_I jump computation, not the input.

New C tests pinning the contract: `ValidateInputNoVth_*` (5 in `src/EventProp.C.Tests/test/validate_input.cpp` — zero/negative/NaN `v_th` accepted, structural errors still caught) and `ZeroVthAcceptedAndIgnored` in `compute_sums.cpp`, `compute_maxima.cpp`, `backward_maxima.cpp`, each running one input with `v_th = 1.0` and `v_th = 0.0` and asserting identical output.

### Fixed — `submodule-linux.patch` was unparseable and `build.sh` swallowed the failure

The patch was truncated mid-hunk. Its `setup.py` hunk header declared `@@ -3,6 +3,7 @@` (6 old lines, 7 new) while the body carried only 3 context + 1 removed + 1 added — the two trailing context lines had been deleted, leaving the counts stale. `git apply` cannot parse the file at all (`corrupt patch at ...:24`), so **no** part of it can be applied, and `--exclude` cannot rescue the good hunks because the failure is a parse error, not a match error.

`build.sh` applied patches with `patch -p1 < "$PATCH_FILE" || true`. `patch` is more forgiving than `git apply`: it patched `eventprop/loss_layer.py`, then aborted at `malformed patch at line 24` — and `|| true` discarded that. Net effect on Linux: the submodule ends up with **only `loss_layer.py` patched**, while `setup.py` (Boost include path, `-fopenmp` link arg), `src/eventprop_cpp.cpp` (the C++17 `<execution>` parallel policies the patch removes for GCC), and the pybind11 2.13.6 bump are all silently absent. The build then fails with an opaque compiler error far from the cause.

The patch is regenerated from its intent — apply the intended edits to a detached worktree at pristine submodule HEAD, then `git diff` — which yields correct hunk headers. The regenerated `src/eventprop_cpp.cpp` hashes to the same `c604ff4..45d9ce9` as the original, confirming those hunks were always correct and only `setup.py` was corrupt. The macOS and Windows patches were structurally sound.

Guards: `build.sh` now calls `scripts/submodule-state.sh apply`, which uses `git apply` (atomic, parse-checked), classifies each patch as applied / applicable / blocked, and dies with instructions on `blocked` instead of continuing. `check-patches` applies every `submodule-*.patch` to a pristine detached worktree — the only context in which a foreign-platform patch can be told apart from upstream drift.

Diagnostic signature: a Linux reference-library build failing on `#include <execution>`, a missing `-fopenmp` link symbol, or a pybind11 2.6.x error against NumPy 2. Check `./scripts/submodule-state.sh status` before blaming the compiler.

### Fixed — the reference library's own TTFS gradient test compared NaN against 0.0

`src/EventProp.Python/test/test_loss_layer.py::TTFSCrossEntropyLossTest::test_numerical_gradient_vs_error`
failed deterministically with a message that reads like a gradient bug and is not one:

```
nan location mismatch:
 ACTUAL: array(0.)      # analytical, from loss.backward()
 DESIRED: array(nan)    # the central difference
```

The test is **vacuous**. `TTFSCrossEntropyLoss.forward` is not a bookkeeping call — it runs a real
LIF forward pass — and the test constructs the layer with no `w_in`, so `LIFLayer` falls back to
`w_dist = GaussianDistribution(w_mean=0, w_std=1)` and the test **seeds nothing**. Measured on this
checkout: 50 fresh instances, **0 emitted a spike**, so every `first_spike_time` was NaN, and
`get_losses` on an all-NaN row returns NaN — that is the numerical leg. The analytical leg is
legitimately `0.0`, because nothing writes into the errors array when nothing spiked.

`numpy.testing.assert_allclose` defaults to **`equal_nan=True`**, so it reports a mismatch only when
the NaNs are in *different* places. The test therefore goes green when both legs are NaN — a
gradient test that passes precisely when it measures nothing. It turned red here only because the
zero-vs-NaN pair happens to differ in NaN-location.

**Fix**: `submodule-tests.patch`, applied on **every** platform, mirroring the five changes the
project's own Cython twin (`src/EventProp.Native.Python/test/test_loss_layer.py`) already carries —
nothing was invented: `np.random.seed(42)`; explicit all-to-all `w = ones(n, n) * 2.0` at all three
construction sites ("diagonal weights don't produce output spikes with single-spike-per-neuron
input"); base times at `(arange(n) + 1) * 0.001`; `t_eps` 1e-8 → 1e-7; and `atol=1e-7`.

Every number reproduces on the Eigen side, which is why the twin's constants transfer unchanged:
with seed + `ones*2.0` the layer fires (`first_spike_times` 0.00612 / 0.00993); one entry per label
is analytically exactly 0 and its central difference is **5.551e-10** — so `atol` is *required*, not
cosmetic, and `rtol=1e-6` alone fails 0.0 against rounding noise — while the worst relative error on
the non-zero entries is **3.5e-9**, four orders inside `rtol`.

**Proven causal, not correlated.** With the macOS patch applied and *only* `submodule-tests.patch`
reversed: `1 failed, 13 passed`. With it applied: **`14 passed`**. (Reverting both patches adds a
second failure, `test_layer_chain::test_gradient_vs_numerical_random_ttfs`, which is the
`dtype=np.int` → `dtype=int` NumPy-2 fix the macOS patch owns — a useful reminder that these patches
are load-bearing together.)

The other two tests in that file were checked for the same vacuity and are **not** affected: both
supply explicit weights and a seed, and produce real signal (spike errors spanning −9.2 to 10.6,
weight gradient `max|g| = 0.6669`), asserted with `assert_almost_equal`, whose default `atol=1e-8`
cannot be satisfied by an all-zero array against a non-zero one.

**The rule.** Any finite-difference test must assert the thing actually *fired* before comparing
gradients, and must not rely on NaN semantics to pass: `equal_nan=True` is the default, so a
scenario that produces no spikes makes both legs NaN and the assertion green. The same reasoning is
already applied to the .NET `LayerContractTests` ("with a firing guard so the check is
non-vacuous") — this is that rule, seen from the test-data side. When a numerical-gradient test
fails, check for a vacuous scenario *before* suspecting the adjoint sweep, and check the noise floor
(`machine_eps / t_eps`) before suspecting the math.

### Fixed — .NET: pointers taken from managed memory outlived their `fixed` pin (SIMD + P/Invoke)

Two families of hot path handed raw pointers to code that runs for a long time, and in both the pin was released **before** the pointer was consumed.

**1. `SimdAlgorithms` (`src/EventProp/`).** Every method derived its base address with `Unsafe.AsPointer(ref MemoryMarshal.GetReference(span))` — a raw address into **unpinned** managed memory. A concurrent GC relocating the array mid-loop makes the loads read a dead copy and the stores land at the stale address. The live array keeps its zero initialisation, and the stale region — already recycled to another live object — is silently overwritten. All 17 sites now use `fixed (double* p = span)`.

**2. The P/Invoke wrappers (`src/EventProp.Native/Api.cs`, `BatchFriendly.cs`).** This was the one causing the test flakiness. `EventPropInput` carries raw `W`/`Times`/`Sources` pointers, and the span overload `EventPropInput.Create(w, nPre, nPost, times, sources, …)` took its `fixed` pins **inside itself** and returned the struct — so the pins were gone by the time the caller invoked the native function. Same shape in `BatchFriendly.BuildInputs`, which self-pinned and returned, leaving all six batch entry points calling native on unpinned pointers. Every wrapper now holds a `fixed` block that spans the native call; the span overload is documented as returning UNPINNED pointers, and `BuildInputs` became `FillInputs(pw, pt, ps, pn, …)` — pointer-based, so the caller is forced to own the pin.

**Why this was so hard to find.** It is a ~30%-of-runs flake under TUnit's parallel scheduler, and it does **not** reproduce in isolation: a 34-million-iteration single-test probe and a 646k-iteration standalone SIMD probe both showed **0 lost writes**. Fixing only the SIMD side changed the failure rate from 5/12 full-suite runs to 4/12 — the pinning was necessary, and not the cause.

**The tool that made it deterministic: `DOTNET_GCStress=0x4`** (compacting GC — forces relocations on nearly every allocation). With the native side unpinned it fails immediately and legibly:

```
failed Li_Forward_ExposesMaximaPerNeuron
  InvalidOperationException: Forward pass failed for sample 0:
  Invalid source neuron index (out of [0, n_pre) range)
```

C is reading a relocated array — the `sources` values no longer index `[0, n_pre)`. With the pins in place: **5/5 GCStress runs at 374/374**, and **12/12 normal runs at 374/374** (previously 4–5 of 12 runs failed). GCStress runs the suite in ~15s against ~2s normal, so it is cheap enough to run ad hoc, not in CI.

**The non-obvious symptom class.** A stale pointer here does not crash — it returns *plausible* numbers. The observed failures were `MANAGED=0 NATIVE_C=2.6559…` (the all-zeros live array), `ComputeMaxima` mismatches on `max_times[...]`, and `ComputeSpikes` spike counts off by 1 (8 vs 90). Two of these look like numerical/parity bugs in the algorithm, which is where the search went first. When a managed-vs-native parity test fails **intermittently**, suspect the marshalling boundary before the math.

Regression guards: `src/EventProp.Tests/PinningStressTests.cs` — `FmaRow_WritesLandUnderConcurrentGcPressure` and `NativeForward_IsBitIdenticalUnderConcurrentGcPressure`, each running a bounded window of concurrent allocation against the operation. **These are probabilistic tripwires, not proofs** — they do not reliably fail if the pinning regresses (see the reproduction rates above). The AGENTS.md post-mortem and `DOTNET_GCStress=0x4` are the real record; the tests are the cheap net.

### Fixed — BenchmarkDotNet adapter read its CSV with the locale encoding

The first full Windows run completed all three suites and then lost the .NET one:

```
ValueError: unparsable Mean: '169.8 ╬╝s'
```

BenchmarkDotNet writes the Mean column as **UTF-8** — `169.8 <CE BC>s`, where `CE BC` is U+03BC GREEK SMALL LETTER MU. `from_benchmarkdotnet.py` opened the CSV with no `encoding`, so Python used `locale.getpreferredencoding()`, which is **cp1252** on a Western Windows; those two bytes decode as `Î¼`, the unit regex misses, and the adapter raised **after a 16-minute run of 95 benchmarks**. No dotnet record file, so `merge.py` produced a report with no .NET column and the pipeline exited non-zero.

macOS and Linux locales are UTF-8, so the same code is correct there and silently wrong here. `UNIT_TO_NS` already accepts both U+03BC and U+00B5 — the two-spelling problem, one layer above the one that broke.

Fixed at every site, not just the failing one: **all 12 opens** across the six `scripts/bench/` scripts state an encoding. The writes matter as much as the reads — the adapters and `report.py` emit non-ASCII, and a cp1252 write of a committed artifact is mojibake for the next machine. Audit: `grep -n "open(" scripts/bench/*.py | grep -v encoding=` → 12 hits, now 0.

Diagnostic signature: a suite that ran to completion, produced artifacts, and contributes **zero** records, with a `ValueError` naming a mangled unit.

### Fixed — the Eigen core's arch flag was `PRIVATE`: aligned-allocator ABI heap corruption

See *SIMD: One ISA Level* above for the mechanism (`EIGEN_MAX_ALIGN_BYTES` 32 vs 16, header-inlined allocator, STATIC library, consumer TU frees the block). The failure signature is worth repeating because it points nowhere near the cause: `STATUS_HEAP_CORRUPTION` (`0xC0000374`) in `RtlFreeHeap` on the **first** test, with zero program output (the fast-fail discards the stdout buffer), in a process that had just been rebuilt with one added compiler flag.

How it was localised, in order, because each step is cheap: (1) split the two sides — `bench_c` completed, `bench_cpp` fast-failed, so the C library was innocent; (2) `cdb.exe` (Windows SDK Debuggers) with `-c "g; k 25; q"` on a **RelWithDebInfo** build (Release has no PDB, so frames are `eventprop_cpp_tests+0x1292a`; RelWithDebInfo names them) → `handmade_aligned_free` → `DenseStorage::~` inlined at the caller. That is the whole diagnosis in two commands.

Guards: the flag is `PUBLIC` on `eventprop_eigen_core`, and `ParityTests` / `Native.Benchmarks` apply the module's level to their own TUs. The regression is silent on Linux, so the guard there is the parity suite running at all.

### Fixed — the Windows scripts reported success they had not measured

`build.ps1` checked **no exit code at all** and ended with "Build complete"; it also never built `EventProp.C.Tests` or `EventProp.Benchmarks` — the second being the artifact `run_benchmarks.ps1` checks for, so `-Build` could never satisfy the check the same script performs. `run_benchmarks.ps1` printed the green "Report (committed):" line regardless of the pipeline's exit code, pointing at a report a failed run may not have written, and its Cython prerequisite was "the `.pyd` exists", which was true while `import eventprop_c._native` failed.

Both now use presets, check every step, and summarise with a real exit code. Four Windows mechanisms made the old behaviour misleading, and all four are invisible in the code:

- `$ErrorActionPreference = "Stop"` + a native command writing to stderr is a **terminating** `NativeCommandError` in PowerShell 5.1. The prereq probe `python -c "import numpy, pytest, pytest_benchmark"` died with `NativeCommandError` on the very failure it was written to report, so the install hint never printed. Fix: `$PSNativeCommandUseErrorActionPreference = $false` (a no-op on 5.1) and `Continue`.
- PowerShell **drops empty strings** from a native command's argument array. `run_all.py` documents "pass `''` to skip" for `--reports-dir` / `--benchmarks-md`; from PowerShell that is unreachable — `-ReportsDir ''` arrives as a bare `--reports-dir` and the runner silently uses the default. Measured: a scratch run overwrote the committed `BENCHMARKS.md` with a single-suite report. Fix: `-NoReports` / `-NoBenchmarksMd`, which pass the single-token `--reports-dir=` form.
- `where bash` returns `C:\WINDOWS\System32\bash.exe` (**WSL**) before Git's. The same `submodule-state.sh` call succeeded from Git bash and exited 1 from a PATH where System32 precedes Git's `usr\bin` — and Git bash needs that directory on `PATH` for coreutils, failing with `dirname: command not found` without it.
- `Tee-Object` writes **UTF-16LE** on 5.1, so a captured log greps clean and reads as an empty run. Convert with `iconv -f UTF-16LE -t UTF-8`.
- `cmd /c "… && cmake --build … & echo EXIT=%ERRORLEVEL%"` is a **phantom success**: `%ERRORLEVEL%` expands at **parse** time, so it prints the code that was there before the command ran, and `&` runs the echo even when the `&&` chain stopped. It reported `BUILD_C_EXIT=0` twice while `vcvarsall` had failed and **nothing had compiled** — verified afterwards by the missing `.obj` and an old DLL timestamp. Real checks need `if errorlevel 1` in a .bat (or `$LASTEXITCODE` in PowerShell, read immediately after the command).
- A shell's `PATH` is a **snapshot from process start**. A machine-PATH defect fixed in the registry is invisible to every shell opened before the sign-in that applies it, so `vcvarsall` dies parsing the stale value (`\NVIDIA was unexpected at this time`, exit 255) on a machine whose PATH is already repaired. `build.ps1` re-reads `PATH` from the registry for exactly this reason.

`scripts/bench/run_all.py` had one real defect: `--out` / `--reports-dir` / `--benchmarks-md` were used verbatim, and every suite runs with its own cwd, so a relative `--out` is opened relative to the suite. Measured with `--out ../bench-test`: pytest tried to open `../bench-test\kernels.json` under `src/EventProp.Python.Benchmarks` and the python suite failed in 0s. The defaults are built from `SRC` and absolute, which is why it never surfaced. All three are resolved with `os.path.abspath` at parse time.

### Fixed — the two clean scripts had drifted apart, and each reported success

`clean.sh` and `clean.ps1` implement one contract, and each had broken the half the other
covered — while both printed `Done.` and exited 0.

**`clean.ps1`**, four defects, all measured:

- `-Target bogus` printed `Cleaning… Done.` and **exited 0 having removed nothing**. `clean.sh`'s exit-2 guard never landed here, so a typo read as a successful clean.
- The `dotnet` list contained `EventProp.Cpp.Native` (**not a project — the directory does not exist**) and omitted `EventProp.Abstractions`, which has real `bin/` and `obj/`. So `clean.ps1 dotnet`, and `all`, left Abstractions' build output behind forever.
- `-Target` was a single string: targets did not compose, unlike `clean.sh c cpp parity`.
- `Remove-Item` errors are **non-terminating** by default, so a locked `build\` — the normal Windows case while a benchmark exe has the C DLL loaded — printed an error and the script still printed `Done.`

**`clean.sh`**, five:

- `-full` matched only `eventprop_cpp.*.so` / `.dylib`. Measured on Windows: **0 files match, 2 `.pyd` exist** — so it deleted nothing and printed `→ Deleted reference binary (use build.sh to rebuild)`, while AGENTS.md promises `-full` forces a rebuild.
- `eventprop_c/*.so` likewise matched **0** files (2 `.pyd` present), so `clean.sh python` **left the compiled extension in place** — the stale-extension class survived a clean.
- `_native.c` was never removed, and that defeats `build.sh`'s regeneration rule ("regenerate when `_native.pyx` is newer than `_native.c`"): after a clean the surviving `_native.c` is the newer file, so Cython is skipped and the "clean" build compiles the stale generated C — the exact artifact the ABI check refuses to import on.
- `pip uninstall` ran on the ambient `pip` (3.14 on this machine) while the suite runs on 3.11, so "cleaned" can describe an interpreter the suite never uses.
- `eventprop/**/*.pyc` is decorative — `globstar` is **off** by default, so it acts as `eventprop/*/*.pyc` (the `find` covers it). Plus: no target for the pipeline's own outputs (`bench-results/` 684K, `BenchmarkDotNet.Artifacts/` **2.5M**, `baselines/` 136K), and `rm -rf .venv/` in the benchmarks step, where a venv is not a build output.

**The defect the test of the fix found.** Declaring `$Targets` **last** with
`ValueFromRemainingArguments` makes a bare word bind to the first bindable positional —
which was `$PythonExe`. Running `.\clean.ps1 bogus` therefore set the interpreter to
`bogus`, left `$Targets` empty (so it defaulted to `all`), and **ran a full clean: 39 paths
removed**, including the compiled extension, `_native.c` and 6 baseline JSONs. `.\clean.ps1 c`
would have done the same. Fixed by declaring `$Targets` as `Position = 0` with the
remaining-arguments binding (bare words → targets, `-PythonExe` named-only) and validating
that `PythonExe` resolves **before** the first removal.

**Caught by the new `--dry-run`:** the `__pycache__` walk recursed **into `.venv`** — it
would have deleted numpy's and pytest's caches inside the user's virtualenv. A virtualenv is
now pruned from that walk, and never removed.

Both scripts now: compose targets, exit 2 on an unknown target, match `.pyd`/`.dll`/`.so`/`.dylib`,
remove `_native.c`, take the interpreter (`--python` / `-PythonExe`), collect removal failures
into a non-zero exit, and add a `results` target for the pipeline's regenerable outputs — with
`--dry-run` / `-DryRun` to inspect first. **No target touches `bench-reports/` or
`BENCHMARKS.md`**, which are committed; that boundary is the guard against the subset-run
overwrite trap.

Verification after the accidental clean: full rebuild, then **C 220/220**, **parity 105/105**,
**eventprop_c 29/29**, **.NET 382/382**. `--dry-run` lists Abstractions' `bin`/`obj`, both
`.pyd`, `_native.c` and the 6 baselines, and no `.venv` path; an unknown target exits 2 on both
platforms.

**The rule:** a destructive script needs a dry run and a real exit code, and the two
platform scripts are one contract — a fix in one is a fix in both, or the platforms measure
different trees.

### Fixed — the MSVC environment import leaked `Platform=X64` into the .NET build

`run_benchmarks.ps1 -Full -Build` reported
`MISSING: src\EventProp.Benchmarks\bin\Release\net10.0\EventProp.Benchmarks.dll` **immediately
after `build.ps1` printed "Build complete"**, with step 9 (".NET (libraries, tests,
benchmarks)") green. Both statements were true: the binary existed, at
`bin\x64\Release\net10.0\`.

MSBuild's `$(Platform)` inference reads the **environment**, and
`Enter-VisualStudioEnvironment` (added in `8ef41a9`) imports the whole vcvars environment into
the script's process — including `Platform=X64`. Every later step inherits it, so the
`dotnet build` steps in that same process wrote the platform-qualified layout while the
prerequisite check and `run_all.py` named the framework-dependent path.

Why it hid through a successful 47-minute run: `bin\Release\net10.0\` already existed from an
IDE build, so the path resolved. It surfaced the moment `clean.ps1` deleted `bin\` and the
rebuild produced the `x64` layout.

Fix, in three layers:

- `build.ps1` strips `Platform` from the process environment **globally**, right after importing vcvars — the CMake presets do not want it either.
- `run_benchmarks.ps1` probes `bin\Release\net10.0`, `bin\x64\Release\net10.0`, `bin\AnyCPU\Release\net10.0` and the Debug variants, and **prints the layout it found**.
- `run_all.py` gained `dotnet_dll_path()` with the same probe, prints `.NET binary: <relpath>`, and hangs the per-job cleanup glob off the located directory.

The consumer-side probe is not redundancy: **any** Visual Studio developer prompt sets
`Platform=X64`, so a user who runs `dotnet build` from one gets the `x64` layout with none of
these scripts involved. Naming one layout in a consumer is the trap; probing is the fix.

Also fixed in the same pass: with `-NoReports` the failure text said "a report exists at
`bench-reports\report_<tag>.md`, … from this FAILED run", which misattributes the previously
committed report — that run writes no report. It now says the file is a previous run's and was
not modified.

Verification: `build.ps1 release` → `bin\Release\net10.0\EventProp.Benchmarks.dll`;
`run_benchmarks.ps1 -Full -ReportOnly -NoReports -NoBenchmarksMd` prints
`.NET : src\EventProp.Benchmarks\bin\Release\net10.0\EventProp.Benchmarks.dll` and
`artifacts: ok` and proceeds into the pipeline; running the located dll directly executes
(`Bench_TrainingLoop_Tiny` Managed **4.124 µs** / Native **18.305 µs**, ShortRun).

**The rule:** importing a vendor environment into your own process means every later step
inherits it. Enumerate what it sets (`set` after `vcvarsall`), and strip the variables your
other tools interpret — `Platform` for MSBuild, `PLATFORM`/`VSINSTALLDIR` for CMake.

### Fixed — three C++ projects would not link on Linux: `<execution>` needs TBB, and the shared library linked anyway

`EventProp.Cpp.Wrapper/eventprop_eigen_core_stripped.cpp` uses `std::sort(std::execution::par_unseq, …)`
and `std::find_if(std::execution::par_unseq, …)`. libstdc++ 12+ implements those policies over
Intel oneTBB, and nothing in this repo linked it — `git grep -i tbb` over the tracked tree
returned zero hits outside the read-only submodule. Three executables died at link
(`eventprop_cpp_tests`, `parity_test`, `bench_cpp`) with `undefined reference to
tbb::detail::r1::allocate/spawn/execute_and_wait/…`; `bench_c` was fine, because it links only
the C library.

The link errors were not the problem worth recording. **`libeventprop_cpp_wrapper.so` linked
successfully** — a shared library on Linux may keep undefined symbols — reported `Built
target`, and could not be loaded:

```
OSError: .../libeventprop_cpp_wrapper.so: undefined symbol:
         _ZN3tbb6detail2r15spawnERNS0_2d14taskERNS2_18task_group_contextE
```

14 undefined `tbb::` symbols in an artifact a green build step had just handed to its
consumers. That is this repo's recurring class — a step whose final line is a claim rather
than a measurement — and the loud failure (the executables) was what pointed at the quiet one.

**Fix**: `cmake/EventPropPstl.cmake`, called by all three projects — PUBLIC on
`eventprop_eigen_core` (a STATIC library's link requirement is its consumers' requirement, which
is also what makes the SHARED wrapper resolve instead of deferring), `INTERFACE_LINK_LIBRARIES`
on the imported stand-in in ParityTests, and explicit link items for `bench_cpp` /
`bench_rootfind` (see *PSTL: what `<execution>` needs linked*).

**Why not delete the parallel policies instead?** `submodule-linux.patch` does exactly that to
the reference library's `src/eventprop_cpp.cpp` — which is why the pybind11 build has never
needed TBB — and porting that edit into the convenience copy would have been one line. It was
rejected: the wrapper is shared with macOS, where the same source compiles today, so removing
`<execution>` changes the platform the constraint said not to change, and it would silently
serialise the benchmarked path on Linux while the C side runs OpenMP-parallel. Adding a link
input where one is missing is the additive fix; deleting a construct to dodge a missing
dependency is not.

**Proven by construction, not by reading**: a six-line reproduction of the same two calls fails
with the identical error class and links cleanly with `-ltbb`; `libtbb-dev` 2021.11.0 was
already installed; the wrapper `.so` now carries `NEEDED libtbb.so.12`, `dlopen`s, and
`eventprop_cpp_tests` passes 25/25.

Diagnostic signatures: `undefined reference to tbb::detail::r1::*` on GCC 12+ whenever a TU
includes `<execution>`; a shared library that builds green and `dlopen`s to `OSError`;
and `nm -DC` still showing those symbols as undefined **after** the fix, which is correct —
they resolve at load through the `NEEDED` entry, so the load test is the check and `nm` is not.

Regression guards: the module's three branches are each exercised (finds TBB / `FATAL_ERROR`
naming `libtbb-dev` / empty with **zero probes** under `-DCMAKE_SYSTEM_NAME=Darwin`), and the
suites that consume it run — C++ 25/25, parity 105/105, `bench_cpp` executing.

### Fixed — the Cython step reported two failures for one missing package, and would have compiled a stale source

On a Linux interpreter without Cython, `build.sh` step 8 printed `No module named cython`,
recorded `FAILED (exit 1): cython _native.pyx`, and then ran `pip install -e .[dev]` anyway,
against a `_native.c` that had never been generated. The summary listed **five** failed steps,
and **two of them were this one missing package**; the redundant one was a 98-line setuptools
traceback whose headline — `cc1: fatal error: eventprop_c/_native.c: No such file or directory`
— named a file, not the cause, several screens below it. Cython is a build prerequisite of this
step and nothing
fetches it automatically: `setup.py` compiles the generated `.c`, and `build-system.requires`
lists only setuptools and numpy, so pip's isolated build env never cythonises anything either.

**Fix**: probe `import Cython` in the interpreter that will build the extension, and on failure
record **one** failure whose text names the interpreter and gives the command for that
interpreter. Then gate the install on the source having actually been produced **in this run**:

```bash
if [[ $cfile_ready -ne 1 ]]; then          # not: [[ ! -s "$cfile" ]]
  note "$cfile was not regenerated — skipping the editable install rather than"
  note "compiling a stale or missing source (the cython failure above is the cause)"
```

The `cfile_ready` flag, rather than a file test, is the part that matters: if regeneration was
needed and did not happen, a `_native.c` that **is** present is a **stale** one, and compiling
it is exactly the stale-extension class `_native.pyx`'s ABI check refuses to import on. A
`-s "$cfile"` test would have skipped the failure and printed a green editable install over
headers from a previous build.

**Verified, all three states**, by running the step's own text (extracted verbatim from
`build.sh`, so the harness cannot drift from the script): stale `_native.c` + no Cython → one
failure, one named skip, no compile; no `_native.c` at all → same; same state with Cython
present → regenerates 1.76 MB of C and installs, no failures. The skip appears in the summary,
because a silent skip is the defect this script was rewritten to remove.

Related: the summary printed `Skipped:` only on the failure path, so `--skip-python` on a green
build claimed "Build complete" while naming neither of the two projects it had not touched.
Both paths print it now.

### Fixed — `import eventprop_c` failed on Linux after a green build: no rpath, and the escape hatch was inert

`_native*.so`'s `DT_NEEDED` is the bare soname `libeventprop_c_api.so`. `setup.py` gave the
darwin branch `-Wl,-rpath,…` and the `else` branch `extra_link_args = []`, so on Linux the only
thing that could find the library was `LD_LIBRARY_PATH` — which the sole consumer,
`scripts/bench/run_all.py`, happens to export for its own reasons. So the extension imported
fine inside the benchmark pipeline and failed everywhere else, including immediately after
`build.sh` printed a green editable install:

```
ImportError: libeventprop_c_api.so: cannot open shared object file: No such file or directory
```

`_library_path.py`'s docstring said why nobody had to fix this: *"On POSIX this module does
nothing: the extension embeds an rpath at build time"* — true on macOS, false on Linux, and
written confidently enough that the file itself argued against the fix it needed. It also made
`EVENTPROP_C_LIB_DIR`, the documented escape hatch, silently inert on Linux: a variable that
appears to configure the package while doing nothing is its own defect class, since the reader
reasons from it as if it were load-bearing.

**Fix** (both Linux-only; macOS and Windows paths untouched): link the rpath in `setup.py`
— absolute, derived from `__file__`, because the loader resolves it at import time in a process
that may have started anywhere — and preload the library in `_library_path.py` via
`ctypes.CDLL(lib, RTLD_GLOBAL)` per candidate directory, the POSIX counterpart of
`os.add_dll_directory`. A library already loaded with `RTLD_GLOBAL` satisfies a later
extension's `DT_NEEDED` by soname, which is what makes `EVENTPROP_C_LIB_DIR` real on Linux and
covers the case an absolute rpath cannot: an installed `eventprop_c`, or a checkout moved after
the extension was built.

**Measured in isolation**, because "it imports now" does not say which mechanism did it: the
bare soname fails to load before the preload and loads after it; repeated registration is
idempotent; `import eventprop_c` works from `CWD=/` with `LD_LIBRARY_PATH` unset; RUNPATH is
present in the built extension.

Regression guards: `test/test_library_path.py::LinuxLibraryLookupTest` (3 tests) plus the
cross-platform `test_extension_imports_and_exposes_the_batch_api`, which is the one that had
been failing — worth noting, because that test lives in the suite nobody ran on Linux.

### Fixed — "already built" over an unloadable reference binary: step 7 trusted file existence

`build.sh` step 7 decides whether to rebuild `EventProp.Python`'s pybind11 extension from the
filename's ABI tag. Correct about the interpreter, silent about everything else: the binary in
this checkout was dated three months earlier, built before `submodule-linux.patch` put
`-fopenmp` in the link args, so it carried two undefined `omp_*` symbols. Step 7 printed
`→ eventprop.eventprop_cpp already built (…cpython-312-x86_64-linux-gnu.so)` for a library that
could not be loaded, and the failure surfaced in `tests/test_paper_parity.py` — the suite that
compares the two backends on trained checkpoints:

```
ImportError: …/eventprop_cpp.cpython-312-x86_64-linux-gnu.so: undefined symbol: omp_get_thread_num
```

The same lesson this file already records for a zero-byte `_native.c` — an existence test cannot
see an unusable artifact — applied one step earlier, to a bigger file.

**Fix**: ask the interpreter. If a matching-tag extension exists but
`import eventprop.eventprop_cpp` fails, print the loader's own line and rebuild:

```
eventprop/eventprop_cpp…so matches this interpreter but does NOT import — rebuilding
  reason: ImportError: …undefined symbol: omp_get_thread_num
```

Quoting the loader matters: a missing symbol names the library that was never linked, which is
the entire diagnosis, and otherwise it is three steps and one test-suite failure away from the
cause. `PYTHONPATH` for the probe is joined with `os.pathsep` asked of the interpreter, not a
hardcoded `:` (the `run_all.py` defect on Windows). The check is deliberately **not** a Linux
special case: an extension that imports is skipped exactly as before on every platform, and one
that does not is rebuilt with a line saying why; if some platform's in-place extension ever
fails the probe for a reason the step cannot see, the cost is a rebuild a fresh clone pays
anyway — never a green "already built" over a binary that cannot load.

Evidence: the rebuilt binary carries `NEEDED libgomp`, and **paper parity is 7/7 on Linux** —
YinYang TTFS and MNIST VMax checkpoints through both backends, the only suite that exercises
real trained weights. The probe's **happy path** is what every build since has taken: step 7
prints `→ eventprop.eventprop_cpp already built …` and does not rebuild, so the check is not a
tax on incremental builds — that line is now a statement about the interpreter's opinion of the
binary rather than about the filename.

### Fixed — Windows reference build: pybind11 had no required version, so step 1 checked out the one that cannot compile

`build.ps1 release` failed at step 7 on a fully clean tree — `pip install -e .` for
`EventProp.Python` died inside pybind11:

```
pybind11/detail/type_caster_base.h(448): error C2027: use of undefined type '_frame'
pybind11/pybind11.h(2128):               error C2039: 'frame': is not a member of '_ts'
pybind11/pybind11.h(2132):               error C2660: 'PyDict_GetItem': function does not take 1 arguments
```

That is pybind11 2.6.3.dev1's pre-3.11 frame C API (`frame->f_code`,
`PyThreadState_Get()->frame`) compiled against headers that made `struct _frame`
opaque — Python 3.11.9 on MSVC 14.52 with NumPy 2.4.6.

**Why the version was wrong, and why the script made it wronger.** The macOS row of the
pybind11 table existed; the Windows row did not. `pybind11_required()`'s `*` arm returned
`PYBIND11_RECORDED`, and `apply_pybind11()` does `git checkout` when `HEAD != required` —
so `build.ps1` step 1 was not merely omitting the requirement, it was **actively checking
out the version that cannot compile**. Measured: with the nested submodule at 2.13.6,
`submodule-state.sh apply` printed `→ pybind11 v2.13.6 -> v2.6.2-41-g54430436`.

No MSVC flag or setuptools version fixes it — 2.6.3.dev1 predates the interpreter, so the
only fix is the pybind11 version. `PYBIND11_REQUIRED_WINDOWS=a2e59f0e` (2.13.6, the Linux
commit; no Windows patch moves the gitlink, so the constant is the record) plus a
`msys*|cygwin*|mingw*` arm in `pybind11_required()`, using the same OSTYPE spellings as
the `PATCH_FILE` case above. This file is on the Windows build path — `build.ps1` step 1
runs it through Git-for-Windows bash (`Get-GitBash`, not `System32\bash.exe`) — so the fix
lands where the state is owned rather than in a second `git checkout` that step 1 would
undo on the next run.

**A second platform-blind test in the same script:** `status` globbed only
`eventprop_cpp.*.so`, so on Windows it printed `reference binary: not built` next to a
built, importable `eventprop_cpp.cp311-win_amd64.pyd` — the `.so`-only glob defect that
made `clean.sh python` leave the Windows extension in place, in a different file. It now
probes `.so`, `.dylib`, `.pyd`, `.dll`. Whether the binary *loads* stays `build.ps1`
step 7's import probe, which is the right owner of that question.

**Measured after the fix**: downgrade to `54430436` by hand → `apply` restores `a2e59f0e`;
`verify` and `check-patches` green (all four patches apply to a pristine worktree);
`build.ps1 release -SkipSubmodule` ends `Build complete`, exit 0; `import
eventprop.eventprop_cpp` OK; paper parity 7/7 on the trained YinYang and MNIST checkpoints.

**Diagnostic signature**: `error C2027: use of undefined type '_frame'` (or the same class
of frame/C-API errors on Clang or GCC) inside `type_caster_base.h` or `pybind11.h`. Check
`./scripts/submodule-state.sh status` and compare `HEAD:` against `required:` before
blaming the toolchain — the failure is in the reference build, three steps from the cause.

### Fixed — the clean scripts counted paths they had not removed

`clean.ps1 -Full` printed `Clean complete (27 path(s) removed)` and exited 0 while **8 of
the 10 `.NET` `bin`/`obj` paths were still on disk, files included** — `obj\Debug\net10.0\`
still held `.cs` and `.cache` files, not merely an empty shell. A second `clean.ps1 dotnet`
pass removed all 8 and reported 8. No `FAILED` line was ever printed.

`Remove-Tree` treated "`Remove-Item` returned without error" as "removed". On Windows those
are different claims: a file another process opened with `FILE_SHARE_DELETE` — which is what
MSBuild's and Roslyn's persistent build nodes do to `obj\` and `bin\` — enters
**delete-pending**, so the unlink is queued, the call succeeds, and the path stays visible
until the last handle closes. Five `dotnet.exe` build-server processes were running on this
machine throughout.

Both scripts now test the post-condition instead of reporting the attempt:

- `clean.ps1` — `Remove-Tree` and `Remove-Matched` re-`Test-Path` after a non-throwing
  removal; a surviving path goes to `$Failures` (so the run prints `Clean INCOMPLETE` and
  exits 1), and the file count prints `$gone/$($files.Count)` — removals against matches,
  not matches dressed as removals.
- `clean.sh` — the same `[[ -e ]]` post-condition in `rm_tree` and `rm_find`, and
  `dotnet build-server shutdown` before the `.NET` step in both scripts (best effort: no
  `dotnet` on PATH must not fail a clean of everything else).

**Honest scope of the POSIX half**: delete-pending is a Windows phenomenon, so on
Linux/macOS the post-condition is inert and `dotnet build-server shutdown` there is
**unmeasured** — mirrored for the one-contract rule, and `bash -n clean.sh` plus
`clean.sh --dry-run` were run here under MINGW, which is not a Linux run. Treat the POSIX
leg as pending its first macOS/Linux `clean.sh -full`.

The rule: a destructive script needs a post-condition, not a return code — the return code
is the *attempt*, and on Windows even a successful attempt can leave the tree in place.

### Fixed — the wrapper's C API had no write bound, so its own test binary overran its heap

`eventprop_capi_tests.exe` died with an access violation (exit 139) and, because stdout is
block-buffered when redirected, **no test output at all** — it read as a build problem,
not a heap overflow. cdb on a RelWithDebInfo build named it:

```
eventprop_capi_tests!eventprop_compute_spikes(...)   <-- mov qword ptr [r15+rcx*8+10h],rax  (AV)
eventprop_capi_tests!test_compute_spikes_very_low_threshold(void)+0x17f
eventprop_capi_tests!main(void)+0x9b
```

**Root cause, not a fixture typo.** `eventprop_compute_spikes` in `eventprop_c_api.h` took
output pointers with **no capacity argument**, so it could not bounds-check, and every
caller sized its buffer from a bound that does not exist — `max_out = 50*(n_spikes+1)*n_post`
in `tests_capi.cpp`, and `max_out = n_spikes * n_post` under the comment *"max possible
spikes = n_spikes * n_post"* in `benchmarks/eventprop_benchmarks.cpp` (that whole project is
now deleted — see the Project Map — but it held the same wrong assumption). A
post-synaptic neuron may cross `v_th` any number of times inside one input interval, so the
output count is unbounded in the inputs. Measured in the fixture's own binary at 50x20
neurons / 100 input spikes: **24,046** output spikes at `v_th=0.1` and **240,986** at
`v_th=0.01`, against a 101,000-element buffer — 2.4x past the end, three arrays deep.

**Fix — the API carries the bound now** (`eventprop_c_api.h`): `int max_out` is the last
argument of `eventprop_compute_spikes`; writes are clamped to it, and the return value is
the count the call *would* have written, so a caller that cannot bound its output does
`n = call(cap); if (n > cap) resize(n); call(n);`. `eventprop_backward_spikes` /
`backward_maxima` / `compute_maxima` need no capacity — their output sizes (`n_pre*n_post`,
`n_input_spikes`, `n_post`) are already arguments.

Callers updated: `tests_capi.cpp` — 8 sites pass their capacity, each followed by
`ASSERT_FITS(n, max_out)`, which `return`s, so nothing downstream can read past what was
written. `eventprop_capi_tests` now reports **7/7 passed, exit 0** (it segfaulted before).

The only other in-tree consumer of the shim was `Cpp.Wrapper/benchmarks/`, which is
**deleted in the same change** rather than repaired: nothing configured or ran it, its
generators were a fourth copy without `SortTimesSources`, and it did not compile. Verified
before the deletion that the repair was sound (`eventprop_benchmarks.exe` built with
google/benchmark v1.9.1 via FetchContent, clang-cl diagnostics identical to pristine HEAD),
so the deletion removes dead weight rather than a working alternative — the maintained
`EventProp.Native.Benchmarks` suite is the one with the data contract, the JSON baselines
and the cross-suite pipeline.

**Also removed: 13 `fprintf(stderr, "DEBUG_API: …")` calls plus a
`CaptureStackBackTrace(0, 16, …)` per `eventprop_backward_spikes` invocation**, committed in
`eventprop_c_api.cpp` with a `tests_capi.cpp` twin of 70 `DEBUG:` lines. Instrumentation from
one debugging session, left in the hot path of a published wrapper: it wrote to stderr on
every call, put a stack walk into any measurement taken through the shim, and swallowed the
process's own output when it crashed.

**Diagnostic signature**: a Windows test exe that exits 139 with no stdout (buffer lost), or
an AV inside `eventprop_compute_spikes`. `cdb -g -G -c "g; kP 25; q"` on a RelWithDebInfo
build names the frame in two commands; Release has no PDB and gives offsets only.

**Not wired into any script**: neither `eventprop_cpp_tests` nor `eventprop_capi_tests` is
run by `build.sh`, `build.ps1`, or the benchmark pipeline — they are built and left. Running
them is a documented command (see *Wrapper C++ Tests* above); wiring them into the build
scripts is an open choice, not an existing behaviour to rely on.

### Fixed — `bench_rootfind`: the bracket list outlived the spike rows it pointed into

`./build/bench_rootfind` exited **139** on its first benchmark, after printing
`RootFind/C/Small 0.000 ns 0.000 ns 1000000000000 iterations` and nothing else. Nothing runs
that binary — `run_benchmarks.sh`, `run_benchmarks.ps1` and `run_all.py` execute `bench_c` and
`bench_cpp` only — so it had never been executed since it was written. The Project Map's
"`bench_rootfind` already covers its root-finder micro-benchmarks" is exactly the sentence that
lets a binary go unrun.

**Root cause, not a fixture typo.** `gen()` returned `std::vector<TestCase>`, and each `TestCase`
held a `VoltageCtx` **by value** whose `sum0` / `sum1` / `post_spike_times` members were `.data()`
pointers into `std::vector` storage **local to `gen()`**. All three were freed the moment `gen()`
returned, so every benchmark iteration dereferenced dead memory. AddressSanitizer on the pre-fix
source, first iteration:

```
ERROR: AddressSanitizer: heap-use-after-free on address 0x7ffff72c0c00
READ of size 8 at ... thread T0
SUMMARY: AddressSanitizer: heap-use-after-free bench_rootfind.cpp:21 in compute_v
```

**The rule** is the one `eventprop_c`'s zero-copy views already obey: *a borrowed pointer must not
outlive the function that borrowed it.* `gen()` now returns a `SumRows` that owns `sum0` / `sum1` /
`post_times` together with an **index-based** `Bracket` list (`i`, `nrn`, `a`, `b`) and the
per-fixture constants, and `make_cases()` builds each `VoltageCtx` from a fixture the caller holds,
in the caller's own scope. Deliberately **not** "return the fixture and rely on the vector move
preserving the buffers": every real implementation preserves them, the standard does not promise
it, and silent reliance on that is how the file got here.

**A second defect sat on the same line** — `RowMatrixXd::Random(n_pre, n_post).array() * 0.5`.
`Eigen::Random()` is not the generator's RNG: it reads the global `std::rand()` stream (see the
benchmark data contract above), so the weights were neither reproducible nor in the documented
U[0.5, 2.0) range. That is covered there; both are fixed together.

**An empty row must be visible, not fast.** Each row now publishes `state.counters["cases"]`. An
empty bracket list is a legitimate outcome of the scan, but it times an empty loop, and google
benchmark then reports `0.000 ns` with an absurd iteration count — which reads as "the C root finder
is infinitely fast". Same reason `bench_main.cpp` publishes `c_error` / `c_fwd_error`. Exclude
`cases=0` rows from any ratio.

**Measured before/after.** Before: `C/Small` was `cases=0` at 0.28 ns (nothing measured) and
`Boost/Small` was 2 brackets in one invocation, 0 in another. After: `C 100/80/100` and
`Boost 100/80/100`, identical across consecutive runs, so a C-vs-Boost row pair solves the *same*
bracket set — the minimum for that comparison to mean anything. ASan on the fixed source reports no
error on any of the six rows; exit 139 became exit 0.

**Guard, and its limit.** `bench_rootfind` is now the one target in the repo compiled with
`-Wall -Wextra` (`/W4` on MSVC): for a binary outside the pipeline the compiler is the only
mechanical guard it can have. It is scoped to that target on purpose — `bench_main.cpp` has 6 of
*our own* warnings under those flags (`-Wunused-parameter`, `-Wunused-variable`) and enabling it
there means fixing them first, which is a separate change. Eigen remains a plain `-I` include:
measured across all seven Eigen-including TUs in the repo, **zero** warnings even with
`-Wall -Wextra` here, so a `SYSTEM` include would be decorative config. Add `SYSTEM` at the point
Eigen's own templates actually warn (v3.3.90 in the read-only submodule trips GCC 13's default-on
`-Wdeprecated-enum-enum-conversion` at `CoreEvaluators.h:738` — it did, until the `Random()` call
above was removed), and do not reach for `-Wno-`, which would mute that diagnostic in our code too.
