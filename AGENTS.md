# AGENTS.md — EventProp

Constraints that are invisible from any single file. Everything else has a better owner:

| You want | Go to |
|---|---|
| Flags/targets for `clean`, `build`, `run_benchmarks` | `./clean.sh --help`, `./build.sh --help`, `./run_benchmarks.sh --help` (and the `.ps1` `param()` blocks) |
| Why a failure looks nothing like its cause; symptom → defect | [`docs/POSTMORTEMS.md`](docs/POSTMORTEMS.md) |
| Benchmark record format, validity rules, provenance | [`BENCHMARK_SCHEMA.md`](BENCHMARK_SCHEMA.md) |
| Struct layouts, error codes, function contracts | `src/EventProp.C/eventprop.h` and `eventprop_internal.h` |
| ISA level / parallel-std-lib link rules per project | `cmake/EventPropSimd.cmake`, `cmake/EventPropPstl.cmake` |

## 1. You are on one OS; the other two are real

This project ships and is measured on **macOS (arm64)**, **Linux (x86_64, GCC/Clang)** and
**Windows (x86_64, MSVC)**. A session runs on exactly one of them and can *prove* things
about only that one.

Start by naming your platform: `uname -s -m` (`Darwin arm64`, `Linux x86_64`), or on Windows
`$env:OS` / `[Environment]::OSVersion`. Then classify your change:

| Class | Examples | What you may do |
|---|---|---|
| **Shared** | algorithm sources (`eventprop_*.c`, `EventPropMath.cs`), public headers, CMake modules both platforms include, benchmark data contract | Change it, build and test here, and **state that the other two platforms are unverified by you** |
| **Scoped, with a counterpart** | `build.sh` ↔ `build.ps1`; `clean.sh` ↔ `clean.ps1`; `run_benchmarks.sh` ↔ `run_benchmarks.ps1`; `submodule-macos.patch` ↔ `-linux.patch` ↔ `-windows.patch` | Edit **your** platform's file only. Editing the counterpart — or changing the shared half so it behaves differently there — **requires user confirmation**: you cannot validate it |
| **Conditional inside a shared file** | `if(APPLE)` / `if(WIN32)` / `if(UNIX)`, `#ifdef _MSC_VER`, `sys.platform`/`os.name` branches in `scripts/bench/` | Change the branch for the OS you are on; leave the others byte-identical unless asked |

Consequences, all of them learned the expensive way:

- **Do not "fix" another OS from this one.** A Windows- or Linux-path edit made from macOS is
  unverifiable guesswork. Describe what you think is wrong and what you'd change, and ask.
- **The per-platform scripts are one contract, not parallel implementations.** Every flag,
  target, probe and exit code exists on every platform it applies to. A fix landing in one
  half is exactly how two platforms end up measuring different trees. If you can only change
  one, say so in the commit message rather than letting the pair drift silently.
- **State what you validated and where you did not.** "Release build green, C suite + parity +
  .NET suite passing on darwin-arm64; Windows and Linux untouched by this change and
  unverified here" is a complete statement. Silence about platform gets read as "everywhere".
- **Never assert a cross-platform claim you have not measured there.** This repository has
  carried comments that were true on macOS and false on Linux — the `_library_path.py` rpath
  claim is in the archive precisely because the file itself argued against the fix it needed.
- **Artifact layout is not portable.** MSVC's default CMake generator is *multi-config*
  (`build/Release/`, `build/Debug/`); Unix Makefiles is single-config (`build/`). Any consumer
  that hardcodes one layout is a latent bug — probe which directory actually holds the binary.
- **Python bindings load differently on each OS.** Since Python 3.8, `PATH` is *excluded* from
  extension-module dependency resolution on Windows (`os.add_dll_directory` is the mechanism);
  on Linux a bare-soname `DT_NEEDED` is invisible unless an rpath, a preload, or
  `LD_LIBRARY_PATH` names the directory; macOS links an rpath at build time. All three live in
  `src/EventProp.Native.Python/eventprop_c/_library_path.py`. Setting a variable that the
  loader ignores is how a platform ends up working only inside one caller.

## 2. What this project is

**EventProp** computes *exact* gradients through spike times for spiking neural networks: a
LIF neuron with synaptic filtering, differentiated with the adjoint method from control
theory. The algorithm has no numerical approximation of the spike — threshold crossings are
root-found, and the backward pass is a time-reversed adjoint sweep.

`src/EventProp.Python/` is the **reference library** and a **read-only git submodule**.
Everything else in the repository mimics its API and must reproduce its numbers to tolerance.
Treat it as a specification you may not edit: local build modifications are owned by
`scripts/submodule-state.sh` (§7), not by you.

Because two independent implementations of the same maths exist, **parity is a first-class
deliverable**, not a test detail: C vs Eigen, managed .NET vs C, pybind11 vs Cython, and both
Python backends against real trained checkpoints.

## 3. Project map

| Path | Language | Role |
|---|---|---|
| `src/EventProp.Python/` | Python + C++ (pybind11/Eigen) | Reference library — read-only submodule |
| `src/EventProp.C/` | C | `libeventprop_c_api`, batch-first public API |
| `src/EventProp.Cpp.Wrapper/` | C++ (Eigen) | Convenience copy of the reference core; source of the parity side |
| `src/EventProp.Native/` | C# | P/Invoke bindings (`LibraryImport`) + native `LIFLayer`/`LILayer` |
| `src/EventProp.Abstractions/` | C# | Shared `Layer` contract + data carriers, implemented by both backends |
| `src/EventProp/` | C# | Fully managed implementation of the reference library |
| `src/EventProp.Native.Python/` | Python + Cython | Python layer over the C library; also holds the paper-parity suite |
| `src/EventProp.C.Tests/` | C++ (GoogleTest) | C library unit tests |
| `src/EventProp.Tests/` | C# (TUnit) | Correctness + managed↔native parity |
| `src/EventProp.Native.ParityTests/` | C++ | Standalone C vs Eigen parity |
| `src/EventProp.Native.Benchmarks/` | C++ (Google Benchmark) | C vs C++ kernels; also `bench_rootfind` |
| `src/EventProp.Benchmarks/` | C# (BenchmarkDotNet) | managed vs P/Invoke |
| `src/EventProp.Python.Benchmarks/` | Python (pytest-benchmark) | pybind11 vs Cython; kernel suite emits the `impl: "C"` chain link |
| `src/EventProp.Web/` | Python (Streamlit) | "EventProp Visualizer"; not wired into `build.sh` |
| `scripts/bench/` | Python (stdlib only) | Cross-suite pipeline: run + adapt + merge + report |
| `cmake/` | CMake | `EventPropSimd.cmake`, `EventPropPstl.cmake` — shared decisions |

## 4. Layout

```
eventprop.net/
├── build.sh / build.ps1        # 9 steps, one contract, per-step exit codes
├── clean.sh / clean.ps1        # composable targets, --dry-run, real exit code
├── run_benchmarks.sh / .ps1    # cross-suite benchmark runner (NOT a default action)
├── Directory.Build.props       # warnings are errors for every .NET project
├── scripts/submodule-state.sh  # sole owner of the reference submodule's local state
├── scripts/bench/              # run_all.py + one adapter per tool + merge + report
├── cmake/                      # SIMD level, PSTL link input
├── bench-reports/              # COMMITTED per-platform reports (report_<tag>.md)
├── BENCHMARKS.md               # COMMITTED "latest run" copy
└── src/                        # see project map
```

## 5. Conventions

- **C#**: `net10.0`, root namespace `CW.EventProp`, nullable + implicit usings on. `LangVersion`
  is pinned once in `Directory.Build.props` — never per csproj — because it is derived from the
  TFM, so a lowered or multi-targeted TFM silently changes the language level. That file also
  owns warnings-as-errors. Tests use
  **TUnit** on Microsoft Testing Platform — `dotnet run`, never `dotnet test`, and do not
  re-add `Microsoft.NET.Test.Sdk` to "fix" discovery.
- **Warnings are errors everywhere.** C: the `werror` preset. C++/.NET: `werror`-equivalent
  flags, and `Directory.Build.props` sets `TreatWarningsAsErrors` for every .NET project
  (including analyzer diagnostics such as TUnit's). When a warning must be tolerated, scope it
  (`WarningsNotAsErrors`, `/wdNNNN`, `-Wno-…` on one target) — never drop the rule.
  Third-party code is not held to the rule, but a warning from it is silenced **by name on that
  target** rather than left to dirty the build: googletest's own source emits one
  `-Wcharacter-conversion` under clang/AppleClang, suppressed on the `gtest`/`gtest_main`/
  `gmock` targets in `src/EventProp.C.Tests/CMakeLists.txt` behind a `check_cxx_compiler_flag`
  probe. Probe the **positive** flag — compilers accept unknown `-Wno-…` spellings, so probing
  the negated form reports "supported" on a compiler that has never heard of the warning (GCC
  has none; MSVC takes no `-W` flags, so both get nothing). `SYSTEM` includes cannot reach it:
  it fires while googletest compiles its own source, where there is no consumer involved. A full
  nine-step build therefore emits zero warnings.
- **CMake floor 3.21**, one `CMakePresets.json` per project (never at the repo root — a
  configure preset's source dir is always its own directory). Presets are the supported way to
  drive the build; `mkdir build && cmake ..` survives only as a documented fallback. One
  standalone project (`src/EventProp.Cpp.Wrapper/benchmarks/`) has no presets — configure it by
  hand.
- **Coverage**: TUnit `--coverage --coverage-output-format=cobertura` + ReportGenerator.

## 6. The three actions

`--help` owns the flag lists; this section states only what `--help` cannot say.

### `clean.sh` / `clean.ps1`
- Targets **compose** (`c cpp parity`), an unknown target is **exit 2**, and removal failures
  are collected into a non-zero exit. Use `--dry-run` / `-DryRun` before `-full`.
- **No target ever touches `bench-reports/` or `BENCHMARKS.md`** — those are committed
  artifacts, not outputs. `results` stops at that boundary.
- A project `.venv` is not a build output and is never removed; a virtualenv's `__pycache__`
  is not either.
- `-full` also removes the reference binary, forcing a rebuild, and only takes effect when the
  `python`/`all` target is selected.
- After a clean, .NET `bin/obj` may reappear within seconds: that is the IDE's design-time
  build, not a failed clean. Check the clean log's `FAILED` lines instead.

### `build.sh` / `build.ps1`
- Nine steps in a **fixed dependency order**, because the consumers import a *pre-built*
  library out of a sibling project's `build/` directory: C library → Eigen core → C.Tests /
  parity / native benchmarks → Python layers → .NET. Configuring a consumer before its
  dependency fails with a `FATAL_ERROR` naming the missing library.
- **`ENABLE_ASAN` must be stated in both directions in every preset.** It is a cache variable:
  run `--preset asan` then `--preset release` without it and you get a silently
  ASan-instrumented *static* library that breaks the parity tests and poisons timings. Any
  option a preset toggles follows this rule.
- `werror` builds into `build-werror/` so it never clobbers the release build.
- **A failed step is reported, never swallowed.** No `|| true`, no output to `/dev/null`. The
  summary names failures *and* skips.
- `--python` / `-PythonExe` chooses the interpreter that receives the Python layers — and the
  one that will **run** the suites. On macOS with Homebrew Python and a system `python3`, name
  the interpreter explicitly; the built extensions are tagged for one version.
- Cython is a **build prerequisite** of the Cython step, not a dependency the build fetches;
  the step probes for it and names the interpreter that lacks it. `_native.c` is untracked and
  must be regenerated when `_native.pyx` is newer, or a "clean" build compiles stale C.
- On Linux: `libtbb-dev` is required (§7 PSTL), and the reference submodule needs
  `scripts/submodule-state.sh apply` — a bare `pip install` does not do it.
- Windows: `build.ps1` imports the vcvars environment via vswhere + a temp `.bat`, and
  **strips `Platform` from the process environment afterwards**, because MSBuild infers
  `$(Platform)` from it and would write `bin/x64/Release/…` where every consumer probes for
  `bin/Release/…`. Importing a vendor environment means auditing what it sets.

### `run_benchmarks.sh` / `run_benchmarks.ps1` — **not a default action**
Run it only when the user asks for benchmarks. It takes tens of minutes and **a subset run
rewrites the committed reports at the same path**: `--suites python` (or `--report-only`)
replaces `BENCHMARKS.md` and `bench-reports/report_<tag>.md` with single-suite data and exits
0. Scratch runs must pass `--no-reports --no-benchmarks-md` (`-NoReports -NoBenchmarksMd`).
- **Two scripts are named `run_benchmarks.sh`.** The repo-root one is this cross-suite runner;
  `src/EventProp.Native.Benchmarks/run_benchmarks.sh` runs the native C-vs-C++ suite alone and
  calls Google Benchmark's `compare.py`. Same name, two scopes, two directories — and the
  Windows legs are `.ps1` and `.bat` for the same reason.
- Prerequisites are **verified, not built**: executables exist in the right layout, the .NET
  benchmark dll is found, and `import eventprop_c._native` actually succeeds — not "the
  extension file exists".
- Results are tagged by platform (`darwin-arm64`, `linux-x86_64`, `windows-amd64`) and
  `merge.py` **refuses to merge across hosts**. Cross-machine comparison is a diff between two
  tagged reports; compare **per-pair geomeans, never the µs columns**.
- Every record carries provenance (`host`, `cpu`, `tool_version`, `vcs_rev`, `vcs_dirty`,
  `submodule_dirty`, `untracked`). `submodule_dirty: true` is **expected** — the read-only
  reference library must carry patches to compile at all.
- **A committed report is not a measurement of the current tree.** Check its `vcs_rev` and
  date; re-run before quoting it.
- **XL is heavy, and `min_time` does not save you.** Google Benchmark's `min_time` is
  *relative*, so a multi-second iteration still costs one full iteration: an XL burst
  `BackwardSpikes` row is ~16 s per iteration. Quick mode skips XL for that reason; keep it out
  of exploratory runs.
- Changing the BenchmarkDotNet version **invalidates every .NET number** historically compared.
  Measured effect between two BDN lines: ~8–12% on the headline managed/native `Ratio` and up to
  ~15% on nanosecond-scale individual rows, against a repeat-run noise floor of ~0.6% mean
  |delta| — roughly an order of magnitude above noise and fully reproducible, which is why the
  version band is pinned deliberately. Take an upgrade in one visible commit, re-baseline, and
  quote the version when quoting numbers: BenchmarkDotNet records
  `HostEnvironmentInfo.BenchmarkDotNetVersion` in every report. Don't adopt a preview first —
  preview → stable moves the numbers a second time.

### Tests
```bash
# C suite                # C++ core + C-API shim
cd src/EventProp.C.Tests && ctest --preset release
cd src/EventProp.Cpp.Wrapper/build && ./eventprop_cpp_tests && ./eventprop_capi_tests
# Parity (needs the release presets of EventProp.C and EventProp.Cpp.Wrapper first)
cd src/EventProp.Native.ParityTests && ./build/parity_test ../../EventProp.C/build/libeventprop_c_api.dylib
# .NET (TUnit/MTP: -c before --, TUnit flags after --)
cd src/EventProp.Tests && dotnet run -c Release --
# Python: Cython layer + paper parity, then the reference library's own suite
cd src/EventProp.Native.Python && PYTHONPATH="$PWD/../EventProp.Python:$PWD" python3 -m pytest test tests test_backward.py
cd src/EventProp.Python && PYTHONPATH="$PWD:$PWD/test" python3 -m pytest test   # the second entry is required: sibling-module import
# Visualizer, headless (no browser): numbers, then pages
cd src/EventProp.Web && .venv/bin/python selfcheck.py && .venv/bin/python smoketest.py
```
- The Visualizer imports the **pre-built** extensions of two sibling projects, so its
  interpreter must match the ABI tag they carry — the same question `build.sh` asks before
  deciding to rebuild. `.python-version` pins the interpreter and
  `src/EventProp.Web/eventweb/runtime.py` refuses a mismatch with a readable message.
  **How to build and start it is owned by `src/EventProp.Web/README.md`** (Setup / Run, both
  POSIX and PowerShell): the app is not a step in `build.sh`/`build.ps1`, so building the
  libraries and installing the app are two separate commands, and `build.ps1 -PythonExe`
  / `build.sh --python` must name the app's interpreter or step 7 installs the Cython
  layer somewhere the app cannot see.
  `selfcheck.py` is the only thing that checks the app's *science* (C library vs the
  accuracy the reference recorded for the same weights, exact gradient vs finite
  differences); nothing else in the repo touches `EventProp.Web`.
- `parity_test`'s library-path argument is **decorative** — the target links the library and
  never reads `argv`. Passing a different `.dylib`/`.so` proves nothing.
- On Windows the parity exe lands in `src/EventProp.C/build/Release/Release/` (its
  `RUNTIME_OUTPUT_DIRECTORY` *plus* the multi-config generator's config dir) and needs
  `eventprop_c_api.dll` beside it or on `PATH`. MSYS bash reports an executable whose dependent
  DLL cannot resolve as *"No such file or directory"* (exit 127), which reads like a missing
  build.
- A TUnit `--treenode-filter` pattern is exactly three segments `/Ns/Class/Method`; a
  four-segment or wrong-slot pattern is accepted and matches **zero** tests, reporting an empty
  suite. `--list-tests` shows real names.
- `src/EventProp.Native.Python/tests/test_paper_parity.py` is the **only suite
  that runs real trained weights** (the `src/EventProp.Python/paper/` checkpoints) through both
  backends. Synthetic fixtures miss what it catches.
- Managed-vs-native parity that fails **intermittently** is a marshalling problem, not a math
  problem. Reproduce with `DOTNET_GCStress=0x4 dotnet run -c Release --` (compacting GC forces
  relocations; ~8× slower, cheap enough to run ad hoc).
- The reference library's own suite is run by nothing in CI; it is the only suite checking the
  pybind11 backend against hand-written expectations.

## 7. Invariants

Each line is a rule you cannot see from the file it governs. Detail and evidence:
[`docs/POSTMORTEMS.md`](docs/POSTMORTEMS.md).

**Numerics and the shared contract**
- **Every spike list fed to the library is non-decreasing**, and times are **non-negative**.
  `eventprop_validate_input_no_v_th` enforces both (`60`, `7`) for `EventPropInput.times`; a
  post-spike list the caller owns (the backward pass's `SpikeOutput`, both diagnostic traces) goes
  through **`eventprop_validate_post_spikes`**, which adds the source bound against `n_post`.
  One helper on purpose: the consumers differ in cursor *direction* (forward current rebuild,
  backward adjoint sweep with an early break), but every one of them **silently truncates instead
  of failing**, so per-site loops drift — the voltage trace used to accept an unsorted list the
  lambda-I trace rejected, same field, same array. The Eigen reference has the same requirement
  and does **not** check it — it returns plausible garbage. C is the stricter *diagnostic*, not the
  divergent implementation; on sorted input the two agree to root-finder tolerance. Pinned by
  `ComputeSpikes.UnsortedTimesRejected` and `ComputeSpikes.EqualTimesAccepted` (equal times are the
  legitimate simultaneous-spike case), and on the post-spike side by
  `BackwardSpikes.UnsortedOutputSpikeTimesReturnsError` plus the `*Rejected` cases in
  `DiagVoltageTrace` / `DiagLambdaITrace`.
- The lambda-I sweep's **first decay is `exp(largest_time / tau_mem)`** — `previous_t` starts at
  `largest_time` and the newest spike's backward time is 0. It multiplies an all-zero `lambda_v`,
  so it is a no-op while finite and **NaN past ~709** (`0 * inf`), which then poisons every sample
  with `EVENTPROP_OK`. Output of `compute_spikes` can never reach it (`validate_input` bounds
  `max(times)/tau_syn`, `tau_syn < tau_mem`, output times never exceed the last input time), so
  the only way in is a **caller-built spike list running past the input span** — a replay or a UI
  cursor, which is what both bindings expose. `lambda_sweep_prepare` therefore rejects
  `largest_time / tau_mem >= EVENTPROP_EXP_OVERFLOW_THRESHOLD` (`13`). The Eigen reference
  overflows identically and says nothing; no parity fixture gets near the bound (~2.5), so the
  guard cannot split the backends on any input the reference can evaluate.
- `sum0[i]`/`sum1[i]` include every input spike up to `times[i-1]`, so interval *i*'s voltage
  model is valid only for `t >= times[i-1]`. Bracket floors must respect that in the **last**
  interval too — that is where spurious spikes come from.
- `v_th` is **not** validated on entry points that never read it (`ComputeSums`,
  `ComputeMaxima`, `backward_maxima`); the validator is split precisely so it can't drift.
- Finite-difference gradient tests must assert the scenario **fired** before comparing
  gradients and must not lean on NaN semantics (`assert_allclose` defaults to
  `equal_nan=True`, so a no-spike scenario makes both legs NaN and passes). Check the noise
  floor `~machine_eps / t_eps` before suspecting the math; where the analytical gradient is
  legitimately zero, `atol` is required, not cosmetic.
- Scalar vs SIMD builds must agree with the parity tolerances, not "roughly": a SIMD level
  mismatch between two builds being compared measures codegen, not the algorithm.

**Build-system invariants that are actually ABI invariants**
- **The Eigen core's arch flag is `PUBLIC`, not a performance knob.** `EIGEN_MAX_ALIGN_BYTES`
  is 32 with AVX/AVX2 and 16 without, the allocator is header-inlined, and the core is a
  STATIC library whose consumers destroy what it allocated — `PRIVATE` yields
  `STATUS_HEAP_CORRUPTION` in the *caller's* inlined destructor. Projects consuming the
  **prebuilt** core must restate the same level (`IMPORTED` targets carry no usage
  requirements).
- One SIMD level is decided once by `cmake/EventPropSimd.cmake` — a CPUID **and XGETBV** probe,
  because a CPUID bit says the hardware has it and XCR0 says the OS preserves the state. It is
  shared by four projects so the parity comparison compares like-built code. Detection is not
  enough: flag spelling differs (`-mavx2` vs `/arch:AVX2`, which `cl` ignores with only a
  warning), and x64 MSVC predefines neither `__SSE2__` nor `__AVX2__`. On Windows CMake maps `/arch:AVX2` to
  `<EnableEnhancedInstructionSet>` in the `.vcxproj`, so grep for *that* — grepping
  `arch:AVX2` finds nothing in a build that does have the flag.
- `cmake/EventPropPstl.cmake` returns **empty on Apple and Windows before any probe** — that is
  what those validated builds measure, and it is what makes it true that the module cannot
  change those platforms. On libstdc++ 12+, `<execution>` needs oneTBB linked, and a **shared**
  library with those symbols undefined still links and then fails to load. Everything past the
  gate is decided by compiling, because the answer differs on one machine between GCC/libstdc++
  and clang/libc++.
- **C11 atomics on MSVC need `/experimental:c11atomics` on top of `/std:c11`, and the build
  probes rather than version-gates** — a version test keeps passing the flag after a toolset
  drops it, and probing at the default standard reports "no atomics" about a toolset that has
  them.

**ABI**
- Every consumer of the C library compiles struct layouts **into its own binary** (Cython
  `cdef extern`, .NET `[StructLayout]`, C++ headers). A header change that moves an offset is
  invisible at compile **and** link time and returns plausible garbage. `eventprop_abi.c`
  exports the version and `sizeof` of each public struct; `_native.pyx` checks at import,
  `EventProp.Native` checks in layer constructors and at `Batch.ValidateFlat`, the shared
  choke point for the six friendly batch entry points.
  **Bump `EVENTPROP_ABI_VERSION` on any public struct layout change** — including appends,
  which change `sizeof`. The name list is duplicated in four consumers on purpose: adding a
  struct must fail a test in every consumer, so no consumer silently stops covering it. Pinned by
  `src/EventProp.Tests/AbiParityTests.cs`, `src/EventProp.Native.Python/test/test_abi.py` —
  which ctypes-loads the library and so compares offsets across binaries rather than
  tautologically — and `src/EventProp.C.Tests/test/abi_version.cpp`.
- `eventprop.h` is the source of truth for struct layouts and error codes; AGENTS.md and
  bindings are downstream of it.

**Memory and ownership (C)**
- Single `goto cleanup_and_return`; **one specific error code per allocation site**, never a
  generic one; heap allocations grouped into a context struct with a single free function.
- **Never two `realloc`s in a row without checking the first** — check, assign, then allocate
  the next, undoing the first on failure.
- All public free functions are in `eventprop_memory.c`, take their struct **by value**, and
  exist for every heap-backed type. Output free functions run only on the error path.
- Batch structs expose an atomic `has_errors` flag set with relaxed ordering (all threads store
  the same value; the parallel region's barrier provides visibility). Details belong to
  `error_no` and per-sample errors.
- `SpikeOutput` carries only `times` + `sources`. The synaptic current at each output spike is
  **rebuilt** by the backward pass with one **ascending** cursor, scaled once at the end —
  rebuilding it time-reversed loses ~6 significant digits per post spike at benchmark sizes.
  Post-spike error signals are a **caller-owned argument** (`post_spike_errors`), not a
  library-allocated field.

**Python bindings**
- Wrappers own the heap memory (`__dealloc__` calls the matching free function) and hand out
  **zero-copy numpy views**. Anything that hands a view to a caller must keep the wrapper alive
  as long as the view is readable — the ancestor layer's backward runs *after* the local result
  would go out of scope, so the result is stored on `self`.
- `abi_check()` runs at import and refuses to import on a mismatch: a stale extension is memory
  corruption, not staleness.
- Layers chain through `ancestor_layer`; the C# design has no counterpart (§10).

**C# / .NET**
- `Layer.Backward()` **accumulates** and never zeroes, matching Python (`ZeroGrad()` is the
  only zeroing). Pinned by `src/EventProp.Tests/LayerContractTests.cs`, which runs the same
  scenarios against **every**
  backend — a new backend must satisfy them all.
- `SpikesBatch`/`MaximaBatch` are `ref struct`s and cannot be cached on a field, so the native
  layers **recompute** the forward batch in `Backward()` (deterministic for the same weights).
- The C# chain is explicit `Layer.Chain<TA,TB>`; there is no `ancestor_layer` and no ancestor
  recursion — deliberate divergence from Python, not a missing feature.
- **A pointer taken from managed memory must be pinned for as long as the callee uses it.**
  `fixed` inside a helper that *returns* a pointer is the bug shape: the pin is gone before the
  native call. Span-based `EventPropInput.Create(...)` overloads return **UNPINNED** pointers
  by contract; callers own the pin.
- Hot paths are allocation-free by design: `MatrixView` (`ref struct` row-major views),
  `StridedSpan`, `ScratchBuffers`, and one growable `PostSpikeRow` per neuron — a neuron may
  cross threshold any number of times, so spikes cannot live at a fixed stride. `Grow` must
  copy live spikes (`ArrayPool.Rent` hands back dirty memory) and must **not** return
  superseded arrays to the pool while the sweep still caches them.

**Submodule (read-only)**
- Never edit `src/EventProp.Python`. `scripts/submodule-state.sh` is the sole owner of its
  local state: `status` / `apply` / `verify` / `check-patches` / `revert`. Patches apply with
  `git apply`, never `patch -p1`, never behind `|| true`.
- Two classes of local modification: per-**platform** toolchain patches (exactly one active)
  and `submodule-tests.patch`, which answers a question with the same answer on every OS and
  so is applied on all three. Fixing a cross-platform defect in one platform's patch is how
  the other two keep running the broken thing.
- **pybind11 is a nested submodule whose required version is platform-specific**, and a plain
  `git submodule update --init --recursive` silently downgrades it to something that cannot
  build. `submodule-state.sh apply` restores it; `check-patches` (a glob, so new patches are
  covered) tells you whether upstream moved — run it after any submodule update. A dirty
  `src/EventProp.Python` entry in `git status` is **expected** and never committed.

**Benchmark data contract** (three suites, one canonical generator, three copies)
- Sizes Tiny 10/5/20, Small 50/20/100, Medium 200/80/500, Large 500/200/1000 (+XL in native and
  .NET); `tau_mem 0.02`, `tau_syn 0.005`, `v_th 1.0`, **weights U[0.5, 2.0)** in all three.
- The generators must uphold: non-decreasing times (or the run measures a validation
  short-circuit), every spike inside `[0, 1)` (or `max_time/tau_syn` overflows the exponential
  and the call errors), and exactly `n_spikes` written (no zero-initialised leftovers piling up
  at `t = 0`).
- The three copies — `bench_main.cpp`, `SpikePatterns.cs`, `conftest.py` — **must stay in
  sync**; changing one pattern without the others silently changes what cross-suite ratios
  mean. Burst is deterministic and bit-identical across suites; uniform/poisson are identical
  only in distribution, so compare **ratios**, not counts.
- "poisson" is a misnomer shared by all three: times are exponential order statistics
  concentrated in tens of ms, not a homogeneous process over `[0,1)`.
- Rows with `output_spikes == 0` are legitimate but time an empty loop — exclude them from any
  geomean. Non-zero `c_error`/`c_fwd_error` counters mean the row measures an error path.

## 8. C library style essentials

- Strict warnings, zero tolerated (§5). Extract magic thresholds to named constants at the top
  of a file; use designated initializers `{ .field = value }`.
- MSVC specifics, all probe-derived rather than version-gated: the atomics probe (§7);
  `/wd4756` is deliberate because UCRT's `INFINITY` folds to a constant expression that
  overflows at every *use site*, and the suppression is by name so the rule stays meaningful;
  a SIMD build guards `dot_product_simd`'s unreachable scalar tail with the same `#if` as the
  vector declarations, because `/W4` reports C4702 and clang does not.
- ASan presets are Debug-only with `cleanFirst` (the library flips SHARED→STATIC in the same
  `build/` directory the test project hardcodes). `ASAN_OPTIONS=detect_leaks=1` is **not** in
  the presets: `gtest_discover_tests` runs the test binary at *build* time and inherits the
  configure preset's environment, and macOS ASan refuses `detect_leaks`. Supply it at run time.

## 9. Python bindings essentials

Thin Cython wrappers (`src/EventProp.Native.Python/eventprop_c/_native.pyx`) over the C library, with managed heap-backed
wrapper classes (`_SpikesBatchOutput`, `_MaximaBatchOutput`, `_AccumulatedBackwardResult`,
`_DropoutBatchOutput`) that own C memory and expose zero-copy views. The two bindings take
structurally different arguments **on purpose** (pybind11: a `SpikesVector` of 2D arrays;
Cython: flat arrays + `n_per_sample`) — in the kernel benchmarks that difference *is* the
measurement; don't "equalise" it. `compute_maxima_batch` requires `v_th` positionally even
though the maxima path never reads it; the layers pass `inf`.

## 10. .NET architecture essentials

`EventProp.Abstractions` holds the `Layer` contract as an **abstract class** (not an
interface) because the contract passes concrete reference types and shares state — mirroring
the reference library's `ABC`-based `layer.py`. Both backends implement it and the contract
suite is run against every backend, so a new backend cannot drift from the existing pair.
`EventProp.Native`'s friendly wrappers (`BatchFriendly.cs`) marshal flat arrays to an
`EventPropInput[]` on the stack so callers never build per-sample structs; loss-derivative
plumbing differs by loss type (spike loss reads a caller-supplied pointer array of pinned
managed errors; maxima loss reads the library-allocated `errors` inside the native
`MaximaOutput`, so the layer copies into it before the accumulate call).

The contract suite is `src/EventProp.Tests/LayerContractTests.cs` (each scenario loops every
backend, a `LayerBackend` record carrying the backend-specific plumbing); managed↔native layer
parity is `src/EventProp.Tests/NativeLayerParityTests.cs`. TUnit has **no public
`PropertySource`** — verified against 1.45 and 1.75, where the `PropertySource` strings in
`TUnit.Core.dll` are internal — and `MethodDataSource` on a non-static test needs
`[CombinedDataSources]` (TUnit0070), which is why the suite loops backends inside the test body
rather than parameterising.

## 11. Keeping this file honest

It rotted once, loudly, by owning facts it could not keep current. So:

- **No dates, no commit hashes, no test counts, no "validated on <platform> on <date>"
  claims.** Those are `git log`'s and commit messages' job. A count in a sentence is a
  countdown to being wrong.
- **One owner per fact.** If `--help`, a header, a CMake module or `BENCHMARK_SCHEMA.md`
  already states it, link it — do not restate it. Restated facts drift apart and then the doc
  actively misleads.
- **Any sentence naming a flag or target must be derivable from `--help`.**
- Prefer *rule + pointer to the file that owns the detail* over narrative. The narrative goes
  to `docs/POSTMORTEMS.md`.
- When you delete or rename something, grep for what vouched for it. Docs that justify a file's
  existence by naming a caller are the last thing to notice when it goes.
- If following this file produces a wrong result, fix **this file** in the same change and say
  so — that is the only mechanism that keeps it trustworthy.
