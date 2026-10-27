# AGENTS.md — EventProp

## Project Overview

This repository implements the **EventProp** algorithm — an event-based backpropagation method for training Spiking Neural Networks (SNNs). It computes exact gradients through spike times using the Leaky Integrate-and-Fire (LIF) neuron model with synaptic filtering, leveraging the adjoint method from control theory.

### Reference Library

**`EventProp.Python/`** is a git submodule of the original Python + C++ (Eigen/pybind11) EventProp library. It is the **reference implementation** — everything else in this repo should mimic its API and produce matching input/output.

### Project Map

| Project | Language | Role |
|---------|----------|------|
| **EventProp.Python/** | Python + C++ (Eigen/pybind11) | Reference library (git submodule, read-only) |
| **EventProp.C/** | C | Pure C reimplementation with a batch-first API (per-sample input arrays) |
| **EventProp.Cpp.Wrapper/** | C++ (Eigen) | Convenience copy of the C++ core. Used for native benchmarks and parity tests |
| **EventProp.Cpp.Wrapper/benchmarks/** | C++ (Google Benchmark) | **Deleted.** Was a standalone project (no presets, `project()` of its own) holding `eventprop_benchmarks` + `micro_benchmarks`, configured by nothing — no build script, no preset, no CI (the repo has none), only an AGENTS.md sentence saying "configure it by hand". It duplicated kernels `EventProp.Native.Benchmarks` already measures (and `bench_rootfind` already covers its root-finder micro-benchmarks) while carrying a **fourth** copy of the spike-pattern generators **without `SortTimesSources`**, so its `burst` rows measured unsorted input — which this file's own contract calls a different network, not a comparable workload. By the time it was noticed `eventprop_benchmarks.cpp` no longer compiled (3 calls on the pre-`spike_errors` `eventprop_backward_spikes`) and `micro_benchmarks.cpp` never did (missing `<functional>`, `BrentRootFinder` arity drift). Content last touched 2026-05-27 in a directory-rename commit. Its one measurement nothing else makes — Eigen-core thread scaling — can be ported into `bench_main.cpp` if anyone ever wants it |
| **EventProp.Native.Python/** | Python + Cython | Python layer using C library via Cython bindings (`eventprop_c/_native.pyx`) |
| **EventProp.Native/** | C# (.NET 10) | P/Invoke bindings wrapping EventProp.C via `LibraryImport` |
| **EventProp.Abstractions/** | C# (.NET 10) | Shared `Layer` contract + data carriers, implemented by both backends (no package deps) |
| **EventProp/** | C# (.NET 10) | Self-contained, fully managed .NET implementation of EventProp.Python |
| **EventProp.C.Tests/** | C++ (Google Test v1.15.2) | Unit tests for EventProp.C library (220 tests, ASan-compatible) |
| **EventProp.Tests/** | C# (.NET 10) | TUnit test suite for .NET projects (correctness + parity) |
| **EventProp.Benchmarks/** | C# (.NET 10) | BenchmarkDotNet project comparing native vs managed .NET libraries |
| **EventProp.Native.Benchmarks/** | C++ (Google Benchmark) | Benchmarks comparing EventProp.C vs EventProp.Python C++ code |
| **EventProp.Native.ParityTests/** | C++ (CMake) | Standalone parity tests comparing C vs Eigen implementations |
| **EventProp.Python.Benchmarks/** | Python (pytest-benchmark) | Benchmarks comparing EventProp.Python vs EventProp.Native.Python |
| **EventProp.Web/** | Python (Streamlit) | "EventProp Visualizer" — interactive forward/backward pass explorer. **Active; further work planned.** Has its own `requirements.txt`; not wired into `build.sh` |
| **EventProp.Python.GoogleBenchmarks/** | Python | **Deleted.** Was two standalone `benchmark*.py` scripts porting `bench_main.cpp` cases to `google_benchmark` in Python. Its data was **incompatible** with the shared contract (weights U[2.5, 10.0] vs U[0.5, 2.0], bypassed `LIFLayer`), so its numbers were not comparable to any other suite — see the data contract |

## Solution Structure

```
eventprop.net/
├── build.sh                    # Build all targets (C, C++, Python, .NET)
├── clean.sh                    # Clean all or individual targets
├── scripts/
│   └── submodule-state.sh      # Owns the read-only submodule's local modifications
├── src/
│   ├── EventProp.Python/       # Reference library (git submodule, read-only)
│   ├── EventProp.C/            # Pure C implementation (libeventprop_c_api)
│   │   ├── eventprop.h         # Public API + single-sample functions
│   │   ├── eventprop_validation.c  # Input validation + error strings
│   │   ├── eventprop_simd.c/h      # SIMD vector helpers (AVX2/NEON, exported non-static)
│   │   ├── eventprop_memory.c      # Public output-freeing functions
│   │   ├── eventprop_sums.c        # Cumulative exponential sums
│   │   ├── eventprop_maxima.c      # Maxima + voltage-based backward pass
│   │   ├── eventprop_spikes.c      # Forward pass (TOMS748/Brent root finding)
│   │   ├── eventprop_backward.c    # Adjoint (lambda) sweep backward pass
│   │   ├── eventprop_internal.h    # Shared constants, VOLTAGE macro, context types
│   │   ├── eventprop_batch.h/.c    # Batch-first API (per-sample input arrays)
│   │   ├── eventprop_diag.h/.c     # Diagnostic functions (voltage traces, lambda sweeps)
│   │   └── CMakeLists.txt          # Build config: strict warnings, SIMD, OpenMP, ASan
│   ├── EventProp.Cpp.Wrapper/  # C++ Eigen core (convenience copy, no pybind11)
│   ├── EventProp.Native/       # .NET P/Invoke bindings (C library) + native LIFLayer/LILayer
│   ├── EventProp.Abstractions/ # Shared Layer contract (SpikesData, *Parameters, WeightInitializer)
│   ├── EventProp.C.Tests/      # Google Test v1.15.2 suite for C library (220 tests)
│   ├── EventProp/              # Managed .NET implementation
│   ├── EventProp.Tests/        # TUnit test suite (correctness + parity)
│   ├── EventProp.Benchmarks/   # BenchmarkDotNet (native vs managed .NET)
│   ├── EventProp.Native.Python/  # Python + Cython bindings (C library)
│   │   └── eventprop_c/_native.pyx  # Thin Cython wrappers, zero-copy numpy views
│   ├── EventProp.Native.Benchmarks/  # Google Benchmark (C vs C++)
│   ├── EventProp.Native.ParityTests/ # Standalone parity tests (C vs Eigen)
│   ├── EventProp.Python.Benchmarks/  # pytest-benchmark (pybind11 vs Cython)
│   └── EventProp.Web/                # Streamlit visualizer
├── cmake/
│   ├── EventPropSimd.cmake    # one SIMD level (CPUID+XCR0 probe), shared by 4 projects
│   └── EventPropPstl.cmake    # what <execution> needs LINKED (probed; empty on Apple/Windows)
├── scripts/bench/              # cross-suite benchmark pipeline (see below)
├── coverage-report/            # Generated HTML coverage reports (gitignored)
└── AGENTS.md                   # This file
```

Each of the five preset-driven CMake projects also carries a `CMakePresets.json` (see [CMake Presets](#cmake-presets)).

## Conventions

- **Target framework**: .NET 10.0 (`net10.0`)
- **Root namespace**: `CW.EventProp` (library), `CW.EventProp.Tests` (tests)
- **Nullable**: enabled across all projects
- **Implicit usings**: enabled
- **Test framework**: TUnit 1.75+ (Microsoft Testing Platform) for .NET; Google Test v1.15.2 for C/C++ (`EventProp.C.Tests/`)
- **No `Microsoft.NET.Test.Sdk`**: TUnit runs on Microsoft Testing Platform and the suite is invoked with `dotnet run`, not `dotnet test`. The VSTest adapter is unused, and carrying it pinned the transitive graph to `Microsoft.TestPlatform.ObjectModel` / `TestHost` and `Microsoft.CodeCoverage` 17.x. Removing it leaves one top-level package, lifts MTP to 2.5.1 and CodeCoverage to 18.12.0, and leaves both the 372 tests and the `--coverage --coverage-output-format=cobertura` run intact. Do not re-add it to "fix" test discovery.
- **Coverage**: TUnit `--coverage --coverage-output-format=cobertura` + ReportGenerator
- **CMake**: floor is `3.21` in all six first-party CMake projects; five ship a `CMakePresets.json` (see [CMake Presets](#cmake-presets))

## Building

`build.sh` (macOS/Linux) and `build.ps1` (Windows) implement **one contract**, in the same
nine steps and the same dependency order the hardcoded import paths require (C library →
Eigen core → C.Tests / parity / native benchmarks → Python → .NET): presets, not
`mkdir build && cmake ..`; a per-step exit code; a failure list; a real exit code. Keeping
the two scripts the same shape is the point — a fix that lands in one and not the other is
how the two platforms end up measuring different things. Both build `EventProp.C.Tests`
and `EventProp.Benchmarks`; both regenerate `_native.c` when `_native.pyx` is newer (it is
untracked, so a fresh clone has none and an edited `.pyx` silently does not rebuild); both
install the Python layers into the interpreter that will **run** the suite
(`run_all.py` invokes every tool with `sys.executable`), treating a project `.venv` as a
note rather than a precondition.

```bash
# Build everything — Release mode
./build.sh release

# Build everything — Debug mode
./build.sh debug

./build.sh --clean                   # delete each build dir first (cache vars leak)
./build.sh --python python3.11       # which interpreter gets the Python layers
./build.sh --skip-python             # skip the reference library and the Cython bindings
./build.sh --skip-submodule          # skip scripts/submodule-state.sh apply

# Clean all build outputs
clean.sh              # removes everything (all targets)
clean.sh c            # only C library (build/ and build-werror/, plus EventProp.C.Tests/build/)
clean.sh cpp          # only C++ wrapper
clean.sh python       # Python package, Cython bindings and Python benchmarks
clean.sh dotnet       # only .NET projects (incl. EventProp.Abstractions)
clean.sh parity       # only parity tests
clean.sh benchmarks   # only native (C vs C++) benchmarks
clean.sh results      # bench-results/, BenchmarkDotNet.Artifacts/, baselines/*.json, caches
clean.sh -full        # also delete the reference binary, forcing a rebuild
clean.sh --dry-run -full            # print what -full would remove, remove nothing
clean.sh --python python3.11 python # which interpreter's pip uninstall runs
```

Targets **compose** — `clean.sh c cpp parity` cleans all three, and an unrecognised argument is a hard error (exit 2), not a silent no-op. `-full` takes effect only when `python`/`all` is selected, since the reference binary lives in the Python target.

`clean.ps1` is the **same contract**, not a parallel implementation: targets compose
(`.\clean.ps1 c cpp parity`), an unknown target is exit 2, every platform suffix is matched
for the compiled extensions (`.pyd`/`.dll`/`.so`/`.dylib`), `_native.c` is removed, removal
failures are collected into a non-zero exit, and `-DryRun` / `-PythonExe` match
`--dry-run` / `--python`. The two scripts had drifted in both directions, each correct only
on the platform the other covers, and both were silent about it — measured, not assumed; see
*Fixed — the two clean scripts had drifted apart, and each reported success*.

**No clean target ever touches `bench-reports/` or `BENCHMARKS.md`** — they are committed
artifacts. `results` stops at that boundary: it removes the regenerable measurements
(`bench-results/`, `BenchmarkDotNet.Artifacts/`, `baselines/*.json`, caches) that a
47-minute run produces, which is why it ships with `--dry-run`. A project `.venv` is not a
build output and is never removed.

**A failed step is reported, not swallowed.** The version of `build.sh` this replaced piped the three Python installs to `/dev/null` with `|| true`, and installed the Cython bindings **only if a project `.venv` existed** — while printing `→ eventprop_c (editable install)` unconditionally. On a machine with no venv, nothing was installed and the log said it was; the next thing to touch the missing extension was the benchmark suite, far from the cause. That is the same defect class as the old `build.ps1` (no exit checks at all) and the `patch -p1 … || true` submodule line, and all three are now measured rather than asserted.

**Linux prerequisites — both proved by the first Linux run failing.** `libtbb-dev`
(Debian/Ubuntu; `tbb-devel` Fedora, `onetbb` Arch): GCC 12+'s libstdc++ implements the C++17
parallel execution policies over Intel oneTBB, and `eventprop_eigen_core_stripped.cpp` uses
them, so without the dev package the three C++ consumer projects stop at link — see
*PSTL: what `<execution>` needs linked*. And **Cython in the interpreter that builds the
extension**: `setup.py` compiles the generated `_native.c`, and nothing in
`build-system.requires` fetches Cython, so Cython is a prerequisite of step 8 rather than of
the package. `build.sh` probes for it and names the interpreter that lacks it — see *Fixed —
the Cython step reported two failures for one missing package*.

**Windows** — `build.ps1` is the same contract for MSVC: presets, the same dependency order, per-step exit codes, and a failure list that ends in a real exit code.

```powershell
.\build.ps1 release                          # C -> Eigen core -> C.Tests -> Parity -> Benchmarks
.\build.ps1 debug                            # C.Tests skipped: its presets are release + asan
.\build.ps1 -Clean                           # delete each build dir first (flag changes, cache leaks)
.\build.ps1 -PythonExe python                # which interpreter gets the two Python layers
```

It builds `EventProp.C.Tests` and `EventProp.Benchmarks` as well; the .NET benchmark dll is a prerequisite `run_benchmarks.ps1` checks for, so a build script that omits it can never satisfy its own check. The Python layers install into the interpreter that will **run** the suite (`run_all.py` invokes every tool with `sys.executable`), not into a `.venv` nobody uses — which is why the interpreter is a parameter and is printed.

Two things the script has to do for the Python extension, both measured:

- **It imports the vcvars environment.** CMake finds MSVC by itself (the generator probes the registry), so steps 2–6 work in a plain shell; setuptools' own MSVC probe does not, so step 8 died with `DistutilsPlatformError: Microsoft Visual C++ 14.0 or greater is required` while the same `pip install -e .` succeeded in a shell where `vcvarsall` had been called. `Enter-VisualStudioEnvironment` locates `vcvarsall.bat` with **vswhere** (the path contains the product year and the Insider channel, so hardcoding it is a portability bug), runs it through a **temp .bat** (`cmd /c "call \"…\" && set"` as one quoted string is the classic quoting trap — exit 255), and imports the whole environment. It also **re-reads `PATH` from the registry** first: the process `PATH` is a snapshot from shell start, and a machine-PATH defect fixed in the registry is invisible to every shell opened before the sign-in that applies it — that is how vcvars died with `\NVIDIA was unexpected at this time` on a machine whose PATH had already been repaired.
- **It regenerates `_native.c` when `_native.pyx` is newer.** The extension compiles `_native.c`, Cython generates it from `_native.pyx`, and `_native.c` is **not tracked**. So a fresh clone has no `_native.c` at all, and an edited `.pyx` silently does not rebuild — the second is the stale-extension class the ABI check now refuses to import on (see *ABI: One Version Stamp, Checked by Every Consumer*).

### CMake Presets

All five first-party CMake projects declare `cmake_minimum_required(VERSION 3.21)` and **all five ship a `CMakePresets.json`** — `EventProp.C`, `EventProp.C.Tests`, `EventProp.Cpp.Wrapper`, `EventProp.Native.ParityTests`, `EventProp.Native.Benchmarks`. There used to be a sixth — `EventProp.Cpp.Wrapper/benchmarks/`, a standalone project with no presets that nothing configured; it is deleted, with the reasons in the Project Map. Presets are the supported way to drive all five; the hand-typed `mkdir build && cd build` sequences are kept only as a fallback.

Presets live **one per project**, not at the repo root: a configure preset has no `sourceDir` field, so its source directory is always the directory containing its own `CMakePresets.json`. Run preset commands from the project directory, and `cmake --list-presets` there to see what is available.

| Project | Configure / build presets |
|---|---|
| `EventProp.C` | `release`, `debug`, `asan`, `werror` |
| `EventProp.Cpp.Wrapper` | `release`, `debug` |
| `EventProp.C.Tests` | `release`, `asan` — plus matching `testPresets` for `ctest --preset` |
| `EventProp.Native.ParityTests` | `release`, `debug` |
| `EventProp.Native.Benchmarks` | `release` |

Every preset writes to `${sourceDir}/build`, which is what the cross-project imports hardcode: `EventProp.C.Tests`, `EventProp.Native.ParityTests` and `EventProp.Native.Benchmarks` all consume a **pre-built** library out of `EventProp.C/build` and `EventProp.Cpp.Wrapper/build`. Build order therefore matters:

```bash
# 1. C library, 2. Eigen core, 3. consumers
cd src/EventProp.C                 && cmake --preset release && cmake --build --preset release
cd ../EventProp.Cpp.Wrapper        && cmake --preset release && cmake --build --preset release
cd ../EventProp.C.Tests            && cmake --preset release && cmake --build --preset release && ctest --preset release
cd ../EventProp.Native.ParityTests && cmake --preset release && cmake --build --preset release
cd ../EventProp.Native.Benchmarks  && cmake --preset release && cmake --build --preset release
```

- `werror` (`EventProp.C` only) sets `CMAKE_COMPILE_WARNING_AS_ERROR=ON`, making the "code must compile with zero warnings" rule mechanical instead of advisory. It builds in `build-werror/` so it never clobbers the release build.
- `asan` presets set `ENABLE_ASAN=ON` with `CMAKE_BUILD_TYPE=Debug` and `cleanFirst: true`, because the library switches SHARED→STATIC inside the same `build/` directory the test project hardcodes.
- **Every non-ASan preset sets `ENABLE_ASAN=OFF` explicitly.** It is a cache variable, so running `--preset asan` and then `--preset release` leaves the cache at `ON` and the release build silently produces an ASan-instrumented **STATIC** library — which breaks the parity tests (they import a SHARED library) and poisons benchmark timings. The same rule applies to any option a preset toggles: state it in both directions, or the previous preset leaks through.
- Switching `asan` → `release` leaves the ASan `libeventprop_c_api.a` behind in `build/`. Harmless (the test project picks `.a` vs `.dylib` from `ENABLE_ASAN`), just stale; `clean.sh c` removes it.
- `CMakeUserPresets.json` is gitignored — use it for machine-local overrides.
- The floor is 3.21 for presets schema v3 (`condition`, `installDir`, `${hostSystemName}`) while staying inside stock distro packages (Ubuntu 22.04 ships 3.22, Debian 12 ships 3.25). Schema v4's `include` (3.23) would allow one root file sharing base presets across projects, but it costs Ubuntu 22.04 support, so it is not used.

### Submodule Patches

The reference submodule is **read-only**, and it needs local modifications to build on a current toolchain. `scripts/submodule-state.sh` is the single owner of that state:

```bash
./scripts/submodule-state.sh status          # what is applied, and what pybind11 is at
./scripts/submodule-state.sh apply           # idempotent; build.sh calls this
./scripts/submodule-state.sh verify          # non-zero exit if the submodule is not buildable
./scripts/submodule-state.sh check-patches   # do the patches survive upstream drift?
./scripts/submodule-state.sh revert          # pristine submodule (reference library will NOT build)
```

There are **two classes** of local modification, and the difference is which question the
patch answers: `submodule-<platform>.patch` answers a *toolchain* question, so exactly one of
them is active per machine; `submodule-tests.patch` answers a question that has the same answer
on every OS, so it is applied on all three.

| Patch | Platform | Fixes |
|-------|----------|-------|
| `submodule-macos.patch` | macOS (AppleClang + Homebrew) | Homebrew include paths, `-Xpreprocessor -fopenmp`, link args, NumPy deprecation |
| `submodule-linux.patch` | Linux (GCC/Clang) | Boost include path, `-fopenmp` link arg, remove C++17 `<execution>` parallel policies, NumPy deprecation, pybind11 bump |
| `submodule-windows.patch` | Windows (MSVC) | `setup.py` link args, `eventprop_cpp.cpp` / `eventprop_cpp.h` adjustments |
| `submodule-tests.patch` | **All platforms** | `test/test_loss_layer.py` TTFS numerical-gradient test: seed, explicit all-to-all weights, base times at 0.001, `t_eps` 1e-7, `atol` 1e-7 |

Patches are applied with `git apply`, never `patch -p1`, and never behind `|| true`.

**`submodule-tests.patch` is deliberately not parked inside a platform patch.** The defect it
fixes is identical on Linux and Windows — a reference-library test that compares NaN against a
legitimate `0.0` — and a per-platform file is where a bug goes to hide: fixing it in the macOS
patch alone would leave the other two platforms running the broken test while `status` on this
machine read "applied". See *Fixed — the reference library's own TTFS gradient test compared NaN
against 0.0* for the measurement, and why the fix is the five changes the project's Cython twin
already had rather than anything new.

It touches `test/test_loss_layer.py`, which no platform patch touches, so the two classes never
compete for a hunk. That is verified, not assumed: `check-patches` drift-checks every
`submodule-*.patch`, and each platform patch was applied **sequentially with** the tests patch in
a pristine worktree and reversed cleanly — macOS, Linux and Windows, all six directions.

**pybind11 is a nested submodule, and its required version is platform-specific.** It belongs to the same local-modification contract as the patches — it is not accidental drift:

| | pybind11 |
|---|---|
| recorded by the submodule | `54430436` — 2.6.3.dev1 (2021). **Does not build**: no NumPy 2 support, predates modern Clang/GCC and setuptools |
| required on macOS | `6079989c` — 3.1.0a0. No patch touches the gitlink, so `PYBIND11_REQUIRED_MACOS` in the script is the only record of it |
| required on Linux | `a2e59f0e` — 2.13.6. Carried inside `submodule-linux.patch` (it moves the gitlink), so the script reads it out of the patch |
| required on Windows | `a2e59f0e` — 2.13.6 (the Linux commit). No Windows patch touches the gitlink, so `PYBIND11_REQUIRED_WINDOWS` in the script is the only record of it. Before that constant existed the script's `*` arm returned `PYBIND11_RECORDED` and `apply` **checked out the broken version** — see *Fixed — Windows reference build: pybind11 had no required version, so step 1 checked out the one that cannot compile* |

A plain `git submodule update --init --recursive` silently downgrades pybind11 to 2.6.x and breaks the reference build. `scripts/submodule-state.sh apply` restores it.

**Drift test.** `check-patches` applies every `submodule-*.patch` (a glob, so a new patch file is
drift-checked the moment it exists rather than waiting for someone to remember a hardcoded list)
to a detached worktree at pristine submodule HEAD. On the live tree a foreign-platform patch reads as "blocked" because the active patch owns the same hunks, so only the worktree test separates "not for this platform" from "upstream moved". Run it after any submodule update.

The submodule is read-only — patches must be reapplied after pulling from upstream.

**No `.orig` / `.rej` litter, and that is a `git apply` property worth keeping.** The submodule
carried `setup.py.orig`, `src/eventprop_cpp.cpp.orig`, `eventprop/loss_layer.py.orig` and
`src/pybind11.rej` (dated Jun 18) — artifacts of the `patch -p1` era this script left behind.
`git apply` creates neither backups nor reject files, so nothing regenerates them, and their
presence is misleading: they read as "a hunk failed here" when the tree is in fact fully
patched. Removed after verifying each was a subset of the current patched file (the only line
missing from an `.orig` was the line the patch replaces) and that the `pybind11.rej`'s demanded
commit — `a2e59f0e`, which the rejected `setup.py` hunk could not apply — is what
`src/pybind11` actually sits on. `submodule-state.sh verify` and `check-patches` pass after the
removal, and `verify` is the check that would catch a genuinely lost hunk.

## Running Tests & Coverage

### .NET Tests (TUnit on .NET 10)

```bash
# All tests — Release mode (required for parity tests)
cd src/EventProp.Tests && dotnet run -c Release --

# Filtered — the tree is /Namespace/ClassName/MethodName
cd src/EventProp.Tests && dotnet run -c Release -- --treenode-filter "/*/*/BackwardSpikesTests/*"
```

> **Note**: TUnit uses Microsoft Testing Platform (MTP), not VSTest. `dotnet test` does NOT work. The `-c` flag goes BEFORE `--`, TUnit flags go AFTER `--`. Use `--treenode-filter` instead of `--filter`.
>
> **Note**: a treenode pattern is exactly three segments — `/Namespace/ClassName/MethodName`. A four-segment pattern, or a method-name fragment in the class slot, is **accepted and matches nothing**, reporting `total: 0` as if the suite were empty. `"/*/*/BackwardSpikes/*"` is the classic instance and matches 0 tests; the working form is `"/*/*/BackwardSpikesTests/*"` (6 tests). A filter returning 0 is a pattern bug, not an empty suite. `--list-tests` shows the names.

### C Library Unit Tests (Google Test)

```bash
# Presets (preferred)
cd src/EventProp.C.Tests
cmake --preset release && cmake --build --preset release
ctest --preset release            # 220 tests; gtest_discover_tests registers them with CTest
./build/eventprop_tests           # or run the binary directly
```

```bash
# Manual equivalent
cd src/EventProp.C.Tests && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./eventprop_tests
```

**With AddressSanitizer** (Debug builds only, detects use-after-free, buffer overflows):
```bash
cd src/EventProp.C        && cmake --preset asan && cmake --build --preset asan
cd ../EventProp.C.Tests   && cmake --preset asan && cmake --build --preset asan
ctest --preset asan
```
> **Note**: On macOS, ASan automatically builds a static library (`.a`) instead of shared (`.dylib`) due to AppleClang linker limitations. The `asan` build presets use `cleanFirst` because the library flips SHARED→STATIC in the shared `build/` directory.

> **Note**: `ASAN_OPTIONS=detect_leaks=1` is deliberately **not** in the presets. `gtest_discover_tests` runs the test executable at *build* time and inherits the configure preset's `environment`; macOS ASan aborts on `detect_leaks` ("not supported on this platform"), which fails discovery and takes the whole build down. On Linux, supply it at run time: `ASAN_OPTIONS=detect_leaks=1 ctest --preset asan`.

### C/C++ Parity Tests (C vs Eigen)

```bash
cd src/EventProp.Native.ParityTests
cmake --preset release && cmake --build --preset release
./build/parity_test ../../EventProp.C/build/libeventprop_c_api.dylib
```
> **Note**: requires the `release` presets of `EventProp.C` (SHARED library) and `EventProp.Cpp.Wrapper` (Eigen core) — both are imported from their `./build` directories at configure time, so configuring before them fails with a `FATAL_ERROR` naming the missing library.

> **Note (Windows)**: the exe lands in `src/EventProp.C/build/Release/Release/` — its `RUNTIME_OUTPUT_DIRECTORY` plus the multi-config generator's config dir, not `ParityTests/build`. It links `eventprop_c_api.dll`, and MSYS bash reports a Windows exe whose DLL cannot be resolved as **"No such file or directory" (exit 127)**, which reads like a missing build product. Run it from a directory containing the DLL, or with that directory on `PATH` (a native process honours `PATH`; a Python extension does not — see *Python bindings on Windows*).

**The runnable Windows command** — verified 105/105, and printed by `build.ps1` at the end of a successful build so it is discoverable where the build left the artifacts (mirroring `build.sh`'s `Run parity:` line). From the repo root:

```powershell
$env:PATH="$PWD\src\EventProp.C\build\Release;$env:PATH"
& "src\EventProp.C\build\Release\Release\parity_test.exe"
```

**The library-path argument is decorative.** `parity_main.cpp` mentions `argv` exactly once — its own `main` signature — because `parity_test` links `eventprop_c_api` as an `IMPORTED SHARED` target, so the *loader* chooses the DLL and `PATH` decides. The `../../EventProp.C/build/libeventprop_c_api.dylib` argument in the macOS command above is ignored for the same reason; it survives from the era before the imported target. `run_parity.bat` is **deleted** for this whole class of reason: three lines that hardcoded `G:\eventprop.net`, were referenced by no script, doc or CI file, passed that dead argument, and never put the DLL directory (one level *above* the exe) on `PATH` — so it could not have worked even on the machine it named.

### Wrapper C++ Tests (`eventprop_cpp_tests`, `eventprop_capi_tests`)

`EventProp.Cpp.Wrapper` builds two Google-Test-free standalone binaries (plain `main`,
`Results: N/N passed`, exit code = failures). `build.sh`/`build.ps1` build them and do not
run them, so this is the command:

```bash
cd src/EventProp.Cpp.Wrapper/build/Release      # macOS/Linux: ./eventprop_cpp_tests etc.
PATH="/path/to/src/EventProp.C/build/Release:$PATH" ./eventprop_cpp_tests    # 25/25, Eigen core
PATH="/path/to/src/EventProp.C/build/Release:$PATH" ./eventprop_capi_tests   # 7/7, the C shim
```

`eventprop_cpp_tests` exercises the Eigen core; `eventprop_capi_tests` exercises
`eventprop_c_api.cpp`, the flat-array shim over it — **not** the shipped `EventProp.C`
library, which is a different implementation with its own 220 tests. On Windows both are
classic `RUNTIME_OUTPUT_DIRECTORY` exes next to their `.lib` outputs, and `PATH` must name
the directory holding `eventprop_c_api.dll`.

### Reference Library's Own Tests (read-only submodule)

```bash
# 14 tests. NOTE the second PYTHONPATH entry — it is not optional.
cd src/EventProp.Python
PYTHONPATH="$PWD:$PWD/test" python3 -m pytest test -q
```

> **Note**: three of the four modules do `from test_lif_layer import ...`, a sibling-module import.
> `test/__init__.py` exists, so pytest imports them as `test.test_*` and that sibling name is
> unreachable — every one of them then errors at **collection** with
> `ModuleNotFoundError: No module named 'test_lif_layer'`. That is an invocation detail of the
> upstream layout, not a library defect, and it is why `test/` goes on `PYTHONPATH` alongside the
> package root. Do not "fix" it by editing the read-only submodule or deleting its `__init__.py`.

> **Note**: this suite is the reference implementation's own upstream tests. It is not run by
> `build.sh`, `run_benchmarks.sh` or `scripts/bench/run_all.py` — it is here because it is the only
> suite that exercises the pybind11 backend against hand-written expectations, and because one of
> its tests was silently measuring nothing until `submodule-tests.patch`; see *Fixed — the reference
> library's own TTFS gradient test compared NaN against 0.0*.

### Native Benchmarks (C vs C++) — Standard Workflow

**Use the benchmark scripts** (`run_benchmarks.bat` on Windows, `run_benchmarks.sh` on Linux/macOS). These handle building, running both benchmarks, saving JSON to `baselines/`, and invoking Google Benchmark's `compare.py` for human-readable comparison reports.

```bash
# Build first (if not already built) — presets preferred
# Requires the release presets of EventProp.C and EventProp.Cpp.Wrapper first
cd src/EventProp.Native.Benchmarks && cmake --preset release && cmake --build --preset release

# Manual equivalent
cd src/EventProp.Native.Benchmarks && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .

# Run full suite and compare (Windows)
cd src/EventProp.Native.Benchmarks
cmd /c "run_benchmarks.bat"

# Run full suite and compare (Linux/macOS)
cd src/EventProp.Native.Benchmarks
./run_benchmarks.sh
```

**Script modes:**

| Command | Action |
|---------|--------|
| `run_benchmarks` (no args) | Run C + C++, save JSON, auto-compare latest results |
| `run_benchmarks c` | Run only C benchmark |
| `run_benchmarks cpp` | Run only C++ benchmark |
| `run_benchmarks compare` | Compare last saved results (no re-run) |
| `run_benchmarks compare A.json B.json` | Compare two specific JSON files |

**Environment variables** (set before calling):

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_TIME` | `0.1s` | Minimum benchmark time per test (`--benchmark_min_time`) |
| `FILTER` | *(none)* | Regex filter for benchmarks (`--benchmark_filter`) |

**Example — quick run of a single benchmark family:**
```bash
# Windows
cmd /c "set MIN_TIME=0.05s & set FILTER=ComputeSpikes/Tiny & run_benchmarks.bat"

# Linux/macOS
MIN_TIME=0.05s FILTER=ComputeSpikes/Tiny ./run_benchmarks.sh
```

**Output:**
- JSON results saved to `src/EventProp.Native.Benchmarks/baselines/bench_c_<id>.json` and `bench_cpp_<id>.json`
- Comparison report printed to stdout via Google Benchmark's `compare.py`
- Results are gitignored (`baselines/*.json`) but scripts are tracked

> **Note**: Full suite runs can take several minutes. Use `FILTER` for quick validation of specific benchmark families.

**Benchmark data contract** (`bench_main.cpp`): every generator ends with `SortTimesSources`, because the C library rejects non-decreasing input spike times (`EVENTPROP_UNSORTED_SPIKE_TIMES`, 60) while the Eigen reference does not validate. `GenerateBurstData` fills its arrays neuron by neuron, so its bursts interleave — without the sort the C side returns 0 spikes in ~0 ns and the C-vs-C++ comparison measures a validation short-circuit (it did: `ComputeSpikes/Small/burst` reported 36 ns against the Eigen side's 83 µs). The C benchmarks also publish `c_error` / `c_fwd_error` / `c_bwd_error` counters; a non-zero value means the row measures an error path, not the algorithm. Rows with `output_spikes == 0` are legitimate (high threshold, no neuron fires) but time an empty loop — `BackwardSpikes/Tiny|Small/uniform` are in that regime, so their ratios are overhead-on-noise and should be excluded from any geomean.

**Burst at Medium/Large/XL** is registered behind `#ifndef _MSC_VER` for `ComputeSpikes`, `BackwardSpikes` and `FullTrainingStep`, and unconditionally for `ComputeSums` (which has no root finding, so the Toms748 caveat never applied). Probed on macOS: both backends complete with matching spike counts (Medium 10669, Large 58287, XL 771544) and `c_err = 0`. Burst is the heaviest workload in the suite — XL produces ~770k output spikes against ~56k for XL poisson — so it is the best stress on the forward pass's per-neuron spike-row growth. XL costs ~16 s per iteration for `BackwardSpikes`, so keep it out of quick runs (`FILTER` excludes `XL`).

### Python Benchmarks (pytest-benchmark)

```bash
cd src/EventProp.Python.Benchmarks && pytest test_benchmarks.py -v
```

`test_kernel_benchmarks.py` is the separate suite that measures pybind11 vs Cython on the
raw C kernels with **no layer** — it isolates binding overhead, which the layer suite
fuses into the layer number and cannot separate. It also emits `impl: "C"` kernel records,
which is what closes the cross-suite chain (Eigen ↔ C ↔ Managed ↔ pybind11/Cython).

```bash
cd src/EventProp.Python.Benchmarks
PYTHONPATH=../EventProp.Python:../EventProp.Native.Python \
  pytest test_kernel_benchmarks.py --benchmark-only --benchmark-json=results_kernels.json
```

The two bindings take structurally different arguments **on purpose** — pybind11 wants a
`SpikesVector` of 2D arrays, Cython wants flat arrays plus `n_per_sample`. That difference
is the measurement; equalizing it would destroy the benchmark. `compute_maxima_batch`
requires `v_th` as a 9th positional argument even though the maxima path never reads it;
`LILayer` passes `float('inf')` and the kernel benchmarks match that.

### Cross-Suite Report (one command)

```bash
./build.sh release
cd src/EventProp.Native.Benchmarks && cmake --preset release && cmake --build --preset release && cd ../..
scripts/bench/run_all.py --quick          # Tiny/Small only
scripts/bench/run_all.py                  # full run, includes XL
```

`run_benchmarks.sh` (macOS/Linux) and `run_benchmarks.ps1` (Windows) wrap that runner with
the same contract: they verify the prerequisites by checking what the suites actually need
(native executables in `build/`, `build/Release/` and `build/Debug/`; the .NET benchmark
dll; the C library; and an **actual `import eventprop_c._native`**, not "the extension file
exists"), refuse to run when something is missing rather than building it silently, pass
through `--suites` / `--tag` / `--out` / `--sweeps` / `--reports-dir` / `--benchmarks-md`,
and print the committed-report line only when the runner exited 0 **and** the file exists.
The import probe is the load-bearing check: it runs with the same `PYTHONPATH` the runner
uses, and on Windows the extension cannot load without `os.add_dll_directory` (PATH is
excluded since Python 3.8), so a file-existence check reported ready while the suite could
not start.

**Two scripts share the name `run_benchmarks.sh`.** The repo-root one is this cross-suite
runner; `src/EventProp.Native.Benchmarks/run_benchmarks.sh` runs the native C-vs-C++ suite
alone and calls Google Benchmark's `compare.py`. Same name, two scopes, two directories.

**A subset run rewrites the committed report at the same path.** `--suites native` (or
`--report-only`) merges whatever records are in `--out` and writes
`bench-reports/report_<tag>.md` **and** `BENCHMARKS.md` from that subset — a one-suite
report replacing the three-suite one, with a green exit. Measured: a scratch
`--report-only --out <scratch>` overwrote both committed files and exited 0. Both wrappers
now print that warning before running; the escape hatch is `--no-reports --no-benchmarks-md`
(PowerShell: `-NoReports -NoBenchmarksMd`), and that is what a scratch run should use.

Writes `bench-results/records_<tag>_*.json`, `merged_<tag>.json`, `report_<tag>.md`, and
regenerates `BENCHMARKS.md` plus `bench-reports/report_<tag>.md`. `<tag>` is the platform —
`darwin-arm64`, `linux-x86_64`, `windows-amd64` — taken from the same
`platform.system()/machine()` fields the adapters put in `meta.host`, so a filename prefix
and the host string that `merge.py`'s cross-host guard compares cannot disagree. Results
from several machines therefore coexist in one directory without overwriting each other;
`--tag <tag>` selects which set `--report-only` merges, and a tag with no records prints the
expected filenames rather than a bare "no record files".

**What is committed, and what is not:**

| Path | Git | Why |
|---|---|---|
| `bench-results/` (records, `merged_<tag>.json`) | **ignored** | machine-specific, and `merge.py` refuses to merge across hosts — a committed record file is a measurement nobody on another machine can use |
| `bench-reports/report_<tag>.md` | **committed** | the report is the artifact that survives being read on a different machine; deterministic (byte-identical across runs of the same records) and ~24K |
| `BENCHMARKS.md` | **committed** | "latest run" convenience copy, written from the same `merged_<tag>.json` as the tagged report, so the two cannot drift |
| `EventProp.Native.Benchmarks/baselines/*.json` | ignored | the native suite's own raw output |

Tagged reports are what make the **cross-machine comparison a diffable artifact**: a Windows
run and a macOS run coexist as `report_windows-amd64.md` and `report_darwin-arm64.md`
instead of overwriting one file. Compare their per-pair geomeans; never the µs columns.

**Committed reports go stale, so the report says what it describes.** `merge.py` aggregates
`date`, `vcs_rev` and the vcs flags from record meta into top-level provenance, and the report
header prints them above a "this file is generated and goes stale" banner. A committed report
at an old `vcs_rev` is not a measurement of the current committed state — re-run before
quoting it. Multiple `vcs_rev` or `date` values render with a "records span multiple
commits/dates" note, which means the record set was not one run.

**`vcs_dirty` means tracked modifications, and that is a recent change.** It used to be
`bool(git status --porcelain)`, which is true on every machine that can run the python suite:
the read-only reference submodule must carry local patches to compile at all
(`scripts/submodule-state.sh` owns them), and untracked tool output counts too. A flag that is
true on every run cannot gate the "not a committed state" claim the banner makes — the
committed darwin report says `vcs_dirty: True` for exactly this reason, not because anyone was
editing source. The three signals are now separate fields, aggregated to a tri-state
(`true` / `false` / `mixed`) where applicable: `vcs_dirty` (tracked only), `submodule_dirty`
(expected `true`; stated as a fact, not a warning) and `untracked` (a count). All three come
from `scripts/bench/provenance.py`, shared by every adapter, and are `null` when no git
repository is reachable — the absence of a repository is not a clean tree. `provenance.py`
pins its git calls to the superproject top level: measured, run from `scripts/bench`,
`git status` reports the submodule as `../../src/EventProp.Python` while `git submodule status`
reports `src/EventProp.Python`, so the path comparison missed and `submodule_dirty` read
`false` from a subdirectory and `true` from the root — same repository, same state, two
answers. Records written before the split carry the old definition and lack the two new
fields; `merge.py` omits those lines rather than inventing them.

Prerequisites are **not** auto-built — a silent build here would hide a compile failure as a
benchmark result. See `BENCHMARK_SCHEMA.md` for the record format,
the validity rules, and the provenance each tool does and does not supply.

Flags: `--quick` (Tiny/Small, skips XL), `--suites native dotnet python`, `--report-only`
(re-merge existing records), `--out DIR`, `--tag TAG`, `--reports-dir DIR` (pass `''` to
skip), `--benchmarks-md PATH` (pass `''` to skip). A suite that fails is reported and the
run continues with the other suites' data, exiting non-zero.

The pipeline is six Python scripts in `scripts/bench/` (stdlib only — no pip install):
`run_all.py` (orchestrator), one adapter per tool (`from_google_benchmark.py`,
`from_benchmarkdotnet.py`, `from_pytest_benchmark.py`), `merge.py` (join + validity),
`report.py` (tables). They replace the five ad-hoc `print_*.py` scripts in
`EventProp.Python.Benchmarks/`.

**XL is not the slow part.** Google Benchmark's `min_time` is relative, so a 14 s
iteration takes the minimum of ONE iteration. Measured on macOS/arm64: native full suite
incl. XL is 162 s per binary; the ~30 min full run is dominated by .NET's 75 benchmarks.

**That dominance is per-platform, and on Windows it is inverted.** Full run, Windows/x64
(10 cores), 2026-10-16 at `8ef41a9`: native **7m 04s** (`bench_c` 3m 22s, `bench_cpp`
3m 40s), dotnet **16m 07s** (95 benchmarks, BDN global total 16m 05s), python
**22m 26s** (96 kernel benchmarks), total wall clock **46m 40s**. The python suite is the
longest on Windows — 40% over dotnet, and ~11x its macOS wall clock (~2 min). Not
diagnosed; the number is the finding. Quote timings per platform, never as one number.

### Cross-platform status of the benchmark pipeline

**Validated on macOS/arm64 and Windows/x64. On Linux the builds and every test suite pass;
the benchmark pipeline itself has still not been run there.** The Linux validation
(2026-10-20, WSL2, GCC 13.3, cmake 3.28.3, .NET 10.0.112, Python 3.12.3) is
`./clean.sh -full && ./build.sh release` ending green with C 220/220, parity 105/105, .NET
382/382 and `eventprop_c` + paper-parity 33 passed / 6 Windows-only skips. It found four
defects, all under *Fixed Defects*: the missing TBB link input, the Cython step reporting two
failures for one missing package, the extension with no rpath on Linux, and step 7 printing
"already built" over an unloadable reference binary.

The Windows validation is a full run (all three suites, XL included, 46m 40s) producing
`bench-reports/report_windows-amd64.md`; the per-pair geomeans against
`report_darwin-arm64.md` are in that commit's message. It is also the run that found the
UTF-8/cp1252 adapter defect (see *Fixed — BenchmarkDotNet adapter*), the relative-`--out`
defect, and the Python DLL-loading defect (see *Python bindings on Windows*).

**Fixed — platform-dependent paths now resolve at runtime:**

- **Native binaries.** `native_build_dir()` probes `build/`, `build/Release/`, and
  `build/Debug/` for the actual binaries, preferring `build/Release` on Windows. MSVC's
  multi-config generator writes `bench_c.exe` to `build/Release/`; Unix Makefiles writes
  `bench_c` to `build/`. Detection is by which directory contains the exe, so a
  single-config Windows build (MSYS2/MinGW) also resolves. `native_exe_name()` appends
  `.exe` on Windows. Verified by simulating an MSVC tree.
- **`PYTHONPATH` separator.** Was a hardcoded `:` — on Windows (`os.pathsep` is `;`) that
  silently made the second entry unimportable, so the reference submodule import would fail
  with a confusing `ModuleNotFoundError`. Now `os.pathsep.join(...)`.
- **Library path.** Sets `DYLD_LIBRARY_PATH` (macOS) and `LD_LIBRARY_PATH` (Linux), both
  of which work, and prepends the C build dir to `PATH` (Windows) — which **does not work
  for the Cython extension**, and never did: since Python 3.8, CPython loads extension
  modules with `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, whose search set is the extension's own
  directory, the process directory, the system directories, and directories registered with
  `os.add_dll_directory`. **PATH is excluded.** Measured on the same directory both times:
  PATH pointing at the one holding `eventprop_c_api.dll` gives `ImportError: DLL load failed
  while importing _native`; `os.add_dll_directory` on it imports. The registration lives in
  `eventprop_c/_library_path.py` and runs at package import. The `PATH` entry in
  `run_all.py` is kept for non-Python consumers of that directory (the native benchmark
  exes, which do honour it) and is not what makes the import work. The extension embeds an
  **rpath at build time** (`setup.py`) — on macOS and, since the first Linux run, on Linux
  too; `_native.pyx`'s comment used to claim the rpath covered every platform, which is what
  made `import eventprop_c` fail on Linux behind `run_all.py`'s back. See *Linux: finding the
  C library (rpath, then preload)*.
- **.NET binary path.** `bin/Release/net10.0/EventProp.Benchmarks.dll` is
  framework-dependent and uses the **same layout on Windows and Linux** — no branch needed.
  Stated here because it looks like the kind of path that would need one.

**Linux CPU label: fixed.** `from_google_benchmark.py`'s `_cpu_label` used `platform.mac_ver()`
and a macOS-only `sysctl` helper, so on Linux `meta.cpu` read `"8 cpu @ M MHz"` instead of a
brand string — non-comparable with the other two hosts' records for the same machine class.
`_linux_cpu_brand()` now reads `/proc/cpuinfo` (`model name`, then ARM's `Hardware`), and
`_cpu_label` dispatches on `platform.system()` rather than on `mac_ver()`. That gate was a
latent bug, not a style choice: `platform.mac_ver()` returns a populated tuple on some Linux
distributions, and the old code then ran `sysctl -n machdep.cpu.brand_string` on a Linux
machine to find nothing. Brand strings do not carry a core count, so the label no longer says
"N cpu @ M Hz" once a brand is found — which is the point: `Intel(R) Core(TM) i9-9900K CPU @
3.60GHz` is 1 core at 3.6 GHz as that string reads on macOS, and a count of 16 would have been
the better label. Windows deliberately gets no new brand reader; it keeps its existing
`mhz_per_cpu` fallback and its `report_windows-amd64.md` numbers stay comparable to
themselves. Verified on this box: brand resolves to `12th Gen Intel(R) Core(TM) i7-12700K`,
and a Linux record generated from the real `bench_c`/`bench_cpp` binaries carries
`host: linux-x86_64`, that `cpu`, `tool: 1.9.5`.

**Prerequisite on Linux:** the Python suite needs the reference submodule built, which needs
`scripts/submodule-state.sh apply` (pybind11 `a2e59f0e` / 2.13.6 on Linux, carried inside
`submodule-linux.patch`). `./build.sh release` does this; a bare `pip install` does not.

**Unchanged by design, and the important one:**

- **`merge.py` refuses to merge records from different hosts** — cross-machine absolute
  times are not comparable. A Windows run and a Linux run produce **two separate reports**,
  never one merged table. `--allow-mixed-hosts` yields ratios only. Run the pipeline once
  per machine and compare the **per-pair geomeans**, not the µs columns.
- Provenance is what makes that safe: every record carries `host`, `cpu`, `tool_version`,
  `vcs_rev`, `vcs_dirty`, `submodule_dirty` and `untracked`. Check them before comparing
  two reports.

**Known-good on macOS and Windows:** the CMake presets, `build.ps1` + `run_benchmarks.ps1`
(Windows), `scripts/submodule-state.sh` (runs under Git bash on Windows too), and the .NET
projects. The benchmark *suites* are portable; the *runner's* path assumptions were not,
and are what the Windows run fixed.

`build.sh` has had its first real Linux run and it ended green. Its mechanics were also
exercised on Windows — every CMake, .NET and pip step ran through the new reporting, and the
Python step failed there for Windows reasons (setuptools' vcvars probe under a stale shell
PATH; MSVC env is `build.ps1`'s job) — which is a demonstration that the reporting works, not
that the script is right on Windows. The Linux run is the one that turned "the reporting works"
into "the build is right here": it named four real defects with exit codes instead of printing
"Build complete", and they are all fixed (see *Fixed Defects*).

`run_benchmarks.sh` is newer still. Its prerequisite probes were reproduced by hand on Linux —
every artifact it looks for is present and its `import eventprop_c._native` probe passes — but
the runner itself has never executed the pipeline on any platform from this script.

**To get benchmarks after cloning on Windows:**

```powershell
.\build.ps1 release            # presets, in dependency order, per-step exit codes
.\run_benchmarks.ps1           # quick (Tiny/Small, skips XL)
.\run_benchmarks.ps1 -Full     # full run incl. XL: ~47 min, python is the long one
# scratch run that must not touch the committed BENCHMARKS.md / bench-reports\
.\run_benchmarks.ps1 -Suites python -Out ..\bench-test -NoReports -NoBenchmarksMd
```

**To get benchmarks after cloning on Linux or macOS:**

```bash
./build.sh release                 # presets, dependency order, per-step exit codes
./run_benchmarks.sh                # quick (Tiny/Small, skips XL)
./run_benchmarks.sh --full         # full run, includes XL
./run_benchmarks.sh --build        # ./build.sh release first, then the run
# scratch run that must not touch the committed BENCHMARKS.md / bench-reports/
./run_benchmarks.sh --suites python --out /tmp/bench-test --no-reports --no-benchmarks-md
# a non-default interpreter (the suite runs with the one that launches run_all.py)
./run_benchmarks.sh --python python3.11
```

The manual `cd src/EventProp.Native.Benchmarks && cmake --preset release …` and
`pip install -e .` lines are no longer needed: `build.sh` builds the native benchmark
binaries and installs both Python layers, and `run_benchmarks.sh` checks for them rather
than assuming them.

**Not yet done:** run the pipeline on **Linux** and record its per-pair geomeans alongside
the darwin and windows reports. The builds and suites pass there now, but no
`report_linux-x86_64.md` exists, and no amount of code reading substitutes for the run. One
question it still has to settle first: whether `bench_cpp`'s Tiny family is dominated by
one-off TBB task-scheduler spin-up — measured at ~24 ms for a single `ComputeSpikes/Tiny`
iteration at `--benchmark_min_time=0.01s`, orders of magnitude off the macOS number. **Do not
quote Linux benchmark numbers until that is measured.** The two Linux items the build/suite run
*did* settle: the SIMD level came from the CPUID+XCR0 probe (`AVX512F` rejected, `AVX2`
accepted), and `eventprop_eigen_core`'s `-march=native` PUBLIC propagation is consistent with
`parity_test`'s `-mavx2` at `EIGEN_MAX_ALIGN_BYTES` 32 — proven end to end by parity 105/105
rather than by reading compile_commands.json, which is the only reason that check was worth
listing.

**Both `build.sh` logic fixes are now mirrored in `build.ps1`**, so the two scripts are one
contract again: an import-probe before the reference library's "already built", and a Cython
probe + skip-the-doomed-install in the Cython step. `Get-ImportFailure` is the PowerShell form
of the same question, and takes its `PYTHONPATH` separator from the interpreter (`os.pathsep`
is `;` on Windows — a hardcoded `:` is the defect recorded under *Fixed — BenchmarkDotNet
adapter*). The TBB fix needs nothing there: `eventprop_pstl_link_deps()` returns empty on
`WIN32`.

What that means for the Windows claim, stated as a boundary rather than a pass: validated with
**PowerShell 7.6 on Linux** — file parses clean; the helper's seven cases (success, unknown
module, this tree's `eventprop.eventprop_cpp`, Cython present, Cython absent, missing
interpreter, a package whose import genuinely fails, and `PYTHONPATH` restored afterwards);
all four step-7 branches (imports → "already built" with no rebuild; fails → reason + rebuild;
wrong-interpreter tag → different-interpreter note + rebuild; nothing → silent rebuild); the
step-8 decision table (no Cython + regeneration needed → **one** failure naming the interpreter
+ one skip + install not reached + `_native.c` untouched; Cython present → regenerates for real
and reaches the install; up to date on a Cython-less interpreter → probe does not fire, install
runs). Not validated: Windows PowerShell **5.1**, MSVC, and a real `pip` install on Windows —
the constructs used are 5.1-safe (`$null -eq` comparisons, `@()` arrays, `Env:` drive removal,
`2>&1` + `$LASTEXITCODE`), but "5.1-safe by construction" is not "5.1-tested". One drive-by
portability fix fell out of testing: the Cython call passed the cwd-relative literal
`"eventprop_c\_native.pyx"`; it now passes the absolute `$pyx` the block already computed.
PowerShell normalises `\` to `/` on Unix, which is why that literal — and step 7's
`"eventprop\eventprop_cpp*.pyd"` globs — work on both platforms, and why the step-7 branches
were testable here at all.

### Coverage Report Generation

```bash
# Run with coverage
cd src/EventProp.Tests && dotnet run -c Release -- --coverage --coverage-output-format=cobertura

# Generate HTML report
export PATH="$PATH:$HOME/.dotnet/tools"
reportgenerator -reports:bin/Release/net10.0/TestResults/*.cobertura.xml \
  -targetdir:../../coverage-report -reporttypes:Html
```

## C Library Coding Style

### Error Handling: Single-Return Pattern

All functions use a single `goto cleanup_and_return` at the end. On error, set `result.error_no` and jump to cleanup. Output free functions are only called when `result.error_no != EVENTPROP_OK`, preserving result data for successful returns.

```c
ResultType eventprop_foo(const Input input) {
    ResultType result = {0};
    SomeCtx ctx = {0};  // context groups heap allocations

    if (eventprop_validate_input(input)) return result;

    ctx.data = malloc(...);
    if (!ctx.data) {
        result.error_no = EVENTPROP_SPECIFIC_ALLOC_CODE;
        goto cleanup_and_return;
    }

    /* ... main logic ... */

cleanup_and_return:
    if (result.error_no != EVENTPROP_OK) free_ctx(&ctx);
    return result;
}
```

### Error Codes: Specific, Not Generic

Each allocation site gets its own error code. No generic `EVENTPROP_ALLOCATION_FAILED` — callers should know exactly which allocation failed.

### Context Structs for Cleanup

Group related heap allocations into a context struct with a single free function. This eliminates copy-pasted cleanup across multiple error paths and prevents forgotten frees.

### Sequential Allocation Checks

Never do simultaneous `realloc` calls (e.g., `realloc(a)` then `realloc(b)`). If the first succeeds but the second fails, you lose the original pointer to `a`. Always check each allocation before moving to the next:

```c
double* new_a = realloc(old_a, ...);
if (!new_a) { result.error_no = CODE_A; goto cleanup_and_return; }
a = new_a;

double* new_b = realloc(old_b, ...);
if (!new_b) { free(new_a); result.error_no = CODE_B; goto cleanup_and_return; }  // undo a
b = new_b;
```

### Designated Initializers

Use `{ .field = value }` syntax for struct initialization — remaining fields are zero-initialized per C99 §6.7.8/20.

### Magic Numbers as Named Constants

Extract hardcoded thresholds to named constants at the top of files (e.g., `TOMS748_NEAR_ZERO`, `SPIKE_THRESHOLD_EPS`).

### Compiler Warnings

The CMake build enables strict warnings: `-Wall -Wextra -Wpedantic -Wshadow -Wimplicit-fallthrough -Wformat=2 -Wnull-dereference -Wsign-conversion` (GCC/Clang), `/W4` (MSVC). GCC additionally gets `-Wduplicated-cond`, `-Wmisleading-indentation`, `-Wlogical-op`. Code must compile with zero warnings — now enforced mechanically by the `werror` preset in `EventProp.C`.

MSVC specifics, all probe-derived rather than version-gated:

- **C11 atomics need two switches, and the build probes for them.** `eventprop_batch.h` includes `<stdatomic.h>` in C mode. MSVC defines `__STDC_NO_ATOMICS__` (so `vcruntime_c11_stdatomic.h` `#error`s "C atomic support is not enabled") on every toolset tested up to 14.52 / VS 2026, **including with `/std:c11` and `/std:c17`**; the switch that enables them is `/experimental:c11atomics`. `EventProp.C/CMakeLists.txt` proves it by compiling a probe **at the library's own language standard**: a compiler-version test would keep passing the flag after a toolset dropped it, and a probe at the default standard reports "no atomics" about a toolset that has them (the second failure mode is `__STDC_VERSION__ < 201112L`).
- **`/wd4756` is deliberate.** UCRT defines `INFINITY` as `((float)(_HUGE_ENUF * _HUGE_ENUF))`, so every use folds to a constant expression that overflows and MSVC reports "overflow in constant arithmetic" at the *use site*. The library's uses are the forward pass's "no root in this interval" sentinel, not arithmetic on a huge value. Suppressed by name so the zero-warning rule stays meaningful for everything else.
- **A SIMD build must not leave a dead scalar tail.** `dot_product_simd`'s scalar tail is provably unreachable once a `HAVE_AVX*` / `HAVE_NEON` branch is compiled — every SIMD branch folds the remainder loop into itself and returns, and `simd_available()` is a compile-time constant — so `/W4` reports **C4702 unreachable code** on all four lines. Guarded with the same `#if` as the `vec`/`vec_count` declarations: the shape of that function depends on which branch the target compiles. Clang does not warn (its `-Wunreachable-code` is opt-in), so this is MSVC-only.

### Memory Testing: AddressSanitizer

Both library and test CMakeLists support `-DENABLE_ASAN=ON` for memory error detection (use-after-free, buffer overflow). On macOS this builds a static library instead of shared (AppleClang limitation).

```bash
# Presets (preferred) — library first, then tests, both with ENABLE_ASAN=ON
cd src/EventProp.C        && cmake --preset asan && cmake --build --preset asan
cd ../EventProp.C.Tests   && cmake --preset asan && cmake --build --preset asan
ctest --preset asan
```

```bash
# Manual equivalent
cd src/EventProp.C/build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build .
cd ../EventProp.C.Tests/build && cmake .. -DENABLE_ASAN=ON
cmake --build .
./eventprop_tests
```

## C Library API Design

### Batch-First Architecture

The C library (`eventprop_batch.h`) provides batch operations as the primary interface. All batch functions accept per-sample input arrays and return typed result structs. Callers free via corresponding `_free_` functions.

**Core batch types:**

| Type | Role | Free Function |
|------|------|---------------|
| `SpikesBatch` | Forward spike output (per-sample, borrowed inputs) | `eventprop_free_spikes_batch` |
| `MaximaBatch` | Maxima output (per-sample) | `eventprop_free_maxima_batch` |
| `BackwardsBatch` | Per-sample backward results | `eventprop_free_backwards_batch` |
| `AccumulatedBackwardResult` | Combined backward + summed gradient (`GradientOutput*`) | `eventprop_free_accumulated_backward_result` |
| `DropoutBatch` | Dropout output (per-sample surviving spikes) | `eventprop_free_dropout_batch` |

> **Note:** Batch structs expose an atomic `has_errors` flag (`atomic_int`, set via `atomic_store_explicit(..., memory_order_relaxed)` from OpenMP loops — safe because all threads store the same value and the parallel region's exit barrier provides visibility). Error *details* are in the top-level `error_no` field and per-sample `error_no` values.

### Combined Backward + Accumulate

The primary FFI entry point for backward pass is the combined function — one call, zero intermediate allocations in Python:

```c
AccumulatedBackwardResult eventprop_backward_spikes_batch_accumulate(
    const EventPropInput* inputs, const SpikesBatch spikes_batch,
    const double* const* post_spike_errors);

AccumulatedBackwardResult eventprop_backward_maxima_batch_accumulate(
    const EventPropInput* inputs, const MaximaBatch maxima_batch);
```

Returns `GradientOutput summed_grads` (ready for weight update) + per-sample `.spike_errors` preserved. Per-sample `.grad` buffers are freed after summation.

### Spike Output Ownership

`SpikeOutput` carries only `times` + `sources` (+ `n_spikes`). Two former fields are gone:

- **`currents`** — the synaptic current at each output spike is *derived*, not stored. `eventprop_backward_spikes` rebuilds it with one ascending cursor over `input.times` into a row of `n_post` scratch doubles (`ctx.cur_sum`) plus one double per output spike (`ctx.currents`). The forward pass never needed it beyond that one line, and the backward pass already walks the input spikes, so storing it was pure overhead. The accumulation must run **ascending** and be scaled once at the end — the sweep runs time-reversed, and a row rebased at the sweep time loses six significant digits per post spike at benchmark sizes (gradient drifts ~1e-5 relative).
- **`errors`** — the caller-owned loss/error signal per output spike. It moved to an argument: `eventprop_backward_spikes(input, spike, post_spike_errors)` and `eventprop_backward_spikes_batch(inputs, spikes_batch, post_spike_errors)`, where `post_spike_errors` is an array of `batch_size` `const double*` (one per sample, may be `NULL` for empty samples). The library no longer allocates a buffer the caller only writes into.

Post-spike times must be non-decreasing; the backward pass validates this and returns `EVENTPROP_UNSORTED_SPIKE_TIMES` (60). New allocation codes: `EVENTPROP_ALLOC_BWD_CUR_SUM` (62), `EVENTPROP_ALLOC_BWD_CURRENTS` (63).

`MaximaOutput.errors` is unchanged (still library-allocated), as is `BackwardOutput.spike_errors`.

### Memory Ownership Rules

- All 13 public free functions are consolidated in `eventprop_memory.c` (core, batch, and diagnostic outputs)
- All free functions (batch and single-sample) take their struct **by value**
- Free functions exist for every heap-backed type: `eventprop_free_spike_output`, `eventprop_free_sums_output`, `eventprop_free_maxima_output`, `eventprop_free_backward_output`, `eventprop_free_dropout_output`, `eventprop_free_gradient_output`
- Dimension fields are removed from output structs — dimensions are inferred from input parameters

### ABI: One Version Stamp, Checked by Every Consumer

Every consumer of the C library **compiles its struct layouts into its own binary**: the
Cython extension (`cdef extern struct` in `_native.pyx`), the .NET P/Invoke layer
(`[StructLayout]` in `Api.cs` and `Batch.cs`), and the C++ consumers (the headers). A
header change that moves a field offset is invisible at compile time **and** invisible at
link time: the consumer keeps reading the offsets it compiled, against a library whose
fields moved, and returns plausible-looking garbage. It does not fail to load.

The measured instance is in *Python bindings on Windows*: a `_native.cp311-win_amd64.pyd`
built against July headers, loaded against a library rebuilt from October headers, died
with `Windows fatal exception: access violation` inside `spikes.py __getitem__` via
`loss_layer._find_first_spikes` — 40 minutes into a benchmark run, in a frame with no hint
that a header/binary offset mismatch was the cause. Nothing in the build checked that the
extension was newer than the headers it was compiled from.

**The surface** (`eventprop_abi.c`, declared in `eventprop.h`):

| Symbol | Role |
|---|---|
| `EVENTPROP_ABI_VERSION` (macro) | the layout version. **Bump on any change to a public struct's layout** — field order, type, width, or the width of a field's type. Appending a field changes nothing a consumer reads, but it changes `sizeof`, which the size check reports, so bump for that too. |
| `int eventprop_abi_version(void)` | the version this build was compiled with. A consumer compares it against the macro **as compiled from its own copy of the header**. |
| `size_t eventprop_abi_struct_size(const char* name)` | `sizeof` of a public struct as compiled in the library; **0 for an unknown name**, so a consumer asking about a struct the library does not know fails its own check instead of reading 0 as a match. |

**Where it is checked:** `_native.pyx` runs `abi_check()` at module import and raises
`ImportError` naming both values — the extension refuses to import on a mismatch, so the
failure is at import, not 40 minutes later. `EventProp.Native` runs
`Api.EnsureAbiMatches()` (once per process) from the `LIFLayer`/`LILayer` constructors and
from `Batch.ValidateFlat`, which is the shared choke point for all six friendly batch entry
points, so direct non-layer callers are gated too. The C suite pins the library side:
symbol exported, table **complete**, keys unique, unknown name → 0.

**Why `sizeof` and not just a version int:** the version needs a human to remember to bump
it; the size check is mechanical and fails on exactly the thing that broke (offsets). Both
exist for that reason.

**The 12-struct name list is deliberately duplicated** in `eventprop_abi.c`, `_native.pyx`,
`Api.cs` and the three test suites (`test/abi_version.cpp`, `test/test_abi.py`,
`AbiParityTests.cs`). Adding a struct to the headers must fail a test in **every** consumer,
which is what forces the check to be extended — a single shared list would let one consumer
stop covering the surface in silence.

`.NET` detail worth knowing: `sizeof` works on the `Batch.cs` **ref structs** in an `unsafe`
context (they are unmanaged), which is what makes a single size table possible across plain
and ref structs; `Marshal.SizeOf<T>()` and `Unsafe.SizeOf<T>()` cannot take a ref struct as
a generic argument.

Tests: `AbiParityTests` (8), `test_abi.py` (7, including a **ctypes** load of the DLL, which
reads the library's `sizeof` without consulting the extension's compiled offsets — so the
comparison is genuinely cross-binary, not a tautology), `abi_version.cpp` (6).

### Spike Time Ordering: A Shared Contract, Not a Parity Gap

Every spike list the library consumes must be **non-decreasing**. `eventprop_validate_input_no_v_th` enforces it on input times (`EVENTPROP_UNSORTED_SPIKE_TIMES`, 60) and `eventprop_backward_spikes` enforces it on post-spike times, because the currents there are rebuilt with one forward cursor over the post spikes.

The requirement is not a C-only strictness — it follows from the algorithm both implementations share. `compute_sums` builds prefix sums **in array order** (`sum0.row(i+1) = sum0.row(i) + exp(t_i/tau_syn) * w.row(sources[i])`), and the forward pass then indexes `sum0[k][nrn]` by position while treating `k` as "all input spikes up to `times[k]`". That is only true if `times` is sorted. An unsorted array is therefore not a valid alternative input; it describes a different network.

The Eigen reference has the same requirement and does not check it. Measured on a 4x2, 10-spike input with two interleaved bursts:

| Call | Result |
|---|---|
| C, unsorted | `error_no = 60`, 0 spikes |
| Eigen, unsorted | **3 spikes** — plausible-looking garbage |
| Eigen, sorted | 5 spikes |
| C, sorted | 5 spikes, identical times/sources, max delta **2.8e-15** |

So the C library is the stricter *diagnostic*, not the divergent *implementation*: on sorted input the two agree to root-finder tolerance. The Eigen `backward` does build a sorted index list internally, but that merges input and post spikes for the time-reversed sweep and does nothing for the prefix-sum ordering. Callers must sort; the C library tells you when they didn't, the reference silently returns wrong numbers.

Pinned by `ComputeSpikes.UnsortedTimesRejected` and `ComputeSpikes.EqualTimesAccepted` (`src/EventProp.C.Tests/test/compute_spikes.cpp`), alongside the existing validator and `BackwardSpikes` ordering tests. Non-decreasing (equal times) is valid and is the simultaneous-spike case the parity suite covers.

### SIMD: One ISA Level, Decided by Probe, Shared by Four Projects

`cmake/EventPropSimd.cmake` decides **one** level — `avx512 | avx2 | neon | scalar` — and `EventProp.C`, `EventProp.Cpp.Wrapper`, `EventProp.Native.ParityTests` and `EventProp.Native.Benchmarks` all call it. Override with `-DEVENTPROP_SIMD=scalar|avx2|avx512` (the A/B below used `scalar`). It is shared because the parity suite compares the C library against the Eigen core at `ABS_TOLERANCE 1e-10` / `REL_TOLERANCE 1e-9` on sums, spike times and gradients: if one side is vectorised and the other is not, that suite measures a code-generation difference instead of the algorithm.

**Why a probe replaced `/proc/cpuinfo`.** The old detection read `/proc/cpuinfo` and, on its absence, ran `sysctl`. Neither exists on Windows, so the level stayed "not detected" and the library ran its **scalar fallback on an AVX2 CPU** — `simd_vec_width()` 0 instead of 4 — in the `ComputeSums` prefix-sum inner loop (2 × `fma_row_simd` per input spike, `eventprop_sums.c:65-66`) and the backward pass's λ_I jump (`eventprop_backward.c:233`). The probe is a CPUID + **XGETBV** program run on the host by `check_c_source_runs`, so it asks the same question on every OS. It checks **XCR0 bits 1|2**, not just the CPUID bit: the bit says the hardware has AVX2; the OS has to preserve YMM state for user code to use it. Linux's `/proc/cpuinfo` "avx2" flag already implies OS support (the kernel checks XCR0 before setting it), so a CPUID-only probe would have been *weaker* than the path it replaces. Cross-compiling yields `scalar` with a message — the old probe described the build host, which is the wrong machine in that case.

**Three layers, each of which alone leaves the scalar path in place:**

1. **detection** (above);
2. **flag spelling** — `-mavx2` / `-mavx512f` are GCC/Clang. `cl` ignores them with only `Command line warning D9002 : ignoring unknown option '-mavx2'`, so a detection-only fix defines `HAVE_AVX2` and compiles `_mm256_set1_pd` against the SSE2 baseline. MSVC spells them `/arch:AVX2` / `/arch:AVX512`;
3. **macros** — x64 MSVC predefines **neither `__SSE2__` nor `__AVX2__`**, so `eventprop_simd.c`'s `#if defined(__AVX2__)` auto-detect is dead there. The explicit `HAVE_AVX2` define is the signal that works; with `/arch:AVX2` MSVC does define `__AVX2__`, so the source needs no change.

**The arch flag on the Eigen core is an ABI switch, not a performance knob.** Eigen's `EIGEN_MAX_ALIGN_BYTES` is **32 with AVX/AVX2 and 16 without**, and the aligned allocator is header-inlined. `eventprop_eigen_core` is a **STATIC** library, and consumer TUs include the Eigen headers and destroy the matrices the core returned. Applying the flag `PRIVATE` means the core allocates with the 32-byte convention and the consumer frees with the 16-byte convention → `STATUS_HEAP_CORRUPTION` (`0xC0000374`) in `RtlFreeHeap`, first test, every time, with the `DenseStorage` destructor inlined into the *calling* TU:

```
ntdll!RtlpHeapHandleError -> ucrtbase!free_base
  Eigen::internal::handmade_aligned_free    (Memory.h:118)
  Eigen::DenseStorage<double,-1,-1,-1,1>::~ (DenseStorage.h:452)
  compute_sums (eventprop_eigen_core_stripped.cpp:30, inlined at tests.cpp:51)
```

It is `PUBLIC` now, and the two projects that import the **prebuilt** core (`ParityTests`, `Native.Benchmarks`) apply the same level to their own TUs, because an `IMPORTED` target carries no usage requirements. `INTERFACE` alone is also wrong: it leaves the core's own TU at the baseline — the same mismatch in the other direction, and the identical crash.

**The same mismatch existed on Linux before this was found**: the core was `-march=native` **PRIVATE** and `parity_test` / `bench_c` were not, i.e. `EIGEN_MAX_ALIGN_BYTES` 32 in one TU and 16 in the next. Silent there because glibc does not always notice. The Linux leg of the benchmark run must confirm the `PUBLIC` propagation, which changes consumer codegen on that platform.

**Scalar-vs-AVX2 delta, measured** (50×20, 100 spikes, `tau_mem` 0.02, `tau_syn` 0.005, `v_th` 1.0; same source, two builds): output spike count, source checksum and `max_time` **identical**; spike times differ by 7e-18 absolute (tolerance 1e-10); gradient total by 3e-13 absolute / 2e-14 relative (tolerance 1e-9); maxima values by 3e-15. Three to four orders of headroom — matching the two sides is worth doing, and the mismatch was never something the tolerances would catch.

**Diagnostic signatures:**

- CMake's VS generator maps `/arch:AVX2` to `<EnableEnhancedInstructionSet>AdvancedVectorExtensions2</EnableEnhancedInstructionSet>`, so `grep arch:AVX2 *.vcxproj` finds **nothing** in a build that has the flag. Grep for `EnableEnhancedInstructionSet`.
- A SIMD probe that inherits `CMAKE_*_FLAGS` degrades silently under ASan: the instrumented try-compile binary cannot run without its runtime, so `check_c_source_runs` fails and the level becomes `scalar`.
- The library is compiled for the **build host** with no runtime dispatch — the same contract `-mavx2` on Linux and `-march=native` on macOS already had. The binary requires the ISA it was built for.

**Non-negative times are the same kind of shared contract.** `eventprop_validate_input_no_v_th` rejects `times[i] < 0` with `EVENTPROP_NEGATIVE_SPIKE_TIME` (7); the Eigen reference does not check. It is not C-only strictness either — the forward pass floors the first interval at `t_before = 0.0`, so it cannot bracket a spike before t = 0, and a negative input time is outside the model the prefix sums describe.

This bit two numerical-gradient tests in `src/EventProp.Native.Python/test/test_loss_layer.py`: both take a **central difference** of the loss with respect to a spike time, and the base spike train started at exactly `0.0`, so `spike_idx = 0` was perturbed to `-1e-8`. Only the Cython side failed, because only the C library validates. Fix: start the base trains at `0.001` (`(np.arange(n) + 1) * 0.001`) so the minus leg stays non-negative. The tests' intent — numerical vs analytical spike-time gradient — is unchanged.

While fixing them, a second test-side issue surfaced: `t_eps = 1e-8` is below the conditioning limit for a central difference on O(1) losses. Its absolute noise floor is `~machine_eps / t_eps ≈ 2.2e-8`, which swamps the analytical gradient where that gradient is legitimately **exactly 0** (the last input spike, which no first-spike time depends on), and the assertion used `rtol` with no `atol`. `t_eps = 1e-7` measured the best worst-case agreement (1.0e-9, against 8.1e-9 at 1e-8). When a finite-difference test fails on an entry that is analytically zero, check the noise floor before assuming a gradient bug.

### PSTL: what `<execution>` needs linked — probed, shared by three projects, never on Apple/Windows

`EventProp.Cpp.Wrapper/eventprop_eigen_core_stripped.cpp` sorts and searches with
`std::execution::par_unseq` (lines 259, 274). libstdc++ 12+ implements those policies over
Intel oneTBB: `<execution>` inlines task-scheduler calls (`tbb::detail::r1::spawn`,
`allocate`, `execute_and_wait`, …) and expects a library at link time. Nothing in this repo
named one — `git grep -i tbb` over the tracked tree: zero hits outside the read-only
submodule — so the first Linux build lost three projects at link **and** produced a shared
library that built green and could not be loaded.

`cmake/EventPropPstl.cmake` answers the question once, the way `EventPropSimd.cmake` answers
the ISA question, and `EventProp.Cpp.Wrapper`, `EventProp.Native.ParityTests` and
`EventProp.Native.Benchmarks` all call it:

| Call site | Why it has to be stated there |
|---|---|
| `eventprop_eigen_core` **PUBLIC** | It is a STATIC library, so "what it needs to link" is a consumer requirement: `eventprop_c_api` links the core into a SHARED library (which on Linux keeps undefined symbols instead of failing), and the in-tree test executables inherit it |
| ParityTests: `INTERFACE_LINK_LIBRARIES` on the IMPORTED `eventprop_eigen_core` | An `IMPORTED` target carries no usage requirements — the same reason the ISA level is restated in that project |
| `bench_cpp`, `bench_rootfind` link items | They consume the prebuilt `libeventprop_eigen_core.a` through `find_library`, which carries nothing either |

**The function's first act is returning empty on `APPLE` and `WIN32`, before any probe.** That
is not a guess about stdlib internals: it is what this repository's own validated builds
measure — the same source compiles **and links** on macOS and on Windows/x64 (parity 105/105)
with no TBB anywhere in the tree, because libc++ and the MSVC STL fall back to serial
execution. Returning before the probe is what makes it true that this file cannot change those
platforms, which was the constraint the Linux fix was written under. Verified with
`-DCMAKE_SYSTEM_NAME=Darwin`: zero `Performing Test EVENTPROP_PSTL_*` lines, empty result.

**Everything past that gate is decided by compiling, not by an OS or compiler-version gate**,
because the question has different answers on the same Linux box — GCC against libstdc++
(needs TBB) and clang against libc++ (serial, needs nothing). An OS gate would be wrong for
one of them. The order matters: (1) link with no extra library → nothing needed; (2) else
`find_library(tbb)` and re-probe with it → return it; (3) else `FATAL_ERROR` naming the
package to install. `find_library` rather than `find_package(TBB)` because a probe runs in its
own `try_compile` project that cannot see an imported target created in this one, so the answer
must be spelled identically in the probe and in the build. `-DEVENTPROP_PSTL_LIBRARY=/path/…`
overrides the search. Every probe result is `unset` from the cache at entry, because the
stale-cache failure here is specific and ugly: one configure before `apt install libtbb-dev`
would otherwise keep reporting TBB missing forever.

Two probe details, both found by measuring and both looking trivial until they bite:

- **`check_cxx_source_compiles` does not accept `CMAKE_FLAGS`** — measured on CMake 3.28:
  `Unknown argument: CMAKE_FLAGS`. Since the projects' floor is 3.21, the C++ standard travels
  in `CMAKE_REQUIRED_FLAGS` as `-std=c++17`, reachable only on the non-Apple/non-Windows path
  so the GNU spelling is safe by construction. Probing at the compiler default instead of the
  project's standard reports "no `<execution>`" about a project that has it, because
  `std::execution` is declared at C++17 and later only.
- **The header probe must not link.** On libstdc++ 13 a TU that merely includes `<execution>`
  and calls a **serial** `std::sort` fails to link with the same undefined
  `tbb::detail::r1` symbols as the parallel version — measured. A linking header probe
  therefore reports "this stdlib has no `<execution>`" about the exact toolchain that has it
  and needs TBB: the opposite advice, on every Linux machine the module exists to help. It runs
  with `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY`, which `check_cxx_source_compiles` does
  honour.

The signature worth remembering is the quiet failure, not the link error:

```
[100%] Built target eventprop_c_api        # green
$ python3 -c "import ctypes; ctypes.CDLL('.../libeventprop_cpp_wrapper.so')"
OSError: undefined symbol: _ZN3tbb6detail2r15spawnERNS0_2d14taskERNS2_18task_group_contextE
```

`nm -DC` still lists `U tbb::detail::r1::*` on that library after the fix, correctly: they are
resolved at load through the new `NEEDED libtbb.so.12`. The load test is the check; `nm` on a
shared library is not.

Post-mortem: *Fixed — three C++ projects would not link on Linux: `<execution>` needs TBB*.
Neighbouring platform rule: *SIMD: One ISA Level* — that one is about codegen agreement, this
one about link inputs, and both are shared modules for the same reason (the parity suite
compares two builds that must be built the same way).

### Cross-Suite Benchmark Data Contract

Three suites benchmark the same algorithms — native `bench_main.cpp` (C vs Eigen), .NET `EventProp.Benchmarks` (BenchmarkDotNet, managed vs P/Invoke), and `EventProp.Python.Benchmarks` (pytest-benchmark, pybind11 vs Cython). They share sizes (Tiny 10/5/20, Small 50/20/100, Medium 200/80/500, Large 500/200/1000; native and .NET add XL), taus (0.02 / 0.005), `v_th = 1.0`, and weights U(0.5, 2.0).

**The .NET suite's numbers are BenchmarkDotNet-version-sensitive — measured, not assumed.** macOS/arm64, .NET 10.0.13, `--job short`, Tiny family, 0.15.8 vs 0.16.0-preview.2:

| Benchmark | 0.15.8 | 0.16.0-preview.2 | Delta |
|---|---|---|---|
| `ComputeSpikes_Tiny` Managed | 4500 ns | 4943 ns | **+9.8%** |
| `ComputeSpikes_Tiny` Native | 36032 ns | 35590 ns | −1.2% |
| `ComputeSpikes_Tiny` **Ratio** | **8.01** | **7.20** | **−10%** |
| `BackwardSpikes_Tiny` **Ratio** | **10.53** | **9.31** | **−12%** |
| `BackwardSpikes_Tiny` Managed | 8.2 ns | 9.4 ns | +14.9% |

Noise floor, measured by repeating the *same* version: mean |delta| **0.6%**, max **1.1%** — and the 7.20 ratio reproduced exactly. So the version effect is ~8x the noise floor and fully reproducible, and it lands hardest on the managed-vs-native `Ratio`, which is the headline number this project reports. Nanosecond-scale benchmarks move most, because BDN's overhead evaluation and invocation heuristics changed between the two lines.

The mechanism: BenchmarkDotNet 0.15.x ships **no `net10.0` asset** — a net10.0 project silently consumes `lib/net8.0/BenchmarkDotNet.dll` (0 warnings, nothing in the build says so). 0.16.0 is the first line with `lib/net10.0/`. BDN's job is runtime-specific measurement knowledge, so running the net8 build on a net10 runtime is a real, if small, measurement-tool mismatch.

Consequences:

- The `0.15.*` band is deliberate: it blocks 0.16, so the measurement tool cannot change at restore time with no diff in the commit. `dotnet list package --outdated` says "no updates" for it, which is true only *within* the 0.15 band — it is not evidence about 0.16. Check the flatcontainer API for the real latest.
- Any BenchmarkDotNet version change **invalidates historical .NET benchmark comparisons**. BDN records `HostEnvironmentInfo.BenchmarkDotNetVersion` in every report — quote it when quoting numbers.
- When 0.16.0 goes stable, take it in one visible commit and re-baseline. Do not adopt a preview first: preview → stable moves the numbers a second time.

Spike patterns are generated by **one canonical definition implemented three times**, and the copies must stay in sync:

| Suite | Entry point |
|---|---|
| native | `GenerateBurstData` / `GeneratePoissonData` in `src/EventProp.Native.Benchmarks/bench_main.cpp` |
| .NET | `SpikePatterns.Burst` / `SpikePatterns.Poisson` in `src/EventProp.Benchmarks/SpikePatterns.cs` |
| Python | `generate_burst_spikes` / `generate_poisson_spikes` in `src/EventProp.Python.Benchmarks/conftest.py` |

Every generator upholds, and the Python suite asserts it at generation time (`validate_spike_data` in `conftest.py`):

- **times are non-decreasing** — the C library returns `EVENTPROP_UNSORTED_SPIKE_TIMES` (60) otherwise, and the Cython layer reports `error_no` without raising, so an unsorted array silently measures a validation short-circuit.
- **every spike lands inside [0, 1)** — `max_time / tau_syn` must stay under `EVENTPROP_EXP_OVERFLOW_THRESHOLD` (700). The old .NET burst reached 5.94 s → ratio 1188 → `EVENTPROP_TIMES_TAU_MISMATCH`, so that pattern could not be fed to the native side at all.
- **exactly `n_spikes` are produced, none left unwritten** — a fixed per-neuron burst offset saturates at a handful of bursting neurons, and the remaining slots keep their zero-initialised value: thousands of simultaneous spikes at t = 0. That inflated XL burst to 771k output spikes and made `BackwardSpikes/XL/burst` cost 15.6 s per iteration; the canonical generator uses a computed stagger and yields 757k spikes in 7.9 s.

- **weights are U[0.5, 2.0) in all three suites** — `w_dist(0.5, 2.0)` in `bench_main.cpp`, `0.5 + rng.NextDouble() * 1.5` in `SpikePatterns.cs`, `0.5 + 1.5 * rng.random(...)` in `conftest.py`. The Python suite used to be `0.5 + rng.random(...)`, i.e. U[0.5, 1.5) — **half the intended range, under a docstring claiming .NET parity**. Measured at Medium/uniform: **7768 output spikes vs 10308 (−24.6%)**, so Python-vs-native ratios were comparing materially different workloads, not just different RNGs. Fixed; every Python benchmark number re-baselines.
- **Tiny and Small/uniform produce 0 output spikes** for both the old and the corrected weight range, so those Python rows time an empty forward pass — the same regime already flagged for native `BackwardSpikes/Tiny|Small/uniform`. Exclude them from any geomean.

Burst is deterministic (no RNG in times/sources), so Python and native produce **bit-identical** burst data. Uniform times are identical by construction (only `sources` differ, since `std::mt19937`, `System.Random` and PCG64 cannot agree). Poisson uses the same inverse-CDF form in all three (`-log(1-u)/rate` plus a strictly-increasing fix), so it is distributionally identical. Cross-suite numbers are therefore statistical, not exact — compare ratios, not absolute counts.

Two known caveats: the poisson form concentrates the whole spike train in ~40 ms rather than spreading it over [0, 1) ("Poisson @200 Hz" is a misnomer shared by all three suites), and the Python suite skips Medium+/burst as `SLOW_BURST` — a runtime-cost guard written against the old pathological data, worth re-measuring now that the pattern is sane.

**The poisson misnomer, quantified.** Each spike time is an independent exponential order statistic `-log(1-u)/200`, not a homogeneous Poisson process over [0, 1). Measured (seed 42): max_time ~11 ms / effective ~1.8 kHz at Tiny (20 spikes), ~30 ms / ~3.3 kHz at Small (100), ~46 ms / ~10.8 kHz at Medium (500), ~39 ms / ~25.6 kHz at Large (1000) — 9–128x the 200 Hz label. The pattern key stays `poisson` across all three suites (renaming would break the shared `kPatterns`/`SIZES`/parametrize contract); the generators' docstrings state the real distribution. Cross-suite ratios remain valid because all three use the identical form.

### Layer gradient semantics (C# == Python)

`Layer.Backward()` **accumulates** into `Gradient` and never zeroes it, matching the Python reference (`backward_spikes_batch_cpp(..., self.gradient)` adds; the training loop calls `optimizer.zero_grad()` separately). `ZeroGrad()` is the only zeroing. Call `ZeroGrad()` before `Backward()` for a single-step gradient. This is pinned by `LayerContractTests.Backward_AccumulatesAcrossCalls`, which asserts a second `Backward()` **doubles** the gradient (and guards against a vacuous all-zero gradient by requiring the layer to fire). The managed and native layers implement the same semantic; the native layers add the summed gradient out of the native buffer rather than overwriting.

### Chaining model (intentional C# divergence from Python)

Python threads layers through an `ancestor_layer` field and a `__call__(SpikesVector, Layer)` idiom; its base `backward()` recurses into the ancestor and raises `RuntimeError("Run backward to create errors first.")` when the child has not run. C# has no `ancestor_layer` and no `__call__`; chaining is the explicit `Layer.Chain<TLayerA,TLayerB>` whose `Backward()` calls `A.Backward(); B.Backward()` directly. The ancestor-recursion and its "run backward first" guard have no C# counterpart — this is a deliberate simplification, documented on `LayerChain.Backward`.

## Python Bindings (`_native.pyx`)

### Architecture

Thin Cython wrappers with managed heap-backed wrapper classes that own C memory:

| Wrapper Class | Owns | Provides |
|---------------|------|----------|
| `_SpikesBatchOutput` | heap `EventPropInput[]` + `SpikesBatch*` | Zero-copy views via `.get_spike_times(i)`, `.get_sources(i)`; inputs reused by backward/maxima calls |
| `_MaximaBatchOutput` | heap `EventPropInput[]` + `MaximaBatch*` | Zero-copy views for maxima times/values; inputs reused by backward calls |
| `_AccumulatedBackwardResult` | `AccumulatedBackwardResult*` | `.grad` (flat), `.rows`, `.cols`, `.get_spike_errors(i)` |
| `_DropoutBatchOutput` | `DropoutBatch*` | Zero-copy views via `[i]` → `(times, sources)` tuple |

All wrappers use `__dealloc__` to call the appropriate C free function. Numpy views are created via `PyArray_SimpleNewFromData` (zero-copy).

### Python bindings on Windows: finding the C library, and building the extension

Four Windows-specific facts, all measured:

- **`os.add_dll_directory`, not `PATH`.** `_native.pyd` links `eventprop_c_api.dll`. Since Python 3.8, extension modules are loaded with `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, whose search set is the extension's own directory, the process directory, the system directories, and directories registered with `os.add_dll_directory` — **`PATH` is excluded**. Measured on the same directory both times: `PATH` gives `ImportError: DLL load failed while importing _native`; `os.add_dll_directory` imports. `eventprop_c/_library_path.py` registers `EVENTPROP_C_LIB_DIR` (os.pathsep-split) and then the repo layout — `build/`, `build/Release`, `build/Debug` (Release because MSVC is multi-config) — and `__init__.py` calls it **before** `from ._native import`. The handles are kept in a module global, because `os.add_dll_directory`'s handle *removes* the directory when it is finalised. The registration is a no-op on macOS, which links an rpath at build time; Linux needs its own mechanism and had none — see *Linux: finding the C library (rpath, then preload)*. Pinned by `test/test_library_path.py`.
- **The extension needs `/std:c11` *and* `/experimental:c11atomics`.** `_native.c` includes `eventprop_batch.h` → `<stdatomic.h>`. distutils passes `/TC` with no language standard, so `vcruntime_c11_stdatomic.h(16)` `#error`s "C atomics require C11 or later"; MSVC needs the atomics switch on top of that. `setup.py` supplies the pair. Without them the extension cannot be compiled at all.
- **A stale `.pyd` is a memory-corruption bug, not a staleness nuisance.** Cython compiles the `SpikeOutput` / `MaximaOutput` struct layouts out of `eventprop.h`. An extension built against July's headers and run against October's reads those structs at July offsets: the suite died with `Windows fatal exception: access violation` inside `spikes.py __getitem__` via `loss_layer._find_first_spikes`. **Nothing in the build checked that the extension is newer than the headers** — on any platform. It now does, in three places: `_native.pyx` runs `abi_check()` at import and raises `ImportError` naming both the compiled and the loaded values; `build.ps1` regenerates `_native.c` when `_native.pyx` is newer; and `EventProp.Native` gates its layer constructors and batch entry points on `Api.EnsureAbiMatches()`. See *ABI: One Version Stamp, Checked by Every Consumer*.
- **`vcvarsall.bat` failing is often the machine, not the repo.** Symptom: `pip install -e .` reports "Microsoft Visual C++ 14.0 or greater is required" with VS installed. Cause on this machine: a machine-PATH entry with an unbalanced quote (`C:\Program Files\WSL"`), which makes vcvars' own batch parsing die (`\NVIDIA was unexpected at this time`). Check the raw value — `(Get-Item 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment').GetValue('Path', $null, 'DoNotExpandEnvironmentNames')` — for a `"` before blaming the toolchain, and never fix it with `setx PATH "%PATH%"` (merges machine+user into one `REG_SZ`, expands every `%placeholder%`, truncates at 1024 chars).

### Linux: finding the C library (rpath, then preload)

`_native*.so`'s `DT_NEEDED` entry is the bare soname `libeventprop_c_api.so`. On Linux the
loader tries RUNPATH, then `LD_LIBRARY_PATH`, then the system search paths, and none of those
is the repo's build directory — so a plain `import eventprop_c` after a fully green
`./build.sh release` died with `ImportError: libeventprop_c_api.so: cannot open shared object
file: No such file or directory`. It had gone unnoticed because the only consumer, and the
only person ever importing it, was `scripts/bench/run_all.py`, which exports
`LD_LIBRARY_PATH` for its own reasons. Two changes, both Linux-only:

- **`setup.py` links an rpath** — `-Wl,-rpath,<abs>/src/EventProp.C/build`, absolute and
  derived from `__file__` rather than from the CWD, because the loader resolves an rpath at
  import time in a process that may have started anywhere. This is the mechanism the `darwin`
  branch of that same file already used; the `else` branch had `extra_link_args = []`.
- **`_library_path.py` preloads the library on Linux** — `ctypes.CDLL(lib, RTLD_GLOBAL)` for
  each candidate directory, which is the POSIX counterpart of `os.add_dll_directory`: a library
  already loaded with `RTLD_GLOBAL` satisfies a later extension's `DT_NEEDED` by soname. It is
  what makes `EVENTPROP_C_LIB_DIR` mean something on Linux — the documented escape hatch was
  silently inert there, and a variable that appears to configure the package while doing
  nothing is its own defect class — and it covers what an absolute rpath cannot: an installed
  `eventprop_c`, or a checkout moved after the extension was built. macOS still returns `[]`.

`__init__.py` calls `register_c_library_dirs()` before `from ._native import`, which is what
gives the preload its chance. Pinned by `test/test_library_path.py::LinuxLibraryLookupTest`
(preload ran, idempotent, repo `build/` among the candidates) next to the cross-platform import
test — which is the one that had been failing.

### Layer Chain Flow

1. Python constructs `EventPropInput[]` once and passes it to batch functions
2. All subsequent batch calls receive direct pointers to C-owned memory from previous returns
3. No concatenation, no array rebuilding in Python after initial input
4. Backward pass uses combined accumulate function — one FFI call per layer

### Zero-Copy Output Lifetime

`get_spike_errors(i)` and `get_maxima_errors(i)` return numpy views into C-owned memory. The owner is the wrapper, not the view, so anything that hands a view to a caller must keep the wrapper alive for as long as the view can be read. `LIFLayer.backward` / `LILayer.backward` therefore store the `AccumulatedBackwardResult` on `self._backward_result`: the consumer is the ancestor layer's `backward()`, which runs after the local `result` would otherwise go out of scope. See *Fixed — Cython layers: use-after-free in spike-error propagation*.

### Paper Parity Suite

`src/EventProp.Native.Python/tests/test_paper_parity.py` runs the pre-trained checkpoints in `EventProp.Python/paper/` (YinYang TTFS, MNIST VMax) through **both** backends and compares forwards, predictions and gradients. It is the only suite that exercises real trained weights, so it catches divergences that the synthetic fixtures miss.

Two of its tests were themselves broken and are fixed here — when a parity test fails, check the test before blaming the library:

- `test_forward_predictions` used `np.nanargmin` over `first_spike_times`. A sample where no output neuron fires is an all-NaN row, which the **reference** produces (1 of 64 YinYang samples), and `nanargmin` raises `ValueError` on it. Predictions now use a NaN-safe argmin with `-1` for all-NaN rows, on both sides.
- `test_forward_vmax_values` read `py_loss.values_batch`, which does not exist on the pybind11 `VMaxCrossEntropyLoss`. The two backends expose the same numbers under different names: `py_loss.maxima_batch[i].values` vs `cy_loss.values_batch[i]`.

## Architecture Notes (.NET)

### Shared Layer Contract (`EventProp.Abstractions`)

`EventProp.Abstractions` (assembly `CW.EventProp.Abstractions`, namespace `CW.EventProp`, no dependencies) holds the types both backends share, mirroring the Python library's `ABC`-based `layer.py`:

- `Layer` — abstract base class (not an interface) with abstract `Forward`/`Backward`/`ZeroGrad`/`GetWeights`/`SetWeights`/`NIn`/`NOut` and the nested `LayerChain<TA,TB>`. An abstract class (not `IDisposable`-style interfaces) is used because the contract passes concrete reference types (`SpikesVector`, `SpikesData`) and shares state (`Gradient`, `_inputBatch`, `_ranForward`); the Python reference uses `ABC` + `@abstractmethod` the same way.
- `SpikesData`, `MaximaData`, `SpikesVector`, `MaximaVector` — heap-backed data carriers (ArrayPool-rentable), shared by both backends.
- `LIFLayerParameters`, `LILayerParameters`, `WeightInitializer` (+ Gaussian/Uniform/Diagonal).

`EventProp` (managed) and `EventProp.Native` (P/Invoke) each `ProjectReference` Abstractions and implement `Layer`. The compiler enforces the contract; benchmarks and parity tests run the same scenario against both backends.

### Native Layers (`EventProp.Native.LIFLayer` / `LILayer`)

Thin P/Invoke-backed `Layer` implementations so `EventProp.Benchmarks` compares managed vs native like-for-like. Key design points:

- **Friendly batch wrappers** (`BatchFriendly.cs`): `Batch.ComputeSpikesBatch` / `ComputeMaximaBatch` / `BackwardSpikesBatchAccumulate` / `BackwardMaximaBatchAccumulate` take flat arrays (`w`, concatenated `times`/`sources`, `nSpikesPerSample`) and marshal to an `EventPropInput[]` on the stack, so callers never hand-build per-sample structs. `SpikesBatch`/`MaximaBatch` expose per-sample indexing (`SpikeAt`/`MaximaAt`, plus a `this[int]` indexer on `SpikesBatch`).
- **`ref struct` outputs cannot be stored on a class field.** `SpikesBatch`/`MaximaBatch` are `ref struct`s, so the native layers cannot cache the forward batch across `Forward()`→`Backward()`. `Backward()` recomputes the forward batch (deterministic for the same weights + input) in the same scope and feeds the accumulate call.
- **Loss-derivative plumbing differs by loss type.** For the spike loss, the C backward reads `post_spike_errors` from a caller-supplied pointer array; the layer pins each `SpikesData.Errors` array (`GCHandle`) and passes that array. For the maxima loss, the C backward reads `errors` from the **native** `MaximaOutput` (library-allocated), so `LILayer.Backward` copies the managed `MaximaData.Errors` into the native batch's errors array before the accumulate call.
- Native layers are `IDisposable` (free pinned handles); the managed layers expose `Dispose()` for their scratch buffers.

Parity is pinned by `EventProp.Tests/NativeLayerParityTests.cs` (LIF spikes+gradient, LI maxima+gradient, native batch forward).

### Layer Contract Suite (`EventProp.Tests/LayerContractTests.cs`)

The `Layer` contract is tested against **every** backend, so a new backend must satisfy the same behaviour and the managed/native pair cannot drift. A `LayerBackend` record carries the factory (`CreateLif`/`CreateLi`) plus the backend-specific plumbing (`SetSpikeError`, `GetMaxima`); each scenario runs over `AllBackends()`.

Covered per backend: gradient shape, forward/backward determinism, `ZeroGrad`, backward accumulation (a second `Backward()` doubles the gradient — Python semantics, with a firing guard so the check is non-vacuous), weight span round-trip, weight-length validation (ctor + `SetWeights`), Backward/Output-before-Forward throws, forward-replaces-output, batch forward == per-sample, batch gradient == sum of per-sample gradients, LI maxima shape, LI backward determinism. Cross-backend: LIF/LI/batch gradients match between managed and native (with firing + non-zero-gradient guards), and a **mixed managed↔native `LayerChain`** produces the same output as a homogeneous chain.

`Layer` is `IDisposable` (managed layers own ArrayPool buffers, native layers own pinned handles + native batches); `LayerChain.Dispose()` disposes both children. `SyncAssert` gained a `bool` overload (`IsTrue`/`IsFalse`).

> TUnit has no public `PropertySource` attribute — verified against both 1.45.29 and 1.75.0; the `PropertySource` strings in `TUnit.Core.dll` are internal (`PropertySourceRegistry`, `IPropertySource`). `MethodDataSource` on a non-static test needs `[CombinedDataSources]` (TUnit0070). The suite sidesteps both by looping `AllBackends()` inside each `[Test]` body.

### Zero-Allocation Hot Paths

- **`MatrixView`** — `ref struct` providing zero-copy row-major matrix views over flat `double[]` arrays
- **`StridedSpan`** — column access without allocation (strided read-only span)
- **`ScratchBuffers`** — pre-allocated reusable buffers; callers allocate once and pass in
- **`PostSpikeRow`** — one growable scratch row per post-synaptic neuron (`times` / `currents` / `exp(st/tau_mem)`), rented once from `ArrayPool` and grown ×3 in place. A neuron may cross the threshold any number of times, so its spikes cannot live in a fixed-stride slice of a shared slab; each `Parallel.For` iteration owns exactly one row, so growth needs no atomics

### Biophysical Parameters

All functions share these parameters:
- `tauMem` — membrane time constant (seconds)
- `tauSyn` — synaptic time constant (seconds)
- `vTh` — spike threshold voltage

### Weight Matrix Layout

Row-major: `w[row * cols + col]` where rows = pre-synaptic neurons, cols = post-synaptic neurons.

## Key Algorithms

- **`ComputeSums`**: Pre-computes cumulative exponential-weighted sums for fast membrane potential evaluation
- **`ComputeSpikes`**: Forward pass — finds threshold-crossing times via root-finding (Toms748/Brent)
- **`Backward`**: Adjoint method — time-reversed sweep propagating λ_v (membrane) and λ_i (synaptic) adjoints
- **`ComputeMaxima`**: Finds membrane potential maxima per neuron (for voltage-based loss)

## Fixed Defects — Invariants and Failure Modes

Every entry below is **fixed**; there are no open defects in this list. They are kept because each records a constraint that is not visible in the current code and is easy to re-introduce: an invariant a refactor must preserve, a parity contract shared by both backends, a diagnostic signature for recognising the failure class, and the regression test that pins it. Treat this as a post-mortem log and a checklist for touching the areas named, not as a bug tracker entry.

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

## Platform Support

- **macOS** — developed and tested
- **Linux** — validated end to end 2026-10-20 (WSL2, GCC 13.3, cmake 3.28.3, .NET 10.0.112, Python 3.12.3): `clean.sh -full` → `build.sh release` green, C tests 220/220, parity **105/105**, .NET **382/382**, `eventprop_c` + paper parity 33 passed / 6 Windows-only skips, `bench_c`/`bench_cpp`/`parity_test`/`EventProp.Benchmarks.dll` all present and `run_benchmarks.sh`'s import probe passing; `werror` preset builds the C library with zero warnings on GCC 13.3. **The benchmark pipeline has not been run there** (see *Not yet done*), so there is no `report_linux-x86_64.md` and no Linux timings to quote. Prerequisites beyond the obvious toolchain: `libtbb-dev` and Cython in the interpreter that builds the extension. Two things the earlier Linux-facing work claimed and this run confirmed: the SIMD level comes from the CPUID+XCR0 probe (not `/proc/cpuinfo`), and `eventprop_eigen_core`'s `-march=native` PUBLIC propagation agrees with `parity_test`'s `-mavx2` at `EIGEN_MAX_ALIGN_BYTES` 32 — proven by parity passing, not by reading flags.
  Linux traps that each cost part of a session:
  - GCC 12+ implements `std::execution::par_unseq` over TBB, and a **shared** library with those symbols undefined still links — see *PSTL: what `<execution>` needs linked*.
  - Including `<execution>` breaks the link of even a **serial** `std::sort` TU, so a header probe must be compile-only (`CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY`).
  - `check_cxx_source_compiles` rejects `CMAKE_FLAGS`; the C++ standard has to travel in `CMAKE_REQUIRED_FLAGS`.
  - An extension's bare-soname `DT_NEEDED` is invisible to `import` unless an rpath, a preload, or `LD_LIBRARY_PATH` names the directory — and `run_all.py` setting that variable hides the defect from the one consumer that matters.
- **Windows** — supported via MSVC + CMake; validated end to end 2026-10-16 on MSVC 19.52 / VS 18: C tests 220/220, parity 105/105, .NET 382/382, `eventprop_c` Python 29/29, `werror` preset zero warnings, and a full benchmark run (46m 40s) producing `bench-reports/report_windows-amd64.md`. CMake's VS generator is **multi-config**, so artifacts land in `build\Release\`, not `build\` — every path in the scripts and in `EventProp.Native.csproj` has to say so.
  Windows traps, each of which has already cost a session:
  - `where bash` → `System32\bash.exe` (**WSL**) before Git's; WSL bash cannot see `G:\` as a Windows tree. Resolve Git's bash from `git.exe`'s location and prepend its directory for coreutils.
  - PowerShell 5.1: native stderr + `EAP=Stop` is a terminating `NativeCommandError`; `Tee-Object` writes UTF-16LE; empty-string arguments to native commands are dropped.
  - MSYS bash reports an exe whose dependent DLL is missing as "No such file or directory" (127).
  - Python 3.8+ excludes `PATH` from extension-module dependency resolution (`os.add_dll_directory` is the mechanism); `grep arch:AVX2 *.vcxproj` finds nothing because CMake maps the flag to `EnableEnhancedInstructionSet`.
  - `.vscode/` is **gitignored**, so `settings.json` / `tasks.json` are local to each machine, not shared; `tasks.json` on this machine still encodes the pre-presets manual `rm -rf build && mkdir build` sequence and needs modernising to `cmake --preset`
  - pybind11 has a **platform-specific required version and Windows had none recorded**, so `submodule-state.sh apply` (build.ps1 step 1) checked out 2.6.3.dev1 and step 7 died in pybind11 headers, not in this repo's code. See *Fixed — Windows reference build: pybind11 had no required version*. The `m` in the trap list is not "MSVC is unusual" but "the state script owns a version table and a missing row is a silent downgrade".
  - `clean.ps1` can report `Clean complete … 27 path(s) removed)` / exit 0 with 8 `.NET` paths still present, because Windows delete-pending makes `Remove-Item` succeed while MSBuild/Roslyn build nodes hold handles. It re-tests the path now; see *Fixed — the clean scripts counted paths they had not removed*.
