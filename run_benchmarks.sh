#!/usr/bin/env bash
# run_benchmarks.sh — Run the cross-library benchmark pipeline on macOS and Linux.
#
# Usage:
#   ./run_benchmarks.sh                        # quick run (Tiny/Small, skips XL)
#   ./run_benchmarks.sh --full                 # full run, includes XL
#   ./run_benchmarks.sh --report-only          # re-merge records already on disk
#   ./run_benchmarks.sh --build                # ./build.sh release first
#   ./run_benchmarks.sh --suites dotnet python # subset of suites
#   ./run_benchmarks.sh --sweeps               # include the .NET sweep cells
#   ./run_benchmarks.sh --tag mytag --out /tmp/bench
#   ./run_benchmarks.sh --suites python --no-reports --no-benchmarks-md   # scratch run:
#       writes nothing to the committed BENCHMARKS.md / bench-reports/
#
# THIS IS NOT src/EventProp.Native.Benchmarks/run_benchmarks.sh. That one runs the
# native C-vs-C++ suite alone and calls Google Benchmark's compare.py. This one is the
# repo-root cross-suite runner: native + .NET + Python, normalized into the common
# record schema, merged, and reported. Same name, two scopes, two directories — which is
# worth knowing before reading either.
#
# What it does:
#   1. Verifies the prerequisites by checking what the suites actually need: the native
#      benchmark executables, the .NET benchmark dll, the C library, and an ACTUAL
#      `import eventprop_c` for the Cython layer. It does not build them silently — a
#      build failure must surface as a build failure, not as a benchmark result. Pass
#      --build to run ./build.sh first.
#   2. Runs scripts/bench/run_all.py, which writes:
#        bench-results/records_<tag>_*.json    (gitignored — machine-specific)
#        bench-results/merged_<tag>.json       (gitignored)
#        bench-reports/report_<tag>.md         (COMMITTED)
#        BENCHMARKS.md                         (COMMITTED, "latest run")
#
# Why the Cython prerequisite is an import, not a file check
#   The Windows counterpart used to check that the extension file exists. It existed and
#   `import eventprop_c._native` failed, because the extension links the C library and the
#   loader has to find it — on Windows that needs os.add_dll_directory (PATH is excluded
#   since Python 3.8), on macOS/Linux the extension carries an rpath baked in at build time
#   (setup.py: -Wl,-rpath,<C build dir>). The only check that can tell you the suite will
#   run is the import itself, done with the same PYTHONPATH the runner uses.
#
# macOS note, so it is not a surprise: System Integrity Protection strips DYLD_* from the
# environment of system binaries, so DYLD_LIBRARY_PATH exported by run_all.py does not
# reach a /usr/bin/python3. That is not the mechanism the extension relies on — it has the
# rpath — and the import probe below is what proves the loader is satisfied.
#
# Cross-platform note: merge.py refuses to merge records from different hosts, so this
# produces a SEPARATE report from the Windows run. Compare per-pair geomeans across
# reports, never the absolute µs columns.
#
# Timing expectation (measured on an M3 MacBook Air, full run): native ~9 min, .NET ~21 min,
# python ~2 min. XL is not the slow part — Google Benchmark's min_time is relative, so a
# 14 s iteration takes ONE iteration; the ~30 min is .NET's 75+ benchmarks. On Windows the
# ordering inverts (python 22m 26s is the longest suite, .NET 16m 07s), so quote timings per
# platform.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

FULL=0
REPORT_ONLY=0
DO_BUILD=0
SWEEPS=0
NO_REPORTS=0
NO_BENCHMARKS_MD=0
PYTHON="python3"
TAG=""
OUT=""
REPORTS_DIR=""
BENCHMARKS_MD=""
suites=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)            FULL=1 ;;
    --report-only)     REPORT_ONLY=1 ;;
    --build)           DO_BUILD=1 ;;
    --sweeps)          SWEEPS=1 ;;
    --no-reports)      NO_REPORTS=1 ;;
    --no-benchmarks-md) NO_BENCHMARKS_MD=1 ;;
    --python)          shift; PYTHON="${1:-}" ;;
    --python=*)        PYTHON="${1#--python=}" ;;
    --tag)             shift; TAG="${1:-}" ;;
    --tag=*)           TAG="${1#--tag=}" ;;
    --out)             shift; OUT="${1:-}" ;;
    --out=*)           OUT="${1#--out=}" ;;
    --reports-dir)     shift; REPORTS_DIR="${1:-}" ;;
    --reports-dir=*)   REPORTS_DIR="${1#--reports-dir=}" ;;
    --benchmarks-md)   shift; BENCHMARKS_MD="${1#--benchmarks-md=}" ;;
    --benchmarks-md=*) BENCHMARKS_MD="${1#--benchmarks-md=}" ;;
    --suites)
      shift
      while [[ $# -gt 0 && "${1:0:1}" != "-" ]]; do suites+=("$1"); shift; done
      continue
      ;;
    --suites=*) IFS=', ' read -r -a tmp <<< "${1#--suites=}"; suites+=("${tmp[@]}") ;;
    -h|--help) sed -n '2,52p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; echo "Usage: $0 [--full] [--report-only] [--build] [--sweeps] [--suites ...] [--python EXE] [--tag T] [--out DIR] [--reports-dir DIR] [--benchmarks-md PATH] [--no-reports] [--no-benchmarks-md]"; exit 1 ;;
  esac
  shift
done

# --- 0. Optional build --------------------------------------------------------
if [[ $DO_BUILD -eq 1 ]]; then
  echo "=== build (./build.sh release) ==="
  ./build.sh release
  if [[ $? -ne 0 ]]; then
    echo "build.sh failed - stopping."
    exit 1
  fi
fi

# --- 1. Verify prerequisites --------------------------------------------------
# Single-config generators (Unix Makefiles, Ninja) write to build/; the MSVC generator
# is multi-config and writes to build/Release/. Probe both, matching run_all.py's
# native_build_dir(), so this check cannot disagree with the runner about where the
# binaries are.
echo "=== checking prerequisites ==="
missing=0

find_first() {
  local p
  for p in "$@"; do
    if [[ -f "$p" ]]; then printf '%s\n' "$p"; return 0; fi
  done
  return 1
}

bench_c=$(find_first \
  src/EventProp.Native.Benchmarks/build/bench_c \
  src/EventProp.Native.Benchmarks/build/bench_c.exe \
  src/EventProp.Native.Benchmarks/build/Release/bench_c \
  src/EventProp.Native.Benchmarks/build/Release/bench_c.exe \
  src/EventProp.Native.Benchmarks/build/Debug/bench_c.exe) || bench_c=""
bench_cpp=$(find_first \
  src/EventProp.Native.Benchmarks/build/bench_cpp \
  src/EventProp.Native.Benchmarks/build/bench_cpp.exe \
  src/EventProp.Native.Benchmarks/build/Release/bench_cpp \
  src/EventProp.Native.Benchmarks/build/Release/bench_cpp.exe \
  src/EventProp.Native.Benchmarks/build/Debug/bench_cpp.exe)

if [[ -z "$bench_c" || -z "$bench_cpp" ]]; then
  echo "  MISSING: bench_c / bench_cpp"
  echo "           Build: ./build.sh release   (or ./run_benchmarks.sh --build)"
  missing=1
fi

dotnet_dll=$(find_first \
  src/EventProp.Benchmarks/bin/Release/net10.0/EventProp.Benchmarks.dll \
  src/EventProp.Benchmarks/bin/Debug/net10.0/EventProp.Benchmarks.dll)
if [[ -z "$dotnet_dll" ]]; then
  echo "  MISSING: src/EventProp.Benchmarks/bin/Release/net10.0/EventProp.Benchmarks.dll"
  echo "           Build: ./build.sh release   (it builds EventProp.Benchmarks)"
  missing=1
fi

# The C library, at a path the Cython extension can resolve (its rpath is baked in at
# build time; run_all.py also exports DYLD_LIBRARY_PATH / LD_LIBRARY_PATH for the
# non-Python consumers).
c_lib=$(find_first \
  src/EventProp.C/build/libeventprop_c_api.dylib \
  src/EventProp.C/build/libeventprop_c_api.so \
  src/EventProp.C/build/Release/libeventprop_c_api.dylib \
  src/EventProp.C/build/Release/libeventprop_c_api.so \
  src/EventProp.C/build/Release/eventprop_c_api.dll \
  src/EventProp.C/build/Release/libeventprop_c_api.dll \
  src/EventProp.C/build/Debug/libeventprop_c_api.so \
  src/EventProp.C/build/Debug/eventprop_c_api.dll)
if [[ -z "$c_lib" ]]; then
  echo "  MISSING: libeventprop_c_api.{dylib,so} (looked in build/, build/Release/, build/Debug/)"
  echo "           Build: ./build.sh release"
  missing=1
fi

if [[ $missing -eq 1 && $REPORT_ONLY -eq 0 ]]; then
  echo ""
  echo "Prerequisites missing. Nothing was run."
  echo "Run ./build.sh release first, or: ./run_benchmarks.sh --build"
  exit 1
fi
[[ $missing -eq 0 ]] && echo "  artifacts: ok"

# --- 2. Python ----------------------------------------------------------------
# The reference library (pybind11) and the Cython layer are both imported by the python
# suite, with PYTHONPATH set by run_all.py. Import them the way the runner will, so a
# pass here means the suite can start.
if [[ $REPORT_ONLY -eq 0 ]]; then
  echo "=== checking python ==="
  py_paths="$SCRIPT_DIR/src/EventProp.Python:$SCRIPT_DIR/src/EventProp.Native.Python"
  probe="import numpy, pytest, pytest_benchmark, eventprop, eventprop_c; import eventprop_c._native; print('  python packages + both bindings import ok')"
  if ! PYTHONPATH="$py_paths" "$PYTHON" -c "$probe"; then
    echo "  The python suite cannot run."
    echo "  Packages:  $PYTHON -m pip install numpy pytest pytest-benchmark"
    echo "  Cython:    ./build.sh release   (pip install -e src/EventProp.Native.Python)"
    echo "  Reference: ./build.sh release   (pip install -e src/EventProp.Python)"
    echo "  Interpreter: '$PYTHON' is $("$PYTHON" --version 2>&1) — the suite runs with the"
    echo "  interpreter that launches run_all.py, so pass --python <exe> for the one that"
    echo "  has the two bindings installed."
    echo "  If the failure is a missing/undefined symbol in the extension: it was built"
    echo "  against older headers, and _native.pyx's ABI check refuses to import on that."
    echo "  Rebuild: ./build.sh release"
    exit 1
  fi
  echo "  python: $("$PYTHON" --version 2>&1)"
fi

# --- 3. Run the pipeline ------------------------------------------------------
runner="scripts/bench/run_all.py"
if [[ ! -f "$runner" ]]; then
  echo "runner not found: $runner"
  exit 1
fi

runner_args=("$runner")
if [[ $FULL -eq 1 ]]; then
  echo "=== running FULL suite (includes XL; expect tens of minutes) ==="
else
  runner_args+=(--quick)
  echo "=== running QUICK suite (Tiny/Small native; skips XL and Medium/Large burst) ==="
fi
[[ $REPORT_ONLY -eq 1 ]] && runner_args+=(--report-only)
[[ $SWEEPS -eq 1 ]] && runner_args+=(--sweeps)
if [[ ${#suites[@]} -gt 0 ]]; then runner_args+=(--suites "${suites[@]}"); fi
[[ -n "$TAG" ]] && runner_args+=(--tag "$TAG")
[[ -n "$OUT" ]] && runner_args+=(--out "$OUT")
[[ -n "$REPORTS_DIR" ]] && runner_args+=(--reports-dir "$REPORTS_DIR")
[[ -n "$BENCHMARKS_MD" ]] && runner_args+=(--benchmarks-md "$BENCHMARKS_MD")

# Suppressing the committed outputs. `--reports-dir=` is the single-token form argparse
# reads as an empty value, which is run_all.py's documented "skip" signal.
[[ $NO_REPORTS -eq 1 ]] && runner_args+=("--reports-dir=")
[[ $NO_BENCHMARKS_MD -eq 1 ]] && runner_args+=("--benchmarks-md=")

echo ""
echo "Progress: each suite streams its own output, plus a 60s heartbeat and"
echo "'done in Nm SSs' per command on stderr."
echo ""

# A subset run rewrites the COMMITTED report from whatever records are in --out, so a
# native-only run replaces the three-suite report with a one-suite report AT THE SAME
# PATH. Measured on this machine: `--report-only --out <scratch>` overwrote
# bench-reports/report_windows-amd64.md and BENCHMARKS.md with a native-only report and
# exited 0 — a green run that destroyed the committed artifact. Anything that is not a
# full run should say so out loud and name the escape hatch.
if [[ $NO_REPORTS -eq 0 ]] && { [[ $REPORT_ONLY -eq 1 ]] || [[ ${#suites[@]} -gt 0 ]]; }; then
  echo "  NOTE: partial run. This WILL rewrite the committed"
  echo "  bench-reports/report_<tag>.md and BENCHMARKS.md from a SUBSET of the suites."
  echo "  For a scratch run add:  --no-reports --no-benchmarks-md"
  echo ""
fi

"$PYTHON" "${runner_args[@]}"
code=$?
echo ""

# Derive the tag the same way run_all.py does (platform.system/machine lowercased), so
# this message cannot disagree with the filenames the runner actually wrote.
derived_tag=$("$PYTHON" -c "import platform;print(f'{platform.system().lower()}-{platform.machine().lower()}')" 2>/dev/null)
[[ -z "$derived_tag" ]] && derived_tag="unknown"
[[ -n "$TAG" ]] && derived_tag="$TAG"

report="bench-reports/report_${derived_tag}.md"

if [[ $code -ne 0 ]]; then
  # A suite failing is not fatal by design: the runner reports it, continues with the
  # other suites, and exits non-zero. A partial result is the expected outcome of a
  # first run on a new machine. Do NOT print the green report line here — on a failed
  # run there may be no report at all, and pointing at a file that does not exist (or at
  # a stale one from an earlier run) is how a bad run gets quoted.
  echo "Pipeline exited non-zero ($code). Partial results are expected on a"
  echo "first run on a new machine — read the suite output above before re-running."
  if [[ -f "$report" ]]; then
    echo "A report exists at $report, but it is from this FAILED run — check its"
    echo "provenance header (vcs_rev, vcs_dirty, submodule_dirty) before quoting it,"
    echo "and note it may cover fewer suites than a full run."
  fi
  exit $code
fi

if [[ $NO_REPORTS -eq 1 ]]; then
  echo "Reports suppressed (--no-reports): nothing written to bench-reports/ or BENCHMARKS.md."
elif [[ -f "$report" ]]; then
  printf '  \033[32mReport (committed): %s\033[0m\n' "$report"
else
  echo "Runner exited 0 but $report is absent."
  echo "Check --reports-dir (pass '' to suppress reports) and the run output above."
fi
echo "Raw records (ignored): bench-results/records_${derived_tag}_*.json"
echo ""
echo "Compare this report's per-pair GEOMEANS against the other platforms' reports"
echo "(report_windows-amd64.md, report_darwin-arm64.md). Absolute µs columns are NOT"
echo "comparable across machines — merge.py enforces that."
exit $code
