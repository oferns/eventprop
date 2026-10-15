#!/usr/bin/env bash
# clean.sh — Remove all build outputs from EventProp projects.
# Usage: ./clean.sh [-full] [target ...]
#   targets: all | c | cpp | python | dotnet | parity | benchmarks
#
#   -full    Also delete the EventProp.Python reference binary (forces rebuild)
#
# Targets compose: `./clean.sh c cpp parity` cleans all three. No target means all.
# An unrecognised argument is an error, not a silent no-op.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FULL=false
TARGETS=()
for arg in "$@"; do
  case "$arg" in
    -full|--full) FULL=true ;;
    all)          TARGETS=(all) ;;
    c|cpp|python|dotnet|parity|benchmarks) TARGETS+=("$arg") ;;
    *)
      echo "clean.sh: unknown target '$arg'" >&2
      echo "Usage: ./clean.sh [-full] [all|c|cpp|python|dotnet|parity|benchmarks ...]" >&2
      exit 2 ;;
  esac
done
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=(all); fi

# has <target> — true if that target is selected, or `all` is.
has() {
  local want="$1" t
  for t in "${TARGETS[@]}"; do
    [ "$t" = "all" ] && return 0
    [ "$t" = "$want" ] && return 0
  done
  return 1
}

echo "Cleaning EventProp build outputs (targets: ${TARGETS[*]}, full: $FULL)..."

# ── C Library (EventProp.C) ────────────────────────────────
if has c; then
  echo "[1/8] Cleaning EventProp.C..."
  # EventProp.C.Tests/build was never cleaned by any target, including `all` —
  # ~30 MB of GTest + test objects left behind forever.
  rm -rf "$SCRIPT_DIR/src/EventProp.C/build" "$SCRIPT_DIR/src/EventProp.C/build-werror" \
         "$SCRIPT_DIR/src/EventProp.C.Tests/build" "$SCRIPT_DIR/src/EventProp.C.Tests/build-werror"
fi

# ── C++ Wrapper (EventProp.Cpp.Wrapper) ────────────────────
if has cpp; then
  echo "[2/8] Cleaning EventProp.Cpp.Wrapper..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Cpp.Wrapper/build"
fi

# ── Python Package (EventProp.Python) ──────────────────────
if has python; then
  echo "[3/8] Cleaning EventProp.Python..."
  cd "$SCRIPT_DIR/src/EventProp.Python"
  pip uninstall eventprop -y --quiet 2>/dev/null || true
  rm -rf build/ *.egg-info/ eventprop/__pycache__ eventprop/**/*.pyc
  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
  if [[ "$FULL" == "true" ]]; then
    # Delete reference binary — forces rebuild on next build.sh run
    rm -f eventprop/eventprop_cpp.*.so eventprop/eventprop_cpp.*.dylib
    echo "  → Deleted reference binary (use build.sh to rebuild)"
  fi
fi

# ── Python Cython Bindings (EventProp.Native.Python) ───────
if has python; then
  echo "[4/8] Cleaning EventProp.Native.Python..."
  cd "$SCRIPT_DIR/src/EventProp.Native.Python"
  pip uninstall eventprop_c -y --quiet 2>/dev/null || true
  rm -rf build/ *.egg-info/ eventprop_c/*.so \
         eventprop_c/__pycache__ 
  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── Python Benchmarks (EventProp.Python.Benchmarks) ────────
if has python; then
  echo "[5/8] Cleaning EventProp.Python.Benchmarks..."
  cd "$SCRIPT_DIR/src/EventProp.Python.Benchmarks"
  rm -rf .venv/ build/ *.egg-info/
  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── .NET Projects ──────────────────────────────────────────
if has dotnet; then
  echo "[6/8] Cleaning .NET projects..."
  cd "$SCRIPT_DIR/src"
  # EventProp.Cpp.Native is not a project: src/EventProp.Cpp.Native is an empty,
  # untracked directory. EventProp.Abstractions is a real project and must be here.
  for proj in EventProp EventProp.Abstractions EventProp.Native EventProp.Tests EventProp.Benchmarks; do
    rm -rf "$proj/bin" "$proj/obj"
  done
fi

# ── Parity Tests (C vs Eigen) ──────────────────────────────
if has parity; then
  echo "[7/8] Cleaning parity tests..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Native.ParityTests/build"
fi

# ── C/C++ Benchmark Projects ───────────────────────────────
if has benchmarks; then
  echo "[8/8] Cleaning benchmark projects..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Native.Benchmarks/build"
fi

echo "Done."
