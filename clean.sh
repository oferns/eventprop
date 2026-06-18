#!/usr/bin/env bash
# clean.sh — Remove all build outputs from EventProp projects.
# Usage: ./clean.sh [-full] [all|c|cpp|python|dotnet]
#
#   -full    Also delete the EventProp.Python reference binary (forces rebuild)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parse args: accept -full flag in any position, rest is target
FULL=false
TARGET="all"
for arg in "$@"; do
  case "$arg" in
    -full|--full) FULL=true ;;
    *) TARGET="$arg" ;;
  esac
done

echo "Cleaning EventProp build outputs (target: $TARGET, full: $FULL)..."

# ── C Library (EventProp.C) ────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "c" ]]; then
  echo "[1/8] Cleaning EventProp.C..."
  rm -rf "$SCRIPT_DIR/src/EventProp.C/build"
fi

# ── C++ Wrapper (EventProp.Cpp.Wrapper) ────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "cpp" ]]; then
  echo "[2/8] Cleaning EventProp.Cpp.Wrapper..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Cpp.Wrapper/build"
fi

# ── Python Package (EventProp.Python) ──────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "python" ]]; then
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
if [[ "$TARGET" == "all" || "$TARGET" == "python" ]]; then
  echo "[4/8] Cleaning EventProp.Native.Python..."
  cd "$SCRIPT_DIR/src/EventProp.Native.Python"
  pip uninstall eventprop_c -y --quiet 2>/dev/null || true
  rm -rf build/ *.egg-info/ eventprop_c/*.so \
         eventprop_c/__pycache__ 
  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── Python Benchmarks (EventProp.Python.Benchmarks) ────────
if [[ "$TARGET" == "all" || "$TARGET" == "python" ]]; then
  echo "[5/8] Cleaning EventProp.Python.Benchmarks..."
  cd "$SCRIPT_DIR/src/EventProp.Python.Benchmarks"
  rm -rf .venv/ build/ *.egg-info/
  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── .NET Projects ──────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "dotnet" ]]; then
  echo "[6/8] Cleaning .NET projects..."
  cd "$SCRIPT_DIR/src"
  for proj in EventProp EventProp.Native EventProp.Cpp.Native EventProp.Tests EventProp.Benchmarks; do
    rm -rf "$proj/bin" "$proj/obj"
  done
fi

# ── Parity Tests (C vs Eigen) ──────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "parity" ]]; then
  echo "[7/8] Cleaning parity tests..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Native.ParityTests/build"
fi

# ── C/C++ Benchmark Projects ───────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "benchmarks" ]]; then
  echo "[8/8] Cleaning benchmark projects..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Native.Benchmarks/build"
fi

echo "Done."
