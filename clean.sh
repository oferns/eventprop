#!/usr/bin/env bash
# clean.sh — Remove all build outputs from EventProp projects.
# Usage: ./clean.sh [all|c|cpp|python|dotnet]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-all}"

echo "Cleaning EventProp build outputs (target: $TARGET)..."

# ── C Library (EventProp.C) ────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "c" ]]; then
  echo "[1/5] Cleaning EventProp.C..."
  rm -rf "$SCRIPT_DIR/src/EventProp.C/build"
fi

# ── C++ Wrapper (EventProp.Cpp.Wrapper) ────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "cpp" ]]; then
  echo "[2/5] Cleaning EventProp.Cpp.Wrapper..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Cpp.Wrapper/build"
fi

# ── Python Package (EventProp.Python) ──────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "python" ]]; then
  echo "[3/5] Cleaning EventProp.Python..."
  cd "$SCRIPT_DIR/src/EventProp.Python"
  pip uninstall eventprop -y --quiet 2>/dev/null || true
  rm -rf build/ *.egg-info/ eventprop/*.so eventprop/*.dylib \
         eventprop/__pycache__ eventprop/**/*.pyc
  find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
fi

# ── .NET Projects ──────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "dotnet" ]]; then
  echo "[4/5] Cleaning .NET projects..."
  cd "$SCRIPT_DIR/src"
  for proj in EventProp EventProp.Native EventProp.Cpp.Native EventProp.Tests; do
    rm -rf "$proj/bin" "$proj/obj"
  done
fi

# ── C/C++ Benchmark Projects ───────────────────────────────
if [[ "$TARGET" == "all" ]]; then
  echo "[5/5] Cleaning benchmark projects..."
  rm -rf "$SCRIPT_DIR/src/EventProp.Native.Benchmarks/build" \
         "$SCRIPT_DIR/src/EventProp.Native.ParityTests/build"
  # Remove compiled binaries from benchmarks
  find "$SCRIPT_DIR/src/EventProp.Native.Benchmarks" -maxdepth 1 -type f \( -name "*.o" -o -executable \) ! -name "*.cpp" ! -name "*.c" -exec rm -f {} + 2>/dev/null || true
fi

echo "Done."
