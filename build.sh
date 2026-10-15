#!/usr/bin/env bash
# build.sh — Build all EventProp projects (C, C++, .NET, Python).
# Usage: ./build.sh [debug|release]
#
# The CMake projects are driven by their CMakePresets.json (see AGENTS.md
# "CMake Presets"). Build order matters: the C library and the Eigen core are
# imported pre-built by the tests, parity tests and native benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-release}"

if [[ "$BUILD_TYPE" == "debug" ]]; then
  PRESET="debug"
  CMAKE_BUILD_TYPE="Debug"
  DOTNET_CONFIG="Debug"
elif [[ "$BUILD_TYPE" == "release" ]]; then
  PRESET="release"
  CMAKE_BUILD_TYPE="Release"
  DOTNET_CONFIG="Release"
else
  echo "Usage: $0 [debug|release]"
  exit 1
fi

# configure+build a CMake project through its presets, from the project dir
build_cmake() {
  local dir="$1" preset="$2"
  cd "$SCRIPT_DIR/src/$dir"
  cmake --preset "$preset"
  cmake --build --preset "$preset"
}

echo "============================================"
echo " EventProp Build — $BUILD_TYPE mode"
echo "============================================"

# ── 1. C Library (EventProp.C)
echo ""
echo "[1/9] Building C library (EventProp.C)..."
build_cmake EventProp.C "$PRESET"
echo "  → libeventprop_c_api"

# ── 2. C++ Wrapper Library (EventProp.Cpp.Wrapper)
echo ""
echo "[2/9] Building C++ wrapper library (EventProp.Cpp.Wrapper)..."
build_cmake EventProp.Cpp.Wrapper "$PRESET"
echo "  → libeventprop_cpp_wrapper"

# ── 3. Python Package (EventProp.Python) pybind11 bindings — reference library only
echo ""
echo "[3/9] Building Python package (EventProp.Python)..."
cd "$SCRIPT_DIR/src/EventProp.Python"

# The submodule is read-only and needs local modifications to build here: a
# platform patch plus a pybind11 version bump. scripts/submodule-state.sh owns
# that, is idempotent, and fails loudly — a half-patched submodule surfaces
# later as an opaque compile error inside pybind11/Eigen, far from the cause.
"$SCRIPT_DIR/scripts/submodule-state.sh" apply

# Check if the compiled extension already exists (one-time build)
if ls eventprop/eventprop_cpp.*.so 1>/dev/null 2>&1 || ls eventprop/eventprop_cpp.*.dylib 1>/dev/null 2>&1; then
  echo "  → eventprop.eventprop_cpp already built (skipping)"
  echo "    (re-run after ./clean.sh python -full to rebuild against new patches)"
else
  python3 -m pip install --break-system-packages setuptools numpy pytest > /dev/null 2>&1 || true
  python3 -m pip install --break-system-packages -e . > /dev/null 2>&1 || true
  echo "  → eventprop.eventprop_cpp (editable install)"
fi

# ── 4. Python Benchmarks
echo ""
echo "[4/9] Building Python benchmarks (EventProp.Python.Benchmarks)..."
cd "$SCRIPT_DIR/src/EventProp.Python.Benchmarks"
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  pip install -e . --quiet > /dev/null 2>&1 || true
  deactivate
fi
echo "  → eventprop-python-benchmarks (editable install)"

# ── 5. Python Cython bindings
echo ""
echo "[5/9] Building Python cython bindings (EventProp.Native.Python)..."
cd "$SCRIPT_DIR/src/EventProp.Native.Python"
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  pip install -e . --quiet > /dev/null 2>&1 || true
  deactivate
fi
echo "  → eventprop_c (editable install)"

# ── 6. .NET Libraries (EventProp + EventProp.Native)
echo ""
echo "[6/9] Building .NET libraries..."
cd "$SCRIPT_DIR/src"
rm -rf EventProp/bin EventProp/obj \
       EventProp.Native/bin EventProp.Native/obj

dotnet build EventProp/EventProp.csproj -c "$DOTNET_CONFIG" --nologo
dotnet build EventProp.Native/EventProp.Native.csproj -c "$DOTNET_CONFIG" --nologo
echo "  → CW.EventProp.dll + EventProp.Native.dll"

# ── 7. .NET Tests (EventProp.Tests)
echo ""
echo "[7/9] Building .NET tests..."
cd "$SCRIPT_DIR/src"
rm -rf EventProp.Tests/bin EventProp.Tests/obj

dotnet build EventProp.Tests/EventProp.Tests.csproj -c "$DOTNET_CONFIG" --nologo
echo "  → CW.EventProp.Tests.dll"

# ── 8. Parity Tests (C vs Eigen)
echo ""
echo "[8/9] Building parity tests (C vs Eigen)..."
build_cmake EventProp.Native.ParityTests "$PRESET"
echo "  → parity_test built"

# ── 9. Native Benchmarks (C vs C++)
# Always the release preset: a Debug benchmark measures the compiler, not the algorithm.
echo ""
echo "[9/9] Building native benchmarks (C vs C++)..."
build_cmake EventProp.Native.Benchmarks "release"
echo "  → bench_c + bench_cpp built"

# ── Done ───────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Build complete — $BUILD_TYPE mode"
echo "============================================"
echo ""
echo "Run tests:    cd src/EventProp.Tests && dotnet run -c $DOTNET_CONFIG --"
echo "Run C tests:  cd src/EventProp.C.Tests && cmake --preset release && cmake --build --preset release && ctest --preset release"
echo "Run parity:   cd src/EventProp.Native.ParityTests && ./build/parity_test ../../EventProp.C/build/libeventprop_c_api.dylib"
echo "Run bench C:  cd src/EventProp.Native.Benchmarks/build && ./bench_c --benchmark_min_time=0.1s"
echo "Run bench C++: cd src/EventProp.Native.Benchmarks/build && ./bench_cpp --benchmark_min_time=0.1s"
echo ""
