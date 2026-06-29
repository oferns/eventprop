#!/usr/bin/env bash
# build.sh — Build all EventProp projects (C, C++, .NET, Python).
# Usage: ./build.sh [debug|release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-release}"
CMAKE_BUILD_DIR="build"

if [[ "$BUILD_TYPE" == "debug" ]]; then
  CMAKE_BUILD_TYPE="Debug"
  DOTNET_CONFIG="Debug"
elif [[ "$BUILD_TYPE" == "release" ]]; then
  CMAKE_BUILD_TYPE="Release"
  DOTNET_CONFIG="Release"
else
  echo "Usage: $0 [debug|release]"
  exit 1
fi

echo "============================================"
echo " EventProp Build — $BUILD_TYPE mode"
echo "============================================"

# ── 1. C Library (EventProp.C)
echo ""
echo "[1/9] Building C library (EventProp.C)..."
cd "$SCRIPT_DIR/src/EventProp.C"
rm -rf "$CMAKE_BUILD_DIR"
mkdir -p "$CMAKE_BUILD_DIR" && cd "$CMAKE_BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
cmake --build . --config "$CMAKE_BUILD_TYPE"
echo "  → libeventprop_c_api"

# ── 2. C++ Wrapper Library (EventProp.Cpp.Wrapper)
echo ""
echo "[2/9] Building C++ wrapper library (EventProp.Cpp.Wrapper)..."
cd "$SCRIPT_DIR/src/EventProp.Cpp.Wrapper"
rm -rf "$CMAKE_BUILD_DIR"
mkdir -p "$CMAKE_BUILD_DIR" && cd "$CMAKE_BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
cmake --build . --config "$CMAKE_BUILD_TYPE"
echo "  → libeventprop_cpp_wrapper"

# ── 3. Python Package (EventProp.Python) pybind11 bindings — reference library only
echo ""
echo "[3/9] Building Python package (EventProp.Python)..."
cd "$SCRIPT_DIR/src/EventProp.Python"

# Check if the compiled extension already exists (one-time build)
if ls eventprop/eventprop_cpp.*.so 1>/dev/null 2>&1 || ls eventprop/eventprop_cpp.*.dylib 1>/dev/null 2>&1; then
  echo "  → eventprop.eventprop_cpp already built (skipping)"
else
  # Reset patched files to upstream state, apply platform patch, then build
  git checkout -- eventprop/loss_layer.py setup.py 2>/dev/null || true
  if [[ "$OSTYPE" == "darwin"* ]]; then
    PATCH_FILE="$SCRIPT_DIR/submodule-macos.patch"
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PATCH_FILE="$SCRIPT_DIR/submodule-linux.patch"
  else
    echo "WARNING: Unknown platform ($OSTYPE) — skipping submodule patch"
    PATCH_FILE=""
  fi
  if [[ -n "${PATCH_FILE:-}" && -f "$PATCH_FILE" ]]; then
    patch -p1 < "$PATCH_FILE" || true
    echo "  → Applied $(basename $PATCH_FILE)"
  fi

  # Install required Python dependencies and build
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
cd "$SCRIPT_DIR/src/EventProp.Native.ParityTests"
rm -rf build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
cmake --build . --config "$CMAKE_BUILD_TYPE"
echo "  → parity_test built"

# ── 9. Native Benchmarks (C vs C++)
echo ""
echo "[9/9] Building native benchmarks (C vs C++)..."
cd "$SCRIPT_DIR/src/EventProp.Native.Benchmarks"
rm -rf build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
cmake --build . --config "$CMAKE_BUILD_TYPE"
echo "  → bench_c + bench_cpp built"

# ── Done ───────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Build complete — $BUILD_TYPE mode"
echo "============================================"
echo ""
echo "Run tests:    cd src/EventProp.Tests && dotnet run -c $DOTNET_CONFIG --"
echo "Run parity:   cd src/EventProp.Native.ParityTests/build && ./parity_test ../../EventProp.C/build/libeventprop_c_api.dylib"
echo "Run bench C:  cd src/EventProp.Native.Benchmarks/build && ./bench_c --benchmark_min_time=0.1s"
echo "Run bench C++: cd src/EventProp.Native.Benchmarks/build && ./bench_cpp --benchmark_min_time=0.1s"
echo ""
