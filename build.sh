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

# ── 1. C Library (EventProp.C) ─────────────────────────────
echo ""
echo "[1/5] Building C library (EventProp.C)..."
cd "$SCRIPT_DIR/src/EventProp.C"
rm -rf "$CMAKE_BUILD_DIR"
mkdir -p "$CMAKE_BUILD_DIR" && cd "$CMAKE_BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "  → libeventprop_c_api.dylib"

# ── 2. C++ Wrapper Library (EventProp.Cpp.Wrapper) ─────────
#    Builds Eigen core from stripped copy + wrapper on top.
echo ""
echo "[2/5] Building C++ wrapper library (EventProp.Cpp.Wrapper)..."
cd "$SCRIPT_DIR/src/EventProp.Cpp.Wrapper"
rm -rf "$CMAKE_BUILD_DIR"
mkdir -p "$CMAKE_BUILD_DIR" && cd "$CMAKE_BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "  → libeventprop_cpp_wrapper.dylib"

# ── 3. Python Package (EventProp.Python) ───────────────────
echo ""
echo "[3/5] Building Python package (EventProp.Python)..."
cd "$SCRIPT_DIR/src/EventProp.Python"
pip install -e . --quiet 2>&1 | grep -v "already satisfied" || true
echo "  → eventprop.eventprop_cpp (editable install)"

# ── 4. .NET Libraries (EventProp + EventProp.Native) ───────
echo ""
echo "[4/5] Building .NET libraries..."
cd "$SCRIPT_DIR/src"
rm -rf EventProp/bin EventProp/obj \
       EventProp.Native/bin EventProp.Native/obj \
       EventProp.Cpp.Native/bin EventProp.Cpp.Native/obj

dotnet build EventProp/EventProp.csproj -c "$DOTNET_CONFIG" --nologo
dotnet build EventProp.Native/EventProp.Native.csproj -c "$DOTNET_CONFIG" --nologo
dotnet build EventProp.Cpp.Native/EventProp.Cpp.Native.csproj -c "$DOTNET_CONFIG" --nologo
echo "  → CW.EventProp.dll + EventProp.Native.dll + EventProp.Cpp.Native.dll"

# ── 5. .NET Tests (EventProp.Tests) ────────────────────────
echo ""
echo "[5/5] Building .NET tests..."
cd "$SCRIPT_DIR/src"
rm -rf EventProp.Tests/bin EventProp.Tests/obj

dotnet build EventProp.Tests/EventProp.Tests.csproj -c "$DOTNET_CONFIG" --nologo
echo "  → CW.EventProp.Tests.dll"

# ── Done ───────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Build complete — $BUILD_TYPE mode"
echo "============================================"
echo ""
echo "Run tests:  cd src/EventProp.Tests && dotnet run -c $DOTNET_CONFIG --"
echo "Run parity: cd src/EventProp.Native.ParityTests/build && ./parity_test ../../EventProp.C/build/libeventprop_c_api.dylib"
echo ""
