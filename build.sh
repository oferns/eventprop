#!/usr/bin/env bash
# build.sh — Build all EventProp projects in one command.
# Usage: ./build.sh [debug|release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-release}"
BUILD_DIR="build"

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
echo "[1/3] Building C library (EventProp.C)..."
cd "$SCRIPT_DIR/src/EventProp.C"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "  → libeventprop_c_api.dylib"

# ── 2. C++ Wrapper Library (EventProp.Cpp.Wrapper) ─────────
#    Builds Eigen core from EventProp.Python sources + wrapper on top.
echo ""
echo "[2/3] Building C++ wrapper library (EventProp.Cpp.Wrapper)..."
cd "$SCRIPT_DIR/src/EventProp.Cpp.Wrapper"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "  → libeventprop_cpp_wrapper.dylib"

# ── 3. .NET Projects (EventProp + EventProp.Tests) ─────────
echo ""
echo "[3/3] Building .NET projects (EventProp + EventProp.Tests)..."
cd "$SCRIPT_DIR/src"
rm -rf EventProp/bin EventProp/obj \
       EventProp.Native/bin EventProp.Native/obj \
       EventProp.Cpp.Native/bin EventProp.Cpp.Native/obj \
       EventProp.Tests/bin EventProp.Tests/obj

dotnet build EventProp/EventProp.csproj -c "$DOTNET_CONFIG" --nologo
dotnet build EventProp.Native/EventProp.Native.csproj -c "$DOTNET_CONFIG" --nologo
dotnet build EventProp.Cpp.Native/EventProp.Cpp.Native.csproj -c "$DOTNET_CONFIG" --nologo
dotnet build EventProp.Tests/EventProp.Tests.csproj -c "$DOTNET_CONFIG" --nologo
echo "  → CW.EventProp.dll + CW.EventProp.Tests.dll"

# ── Done ───────────────────────────────────────────────────
echo ""
echo "============================================"
echo " Build complete — $BUILD_TYPE mode"
echo "============================================"
echo ""
echo "Run tests:  cd src/EventProp.Tests && dotnet run -- -c $DOTNET_CONFIG"
echo "Run tests:  cd src/EventProp.Tests && dotnet run -- -c $DOTNET_CONFIG --output Detailed"
echo ""
