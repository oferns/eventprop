#!/usr/bin/env bash
# ============================================================
# run_benchmarks.sh - Run C vs C++ benchmarks and compare
#
# Usage:
#   ./run_benchmarks.sh                  Run both, save to baselines/, show comparison
#   ./run_benchmarks.sh c                Run only C benchmark
#   ./run_benchmarks.sh cpp              Run only C++ benchmark
#   ./run_benchmarks.sh compare          Compare last saved results (no re-run)
#   ./run_benchmarks.sh compare A.json B.json  Compare two specific JSON files
#
# Options (set before calling or inline):
#   MIN_TIME=0.1s    Minimum benchmark time per test (default: 0.1s for quick runs)
#   FILTER=ComputeSpikes/Tiny  Run only matching benchmarks (--benchmark_filter)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/Release"
BASELINES_DIR="$SCRIPT_DIR/baselines"
TOOLS_DIR="$SCRIPT_DIR/build/_deps/benchmark-src/tools"

# Check executables exist
if [[ ! -x "$BUILD_DIR/bench_c" ]]; then
    echo "ERROR: bench_c not found in $BUILD_DIR"
    echo "Run CMake build first: cmake --build build --config Release"
    exit 1
fi
if [[ ! -x "$BUILD_DIR/bench_cpp" ]]; then
    echo "ERROR: bench_cpp not found in $BUILD_DIR"
    echo "Run CMake build first: cmake --build build --config Release"
    exit 1
fi

# Default benchmark options
MIN_TIME="${MIN_TIME:-0.1s}"
BENCH_OPTS="--benchmark_min_time=$MIN_TIME"
if [[ -n "${FILTER:-}" ]]; then
    BENCH_OPTS="$BENCH_OPTS --benchmark_filter=$FILTER"
fi

mkdir -p "$BASELINES_DIR"

# Unique ID for output files
DATE_STAMP=$(date +%Y%m%d_%H%M%S)_$$

# Determine action from first argument
ACTION="${1:-}"

case "$ACTION" in
    compare)
        RUN_C=false; RUN_CPP=false; DO_COMPARE=true ;;
    c)
        RUN_C=true; RUN_CPP=false; DO_COMPARE=false ;;
    cpp)
        RUN_C=false; RUN_CPP=true; DO_COMPARE=false ;;
    *)
        RUN_C=true; RUN_CPP=true; DO_COMPARE=true ;;
esac

# Run C benchmark
if $RUN_C; then
    echo ""
    echo "========================================"
    echo "Running C benchmark..."
    echo "========================================"

    # Ensure library is findable (LD_LIBRARY_PATH or rpath)
    export LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}"

    C_OUTPUT="$BASELINES_DIR/bench_c_${DATE_STAMP}.json"
    echo "Output: $C_OUTPUT"

    cd "$BUILD_DIR"
    ./bench_c --benchmark_format=json --benchmark_out="$C_OUTPUT" $BENCH_OPTS

    if [[ $? -ne 0 ]]; then
        echo "ERROR: C benchmark failed!"
        exit 1
    fi
fi

# Run C++ benchmark
if $RUN_CPP; then
    echo ""
    echo "========================================"
    echo "Running C++ benchmark..."
    echo "========================================"

    CPP_OUTPUT="$BASELINES_DIR/bench_cpp_${DATE_STAMP}.json"
    echo "Output: $CPP_OUTPUT"

    cd "$BUILD_DIR"
    ./bench_cpp --benchmark_format=json --benchmark_out="$CPP_OUTPUT" $BENCH_OPTS

    if [[ $? -ne 0 ]]; then
        echo "ERROR: C++ benchmark failed!"
        exit 1
    fi
fi

# Compare results
if $DO_COMPARE; then
    # Find the two most recent JSON files if not specified
    BASELINE=""
    CONTENDER=""

    if [[ $# -lt 2 ]]; then
        # Auto-find latest files
        BASELINE=$(ls -1t "$BASELINES_DIR"/bench_c_*.json 2>/dev/null | head -1) || true
        CONTENDER=$(ls -1t "$BASELINES_DIR"/bench_cpp_*.json 2>/dev/null | head -1) || true

        if [[ -z "$BASELINE" ]]; then
            echo "ERROR: No C benchmark results found in $BASELINES_DIR"
            exit 1
        fi
        if [[ -z "$CONTENDER" ]]; then
            echo "ERROR: No C++ benchmark results found in $BASELINES_DIR"
            exit 1
        fi
    elif [[ $# -eq 2 ]]; then
        BASELINE="$2"
        CONTENDER=$(ls -1t "$BASELINES_DIR"/bench_cpp_*.json 2>/dev/null | head -1) || true
    else
        BASELINE="$2"
        CONTENDER="$3"
    fi

    # Resolve paths (try baselines/ first, then SCRIPT_DIR)
    if [[ -f "$BASELINES_DIR/$BASELINE" ]]; then
        FULL_BASELINE="$BASELINES_DIR/$BASELINE"
    elif [[ -f "$SCRIPT_DIR/$BASELINE" ]]; then
        FULL_BASELINE="$SCRIPT_DIR/$BASELINE"
    else
        echo "ERROR: Cannot find $BASELINE"
        exit 1
    fi

    if [[ -f "$BASELINES_DIR/$CONTENDER" ]]; then
        FULL_CONTENDER="$BASELINES_DIR/$CONTENDER"
    elif [[ -f "$SCRIPT_DIR/$CONTENDER" ]]; then
        FULL_CONTENDER="$SCRIPT_DIR/$CONTENDER"
    else
        echo "ERROR: Cannot find $CONTENDER"
        exit 1
    fi

    echo ""
    echo "========================================"
    echo "Comparing: $(basename "$FULL_BASELINE") vs $(basename "$FULL_CONTENDER")"
    echo "========================================"

    cd "$TOOLS_DIR"
    python3 compare.py benchmarks "$FULL_BASELINE" "$FULL_CONTENDER" --no-utest || {
        echo ""
        echo "WARNING: compare.py exited with code $?"
        echo "This may be due to benchmark name mismatches between runs."
    }
elif ! $DO_COMPARE; then
    echo ""
    echo "Done. Use ./run_benchmarks.sh compare to view comparison."
fi

cd "$SCRIPT_DIR"
exit 0
