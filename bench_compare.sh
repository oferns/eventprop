#!/bin/bash
# Run C vs C++ benchmarks multiple times and aggregate with median CPU times.
# Usage: bench_compare.sh [num_runs] [min_time_s]
set -euo pipefail

# Detect platform for correct binary paths and extensions
if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == CYGWIN* ]] || [[ "$(uname -s)" == MSYS* ]]; then
    BENCH_SUBDIR="Release"
    BENCH_EXT=".exe"
else
    # Linux/macOS — single-config CMake puts binaries directly in build/
    BENCH_SUBDIR=""
    BENCH_EXT=""
fi

NUM_RUNS="${1:-3}"
MIN_TIME="${2:-0.1}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/src/EventProp.Native.Benchmarks/build"
OUT_DIR="$BUILD_DIR/bench_runs"
BENCH_C="$BUILD_DIR/${BENCH_SUBDIR:+$BENCH_SUBDIR/}bench_c${BENCH_EXT}"
BENCH_CPP="$BUILD_DIR/${BENCH_SUBDIR:+$BENCH_SUBDIR/}bench_cpp${BENCH_EXT}"

# Copy C library DLL next to executable (Windows needs it in same dir or PATH)
if [ -f "$REPO_ROOT/src/EventProp.C/build/Release/eventprop_c_api.dll" ]; then
    cp "$REPO_ROOT/src/EventProp.C/build/Release/eventprop_c_api.dll" "$BUILD_DIR/${BENCH_SUBDIR:+$BENCH_SUBDIR}/" 2>/dev/null || true
fi

mkdir -p "$OUT_DIR"

# Filter to Tiny|Small — C segfaults on Medium+ due to OpenMP contention
# (measured on the native suite; not a data bug). Both backends have 36 benchmarks here.
FILTER="Tiny|Small"

echo "Running $NUM_RUNS iterations of C vs C++ benchmarks (min_time=${MIN_TIME}s, filter=$FILTER)"
echo ""

for i in $(seq 1 "$NUM_RUNS"); do
    echo "[C]   Run $i/$NUM_RUNS..."
    cd "$BUILD_DIR"
    if [ -n "$FILTER" ]; then
        ./${BENCH_SUBDIR:+$BENCH_SUBDIR/}bench_c${BENCH_EXT} --benchmark_min_time="${MIN_TIME}s" --benchmark_filter="$FILTER" --benchmark_format=json > "$OUT_DIR/c_run_${i}.json" 2>/dev/null || true
    else
        ./${BENCH_SUBDIR:+$BENCH_SUBDIR/}bench_c${BENCH_EXT} --benchmark_min_time="${MIN_TIME}s" --benchmark_format=json > "$OUT_DIR/c_run_${i}.json" 2>/dev/null || true
    fi

    echo "[CPP] Run $i/$NUM_RUNS..."
    cd "$BUILD_DIR"
    if [ -n "$FILTER" ]; then
        ./${BENCH_SUBDIR:+$BENCH_SUBDIR/}bench_cpp${BENCH_EXT} --benchmark_min_time="${MIN_TIME}s" --benchmark_filter="$FILTER" --benchmark_format=json > "$OUT_DIR/cpp_run_${i}.json" 2>/dev/null || true
    else
        ./${BENCH_SUBDIR:+$BENCH_SUBDIR/}bench_cpp${BENCH_EXT} --benchmark_min_time="${MIN_TIME}s" --benchmark_format=json > "$OUT_DIR/cpp_run_${i}.json" 2>/dev/null || true
    fi
done

echo ""
echo "Aggregating results (median CPU time per benchmark)..."
echo ""

python3 - "$OUT_DIR" "$NUM_RUNS" <<'PYEOF'
import json, sys, os, statistics
sys.stdout.reconfigure(encoding='utf-8')

out_dir = sys.argv[1]
num_runs = int(sys.argv[2])

def load_benchmarks(prefix):
    data = {}
    for i in range(1, num_runs + 1):
        path = os.path.join(out_dir, f"{prefix}_run_{i}.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                obj = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        # Google Benchmark JSON: {"context": ..., "benchmarks": [...]} or flat array
        runs = obj.get("benchmarks", obj) if isinstance(obj, dict) else obj
        for entry in runs:
            name = entry.get("name", "")
            real = entry.get("real_time")
            if name and real is not None and real > 0:
                data.setdefault(name, []).append(real)
    return data

c_data = load_benchmarks("c")
cpp_data = load_benchmarks("cpp")

all_names = sorted(set(c_data.keys()) | set(cpp_data.keys()))

print(f"{'Benchmark':<50} {'C median(μs)':>14} {'CPP median(μs)':>14} {'Ratio C/CPP':>12}")
print("-" * 92)

for name in all_names:
    c_times = c_data.get(name, [])
    cpp_times = cpp_data.get(name, [])
    if not c_times and not cpp_times:
        continue

    # real_time from Google Benchmark JSON is in nanoseconds
    c_med = statistics.median(c_times) / 1e3 if c_times else None  # ns -> μs
    cpp_med = statistics.median(cpp_times) / 1e3 if cpp_times else None

    c_str = f"{c_med:.2f}" if c_med is not None else "N/A"
    cpp_str = f"{cpp_med:.2f}" if cpp_med is not None else "N/A"

    ratio_str = "N/A"
    if c_med and cpp_med and cpp_med > 0:
        ratio = c_med / cpp_med
        ratio_str = f"{ratio:.1f}x"

    print(f"{name:<50} {c_str:>14} {cpp_str:>14} {ratio_str:>12}")

print()
print("Legend: Ratio C/CPP < 1.0 means C is faster, > 1.0 means C++ is faster")
PYEOF

echo ""
echo "Raw JSON files saved in: $OUT_DIR"
