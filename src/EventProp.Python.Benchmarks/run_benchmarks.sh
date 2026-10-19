#!/bin/bash
# Run all benchmarks (Python + .NET) and generate unified HTML report.
#
# Usage:
#   ./run_benchmarks.sh              # Run everything, generate report
#   ./run_benchmarks.sh python       # Only Python benchmarks
#   ./run_benchmarks.sh dotnet       # Only .NET benchmarks  
#   ./run_benchmarks.sh report       # Generate report from existing JSON files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../.."  # eventprop.net/src/
ROOT_DIR="$SCRIPT_DIR/../../.."  # eventprop.net/

PY_JSON="$SCRIPT_DIR/results_py.json"
BDN_RESULTS_DIR="$SRC_DIR/EventProp.Benchmarks/BenchmarkDotNet.Artifacts/results"
REPORT="$SCRIPT_DIR/benchmark_report.html"

run_python() {
    echo "=== Running Python benchmarks (pytest-benchmark) ==="
    cd "$SCRIPT_DIR"
    
    # Ensure both packages are importable
    export PYTHONPATH="$SRC_DIR/EventProp.Python:$SRC_DIR/EventProp.Native.Python:${PYTHONPATH}"
    
    python -m pytest test_benchmarks.py \
        --benchmark-only \
        --benchmark-json="$PY_JSON" \
        --benchmark-min-rounds=3 \
        --benchmark-max-time=0.15 \
        --tb=no -q
    
    echo "Python benchmarks saved to: $PY_JSON"
}

run_dotnet() {
    echo ""
    echo "=== Running .NET benchmarks (BenchmarkDotNet) ==="
    cd "$SRC_DIR/EventProp.Benchmarks"
    
    # Run all core benchmarks (skip XL for speed, add --filter if needed)
    dotnet run -c Release \
        -- --exporters json \
        --iteration-time 250ms
    
    echo ".NET results saved to: $BDN_RESULTS_DIR/"
}

generate_report() {
    echo ""
    echo "=== Generating unified HTML report ==="
    cd "$SCRIPT_DIR"
    
    # Collect all JSON files (check multiple locations)
    PY_FILES=""
    if [ -f "$PY_JSON" ]; then
        PY_FILES="$PY_JSON"
    elif ls .benchmarks/*/0002_baseline.json 1>/dev/null 2>&1; then
        PY_FILES=$(ls .benchmarks/*/0002_baseline.json 2>/dev/null | head -1)
    elif ls .benchmarks/*/0*.json 1>/dev/null 2>&1; then
        # Try largest file first (most likely to have full benchmark data)
        PY_FILES=$(ls -S .benchmarks/*/0*.json 2>/dev/null | head -1)
    fi
    
    BDN_FILES=$(find "$BDN_RESULTS_DIR" -name "*-report-full-compressed.json" 2>/dev/null || true)
    
    if [ -z "$PY_FILES" ] && [ -z "$BDN_FILES" ]; then
        echo "No benchmark JSON files found!"
        echo "Run benchmarks first: ./run_benchmarks.sh python && ./run_benchmarks.sh dotnet"
        exit 1
    fi
    
    # Generate report from all available JSON files
    python generate_report.py $PY_FILES $BDN_FILES
    
    echo ""
    echo "Report generated: $REPORT"
    if command -v open &> /dev/null; then
        echo "Opening in browser..."
        open "$REPORT" 2>/dev/null || true
    fi
}

# Main
case "${1:-all}" in
    python)   run_python ;;
    dotnet)   run_dotnet ;;
    report)   generate_report ;;
    all)      run_python && run_dotnet && generate_report ;;
    *)        echo "Usage: $0 [python|dotnet|report|all]" ; exit 1 ;;
esac
