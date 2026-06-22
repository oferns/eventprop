#!/usr/bin/env python3
"""Generate unified HTML benchmark report from pytest-benchmark + BenchmarkDotNet JSON.

Usage:
    # Run Python benchmarks (pytest-benchmark) → JSON
    cd src/EventProp.Python.Benchmarks
    pytest test_benchmarks.py --benchmark-only --benchmark-json=results_py.json \
        --benchmark-min-rounds=3 --benchmark-max-time=0.15

    # Run .NET benchmarks (BenchmarkDotNet) → JSON
    cd src/EventProp.Benchmarks
    dotnet run -c Release -- --exporters json

    # Generate report from both
    python generate_report.py results_py.json ../EventProp.Benchmarks/BenchmarkDotNet.Artifacts/results/*.json

Output: benchmark_report.html (self-contained, no external deps)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# ============================================================
# Data model — unified representation of a single benchmark result.
# ============================================================

class BenchResult:
    """Single benchmark measurement."""
    __slots__ = ('category', 'size', 'pattern', 'backend', 'mean_us', 'min_us', 'max_us', 'stddev_us')

    def __init__(self, category: str, size: str, pattern: str, backend: str, mean_us: float,
                 min_us: float = 0.0, max_us: float = 0.0, stddev_us: float = 0.0):
        self.category = category   # e.g. "LIF Forward", "ComputeSums"
        self.size = size           # e.g. "Small", "Medium", or batch_size "4"
        self.pattern = pattern     # e.g. "uniform", "poisson", "" for non-patterned
        self.backend = backend     # e.g. "pybind11", "Cython", "Managed", "Native(C)"
        self.mean_us = mean_us
        self.min_us = min_us
        self.max_us = max_us
        self.stddev_us = stddev_us


# ============================================================
# Parser: pytest-benchmark JSON (--benchmark-json output).
# Structure: {"benchmarks": [{"name": "...", "params": {...}, "stats": {"mean": 0.001}}]}
# Times are in SECONDS.
# ============================================================

def parse_pytest_json(path: str) -> list[BenchResult]:
    """Parse pytest-benchmark JSON output."""
    with open(path) as f:
        data = json.load(f)

    results = []
    for entry in data.get("benchmarks", []):
        name = entry["name"]
        params = entry.get("params", {})
        stats = entry.get("stats", {})
        mean_s = stats.get("mean")
        if mean_s is None:
            continue

        # Extract category from test class/method name pattern
        # e.g. "test_lif_forward_pybind11[uniform-Small]" → category="LIF Forward", backend="pybind11"
        cat, backend = _parse_pytest_name(name)
        if not cat or not backend:
            continue

        size = params.get("size_name") or params.get("batch_size", "")
        pattern = params.get("pattern", "")

        results.append(BenchResult(
            category=cat,
            size=str(size),
            pattern=pattern,
            backend=backend,
            mean_us=mean_s * 1e6,
            min_us=stats.get("min", 0) * 1e6,
            max_us=stats.get("max", 0) * 1e6,
            stddev_us=stats.get("stddev", 0) * 1e6,
        ))

    return results


def _parse_pytest_name(name: str) -> tuple[str | None, str | None]:
    """Extract (category, backend) from pytest benchmark name.

    Examples:
        test_lif_forward_pybind11[uniform-Small] → ("LIF Forward", "pybind11")
        test_li_backward_cython[poisson-Medium]  → ("LI Backward", "Cython")
        test_ttfs_step[4]                        → ("TTFS Step", "pybind11" or "Cython")
    """
    # Remove param suffix [...]
    base = re.sub(r'\[.*\]', '', name)

    # Map backend keywords
    if 'pybind11' in base:
        backend = "pybind11 (Eigen)"
    elif 'cython' in base or 'ctypes' in base:
        backend = "Cython (C lib)"
    else:
        return None, None

    # Map operation keywords → human-readable category
    cat_map = [
        ('lif_forward', 'LIF Forward'),
        ('lif_backward', 'LIF Backward'),
        ('li_forward', 'LI Forward'),
        ('li_backward', 'LI Backward'),
        ('ttfs_forward', 'TTFS Forward'),
        ('ttfs_backward', 'TTFS Backward'),
        ('vmax_forward', 'VMax Forward'),
        ('vmax_backward', 'VMax Backward'),
        ('batch', 'Batch Forward'),
        ('dropout', 'Dropout'),
        ('sweep_spikes', 'Sweep (fixed neurons, varying spikes)'),  # param = spike count
        ('sweep_neurons', 'Sweep (fixed spikes, varying neurons)'),  # param = size name
        ('training_step', 'Training Step'),  # size extracted from param
        ('ttfs_step', 'Training Step (TTFS)'),
        ('vmax_step', 'Training Step (VMax)'),
    ]

    for key, label in cat_map:
        if key in base.lower():
            return label, backend

    return None, None


# ============================================================
# Parser: BenchmarkDotNet JSON.
# Structure varies by version but typically has "Benchmarks" array with Statistics.Mean in ns.
# ============================================================

def parse_bdn_json(path: str) -> list[BenchResult]:
    """Parse BenchmarkDotNet JSON output.
    
    Structure: {"Benchmarks": [{"FullName": "Bench_ComputeSums_Small.Managed", 
                               "Statistics": {"Mean": 1641.5, ...}, ...}]}
    Times are in NANOSECONDS.
    """
    with open(path) as f:
        data = json.load(f)

    results = []
    benchmarks = data.get("Benchmarks", [])

    for entry in benchmarks:
        stats = entry.get("Statistics") or {}
        mean_ns = stats.get("Mean")
        if mean_ns is None:
            continue

        # FullName: "EventProp.Benchmarks.Bench_ComputeSums_Small.Managed"
        full_name = entry.get("FullName", "") or entry.get("DisplayInfo", "")
        cat, size, backend = _parse_bdn_name(full_name)
        if not cat or not backend:
            continue

        results.append(BenchResult(
            category=cat,
            size=size,
            pattern="uniform",  # .NET benchmarks use uniform only
            backend=backend,
            mean_us=mean_ns / 1000.0,  # ns → μs
            min_us=stats.get("Min", 0) / 1000.0,
            max_us=stats.get("Max", 0) / 1000.0,
            stddev_us=stats.get("StdError", stats.get("StdDev", 0)) / 1000.0,
        ))

    return results


def _parse_bdn_name(full_name: str) -> tuple[str | None, str | None, str | None]:
    """Extract (category, size, backend) from BenchmarkDotNet FullName.
    
    Examples:
        "EventProp.Benchmarks.Bench_ComputeSums_Small.Managed" → ("ComputeSums", "Small", "Managed (.NET)")
        "Bench_LIF_Forward_Medium.Native" → ("LIF Forward", "Medium", "Native (C lib)")
    """
    # Split on dot to get class and method
    parts = full_name.rsplit('.', 1)
    if len(parts) != 2:
        return None, None, None
    
    class_part, method = parts[0], parts[1]
    
    # Extract backend from method name
    backend_map = {
        'Managed': 'Managed (.NET)',
        'Native': 'Native (C lib)',
        'Managed_PerSampleLoop': 'Managed (.NET)',
        'Native_BatchApi': 'Native (C lib)',
    }
    backend = backend_map.get(method)
    if not backend:
        return None, None, None

    # Parse class name: Bench_Category_Size → category, size
    # Also handle batch benchmarks and sweep spike counts (numeric suffixes)
    m = re.search(r'Bench_(.+?)_(Small|Medium|Large|Tiny|XL)$', class_part)
    if not m:
        m = re.search(r'Bench_(Batch_.+?)_(\d+)$', class_part)  # batch sizes
    if not m:
        m = re.search(r'Bench_(.+?)_(\d+)$', class_part)  # sweep spike counts, etc.
    if not m:
        return None, None, None

    raw_cat = m.group(1)  # e.g. "ComputeSums", "LIF_Forward"
    size = m.group(2)

    # Map to human-readable category
    cat_map = {
        'ComputeSums': 'ComputeSums',
        'ComputeSpikes': 'ComputeSpikes (Forward)',
        'BackwardSpikes': 'Backward (Spikes)',
        'LIF_Forward': 'LIF Forward',
        'LIF_Backward': 'LIF Backward',
        'TrainingLoop': 'Training Step',  # matches Python TestTrainingStepLIF
        'Batch_TrainingStep': 'Training Step (Small)',  # legacy batch benchmarks
        'Batch_Forward': 'Batch Forward',
        'TrainingLoopSweepSpikes': 'Sweep (fixed neurons, varying spikes)',
        'TrainingLoopSweepNeurons': 'Sweep (fixed spikes, varying neurons)',
    }

    return cat_map.get(raw_cat, raw_cat.replace('_', ' ')), size, backend


# ============================================================
# HTML report generation.
# ============================================================

def generate_html(results: list[BenchResult], output_path: str = "benchmark_report.html"):
    """Generate self-contained HTML report from benchmark results."""
    # Group by category → size/pattern → backend
    grouped: dict[str, dict[tuple[str, str], dict[str, BenchResult]]] = {}
    for r in results:
        if r.category not in grouped:
            grouped[r.category] = {}
        key = (r.size, r.pattern or "uniform")
        if key not in grouped[r.category]:
            grouped[r.category][key] = {}
        grouped[r.category][key][r.backend] = r

    # Build HTML tables for each category
    tables_html = []
    for cat_name in sorted(grouped.keys()):
        sizes_patterns = sorted(grouped[cat_name].keys())
        backends_all = set()
        for bp in grouped[cat_name].values():
            backends_all.update(bp.keys())
        backends_sorted = sorted(backends_all)

        rows_html = []
        for size, pattern in sizes_patterns:
            cells = [f"<td>{size}</td>", f"<td>{pattern}</td>"]
            row_data = grouped[cat_name][(size, pattern)]

            # Collect values for ratio calculation
            vals = {}
            for b in backends_sorted:
                if b in row_data:
                    vals[b] = row_data[b].mean_us

            # Find fastest as baseline for ratios
            min_val = min(vals.values()) if vals else 1.0

            for b in backends_sorted:
                if b in row_data:
                    r = row_data[b]
                    ratio = f"{r.mean_us / min_val:.2f}x" if min_val > 0 else "-"
                    bar_width = max(1, int((min_val / r.mean_us) * 100)) if r.mean_us > 0 else 0
                    cells.append(f'<td class="num">{r.mean_us:,.2f}<br><span class="ratio">{ratio}</span></td>')
                else:
                    cells.append('<td class="empty">—</td>')

            rows_html.append("<tr>" + "".join(cells) + "</tr>")

        header_cells = ["<th>Size</th>", "<th>Pattern</th>"]
        for b in backends_sorted:
            short = b.replace(" (Eigen)", "").replace(" (C lib)", "").replace(" (.NET)", "")
            header_cells.append(f'<th>{short}</th>')

        tables_html.append(f"""
        <h2>{cat_name}</h2>
        <table>
          <thead><tr>{"".join(header_cells)}</tr></thead>
          <tbody>{"".join(rows_html)}</tbody>
        </table>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EventProp Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2em; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 2px solid #4a90d9; padding-bottom: 0.5em; }}
  h2 {{ color: #4a90d9; margin-top: 2em; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 1200px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f5f7fa; font-weight: 600; position: sticky; top: 0; }}
  .num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  .ratio {{ color: #888; font-size: 0.85em; }}
  .empty {{ color: #ccc; text-align: center; }}
  tr:hover {{ background: #f9fafb; }}
  .summary {{ display: flex; gap: 1em; margin: 1em 0; flex-wrap: wrap; }}
  .card {{ background: white; padding: 1em 1.5em; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 200px; }}
  .card h3 {{ margin: 0 0 0.5em; font-size: 0.9em; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.8em; font-weight: 700; color: #4a90d9; }}
</style>
</head>
<body>
<h1>EventProp Benchmark Report</h1>
<p>Mean time per call (μs) — lower is better. Ratio shows slowdown vs fastest backend.</p>

<div class="summary">
  <div class="card"><h3>Total Benchmarks</h3><div class="value">{len(results)}</div></div>
  <div class="card"><h3>Categories</h3><div class="value">{len(grouped)}</div></div>
  <div class="card"><h3>Backends</h3><div class="value">{len(set(r.backend for r in results))}</div></div>
</div>

{"".join(tables_html)}

<hr style="margin: 3em 0; border: none; border-top: 1px solid #eee;">
<p style="color: #999; font-size: 0.85em;">Generated by EventProp benchmark report generator</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to {output_path}")


# ============================================================
# Main entry point.
# Usage: generate_report.py pytest_results.json [bdn_results.json ...]
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_report.py pytest_benchmark.json [bdn_result1.json bdn_result2.json ...]")
        sys.exit(1)

    all_results = []

    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        # Detect format by content
        with open(p) as f:
            raw = json.load(f)

        if "benchmarks" in raw and isinstance(raw["benchmarks"], list):
            print(f"Parsing pytest-benchmark: {path}")
            all_results.extend(parse_pytest_json(path))
        elif "Benchmarks" in raw or "Statistics" in raw:
            print(f"Parsing BenchmarkDotNet: {path}")
            all_results.extend(parse_bdn_json(path))
        else:
            print(f"Unknown format: {path}, skipping")

    if not all_results:
        print("No benchmark data found!")
        sys.exit(1)

    print(f"Total results parsed: {len(all_results)}")
    generate_html(all_results, "benchmark_report.html")


if __name__ == "__main__":
    main()