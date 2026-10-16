#!/usr/bin/env python3
"""
from_google_benchmark.py — native suite adapter for the benchmark result schema.

Converts `bench_c` / `bench_cpp --benchmark_format=json` output into the normalized
record format defined in BENCHMARK_SCHEMA.md.

Usage:
    from_google_benchmark.py <bench.json> --impl C     --out records_native_c.json
    from_google_benchmark.py <bench.json> --impl Eigen --out records_native_eigen.json

Name mapping (verified against baselines/bench_c_20261014_200831_48085.json):

    ComputeSpikes/Tiny/uniform/default/real_time   -> benchmark=ComputeSpikes size=Tiny
                                                      pattern=uniform axis=null
    BatchTrainingStep/Large/uniform/default/batch_size:4
                                                   -> benchmark=FullTrainingStep
                                                      size=Large pattern=uniform
                                                      axis={"batch_size":4}
    BM_ComputeSpikesVth/v_th:5                     -> benchmark=ComputeSpikes
                                                      axis={"v_th":0.5} pattern=null
                                                      size=null  (sweep, not a size cell)
    Dropout/Tiny/uniform/real_time                 -> benchmark=Dropout size=Tiny

Notes on real-world shapes the adapter must tolerate:
  * `BM_` prefix appears on the v_th sweeps only; stripped.
  * `default` is a trailing generator tag, not a pattern.
  * `time_unit` is "ns" today; the multiplier is applied, not assumed.
  * `output_spikes` is absent on Dropout and v_th rows -> null, NOT 0. Zero means
    "measured, produced nothing" (empty-loop regime); null means "not applicable".
    Collapsing the two would mark Dropout invalid for the wrong reason.
  * `c_error` / `c_fwd_error` / `c_bwd_error` are the native suite's error counters.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys

SCHEMA_VERSION = 1

# Google Benchmark time_unit -> nanoseconds
TIME_UNIT_TO_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}

SIZES = {"Tiny", "Small", "Medium", "Large", "XL"}
PATTERNS = {"uniform", "poisson", "burst"}

# Families the schema recognizes. Anything else is reported as unmapped rather than
# silently coerced — a new benchmark must be added to the vocabulary deliberately.
FAMILIES = {
    "ComputeSums", "ComputeSpikes", "ComputeMaxima",
    "BackwardSpikes", "BackwardMaxima",
    "FullTrainingStep", "BatchTrainingStep", "Dropout",
}

# BatchTrainingStep is the batched training step; the schema's family vocabulary has
# FullTrainingStep. Kept as an explicit alias so the mapping is visible, not implicit.
FAMILY_ALIASES = {"BatchTrainingStep": "FullTrainingStep"}


def parse_run_name(run_name):
    """Return (benchmark, size, pattern, axis) or None if unmapped."""
    name = run_name
    # Strip the trailing metric tag Google Benchmark appends.
    for tag in ("/real_time", "/cpu_time", "/manual_time"):
        if name.endswith(tag):
            name = name[: -len(tag)]

    parts = name.split("/")
    family = parts[0]
    # v_th sweeps are registered as BM_<Family>Vth, not <Family>/<size>/<pattern>.
    sweep = family.startswith("BM_")
    if sweep:
        family = family[3:]

    if family not in FAMILIES and family not in FAMILY_ALIASES:
        # e.g. ComputeSpikesVth (sweep suffix)
        for f in FAMILIES:
            if family.startswith(f):
                family = f
                break
        else:
            return None

    benchmark = FAMILY_ALIASES.get(family, family)

    size = None
    pattern = None
    axis = None

    rest = parts[1:]
    for p in rest:
        if p in SIZES:
            size = p
        elif p in PATTERNS:
            pattern = p
        elif p == "default":
            continue                                   # generator tag, not a pattern
        elif ":" in p:
            key, _, val = p.partition(":")
            axis = axis or {}
            axis[key] = _coerce(val)
    return benchmark, size, pattern, axis


def _coerce(value):
    """'5' -> 5, '0.5' -> 0.5, '4' -> 4, else the string."""
    try:
        i = int(value)
        return i
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _v_th_from_axis(axis):
    """BM_ComputeSpikesVth/v_th:5 means 0.5 (the benchmark divides by 10)."""
    if not axis or "v_th" not in axis:
        return axis
    v = axis["v_th"]
    return {"v_th": v / 10.0 if isinstance(v, (int, float)) else v}


def _error_code(bench):
    """Native error counters: any non-zero means the row measures an error path."""
    codes = [bench.get(k) for k in ("c_error", "c_fwd_error", "c_bwd_error")]
    codes = [c for c in codes if c is not None]
    if not codes:
        return None
    return int(max(codes))


def _output_spikes(bench):
    """Absent -> None (not applicable). Present and 0 -> 0 (empty-loop regime)."""
    v = bench.get("output_spikes")
    if v is None:
        return None
    return int(v)


def to_records(doc, impl):
    records = []
    unmapped = []
    for bench in doc.get("benchmarks", []):
        if bench.get("run_type") not in (None, "iteration"):
            continue                                   # skip 'aggregate' summary rows
        parsed = parse_run_name(bench["run_name"])
        if parsed is None:
            unmapped.append(bench["run_name"])
            continue
        benchmark, size, pattern, axis = parsed
        axis = _v_th_from_axis(axis)

        unit = bench.get("time_unit", "ns")
        if unit not in TIME_UNIT_TO_NS:
            raise ValueError(f"unknown time_unit {unit!r} in {bench['run_name']}")
        mean_ns = bench["real_time"] * TIME_UNIT_TO_NS[unit]

        notes = []
        iterations = bench.get("iterations")
        if iterations is not None and iterations < 30:
            notes.append("low_iteration_count")

        records.append({
            "benchmark": benchmark,
            "impl": impl,
            "size": size,
            "pattern": pattern,
            "axis": axis,
            "mean_ns": mean_ns,
            "stddev_ns": None,                        # single repetition: no stddev
            "iterations": iterations,
            "output_spikes": _output_spikes(bench),
            "error_code": _error_code(bench),
            "notes": notes,
        })
    return records, unmapped


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _mac_cpu_brand():
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_meta(doc, suite, impl):
    ctx = doc.get("context", {})
    rev = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "suite": suite,
        "tool": "google-benchmark",
        # context.library_version is emitted as "v1.9.5" — strip the leading v.
        "tool_version": str(ctx.get("library_version", "unknown")).lstrip("v"),
        "host": f"{platform.system().lower()}-{platform.machine().lower()}",
        "cpu": _cpu_label(ctx),
        "date": ctx.get("date"),
        "vcs_rev": rev,
        "vcs_dirty": bool(status),
        "impl": impl,
    }


def _cpu_label(ctx):
    """Google Benchmark's mhz_per_cpu is 0 on Apple Silicon, so it cannot be the label.
    Prefer the OS CPU brand string; fall back to the MHz figure on x86."""
    try:
        brand = platform.mac_ver() and _mac_cpu_brand()
    except Exception:
        brand = None
    if brand:
        return brand
    mhz = ctx.get("mhz_per_cpu")
    if mhz:
        return f"{ctx.get('num_cpus', '?')} cpu @ {mhz:.0f} MHz"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("json_file")
    ap.add_argument("--impl", required=True, choices=["C", "Eigen"],
                    help="which implementation this run measured")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json_file, encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except json.JSONDecodeError as e:
            # baselines/ has at least one truncated file (bench_cpp_20260708_130230_61141.json,
            # cut mid-document). A truncated run must not be silently treated as a
            # partial result set — it would under-report families and skew geomeans.
            print(f"ERROR: {args.json_file} is not valid JSON: {e}\n"
                  f"  Likely a truncated benchmark run. Re-run the suite; do not "
                  f"use this baseline.", file=sys.stderr)
            return 2

    records, unmapped = to_records(doc, args.impl)
    meta = build_meta(doc, "native", args.impl)

    out = {"schema_version": SCHEMA_VERSION, "meta": meta, "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"{args.out}: {len(records)} records from {args.json_file}")
    if unmapped:
        print(f"  WARNING {len(unmapped)} unmapped run_name(s), not emitted:", file=sys.stderr)
        for n in sorted(set(unmapped))[:10]:
            print(f"    {n}", file=sys.stderr)
        print("  Add them to FAMILIES/FAMILY_ALIASES — do not coerce silently.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
