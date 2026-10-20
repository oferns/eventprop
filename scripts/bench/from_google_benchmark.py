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

from provenance import vcs_state

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
    # vcs fields come from provenance.py, shared by all three adapters. The old
    # bool(git status --porcelain) was true on every machine that can build the
    # reference library, because the read-only submodule must carry local patches to
    # compile at all — so the flag could never mean "these numbers are not a committed
    # state", which is what report.py's staleness banner uses it for.
    meta = {
        "suite": suite,
        "tool": "google-benchmark",
        # context.library_version is emitted as "v1.9.5" — strip the leading v.
        "tool_version": str(ctx.get("library_version", "unknown")).lstrip("v"),
        "host": f"{platform.system().lower()}-{platform.machine().lower()}",
        "cpu": _cpu_label(ctx),
        "date": ctx.get("date"),
        "impl": impl,
    }
    meta.update(vcs_state())
    return meta


def _linux_cpu_brand():
    """The CPU brand string from /proc/cpuinfo.

    Only x86 (and some ARM vendors) spell it "model name"; older ARM kernels use
    "Hardware", so that is tried second. "Model" is deliberately NOT tried: on ARM boards
    it names the *board* ("Raspberry Pi 4 Model B Rev 1.5"), and putting a board
    identifier in meta.cpu is worse than falling through to the MHz line, because the
    field exists to let two hosts be compared on CPU. When nothing matches, the caller
    falls back — an honest "N cpu @ M MHz" beats a wrong brand.

    Split on the first ':' rather than whitespace: /proc/cpuinfo is a kernel proc file, not
    a config file, and its spacing and value shapes are not stable across kernels.
    """
    for key in ("model name", "Hardware"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    name, sep, value = line.partition(":")
                    if sep and name.strip() == key and value.strip():
                        return value.strip()
        except OSError:
            # No /proc (or no permission): there is no second reader to try either.
            return None
    return None


def _cpu_label(ctx):
    """Google Benchmark's mhz_per_cpu is 0 on Apple Silicon, so it cannot be the label.
    Prefer the OS CPU brand string; fall back to the MHz figure on x86.

    The label's whole purpose is letting a reader compare two hosts, so every platform
    returns the same shape: the bare brand string, exactly what _mac_cpu_brand() has always
    returned on macOS — core count goes in neither.

    Dispatch on platform.system(), not on platform.mac_ver(): mac_ver() returns an
    all-empty-but-non-empty TUPLE on Linux, which is truthy, so the previous
    `platform.mac_ver() and _mac_cpu_brand()` did not gate macOS at all — it called the
    macOS reader on every platform and relied on sysctl being absent elsewhere to paper
    over it. Same visible behaviour today, and a silent surprise the day a sysctl binary
    appears on a Linux box (an empty brand, or worse, a host's answer to the wrong
    question). macOS keeps its exact reader and its exact output.

    Windows gets no brand reader on purpose: mhz_per_cpu is populated there, and a new
    label would make fresh Windows records disagree with the committed
    report_windows-amd64.md's meta.cpu for no measurement benefit.
    """
    reader = {"Darwin": _mac_cpu_brand, "Linux": _linux_cpu_brand}.get(platform.system())
    brand = None
    if reader:
        try:
            brand = reader()
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
