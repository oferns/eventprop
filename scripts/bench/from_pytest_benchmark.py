#!/usr/bin/env python3
"""
from_pytest_benchmark.py — Python suite adapter for the benchmark result schema.

Converts pytest-benchmark `--benchmark-json` output into the normalized record
format defined in BENCHMARK_SCHEMA.md.

Usage:
    from_pytest_benchmark.py <results.json> --out records_python.json

Name mapping (verified against results_kernels.json, pytest-benchmark 5.2.3):

  fullname  test_kernel_benchmarks.py::Test_Kernel_ComputeSpikes::test_pybind11[Tiny-uniform]
    -> benchmark=ComputeSpikes  impl=pybind11  size=Tiny  pattern=uniform

  The family is in the CLASS name (Test_Kernel_<Family>), not in `name`
  (which is just "test_pybind11[Tiny-uniform]"). The layer suite uses
  TestLIF_Forward / TestLIF_Backward / TestLI_* / TestTTFS_* / TestVMax_*.

  impl comes from the method: test_pybind11 / test_cython, and in the layer
  suite also test_pybind11_batch / test_Cython_batch (note the capital C).

Provenance: pytest-benchmark is the only one of the three tools that emits a
complete provenance block — machine_info, commit_info (id, dirty, branch),
datetime, and the plugin version. Use it; do not re-derive it from git.

Kernel vs layer: records from test_kernel_benchmarks.py are kernel families
(ComputeSpikes, ComputeMaxima) and carry notes ["no_layer"] — they call the C
library directly, which is what isolates binding overhead. Records from
test_benchmarks.py are layer families. Keeping them in one vocabulary is the
point: the difference between a kernel cell and a layer cell at the same size
IS the binding + layer cost.
"""

import argparse
import json
import os
import re
import sys

SCHEMA_VERSION = 1

SIZES = {"Tiny", "Small", "Medium", "Large", "XL"}
PATTERNS = {"uniform", "poisson", "burst"}

# Class name -> schema family. Kernel classes are Test_Kernel_<Family>;
# layer classes are TestLIF_Forward, TestTTFS_*, TestVMax_*, and the batch/dropout
# classes in test_benchmarks.py.
CLASS_TO_FAMILY = {
    "Test_Kernel_ComputeSpikes": "ComputeSpikes",
    "Test_Kernel_ComputeMaxima": "ComputeMaxima",
    "Test_Kernel_BackwardSpikes": "BackwardSpikes",
    "Test_Kernel_BackwardMaxima": "BackwardMaxima",
    "TestLIF_Forward": "LIF_Forward",
    "TestLIF_Backward": "LIF_Backward",
    "TestLI_Forward": "LI_Forward",
    "TestLI_Backward": "LI_Backward",
    "TestTTFS_Forward": "TTFS_Forward",
    "TestTTFS_Backward": "TTFS_Backward",
    "TestVMax_Forward": "VMax_Forward",
    "TestVMax_Backward": "VMax_Backward",
    "TestBatch": "Batch_Forward",
    "TestDropout": "Dropout",
    "TestTrainingStep": "TrainingLoop",
    "TestSweep": "TrainingLoopSweep",
    "TestTTFSStep": "TTFS_Step",
    "TestVMaxStep": "VMax_Step",
}

METHOD_TO_IMPL = {
    "pybind11": "pybind11",
    "cython": "Cython",
    "Cython": "Cython",
}

# test_pybind11_batch / test_Cython_batch / test_pybind11_ttfs_step ...
# The parametrize bracket ([Tiny-uniform]) is stripped before matching: anchoring
# this regex on $ against the full test name silently unmapped all 40 kernel rows.
METHOD_RE = re.compile(r"^test_(?P<impl>pybind11|Cython|cython)(?:_(?P<rest>[a-z_]+))?$")

# Suffix from the method name -> family, for classes that host several families.
REST_TO_FAMILY = {
    "batch": "Batch_Forward",
    "dropout": "Dropout",
    "training_step": "TrainingLoop",
    "sweep": "TrainingLoopSweep",
    "ttfs_step": "TTFS_Step",
    "vmax_step": "VMax_Step",
}

KERNEL_FILE_MARKER = "test_kernel_benchmarks"


def parse_fullname(fullname):
    """Return (benchmark, impl, size, pattern, is_kernel) or None."""
    m = re.match(r"^(?P<file>[^:]+)::(?P<cls>[^:]+)::(?P<test>[^:]+)$", fullname)
    if not m:
        return None
    file_part, cls, test = m.group("file"), m.group("cls"), m.group("test")
    is_kernel = KERNEL_FILE_MARKER in file_part

    # Split the parametrize bracket off the method name before matching.
    pm = re.search(r"\[(?P<params>[^\]]+)\]$", test)
    params_str = pm.group("params") if pm else None
    method_name = test[:pm.start()] if pm else test

    mm = METHOD_RE.match(method_name)
    if not mm:
        return None
    impl = METHOD_TO_IMPL.get(mm.group("impl"))
    rest = mm.group("rest")
    if impl is None:
        return None

    # Family: the method's suffix wins (TestBatch has test_pybind11_batch),
    # then the class name.
    benchmark = None
    if rest and rest in REST_TO_FAMILY:
        benchmark = REST_TO_FAMILY[rest]
    if benchmark is None:
        benchmark = CLASS_TO_FAMILY.get(cls)
    if benchmark is None:
        return None

    # Parametrization: [Tiny-uniform] / [Medium-burst] / [4] (batch size)
    size = pattern = None
    axis = None
    if params_str:
        toks = [t.strip() for t in params_str.split("-")]
        for t in toks:
            if t in SIZES:
                size = t
            elif t.lower() in PATTERNS:
                pattern = t.lower()
            elif re.fullmatch(r"\d+", t):
                axis = {"batch_size": int(t)}

    if pattern is None:
        pattern = "uniform"      # the layer suite parametrizes size only
    return benchmark, impl, size, pattern, axis, is_kernel


def to_records(doc):
    records = []
    unmapped = []
    for b in doc.get("benchmarks", []):
        parsed = parse_fullname(b.get("fullname", ""))
        if parsed is None:
            unmapped.append(b.get("fullname", "<no fullname>"))
            continue
        benchmark, impl, size, pattern, axis, is_kernel = parsed

        stats = b.get("stats") or {}
        mean_s = stats.get("mean")
        if mean_s is None:
            continue
        stddev_s = stats.get("stddev")

        notes = []
        if is_kernel:
            notes.append("no_layer")     # calls the C library directly
        rounds = stats.get("rounds")
        if rounds is not None and rounds < 30:
            notes.append("low_round_count")

        records.append({
            "benchmark": benchmark,
            "impl": impl,
            "size": size,
            "pattern": pattern,
            "axis": axis,
            # pytest-benchmark reports SECONDS.
            "mean_ns": float(mean_s) * 1e9,
            "stddev_ns": float(stddev_s) * 1e9 if stddev_s is not None else None,
            "iterations": rounds,
            # Not measured by this suite. None, not 0 — see the .NET adapter note.
            "output_spikes": None,
            "error_code": None,
            "notes": notes,
        })
    return records, unmapped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("json_file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.json_file) as f:
        try:
            doc = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: {args.json_file} is not valid JSON: {e}", file=sys.stderr)
            return 2

    records, unmapped = to_records(doc)

    # pytest-benchmark carries full provenance; use it rather than re-deriving.
    mi = doc.get("machine_info") or {}
    ci = doc.get("commit_info") or {}
    system = (mi.get("system") or "").lower()
    machine = (mi.get("machine") or "").lower()
    host = {"darwin": "darwin"}.get(system, system) or "unknown"
    cpu = ((mi.get("cpu") or {}).get("brand")
           or (mi.get("cpu") or {}).get("name")
           or "unknown")

    meta = {
        "suite": "python",
        "tool": "pytest-benchmark",
        "tool_version": doc.get("version", "unknown"),
        "host": f"{host}-{machine}",
        "cpu": cpu,
        "date": doc.get("datetime"),
        "vcs_rev": (ci.get("id") or "")[:7] or None,
        "vcs_dirty": ci.get("dirty"),
    }

    out = {"schema_version": SCHEMA_VERSION, "meta": meta, "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"{args.out}: {len(records)} records from {args.json_file}")
    if unmapped:
        print(f"  WARNING {len(unmapped)} unmapped fullname(s):", file=sys.stderr)
        for n in sorted(set(unmapped))[:10]:
            print(f"    {n}", file=sys.stderr)
        print("  Add the class/method to CLASS_TO_FAMILY / REST_TO_FAMILY.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
