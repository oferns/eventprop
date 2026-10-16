#!/usr/bin/env python3
"""
merge.py — join normalized benchmark records and compute validity.

Reads schema-v1 record files (see BENCHMARK_SCHEMA.md) and emits one merged table
where every row carries a computed `valid` flag and the reasons it was excluded.

Usage:
    merge.py records_*.json --out merged.json [--allow-mixed-hosts]

Validity is DERIVED here, never authored in a source file. The rules encode facts
that AGENTS.md currently keeps as prose footnotes:

  zero_output_spikes   output_spikes == 0  -> the row times an empty loop
                         (BackwardSpikes/Tiny|Small/uniform; ComputeSpikes/Tiny/uniform)
  peer_zero_output_spikes  this record cannot measure output_spikes (the BDN CSV has no
                         such column, so .NET records are null) but a peer in the same
                         cell measured 0. The zero-spike regime is a property of the
                         CELL, not the tool, so it condemns every impl in it.
  native_error         error_code != 0     -> the row measures an error path, not the
                         algorithm (the unsorted-burst 36 ns vs 83 us case)
  low_iteration_count  a FIXED-iteration tool config measured this row with too few
                         ops. Applies to BenchmarkDotNet `--job short` (IterationCount=5,
                         which is why its Tiny numbers move 8-12% across BDN versions).
                         NOT applied to Google Benchmark, which budgets by wall-clock
                         time: 1 iteration of a 16 s/iteration XL BackwardSpikes is a
                         sound measurement, not a thin one. See LOW_ITERATION_TOOLS.
  sweep                axis is set         -> parameter sweep, not a size cell; kept
                         out of the size table, reported separately
  missing_counterpart  no peer impl for this cell -> no ratio can be computed

`not_like_for_like` (Dropout's C-batch-API vs C++-shim artifact) and
`overhead_dominated` are judgement calls that cannot be derived from the data;
they are declared in EXCLUSION_POLICY below, keyed by family, so the judgement is
visible and auditable rather than a footnote.
"""

import argparse
import json
import math
import os
import sys

SCHEMA_VERSION = 1

# Judgement-based exclusions, declared rather than inferred. Each entry is a
# documented fact from AGENTS.md; changing one is a visible edit, not a silent one.
EXCLUSION_POLICY = {
    # Dropout: "C batch API vs C++ shim — API-shape artifact, not like-for-like"
    "Dropout": "not_like_for_like",
}

# Rows the schema keeps out of quick runs by policy (AGENTS.md: XL costs seconds
# per iteration; XL burst ~16 s/iteration for BackwardSpikes). Report-side: this is
# a run-selection policy, not a per-row validity verdict.
SLOW_FAMILIES = {"BackwardSpikes", "FullTrainingStep"}

# Only tools that fix the iteration count up front can produce a thin measurement.
# Google Benchmark budgets by time, so a low iteration count there is expected on
# heavy rows and is NOT a validity signal. Keyed by meta.suite.
LOW_ITERATION_TOOLS = {"dotnet"}
LOW_ITERATION_FLOOR = 30


def load(path):
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {doc.get('schema_version')!r}, "
                         f"expected {SCHEMA_VERSION}")
    return doc


def cell_key(rec):
    """The join key. axis is serialized so sweeps never collide with size cells.

    An empty axis dict and a null axis are the same cell. The native adapter emits
    axis=null; the .NET adapter emits axis={} when a row has no batch_size/variant.
    Serializing them verbatim split one logical cell into two (verified:
    ComputeSums/Medium/uniform appeared twice, one with C+Eigen and one with
    C+Managed, so no C-vs-Managed ratio could be computed at all).
    """
    axis = rec["axis"]
    axis_json = json.dumps(axis, sort_keys=True) if axis else "null"
    return (rec["benchmark"], rec["size"], rec["pattern"], axis_json)


def derive_reasons(rec, peers, suite=None):
    reasons = []
    if rec.get("output_spikes") == 0:
        reasons.append("zero_output_spikes")
    if rec.get("error_code") not in (None, 0):
        reasons.append("native_error")

    # Propagate a peer's zero_output_spikes to a record that CANNOT measure it.
    # The BenchmarkDotNet report CSV has no OutputSpikes column, so every .NET
    # record carries output_spikes: null and passes the check above clean. The
    # zero-spike regime is a property of the CELL (the workload produces no
    # output spikes at this size), not of the tool, so a peer that can observe it
    # condemns the whole cell. Without this, BackwardSpikes/Tiny/uniform — which
    # AGENTS.md explicitly says to exclude — entered the C-vs-Managed geomean and
    # made the headline say "Managed is 1.18x faster" on a cell where Managed
    # times an empty loop while C does real work.
    if rec.get("output_spikes") is None and any(
            p.get("output_spikes") == 0 for p in peers):
        reasons.append("peer_zero_output_spikes")

    it = rec.get("iterations")
    if (suite in LOW_ITERATION_TOOLS and it is not None
            and it < LOW_ITERATION_FLOOR):
        reasons.append("low_iteration_count")
    if rec.get("axis"):
        reasons.append("sweep")
    policy = EXCLUSION_POLICY.get(rec["benchmark"])
    if policy:
        reasons.append(policy)
    if not peers:
        reasons.append("missing_counterpart")
    return reasons


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-mixed-hosts", action="store_true",
                    help="permit records from different hosts (cross-machine "
                         "absolute times are not comparable; off by default)")
    args = ap.parse_args()

    docs = [load(f) for f in args.files]

    # Provenance guard: refuse to silently mix hosts. Cross-suite absolute times
    # are only comparable on the same machine.
    hosts = sorted({d["meta"]["host"] for d in docs})
    if len(hosts) > 1 and not args.allow_mixed_hosts:
        print(f"REFUSED: records from {len(hosts)} different hosts: {hosts}\n"
              f"  Absolute times across machines are not comparable.\n"
              f"  Re-run on one host, or pass --allow-mixed-hosts for ratios only.",
              file=sys.stderr)
        return 2

    tools = {}
    for d in docs:
        m = d["meta"]
        tools.setdefault(m["suite"], set()).add(f"{m['tool']} {m['tool_version']}")

    # Commit + date + dirty flag, aggregated across the input records. Surfaced in
    # the report header because a COMMITTED report is a durable artifact that
    # outlives the code it describes: without the revision, a reader cannot tell
    # whether a committed report predates a change they care about, and stale
    # benchmark numbers read exactly like fresh ones. Multiple values mean the
    # records were produced at different commits - the report says so rather than
    # picking one.
    def meta_set(field):
        return sorted({d["meta"][field] for d in docs if d["meta"].get(field)})

    vcs_revs = meta_set("vcs_rev")
    dates = meta_set("date")
    # "dirty" is per-record; aggregate as a single tri-state so the report has one
    # answer: all clean, all dirty, or mixed.
    dirty_flags = [bool(d["meta"].get("vcs_dirty")) for d in docs
                   if d["meta"].get("vcs_dirty") is not None]
    if not dirty_flags:
        vcs_dirty = None
    elif all(dirty_flags):
        vcs_dirty = True
    elif not any(dirty_flags):
        vcs_dirty = False
    else:
        vcs_dirty = "mixed"

    # Join on the cell key; collect every impl that measured it.
    # The suite is recorded PER IMPL, not just per document: `low_iteration_count`
    # is tool-specific (it applies to BenchmarkDotNet, not Google Benchmark), and
    # looking it up from a list of document metas — which carry `suite` but no
    # `impl` — never matched, so the rule silently never fired for any .NET record.
    cells = {}
    for d in docs:
        for rec in d["records"]:
            k = cell_key(rec)
            cells.setdefault(k, {"impls": {}, "meta": [], "suites": {}})
            cells[k]["impls"][rec["impl"]] = rec
            cells[k]["meta"].append(d["meta"])
            cells[k]["suites"][rec["impl"]] = d["meta"]["suite"]

    merged = []
    # size/pattern are None for sweep rows, and None does not order against str.
    # Sort on a normalized tuple so the output order is stable and total.
    def sort_key(item):
        benchmark, size, pattern, axis_json = item[0]
        return (benchmark or "", size or "", pattern or "", axis_json)

    for (benchmark, size, pattern, axis_json), cell in sorted(cells.items(), key=sort_key):
        impls = cell["impls"]
        axis = json.loads(axis_json) if axis_json != "null" else None
        rows = {}
        all_reasons = set()
        for impl, rec in impls.items():
            peers = [r for i, r in impls.items() if i != impl]
            suite = cell["suites"].get(impl)
            reasons = derive_reasons(rec, peers, suite)
            all_reasons |= set(reasons)
            rows[impl] = {
                "mean_ns": rec["mean_ns"],
                "stddev_ns": rec.get("stddev_ns"),
                "iterations": rec.get("iterations"),
                "output_spikes": rec.get("output_spikes"),
                "error_code": rec.get("error_code"),
                "invalid_reasons": sorted(set(reasons)),
            }
        merged.append({
            "benchmark": benchmark,
            "size": size,
            "pattern": pattern,
            "axis": axis,
            "impls": rows,
            # A cell is valid only if every impl in it is valid — a ratio built
            # from one valid and one invalid row is not a valid ratio.
            "valid": not all_reasons,
            "invalid_reasons": sorted(all_reasons),
        })

    out = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "hosts": hosts,
            "tools": {s: sorted(t) for s, t in tools.items()},
            "sources": [os.path.basename(f) for f in args.files],
            "vcs_rev": vcs_revs,
            "vcs_dirty": vcs_dirty,
            "date": dates,
        },
        "cells": merged,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    valid = sum(1 for c in merged if c["valid"])
    print(f"{args.out}: {len(merged)} cells, {valid} valid, "
          f"{len(merged) - valid} excluded")
    counts = {}
    for c in merged:
        for r in c["invalid_reasons"]:
            counts[r] = counts.get(r, 0) + 1
    for r, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  excluded by {r}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
