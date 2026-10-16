#!/usr/bin/env python3
"""
from_benchmarkdotnet.py — .NET suite adapter for the benchmark result schema.

Converts BenchmarkDotNet `-report.csv` files into the normalized record format
defined in BENCHMARK_SCHEMA.md.

Usage:
    # one file
    from_benchmarkdotnet.py <X-report.csv> --out records_dotnet.json

    # a whole results directory
    from_benchmarkdotnet.py src/EventProp.Benchmarks/BenchmarkDotNet.Artifacts/results \
        --out records_dotnet.json --bdn-version 0.15.8

Name mapping (verified against the 93 CSVs in BenchmarkDotNet.Artifacts/results):

    file  Bench_ComputeSums_Medium-report.csv
    rows  Method=Managed / Method=Native
      -> benchmark=ComputeSums size=Medium pattern=uniform impl=Managed|PInvoke

    file  EventProp.Benchmarks.Bench_ComputeMaxima_Burst_Large-report.csv
      -> benchmark=ComputeMaxima pattern=burst size=Large

    file  EventProp.Benchmarks.Bench_LIF_Forward_Small-report.csv
      -> benchmark=LIF_Forward size=Small

    file  EventProp.Benchmarks.Bench_Batch_Forward_8-report.csv
      -> benchmark=Batch_Forward axis={"batch_size":8}

    file  EventProp.Benchmarks.Cross_ComputeSums_Tiny_Uniform-report.csv
         Method=Managed / Native / Cpp
      -> STALE. The Cross_* benchmarks and their Cpp method were deleted in
         f9ee29f ("update .NET benchmarks to use new alloc-based P/Invoke API").
         They are skipped with a warning, not mapped: emitting them would put an
         `Eigen` impl in the merged table from a benchmark that no longer exists,
         which is exactly the "managed vs Eigen claim from a managed vs C
         measurement" failure mode.

Method -> impl vocabulary:
    Managed            -> Managed
    Native             -> PInvoke
    Native_BatchApi    -> PInvoke      (notes: batch_api)
    Managed_PerSampleLoop -> Managed   (notes: per_sample_loop)
    Cpp                -> (skipped, see above)
    Precompute_* / FindRoots_* / MergeAndSort_* / Accumulate_* -> micro-benchmarks,
                         not kernel families: skipped as unmapped.

Provenance: the CSV has NO tool_version column (verified across all 93 files).
The BenchmarkDotNet version appears only in the console log
(`// BenchmarkDotNet v0.15.8`). Since a BDN version change moves the headline
.NET Ratio 8-12% against a 0.6% noise floor, tool_version is the single most
important provenance field here and the tool does not supply it. Pass it
explicitly with --bdn-version; omitting it is a hard error, not a silent "unknown".
"""

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys

SCHEMA_VERSION = 1

SIZES = {"Tiny", "Small", "Medium", "Large", "XL"}
PATTERNS = {"uniform", "poisson", "burst"}

# Kernel families and layer families the schema recognizes.
FAMILIES = {
    "ComputeSums", "ComputeSpikes", "ComputeMaxima",
    "BackwardSpikes", "BackwardMaxima",
    "FullTrainingStep", "TrainingLoop", "Dropout",
    "LIF_Forward", "LIF_Backward", "LI_Forward", "LI_Backward",
    "Batch_Forward", "Batch_TrainingStep", "Batch_RawApi",
}

METHOD_TO_IMPL = {
    "Managed": "Managed",
    "Native": "PInvoke",
    "Native_BatchApi": "PInvoke",
    "Managed_PerSampleLoop": "Managed",
    "Native_PerSampleLoop": "PInvoke",
}

# Some benchmark classes put the family in the METHOD name and the pattern in the
# FILE name: Bench_Pattern_Burst has ComputeSums_Managed / ComputeSums_Native.
# Match "<Family>_<Managed|Native>" and let the method override the filename family.
METHOD_FAMILY_RE = re.compile(r"^(?P<family>[A-Za-z]+)_(?P<method>Managed|Native)$")

# Method names that are micro-benchmarks of internal helpers, not kernel families.
UNMAPPED_METHOD_PREFIXES = ("Precompute_", "FindRoots_", "MergeAndSort_", "Accumulate_")

# BDN unit suffixes seen in the Mean column (verified: "μs", "ms", "ns").
UNIT_TO_NS = {"ns": 1.0, "μs": 1_000.0, "µs": 1_000.0, "us": 1_000.0,
              "ms": 1_000_000.0, "s": 1_000_000_000.0, "min": 60e9}


def parse_mean(value):
    """'22.84 μs' / '\"1,109.4 ns\"' / '18.63 ms' -> nanoseconds (float)."""
    s = value.strip().strip('"').replace(",", "")
    m = re.match(r"^([0-9.]+)\s*(ns|μs|µs|us|ms|min|s)$", s)
    if not m:
        raise ValueError(f"unparsable Mean: {value!r}")
    return float(m.group(1)) * UNIT_TO_NS[m.group(2)]


def parse_filename(path):
    """Return (benchmark, size, pattern, axis) or None."""
    stem = os.path.basename(path)
    stem = re.sub(r"-report\.csv$", "", stem)
    stem = re.sub(r"^EventProp\.Benchmarks\.", "", stem)
    stem = re.sub(r"^Bench_", "", stem)

    # Cross_* were deleted in f9ee29f; caller treats None-with-marker as stale.
    stale = stem.startswith("Cross_")
    if stale:
        stem = stem[len("Cross_"):]

    parts = stem.split("_")
    # Filenames capitalize the pattern token (Bench_ComputeMaxima_Burst_Large) while
    # the schema vocabulary is lowercase (burst). Match case-insensitively and emit
    # the lowercase form; a case-sensitive test silently dropped 32 of 93 CSVs.
    def as_size(tok):
        return next((s for s in SIZES if s.lower() == tok.lower()), None)

    def as_pattern(tok):
        return next((p for p in PATTERNS if p.lower() == tok.lower()), None)

    # Trailing integer is a batch size, not a size name.
    axis = None
    if parts and re.fullmatch(r"\d+", parts[-1]):
        axis = {"batch_size": int(parts[-1])}
        parts = parts[:-1]

    benchmark = None
    size = None
    pattern = None

    # Family is the leading token(s) up to a known size/pattern/axis marker.
    for i, p in enumerate(parts):
        s, pat = as_size(p), as_pattern(p)
        if s:
            benchmark = "_".join(parts[:i])
            size = s
            rest = parts[i + 1:]
            break
        if pat:
            benchmark = "_".join(parts[:i])
            pattern = pat
            rest = parts[i + 1:]
            break
    else:
        rest = []
        benchmark = "_".join(parts)

    for p in rest:
        s, pat = as_size(p), as_pattern(p)
        if s and size is None:
            size = s
        elif pat and pattern is None:
            pattern = pat

    # "Pattern" is not a family: Bench_Pattern_Burst carries the family in the METHOD
    # column (ComputeSums_Managed). Return it as a placeholder; to_records() overrides
    # benchmark per row via METHOD_FAMILY_RE, and drops rows that don't supply one.
    #
    # These benchmarks have NO size token — they vary PATTERN at one fixed size, so
    # `pattern` is the varying dimension and belongs in `axis`. Leaving size=None with
    # pattern in the cell key put them in the size table as `size: None` rows, which
    # showed up as a `None 1.19x (6)` per-size geomean entry.
    if benchmark == "Pattern":
        axis = dict(axis or {})
        if pattern:
            axis["pattern"] = pattern
        return (None, size, None, axis), stale

    if benchmark not in FAMILIES and benchmark != "TrainingLoopSweep":
        return None, stale

    if pattern is None:
        # The .NET single-kernel benchmarks are uniform-only (AGENTS.md: "Poisson
        # cells exist only in suite 1"). Defaulting to uniform states that fact.
        pattern = "uniform"
    return (benchmark, size, pattern, axis), stale


def parse_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def is_dry_run_placeholder(rows):
    """True if this CSV looks like `--job Dry` output rather than a measurement.

    A Dry run writes a full report CSV with a placeholder Mean (it executes the
    benchmark once, so Mean == that single iteration). Adapted blindly, those
    become plausible records with real numbers and no validity signal — the worst
    kind of bad benchmark data.

    Signals, taken from a measured comparison of `-j Dry` output against a real
    run of the same benchmarks (NOT guessed):
      * `UnrollFactor` is 16 in a real run, 1 in Dry. This is the discriminator.

    Deliberately NOT used, each for a measured reason:
      * `IterationCount` — carries the job setting ("Default"), not a count.
      * `OutputSpikes` — not a column in these reports at all (which is why the
        .NET records carry output_spikes: null).
      * The MemoryDiagnoser columns — `Alloc Ratio` is populated in Dry output
        (1.00 / 0.00), so "diagnoser columns are empty" is NOT a Dry signature.

    This is why the runner must not wipe the artifacts dir before a run that can
    fail: a wipe plus a failed run leaves placeholders that would adapt into
    fabricated records.
    """
    if not rows:
        return False
    return (rows[0].get("UnrollFactor") or "").strip() == "1"


def to_records(path, rows):
    parsed, stale = parse_filename(path)
    if parsed is None:
        return [], (stale, os.path.basename(path))

    if is_dry_run_placeholder(rows):
        return [], ("dry-run placeholder (UnrollFactor=1; a real run uses 16)",
                    os.path.basename(path))
    benchmark, size, pattern, axis = parsed

    records = []
    for row in rows:
        method = (row.get("Method") or "").strip()
        if method.startswith(UNMAPPED_METHOD_PREFIXES):
            continue

        # Family may come from the method (Bench_Pattern_Burst -> ComputeSums_Managed).
        row_benchmark = benchmark
        m = METHOD_FAMILY_RE.match(method)
        if m and m.group("family") in FAMILIES:
            row_benchmark = m.group("family")
            method = m.group("method")

        if row_benchmark is None:
            continue          # filename gave no family and the method gave no family

        impl = METHOD_TO_IMPL.get(method)
        if impl is None:
            continue                                   # e.g. 'Cpp' — see module docstring

        mean_ns = parse_mean(row["Mean"])
        stddev = row.get("StdDev")
        stddev_ns = parse_mean(stddev) if stddev and stddev.strip() not in ("", "NA") else None

        # BDN's `IterationCount` column carries the JOB SETTING, not a measured
        # count: every row of a `-j Short` run reports "3", and a `-j Dry` run
        # reports "Default". Treating it as `iterations` made merge.py's
        # low_iteration_count rule fire on every .NET row (floor 30 > 3), which
        # excluded all 31 C-vs-Managed cells and left the pair with no geomean.
        # The real iteration count is not in the report CSV, so report None.
        iterations = None

        notes = []
        variant = None
        if method.endswith("_BatchApi"):
            notes.append("batch_api")
            variant = "batch_api"
        if method.endswith("_PerSampleLoop"):
            notes.append("per_sample_loop")
            variant = "per_sample_loop"
        # The .NET P/Invoke path IS the C library, aliased per the schema so the
        # cross-suite chain can close on impl:"C".
        if impl == "PInvoke":
            notes.append("via_pinvoke")

        # Bench_Batch_RawApi_4 registers Native_BatchApi AND Native_PerSampleLoop —
        # two PInvoke rows in one cell that are genuinely different code paths, not a
        # duplicate. The variant goes in `axis` so it becomes part of the join key and
        # merge.py cannot silently overwrite one with the other.
        row_axis = dict(axis) if axis else {}
        if variant:
            row_axis["variant"] = variant

        records.append({
            "benchmark": row_benchmark,
            "impl": impl,
            "size": size,
            "pattern": pattern,
            "axis": row_axis,
            "mean_ns": mean_ns,
            "stddev_ns": stddev_ns,
            "iterations": iterations,
            # BDN reports no spike count and no native error code. None means
            # "not measured", NOT "zero" — collapsing them would mark every .NET
            # row invalid for the wrong reason.
            "output_spikes": None,
            "error_code": None,
            "notes": notes,
        })
    return records, (stale, os.path.basename(path))


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("src", help="a -report.csv, or a directory of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bdn-version",
                    help="BenchmarkDotNet version, e.g. 0.15.8. NOT in the CSV; "
                         "read it from the run log ('// BenchmarkDotNet v0.15.8') "
                         "or `dotnet list package`.")
    args = ap.parse_args()

    if not args.bdn_version:
        print("ERROR: --bdn-version is required.\n"
              "  The BDN report CSV has no version column (verified across all 93 "
              "files).\n"
              "  A BDN version change moves the headline .NET Ratio 8-12% against a "
              "0.6%\n"
              "  noise floor, so recording it is not optional. Get it from the run "
              "log\n"
              "  ('// BenchmarkDotNet v0.15.8') or `dotnet list package`.",
              file=sys.stderr)
        return 2

    if os.path.isdir(args.src):
        if not os.path.isdir(os.path.join(args.src, "results")) and not any(
                n.endswith("-report.csv") for n in os.listdir(args.src)):
            # The usual cause is that the benchmark run never produced artifacts:
            # BDN prompts on stdin for a selection when no filter is given, so a
            # non-interactive run exits having written nothing. Say that, rather
            # than raising Errno 2 from the CSV reader.
            print(f"ERROR: no -report.csv under {args.src}\n"
                  f"  The benchmark run produced no artifacts. Most likely it was "
                  f"invoked without\n  a filter: with many registered benchmarks BDN "
                  f"prints the list and\n  prompts 'Enter the asterisk `*` to select "
                  f"all' on stdin, which a\n  non-interactive runner cannot answer. "
                  f"Pass `-f '*'`.", file=sys.stderr)
            return 2
        files = sorted(f for f in
                       (os.path.join(args.src, n) for n in os.listdir(args.src))
                       if f.endswith("-report.csv"))
    elif os.path.exists(args.src):
        files = [args.src]
    else:
        print(f"ERROR: {args.src} does not exist — the benchmark run produced no "
              f"artifacts (see the -f '*' note above)", file=sys.stderr)
        return 2
    if not files:
        print(f"ERROR: no -report.csv under {args.src}", file=sys.stderr)
        return 2

    records = []
    stale_files = []
    unmapped_files = []
    placeholder_files = []
    for path in files:
        try:
            rows = parse_csv(path)
        except Exception as e:
            print(f"ERROR: cannot parse {path}: {e}", file=sys.stderr)
            return 2
        recs, (stale, name) = to_records(path, rows)
        if stale:
            # Skip the WHOLE file, not just its Cpp rows. These benchmarks were
            # deleted in f9ee29f; their Managed/Native rows are measurements of a
            # benchmark that no longer exists, and leaving them in collides with the
            # live Bench_* files on the same cell key (verified: Cross_ComputeSums_
            # Medium_Uniform contributed a 4708 us PInvoke row against the live
            # 23.26 us one — a 200x outlier in the same cell).
            #
            # `stale` is a reason STRING for every non-None case, not a boolean:
            # the dry-run guard returns one too, and labelling those "stale
            # Cross_*" sends the reader to look for deleted benchmarks.
            if stale is True:
                stale_files.append(name)
            else:
                placeholder_files.append(f"{name} — {stale}")
            continue
        if not recs:
            unmapped_files.append(name)
        records.extend(recs)

    rev = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    meta = {
        "suite": "dotnet",
        "tool": "benchmarkdotnet",
        "tool_version": args.bdn_version,
        "host": f"{platform.system().lower()}-{platform.machine().lower()}",
        "cpu": "unknown",
        "date": None,                                  # BDN CSV has no timestamp
        "vcs_rev": rev,
        "vcs_dirty": bool(status),
    }

    out = {"schema_version": SCHEMA_VERSION, "meta": meta, "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"{args.out}: {len(records)} records from {len(files)} CSV(s)")
    if stale_files:
        print(f"  SKIPPED {len(stale_files)} stale Cross_* file(s) — the Cross_* "
              f"benchmarks and their\n  'Cpp' method were deleted in f9ee29f. "
              f"Emitting them would add an Eigen impl\n  from a benchmark that no "
              f"longer exists:", file=sys.stderr)
        for n in stale_files[:5]:
            print(f"    {n}", file=sys.stderr)
    if placeholder_files:
        print(f"  SKIPPED {len(placeholder_files)} `--job Dry` placeholder file(s) — "
              f"a Dry run writes a\n  full report CSV whose Mean is one invocation, "
              f"not a measurement:", file=sys.stderr)
        for n in placeholder_files[:5]:
            print(f"    {n}", file=sys.stderr)
    if unmapped_files:
        print(f"  WARNING {len(unmapped_files)} file(s) produced no records "
              f"(unmapped family):", file=sys.stderr)
        for n in unmapped_files[:10]:
            print(f"    {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
