# Benchmark Result Schema (v1)

The three benchmark suites measure the same algorithms through different bindings, in
three incompatible output formats:

| Suite | Tool | Native format |
|---|---|---|
| `EventProp.Native.Benchmarks` | Google Benchmark | `--benchmark_format=json` |
| `EventProp.Benchmarks` | BenchmarkDotNet | `-report.csv` / `-report.md` |
| `EventProp.Python.Benchmarks` | pytest-benchmark | `--benchmark-json` |

This document defines **one normalized intermediate format** that all three parse into,
and that a single report generator consumes. It exists to make cross-library comparison
sound, and to make the invalid rows visible instead of silently averaged.

It is a **contract, not a library**: the format is defined here, the adapters live in
`scripts/bench/`, and the report is generated from the merged records.

---

## Why a schema, and not just three reports

`BENCHMARKS.md` is currently hand-written. Three consequences, all visible in the file
as it stands:

1. **The ratio directions disagree.** Suite 1 is `Eigen/C` (>1 = C wins), suite 2 is
   `Native/Managed` (<1 = native wins), suite 3 is `Cython/pybind11` (>1 = Cython slower).
   A single table cannot be sorted, and a reader comparing columns must remember three
   conventions. The schema stores a **canonical** direction and derives the rest.
2. **Invalid rows are marked by hand.** `BackwardSpikes/Small/uniform 19.63*` carries a
   `*` and a footnote. The information that makes it invalid (`output_spikes == 0`) is
   known to the native suite — it publishes `output_spikes` in its JSON — but it never
   reaches the report. The schema makes validity a **computed field**.
3. **Cross-suite claims are unprovenance-able.** The file says "Generated 2026-10-14 on
   macOS (osx-arm64)". AGENTS.md records that a BenchmarkDotNet version change moves the
   headline .NET Ratio by 8–12% and that the noise floor is 0.6%. A date and a CPU family
   are not enough to tell whether two numbers in the same table are comparable. The schema
   requires tool version per record.

---

## Record format

One JSON document per suite run: a `meta` object plus a list of `records`.

```json
{
  "schema_version": 1,
  "meta": {
    "suite": "native",
    "tool": "google-benchmark",
    "tool_version": "1.9.5",
    "host": "macos-arm64",
    "cpu": "Apple M3 Pro",
    "date": "2026-10-14T20:11:02+01:00",
    "vcs_rev": "9059512",
    "vcs_dirty": false
  },
  "records": [
    {
      "benchmark": "ComputeSpikes",
      "impl": "C",
      "size": "Small",
      "pattern": "burst",
      "axis": null,

      "mean_ns": 1234500.0,
      "stddev_ns": 12300.0,
      "iterations": 1353,

      "output_spikes": 10669,
      "error_code": 0,

      "valid": true,
      "invalid_reasons": [],
      "notes": []
    }
  ]
}
```

### `meta` — provenance. Required, no defaults.

| Field | Why it is mandatory |
|---|---|
| `suite` | `native` \| `dotnet` \| `python`. The join key across files. |
| `tool`, `tool_version` | AGENTS.md: BDN 0.15.8 → 0.16.0-preview.2 moved `ComputeSpikes_Tiny` Managed +9.8% and the Ratio −10%. **A version change invalidates historical .NET comparisons.** Without `tool_version` in the record, that is undetectable after the fact. |
| `host`, `cpu` | Cross-suite absolute times are only comparable on the same machine. |
| `date` | ISO-8601, from the tool where available (Google Benchmark emits `context.date`). |
| `vcs_rev`, `vcs_dirty` | A number from a dirty tree is not reproducible. |

### `records[]` — one row per benchmark case

**Identity (the join keys).** These four fields are how a row in one suite is matched to
a row in another. They must be spelled identically across suites — that is the whole
point of the schema.

| Field | Domain | Notes |
|---|---|---|
| `benchmark` | `ComputeSums` `ComputeSpikes` `ComputeMaxima` `BackwardSpikes` `BackwardMaxima` `FullTrainingStep` `Dropout` `LIF_Forward` `LIF_Backward` `LI_Forward` `LI_Backward` `TrainingLoop` `Batch_Forward` | **Kernel family, no size, no pattern, no tool decoration.** Google Benchmark emits `ComputeSpikes/Small/burst/real_time`; BDN emits `Bench_ComputeSpikes_Small`; pytest emits `test_lif_forward_cython[...]`. Each adapter normalizes to this vocabulary. Adding a family = adding it here. |
| `impl` | `C` `Eigen` `Managed` `PInvoke` `pybind11` `Cython` | Which implementation. **This is the field that closes the cross-suite chain** — see below. |
| `size` | `Tiny` `Small` `Medium` `Large` `XL` | From the shared size table (10/5/20, 50/20/100, 200/80/500, 500/200/1000, 2000/500/5000). |
| `pattern` | `uniform` `poisson` `burst` | From the canonical generators. |
| `axis` | `null` or `{"v_th": 0.5}` | For parameter-sweep benchmarks (`BM_ComputeSpikesVth`). Keeps sweeps out of the main table. |

**Measurement.**

| Field | Notes |
|---|---|
| `mean_ns` | **Nanoseconds, always.** Google Benchmark emits `real_time`/`cpu_time` in `time_unit`; BDN emits `Mean` as a human string (`"18.63 ms"`); pytest-benchmark emits seconds. All convert to ns. |
| `stddev_ns` | Population stddev. Google Benchmark: `cpu_time` stddev if present. BDN: `StdDev`. pytest: `stats.stddev`. `null` if the tool reports only a confidence interval. |
| `iterations` | Ops measured. Below ~30, add `"low_iteration_count"` to `notes` — BDN's `--job short` runs at `IterationCount=5`, which is why its Tiny numbers are version-sensitive. |

**Validity inputs.** These exist so `valid` can be computed by the report generator
rather than asserted by a human.

| Field | Notes |
|---|---|
| `output_spikes` | Spikes the call produced. `0` ⇒ the row times an empty loop. The native suite already emits this in its JSON; the .NET and Python adapters must start emitting it. |
| `error_code` | `0` = `EVENTPROP_OK`. Non-zero ⇒ the row measures an **error path**, not the algorithm. The native suite publishes `c_error`/`c_fwd_error`/`c_bwd_error`; AGENTS.md already says a non-zero value means the row is not the algorithm. Promote that to all suites. |

**Validity (computed, not authored).**

`valid` and `invalid_reasons` are **derived by the report generator** from the fields
above plus the shared data contract. They are stored in the merged output, never
hand-written in a source file.

| `invalid_reasons` value | Trigger | Known instance |
|---|---|---|
| `zero_output_spikes` | `output_spikes == 0` | `BackwardSpikes/Tiny\|Small/uniform` — AGENTS.md: "time an empty loop… exclude from any geomean" |
| `native_error` | `error_code != 0` | unsorted burst: C returns 0 spikes in ~0 ns, reported 36 ns vs Eigen's 83 µs |
| `overhead_dominated` | FFI/marshalling cost exceeds kernel cost at this size | `ComputeSums_Tiny` P/Invoke; AGENTS.md: "managed ahead at Tiny where P/Invoke marshalling dominates" |
| `not_like_for_like` | the two `impl`s are not the same operation | `Dropout` 8.17x — "C batch API vs C++ shim — API-shape artifact" |
| `slow_excluded` | excluded from quick runs by policy | XL rows (~16 s/iteration for `BackwardSpikes`) |

A row with a non-empty `invalid_reasons` **must be excluded from every geomean** and
**must appear in the report's excluded table**. Today it appears only if someone
remembers the footnote.

---

## The canonical ratio

Every ratio in the report is computed as **`impl_a / impl_b` where the pair is declared
per family**, and the report states the winner in words rather than by sign convention:

| Family | Declared pair | Report says |
|---|---|---|
| ComputeSums, ComputeSpikes, BackwardSpikes, ComputeMaxima, BackwardMaxima, FullTrainingStep | `Eigen / C` | "C is 1.64× faster than Eigen" |
| same families, .NET | `Managed / PInvoke` | "P/Invoke is 1.09× faster than managed" |
| LIF_*/LI_*/Batch_*/TrainingLoop | `Managed / PInvoke` | as above |
| Python layer cells | `pybind11 / Cython` | "Cython is 1.35× faster than pybind11" |

The stored records contain **no ratio at all** — only `mean_ns` per `impl`. Ratios are a
report-time derivation, so a pair can be added later (e.g. `Managed / Eigen`) without
re-running anything, provided both records exist.

---

## Closing the chain: the shared `C` baseline

Today the suites share no implementation, so no cross-suite ratio is sound:

```
native:   C  ↔  Eigen
dotnet:   Managed  ↔  PInvoke(→C)
python:   pybind11  ↔  Cython(→C)
```

`C` is the only implementation reachable from two suites, but it is never measured in
the same vocabulary as the others. The Python suite has **no core-kernel benchmarks at
all** — all 28 tests are layer/loss level — so binding overhead is unmeasurable there.

**Requirement:** every suite must emit a record with `impl: "C"` for each kernel family,
at the shared sizes, via the same C entry point.

| Suite | How |
|---|---|
| `native` | already has `impl: "C"` |
| `dotnet` | `PInvoke` **is** C — label it `impl: "C"` for kernel families (alias, recorded in `notes: ["via_pinvoke"]`) |
| `python` | add raw `compute_spikes` / `backward_spikes` benchmarks for pybind11 and Cython, no layer. Emits `impl: "C"` twice (once per binding). |

With `impl: "C"` present in all three, the chain closes:
`Eigen ↔ C ↔ Managed` and `pybind11 ↔ C ↔ Cython` are both computable from one merged
table, and binding overhead becomes a first-class number:
`Cython(C) − C(native)` = the Python binding cost, isolated from the layer.

---

## Adapters

`scripts/bench/` (to be added), one per tool, each emitting the record format above:

| Adapter | Input | Notable mapping work |
|---|---|---|
| `from_google_benchmark.py` | `--benchmark_format=json` | `run_name` `ComputeSpikes/Small/burst/real_time` → `benchmark`/`size`/`pattern`; strip `/real_time`; `real_time` × `time_unit` → `mean_ns`; carry `output_spikes`; `context.library_version` → `tool_version` (verified present: emits `v1.9.5`, matching the `GIT_TAG` in `CMakeLists.txt`) |
| `from_benchmarkdotnet.py` | `-report.csv` | `Method` `Bench_ComputeSpikes_Small` → strip `Bench_`, split family/size; `Mean` `"18.63 ms"` → ns; `Native`/`Managed` → `impl`; `Job` → `notes` when `IterationCount` is low. **`tool_version` is NOT in the CSV** — verified: the report CSV has no version column. The BDN version appears only in the console log (`// BenchmarkDotNet v0.15.8`). So this adapter must take the version from `dotnet list package` / the console log, and the runner must capture it. Given that a BDN version change is the single largest known source of .NET measurement drift (8–12% on the headline Ratio), this is the most important provenance field in the whole schema and the one the tool does not supply. |
| `from_pytest_benchmark.py` | `--benchmark-json` | `test_lif_forward_cython[Small-burst]` → `benchmark`/`impl`/`size`/`pattern`; `stats.mean` (s) → ns; `machine_info` → `meta` |

Then `merge.py` (join on `benchmark`/`size`/`pattern`, compute `valid`) and
`report.py` (emit the side-by-side table). These replace the five ad-hoc scripts in
`EventProp.Python.Benchmarks/` (`generate_report.py`, `parse_benchmarks.py`,
`print_full_matrix.py`, `print_native_comparison.py`, `print_poisson_comparison.py`),
which each re-parse one format with bespoke regexes — `print_native_comparison.py`
parses Google Benchmark **text** output with a regex, discarding the JSON that already
has `output_spikes` and provenance.

---

## Acceptance criteria for v1

1. One command runs all three suites and writes three schema files.
2. `merge.py` produces a table where every row has a `valid` flag and a reason.
3. `BENCHMARKS.md` is **generated**, not hand-written, and contains no `*` footnotes —
   every exclusion is a computed `invalid_reasons` value.
4. Every ratio in the report names both implementations in words.
5. Every record carries `tool_version`, so a BenchmarkDotNet bump is visible as a
   discontinuity in the merged history rather than a silent 10% shift.
6. Re-running on a different machine cannot silently mix: the merge refuses records whose
   `host` differs, unless `--allow-mixed-hosts`.
