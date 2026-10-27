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
    "vcs_dirty": false,
    "submodule_dirty": true,
    "untracked": 3
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
| `tool`, `tool_version` | Measured: BDN 0.15.8 → 0.16.0-preview.2 moved `ComputeSpikes_Tiny` Managed +9.8% and the Ratio −10%. **A version change invalidates historical .NET comparisons.** Without `tool_version` in the record, that is undetectable after the fact. |
| `host`, `cpu` | Cross-suite absolute times are only comparable on the same machine. |
| `date` | ISO-8601, from the tool where available (Google Benchmark emits `context.date`). |
| `vcs_rev`, `vcs_dirty` | A number from a modified tracked tree is not reproducible. `vcs_dirty` is **tracked modifications only** — see the two fields below for what it deliberately excludes. |
| `submodule_dirty` | The read-only reference submodule carries local patches that are **required to build it** (`scripts/submodule-state.sh`), so this is `true` on every machine that can run the python suite. It used to be folded into `vcs_dirty`, which made that flag `true` on every run on every machine — and a flag that is always true cannot gate the "these numbers are not a committed state" claim the report makes. |
| `untracked` | Count of untracked files (generated reports, session logs). Counts as dirtiness under the old flag; now reported, not warned about. |

All three vcs fields are `null` when no git repository is reachable — the absence of a
repository is neither a clean tree nor a dirty one, and `false` would assert a check that
never ran. `scripts/bench/provenance.py` computes them for all three adapters, from the
superproject top level: `git status` prints paths relative to cwd while `git submodule
status` prints them relative to the root, so an adapter run from a subdirectory used to
report the submodule as clean.

**Records written before this split carry the old `vcs_dirty` definition**, in which
submodule dirt and untracked files also counted. So `vcs_dirty: true` in an older record
may describe nothing more than the submodule's required patches, and `submodule_dirty`
/ `untracked` are absent from it. Re-running a suite is the only way to get the split
provenance; `merge.py` omits the two new lines rather than inventing them.

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
| `overhead_dominated` | FFI/marshalling cost exceeds kernel cost at this size | `ComputeSums_Tiny` P/Invoke — managed ahead at Tiny where P/Invoke marshalling dominates |
| `not_like_for_like` | the two `impl`s are not the same operation | `Dropout` 8.17x — "C batch API vs C++ shim — API-shape artifact" |
| `slow_excluded` | excluded from quick runs by policy | XL rows — see the measured XL cost below |

A row with a non-empty `invalid_reasons` **must be excluded from every geomean** and
**must appear in the report's excluded table**. Today it appears only if someone
remembers the footnote.

### XL cost, measured (so "is XL too slow?" is not re-answered by guesswork)

Per-iteration cost, C library, `--benchmark_min_time=0.05s`:

| Family | XL per iteration |
|---|---|
| `ComputeSums`, `ComputeMaxima`, `BackwardMaxima` | 1.5–26 ms |
| `ComputeSpikes/XL` | 3.1–4.3 s |
| `BackwardSpikes/XL` | 7.9–10.5 s |
| `FullTrainingStep/XL` | 12.4–13.9 s |

The number that matters is **not** per-iteration cost times an iteration count. Google
Benchmark's `min_time` is **relative**, so for a 14 s iteration it takes the minimum of
**one** iteration — XL does not multiply by iteration count. Measured wall clock:

| Suite | XL | Full suite incl. XL |
|---|---|---|
| native C | 125 s | **162 s** |
| native Eigen | ~1.18x slower | ~190 s |
| dotnet | 6 XL benchmarks; 64 s for a `-j Dry` pass | ~15–20 min at `-j Short` |
| python | **no XL registered** (P2 gap) | ~2 min |

So a full run is ~25 minutes, dominated by .NET, not by XL. `--quick` exists to skip XL
for fast validation, not because XL is unrunnable.

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
dotnet:   Managed  ↔  PInvoke  (P/Invoke into the C library)
python:   pybind11  ↔  Cython  (both into the C library)
```

`C` is the only implementation reachable from more than one suite, but it is reached by
different mechanisms, and those mechanisms are part of the measurement. The Python suite
has **no core-kernel benchmarks at all** — all 28 tests are layer/loss level — so binding
overhead is unmeasurable there.

**Requirement:** every suite must emit a record for each kernel family, at the shared
sizes, via the same C entry point.

| Suite | How |
|---|---|
| `native` | `impl: "C"` — calls the C library directly |
| `dotnet` | `impl: "PInvoke"` — the C library reached through P/Invoke. **Not aliased to `C`** (see below) |
| `python` | add raw `compute_spikes` / `backward_spikes` benchmarks for pybind11 and Cython, no layer. Emits `impl: "Cython"` / `"pybind11"` |

**Do not alias `PInvoke` to `C`.** An earlier revision of this schema prescribed that
alias (`notes: ["via_pinvoke"]`) so the chain would close on one shared name. It made the
cross-suite join work and it made the report lie: the table titled **"C vs Managed"**
silently meant **"P/Invoke vs Managed"**, with P/Invoke marshalling cost sitting on the C
side of the ratio. One impl name carried two measurement contexts.

The chain closes as **measured adjacent pairs** instead, which is strictly more informative:

```
Eigen  ↔  C  ↔  PInvoke  ↔  Managed
         (1.24x)  (1.06x)     (1.05x)
```

The middle pair — same C library, called directly vs through the P/Invoke round-trip — is
the interop-cost measurement the alias hid. It is computable because the suites share
benchmark names for four kernel families (`ComputeSums`, `ComputeSpikes`, `ComputeMaxima`,
`BackwardSpikes`).

---

## Adapters

`scripts/bench/`, one per tool, each emitting the record format above. **All four are
implemented and running**; the mapping work listed here is what each one actually does.

| Adapter | Input | Mapping work as built |
|---|---|---|
| `from_google_benchmark.py` | `--benchmark_format=json` | `run_name` `ComputeSpikes/Small/burst/real_time` → `benchmark`/`size`/`pattern`; strip `/real_time`; `real_time` × `time_unit` → `mean_ns`; carry `output_spikes` and `c_error` → `error_code`; `context.library_version` → `tool_version` (emits `v1.9.5`, matching the `GIT_TAG` in `CMakeLists.txt`) |
| `from_benchmarkdotnet.py` | `-report.csv` | Filename `Bench_ComputeSpikes_Small` → strip `Bench_`/`EventProp.Benchmarks.` prefix, split family/size/pattern (case-insensitively — a case-sensitive match silently dropped 32 of 93 CSVs); `Mean` `"18.63 ms"` → ns; `Native`/`Managed` → `impl`; `--bdn-version` is **required**, see below |
| `from_pytest_benchmark.py` | `--benchmark-json` | `fullname` `test_kernel_benchmarks.py::Test_Kernel_ComputeSpikes::test_pybind11[Tiny-uniform]` → family from the **class** name, impl from the method, size/pattern from the parametrize bracket; `stats.mean` (seconds) → ns; `machine_info` + `commit_info` + `datetime` + `version` → `meta` |
| `merge.py` | N record files | Joins on `benchmark`/`size`/`pattern`(+`axis`), computes `valid` and `invalid_reasons` from `output_spikes`/`error_code`/coverage — never authored |
| `report.py` | merged JSON | Emits the side-by-side tables, per-pair geomeans, exclusion table, sweep table |
| `run_all.py` | — | One command: runs all suites, adapts, merges, reports. `--quick` restricts to Tiny/Small |

They replace four ad-hoc scripts in `EventProp.Python.Benchmarks/`
(`parse_benchmarks.py`, `print_full_matrix.py`, `print_native_comparison.py`,
`print_poisson_comparison.py`), which each re-parsed one format with bespoke regexes
— `print_native_comparison.py` parsed Google Benchmark **text** output with a regex,
discarding the JSON that already has `output_spikes` and provenance. Those four are
**deleted**, along with the committed result JSONs that fed them (`results_sweep.json`,
`lif_results.json`, `sweep_results.json`, `results_training_step.json`).

`generate_report.py` is **kept**: the suite README documents invoking it directly over
JSON you already have, so it is a supported local workflow rather than a superseded
one-off — but what it writes (`benchmark_report.html`) is not read by the pipeline and is
not the committed report. Its other former caller,
`EventProp.Python.Benchmarks/run_benchmarks.sh`, is **deleted**; see the
`run_benchmarks.sh` name-collision note in `AGENTS.md` for why.

### Provenance: what each tool supplies

| Suite | `tool_version` | `date` | `host`/`cpu` | `vcs_rev` |
|---|---|---|---|---|
| native | ✅ `context.library_version` | ❌ | ✅ `context` | adapter via `git` |
| python | ✅ `version` | ✅ `datetime` | ✅ `machine_info` | ✅ `commit_info` (id, **dirty**, branch) |
| dotnet | ❌ **not in the CSV** | ❌ | partial (`platform` module) | adapter via `git` |

pytest-benchmark is the only tool with complete provenance. The gaps are filled
explicitly, never guessed:

- **BenchmarkDotNet version** — the report CSV has no version column; BDN prints it only
to the console. `run_all.py` reads it from `dotnet list package` and passes it as the
adapter's required `--bdn-version`. This is the most important provenance field in the
schema, because a BDN version change moves the headline .NET Ratio 8–12% against a 0.6%
noise floor (AGENTS.md), and 0.15.x ships no `net10.0` asset — a net10 project silently
consumes `lib/net8.0/BenchmarkDotNet.dll`.
- **`date`** — BDN CSVs have no timestamp, so `meta.date` is `null` for that suite rather
than backfilled with the local clock, which would misreport when a measurement was taken.

### Two traps the .NET adapter had to solve

Both were found by running the tools, not by reading their documentation.

**`--job Dry` output is indistinguishable from a measurement by shape.** A Dry run writes
a full report CSV with a plausible `Mean` (it executes the benchmark once, so `Mean` is
that single invocation) and no validity signal, so it adapts into fabricated records that
look real. Detected by `UnrollFactor`: **16 in a real run, 1 in Dry**. Three signals that
look right and are **not** — each checked against real output:

| Candidate signal | Why it fails |
|---|---|
| `IterationCount <= 1` | The column carries the **job setting** (`"Default"`), not a measured count |
| `OutputSpikes` absent | Not a column in these reports at all — real runs lack it too |
| MemoryDiagnoser columns empty | `Alloc Ratio` is populated in Dry output (`1.00` / `0.00`) |

This is also why `run_all.py` does **not** wipe `BenchmarkDotNet.Artifacts` before the
run: a wipe plus a failed run leaves the directory holding placeholders that adapt into
fabricated records. Stale CSVs are skipped by name instead.

**BDN's CLI flags are not what they look like.** `--artifactsPath` is an unknown option;
`BenchmarkSwitcher` exits with a usage dump that looks like a benchmark failure. The flag
is `-a`/`--artifacts`. Likewise `-f` takes glob patterns as **separate** arguments —
`-f "*Tiny* *Small*"` as one argument matches nothing.

---

## Acceptance criteria for v1

Status as built. **All six are met.** Criterion 6 is implemented in `merge.py` and needs
a cross-host run to exercise.

1. **One command runs all three suites and writes three schema files.** — ✅
`scripts/bench/run_all.py`; `--quick` for Tiny/Small, `--suites` to select, `--report-only`
to re-merge. Suites that fail are reported and the run continues with a non-zero exit, so
one broken suite does not lose the others' data.
2. **`merge.py` produces a table where every row has a `valid` flag and a reason.** — ✅
3. **`BENCHMARKS.md` is generated, not hand-written, and contains no `*` footnotes.** —
✅ `run_all.py` writes it from the same `merged.json` as `report.md`, so the two cannot
drift. Every exclusion is a computed `invalid_reasons` value.
4. **Every ratio in the report names both implementations in words.** — ✅ ("C is 1.35x
faster than Eigen")
5. **Every record carries `tool_version`.** — ✅ all three suites.
6. **Re-running on a different machine cannot silently mix.** — ✅ `merge.py` refuses
differing `host` unless `--allow-mixed-hosts`; not yet exercised across two hosts.

### Per-size geomeans are not optional

The report emits a pooled geomean **and** a per-size geomean per pair. The pooled number is
misleading when one side has a fixed per-call cost. Measured on one machine:

| Pair | Tiny | Small | Pooled |
|---|---:|---:|---:|
| C vs Eigen | 1.39x | 1.31x | 1.35x |
| C vs Managed | **0.25x** | **0.72x** | 2.34x |
| Cython vs pybind11 | **0.40x** | **0.99x** | 1.59x |

(The "C vs Managed" row predates the pair split in `18e6e86`; that pair is now
**C via P/Invoke vs Managed**, and the Tiny/Small kernel cells no longer exist in the .NET
suite at all. The row is kept because it is the measurement that motivated per-size geomeans.)
P/Invoke marshalling and Cython binding overhead are fixed costs: they dominate the small
cells and amortize at the large ones, so the pooled figure describes neither size — and for
the two interop pairs the pooled direction is **opposite** to the Tiny direction. C vs Eigen
shows no split, which is the expected control: no interop layer involved. Any future change
to the report must keep the per-size row.
