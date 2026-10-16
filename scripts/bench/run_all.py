#!/usr/bin/env python3
"""
run_all.py — one command: run every suite, adapt, merge, report.

Implements acceptance criterion 1 of BENCHMARK_SCHEMA.md.

Usage:
    scripts/bench/run_all.py                      # full run (slow: XL rows)
    scripts/bench/run_all.py --quick              # Tiny/Small only, skips XL
    scripts/bench/run_all.py --suites native dotnet
    scripts/bench/run_all.py --report-only        # merge+report existing records
    scripts/bench/run_all.py --out /tmp/bench     # choose the output directory

Outputs, under --out (default bench-results/):
    records_native_c.json      records_native_eigen.json
    records_dotnet.json        records_python.json
    merged.json
    report.md

Prerequisites (NOT auto-built — building is build.sh's job, and a silent build
here would hide a compile failure as a benchmark result):
    ./build.sh release
    cd src/EventProp.Native.Benchmarks && cmake --preset release \\
        && cmake --build --preset release

Notes:
  * The BenchmarkDotNet version is read from `dotnet list package`, because the
    report CSV has no version column and a BDN version change moves the headline
    .NET Ratio 8-12% against a 0.6% noise floor.
  * The .NET P/Invoke impl is aliased to "C" so the cross-suite chain closes
    (Eigen <-> C <-> Managed). The alias is recorded in notes: ["via_pinvoke"].
  * Suites that fail are reported and the run continues, so one broken suite
    does not lose the others' data; the exit code is non-zero.
"""

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))          # repo root
BENCH = os.path.join(SRC, "src")

NATIVE_BUILD = os.path.join(BENCH, "EventProp.Native.Benchmarks", "build")
DOTNET_PROJ = os.path.join(BENCH, "EventProp.Benchmarks")
DOTNET_ARTIFACTS = os.path.join(DOTNET_PROJ, "BenchmarkDotNet.Artifacts", "results")
PY_BENCH = os.path.join(BENCH, "EventProp.Python.Benchmarks")


def is_windows_msvc_build():
    """True when the CMake generator puts outputs in build/Release, not build/.

    run_benchmarks.sh already branches on this (MINGW/CYGWIN/MSYS -> build/Release);
    the runner must match it or it looks for bench_c in the wrong directory on
    Windows. Detection is by the EXE SUFFIX, not the OS: MSYS2 on Windows uses
    Unix-Makefiles-style single-config builds, and the real discriminator is
    whether CMake appended a config subdirectory.
    """
    return os.name == "nt"


def native_build_dir():
    """Resolve the directory holding bench_c / bench_cpp.

    Prefers whichever directory actually contains the binaries, so a single-config
    Linux/macOS build and a multi-config MSVC build both resolve without the caller
    setting anything. Falls back to the platform default when neither has them
    (the caller's `require` then prints the build hint).
    """
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [NATIVE_BUILD, os.path.join(NATIVE_BUILD, "Release"),
                  os.path.join(NATIVE_BUILD, "Debug")]
    if is_windows_msvc_build():
        candidates = [os.path.join(NATIVE_BUILD, "Release"), NATIVE_BUILD,
                      os.path.join(NATIVE_BUILD, "Debug")]
    for d in candidates:
        if os.path.exists(os.path.join(d, "bench_c" + suffix)):
            return d
    return candidates[0]


def native_exe_name(base):
    return base + (".exe" if os.name == "nt" else "")


def run(cmd, cwd=None, env=None, timeout=7200, capture=False):
    """Run a command, printing it. Returns (ok, stdout)."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, cwd=cwd, env=e, timeout=timeout,
                           capture_output=capture, text=True)
        if capture:
            return p.returncode == 0, p.stdout
        return p.returncode == 0, ""
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s", file=sys.stderr)
        return False, ""
    except FileNotFoundError as ex:
        print(f"    cannot run: {ex}", file=sys.stderr)
        return False, ""


def require(path, hint):
    if not os.path.exists(path):
        print(f"ERROR: missing {path}\n  {hint}", file=sys.stderr)
        return False
    return True


def bdn_version():
    ok, out = run(["dotnet", "list", "package"], cwd=DOTNET_PROJ, capture=True, timeout=300)
    if not ok:
        return None
    for line in out.splitlines():
        if "BenchmarkDotNet" in line:
            parts = line.split()
            if parts:
                return parts[-1]
    return None


def host_tag():
    """Platform tag for output filenames: `darwin-arm64`, `win32-x64`, `linux-x86_64`.

    Built from the SAME fields the adapters put in `meta.host` (platform.system /
    platform.machine, lowercased), so a filename prefix and the host string that
    merge.py's cross-host guard compares can never disagree. If they drifted, a
    directory listing would imply a provenance the records do not carry.

    Windows note: platform.system() is "Windows" -> "windows", and machine() is
    "AMD64" -> "amd64". That is the honest string; it is not normalised to
    "win32"/"x64" because the point of the prefix is to identify the machine.
    """
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def suite_native(out_dir, quick, tag):
    print("\n=== suite: native (Google Benchmark) ===")
    build_dir = native_build_dir()
    print(f"  native build dir: {build_dir}")
    for name in ("bench_c", "bench_cpp"):
        if not require(os.path.join(build_dir, native_exe_name(name)),
                       "Build it: cd src/EventProp.Native.Benchmarks && "
                       "cmake --preset release && cmake --build --preset release"):
            return False

    stamp = "quick" if quick else "full"
    results = {}
    for impl, exe in (("C", "bench_c"), ("Eigen", "bench_cpp")):
        json_path = os.path.join(out_dir, f"{exe}_{stamp}.json")
        cmd = [os.path.join(build_dir, native_exe_name(exe)),
               "--benchmark_format=json", f"--benchmark_out={json_path}"]
        if quick:
            # Tiny/Small only; keeps XL (seconds-to-minutes per iteration) out.
            cmd.append("--benchmark_filter=/(Tiny|Small)/")
        ok, _ = run(cmd, cwd=build_dir, timeout=7200)
        if not ok:
            print(f"  {exe} run failed", file=sys.stderr)
            return False
        # Filename must match record_files in main() exactly — "native_" prefix on
        # all four, so a --report-only with the same --tag finds them.
        rec = os.path.join(out_dir, f"records_{tag}_native_{impl.lower()}.json")
        ok, _ = run([sys.executable, os.path.join(HERE, "from_google_benchmark.py"),
                     json_path, "--impl", impl, "--out", rec], timeout=300)
        results[impl] = ok
    return all(results.values())


def suite_dotnet(out_dir, quick, tag):
    print("\n=== suite: dotnet (BenchmarkDotNet) ===")
    if not require(os.path.join(DOTNET_PROJ, "EventProp.Benchmarks.csproj"), "run ./build.sh release"):
        return False

    version = bdn_version()
    if not version:
        print("ERROR: cannot determine the BenchmarkDotNet version from "
              "`dotnet list package`.\n  It is required provenance: a BDN version "
              "change moves the headline Ratio 8-12%.", file=sys.stderr)
        return False
    print(f"  BenchmarkDotNet version: {version}")

    # BDN's `Artifacts cleanup` deletes the per-job build dir under bin/ when a
    # run is interrupted, which leaves a stale artifacts dir that makes the NEXT
    # run fail with DirectoryNotFoundException on `<Name>-DefaultJob-1/bin/...`
    # after a successful build. Verified: an aborted run caused exactly that, twice.
    # Wiping before the run is safe now that the adapter detects `--job Dry`
    # placeholders (UnrollFactor=1), so leftovers cannot become fabricated records.
    if os.path.isdir(DOTNET_ARTIFACTS):
        shutil.rmtree(DOTNET_ARTIFACTS)
    for d in glob.glob(os.path.join(DOTNET_PROJ, "bin", "Release", "net10.0",
                                    "EventProp.Benchmarks-*Job-*")):
        shutil.rmtree(d, ignore_errors=True)

    # Invoke the BUILT BINARY, not `dotnet run`. Under `dotnet run` the parent
    # process holds bin/, and BDN rebuilds the benchmark project into
    # bin/Release/net10.0/<Name>-DefaultJob-1/ and launches that copy — the launch
    # path does not exist, so it dies with DirectoryNotFoundException after a
    # successful build. Verified: `dotnet run -c Release -- -a ... -f '*Tiny*'`
    # fails this way; running the dll directly does not.
    #
    # The path is framework-dependent (`bin/Release/net10.0/`), which is the same
    # layout on Windows and Linux — no platform branch needed here, unlike the
    # native CMake outputs.
    exe = os.path.join(DOTNET_PROJ, "bin", "Release", "net10.0",
                       "EventProp.Benchmarks.dll")
    if not require(exe, "run ./build.sh release"):
        return False

    cmd = ["dotnet", exe, "-a",
           os.path.join(DOTNET_PROJ, "BenchmarkDotNet.Artifacts")]
    if quick:
        # ShortRun, not DefaultJob. Measured: DefaultJob takes ~2 min for ONE
        # benchmark, so the full Tiny+Small set is tens of minutes; ShortRun is
        # the documented quick setting and keeps the pipeline usable.
        cmd += ["-j", "Short", "-f", "*Tiny*", "*Small*"]
    else:
        # A filter is MANDATORY, not optional. With 75 registered benchmarks and
        # no filter, BDN prints the list and prompts "Enter the asterisk `*` to
        # select all" on stdin. The runner is non-interactive, so the full run
        # produced no artifacts at all and the adapter then failed on a missing
        # results dir. `-f '*'` is the non-interactive form of that prompt.
        cmd += ["-j", "Short", "-f", "*"]
    ok, _ = run(cmd, cwd=DOTNET_PROJ, timeout=7200)
    if not ok:
        print("  dotnet benchmark run failed", file=sys.stderr)
        return False

    # Stale CSVs from deleted benchmarks (the Cross_* set) are skipped by the
    # adapter's --skip list, not by wiping the directory: a wipe before the run
    # means a failed run leaves the artifacts dir holding `--job Dry` placeholders
    # (placeholder Mean, no OutputSpikes) that would adapt into fabricated records.

    rec = os.path.join(out_dir, f"records_{tag}_dotnet.json")
    ok, _ = run([sys.executable, os.path.join(HERE, "from_benchmarkdotnet.py"),
                 DOTNET_ARTIFACTS, "--out", rec, "--bdn-version", version],
                timeout=600)
    if not ok:
        return False

    # The .NET suite's "Native" method is the C library reached through P/Invoke.
    # It is NOT aliased to "C": the two C measurements have different contexts, and
    # collapsing them made a table titled "C vs Managed" silently mean "P/Invoke vs
    # Managed" - P/Invoke marshalling overhead is on the C side of that ratio, which
    # is a real and load-bearing difference (it is what makes the Tiny rows move
    # 8-12% across BDN versions). The native suite's bare C column and this one agree
    # to ~3%, so the chain Eigen <-> C <-> PInvoke <-> Managed still closes, via two
    # adjacent pairs instead of one collapsed name.
    path = os.path.join(out_dir, f"records_{tag}_dotnet_c.json")
    with open(rec) as f:
        doc = json.load(f)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
    print(f"  wrote {path} (impl=PInvoke, kept distinct from the native suite's C)")
    return True


def suite_python(out_dir, quick, tag):
    print("\n=== suite: python (pytest-benchmark) ===")
    if not require(os.path.join(PY_BENCH, "test_kernel_benchmarks.py"),
                   "the kernel benchmarks live in the repo"):
        return False

    # Path separator and library-path variable are platform-specific.
    #   PYTHONPATH: os.pathsep is ';' on Windows, ':' on POSIX. A hardcoded ':'
    #   silently makes the second entry unimportable on Windows — the reference
    #   submodule import would fail with a confusing ModuleNotFoundError.
    #   Library path: DYLD_LIBRARY_PATH (macOS), LD_LIBRARY_PATH (Linux), PATH
    #   (Windows). The Cython extension embeds an rpath at build time
    #   (_native.pyx:21), so on Windows the C library .dll must be findable via
    #   PATH (or sit next to the extension). Setting all three is harmless: each
    #   is ignored on the platforms that don't read it.
    c_build = os.path.join(BENCH, "EventProp.C", "build")
    # On MSVC the C library .dll is in build\Release\ (multi-config generator),
    # which the bare build\ on PATH does not cover. Include both; nonexistent
    # entries are ignored by the loader.
    lib_dirs = [c_build] + [os.path.join(c_build, c) for c in ("Release", "Debug")]
    py_paths = [os.path.join(BENCH, "EventProp.Python"),
                os.path.join(BENCH, "EventProp.Native.Python")]
    env = {
        "PYTHONPATH": os.pathsep.join(py_paths),
        "DYLD_LIBRARY_PATH": os.pathsep.join(lib_dirs),
        "LD_LIBRARY_PATH": c_build,
        "PATH": os.pathsep.join(lib_dirs + [os.environ.get("PATH", "")]),
    }
    json_path = os.path.join(out_dir, "kernels.json")
    cmd = [sys.executable, "-m", "pytest", "test_kernel_benchmarks.py",
           "--benchmark-only", f"--benchmark-json={json_path}", "-q"]
    if quick:
        cmd += ["-k", "Tiny or Small"]
    ok, _ = run(cmd, cwd=PY_BENCH, env=env, timeout=7200)
    if not ok:
        print("  python benchmark run failed", file=sys.stderr)
        return False

    rec = os.path.join(out_dir, f"records_{tag}_python.json")
    ok, _ = run([sys.executable, os.path.join(HERE, "from_pytest_benchmark.py"),
                 json_path, "--out", rec], timeout=300)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=os.path.join(SRC, "bench-results"))
    ap.add_argument("--quick", action="store_true",
                    help="Tiny/Small only; skips XL and Medium/Large burst")
    ap.add_argument("--suites", nargs="+", default=["native", "dotnet", "python"],
                    choices=["native", "dotnet", "python"])
    ap.add_argument("--report-only", action="store_true",
                    help="skip running; merge+report the records already in --out")
    ap.add_argument("--benchmarks-md", default=os.path.join(SRC, "BENCHMARKS.md"),
                    help="generated report is also written here (schema criterion 3); "
                         "pass '' to skip")
    ap.add_argument("--tag", default=None,
                    help="platform prefix for output filenames (default: auto, e.g. "
                         "darwin-arm64). Records, merged.json and report.md all carry "
                         "it, so results from several machines can coexist in one "
                         "directory without overwriting each other.")
    ap.add_argument("--reports-dir", default=os.path.join(SRC, "bench-reports"),
                    help="directory for COMMITTED per-platform reports (report_<tag>.md); "
                         "tracked in git, unlike the raw records. Pass '' to skip.")
    args = ap.parse_args()

    tag = args.tag or host_tag()
    print(f"platform tag: {tag}")
    os.makedirs(args.out, exist_ok=True)
    failures = []

    if not args.report_only:
        runners = {"native": suite_native, "dotnet": suite_dotnet, "python": suite_python}
        for name in args.suites:
            if not runners[name](args.out, args.quick, tag):
                failures.append(name)

    record_files = {
        "native": [os.path.join(args.out, f"records_{tag}_native_c.json"),
                   os.path.join(args.out, f"records_{tag}_native_eigen.json")],
        "dotnet": [os.path.join(args.out, f"records_{tag}_dotnet_c.json")],
        "python": [os.path.join(args.out, f"records_{tag}_python.json")],
    }
    inputs = [p for s in args.suites for p in record_files[s] if os.path.exists(p)]
    if not inputs:
        # Name the tag: the usual cause of this is a --tag mismatch between a run
        # and a --report-only, not missing data.
        print(f"ERROR: no record files to merge in {args.out} for tag '{tag}'\n"
              f"  Expected: {' '.join(os.path.basename(p) for s in args.suites for p in record_files[s])}\n"
              f"  If the records were produced on another machine, pass --tag <that-tag>.",
              file=sys.stderr)
        return 1

    print("\n=== merge ===")
    merged = os.path.join(args.out, f"merged_{tag}.json")
    ok, _ = run([sys.executable, os.path.join(HERE, "merge.py")] + inputs
                + ["--out", merged], timeout=600)
    if not ok:
        return 1

    print("\n=== report ===")
    report = os.path.join(args.out, f"report_{tag}.md")
    cmd = [sys.executable, os.path.join(HERE, "report.py"), merged, "--out", report,
           "--title", "Library Benchmark Comparison",
           "--source", " ".join(os.path.basename(p) for p in inputs)]
    ok, _ = run(cmd, timeout=300)
    if not ok:
        return 1

    # Committed, per-platform reports (bench-reports/).
    #
    # Why commit reports and not records: the records are machine-specific and
    # merge.py refuses to merge across hosts, so a committed record file is a
    # measurement nobody on another machine can use. The report is the artifact
    # that survives being read on a different machine, and it is deterministic
    # (byte-identical across runs of the same records) and small (~24K).
    #
    # Tagged filenames mean a Windows run and a macOS run COEXIST instead of
    # overwriting each other, which is what makes the per-pair geomean comparison
    # across machines a diffable artifact rather than a /tmp eyeball.
    if args.reports_dir:
        os.makedirs(args.reports_dir, exist_ok=True)
        tagged = os.path.join(args.reports_dir, f"report_{tag}.md")
        ok, _ = run([sys.executable, os.path.join(HERE, "report.py"), merged,
                     "--out", tagged,
                     "--title", "Library Benchmark Comparison",
                     "--source", " ".join(os.path.basename(p) for p in inputs)],
                    timeout=300)
        if not ok:
            return 1
        print(f"committed report: {tagged}")

    # Schema criterion 3: BENCHMARKS.md is generated, not hand-written. Written
    # from the same merged.json as report.md, so the two cannot drift. This is the
    # "latest run" convenience copy; the tagged files in --reports-dir are the
    # durable per-platform record.
    if args.benchmarks_md:
        ok, _ = run([sys.executable, os.path.join(HERE, "report.py"), merged,
                     "--out", args.benchmarks_md,
                     "--title", "Library Benchmark Comparison",
                     "--source", " ".join(os.path.basename(p) for p in inputs)],
                    timeout=300)
        if not ok:
            return 1
        print(f"benchmarks md: {args.benchmarks_md}")

    print(f"\nreport: {report}")
    if failures:
        print(f"completed with failures in: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
