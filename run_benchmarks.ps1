# run_benchmarks.ps1 - Run the cross-library benchmark pipeline on Windows.
#
# Usage:
#   .\run_benchmarks.ps1                        # quick run (Tiny/Small, skips XL)
#   .\run_benchmarks.ps1 -Full                  # full run, includes XL
#   .\run_benchmarks.ps1 -ReportOnly            # re-merge records already on disk
#   .\run_benchmarks.ps1 -Build                 # build.ps1 release first
#   .\run_benchmarks.ps1 -Suites dotnet python  # subset of suites
#   .\run_benchmarks.ps1 -Sweeps                # include the sweep cells
#   .\run_benchmarks.ps1 -Tag mytag -Out D:\tmp # where records/reports go
#   .\run_benchmarks.ps1 -Suites python -NoReports -NoBenchmarksMd   # scratch run:
#       writes nothing to the committed BENCHMARKS.md / bench-reports\
#
# What it does:
#   1. Verifies the prerequisites by checking what the suites actually need: the
#      native benchmark executables, the .NET benchmark dll, the Python packages, and
#      an ACTUAL `import eventprop_c` for the Cython layer. It does not build them
#      silently - a build failure must surface as a build failure, not as a benchmark
#      result. Pass -Build to invoke build.ps1 first.
#   2. Runs scripts\bench\run_all.py: three suites, normalized into the common record
#      schema, merged, and reported:
#        bench-results\records_windows-amd64_*.json   (gitignored)
#        bench-results\merged_windows-amd64.json      (gitignored)
#        bench-reports\report_windows-amd64.md        (COMMITTED)
#        BENCHMARKS.md                                (COMMITTED, "latest run")
#
# Why the Cython prerequisite is an import, not a file check
#   The previous version checked that _native*.pyd exists. On Windows that file
#   existed and `import eventprop_c._native` failed, because since Python 3.8 CPython
#   loads extension modules with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS - the search set is
#   the extension's own directory, the process directory, the system directories, and
#   directories registered with os.add_dll_directory. PATH is NOT in that set, so the
#   PATH entry this script used to document cannot work. The registration now lives in
#   eventprop_c/_library_path.py and runs at package import; the only check that can
#   tell you the suite will run is the import itself.
#
# Cross-platform note: merge.py refuses to merge records from different hosts, so this
# produces a SEPARATE report from the macOS/Linux runs. Compare per-pair geomeans
# across reports, never the absolute us columns.

param(
    [switch]$Full,
    [switch]$ReportOnly,
    [switch]$Build,
    [switch]$Sweeps,
    [switch]$NoReports,
    [switch]$NoBenchmarksMd,
    [string]$PythonExe = "python",
    [string]$Tag,
    [string]$Out,
    [string]$ReportsDir,
    [string]$BenchmarksMd,
    [string[]]$Suites
)

# Windows PowerShell 5.1 turns a native command's stderr into a terminating
# NativeCommandError when $ErrorActionPreference is "Stop" - so the prereq probe
# `python -c "import numpy, ..."` died with NativeCommandError on the very failure it
# was written to report, and the friendly install hint never printed. PS 7.3+ has an
# explicit switch; setting it is a no-op on 5.1.
$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# --- 0. Optional build --------------------------------------------------------
if ($Build) {
    Write-Host "=== build (build.ps1 release) ===" -ForegroundColor Cyan
    & "$ScriptDir\build.ps1" -BuildType release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "build.ps1 failed (exit $LASTEXITCODE) - stopping." -ForegroundColor Red
        exit 1
    }
}

# --- 1. Verify prerequisites --------------------------------------------------
# MSVC's CMake generator is multi-config, so artifacts land in build\Release\ (and
# build\Debug\), not build\. run_all.py probes all three; this check must match,
# including Debug, which the previous version did not look for.
Write-Host "=== checking prerequisites ===" -ForegroundColor Cyan
$missing = @()

function Find-Artifact([string[]]$relPaths) {
    foreach ($r in $relPaths) {
        $p = Join-Path $ScriptDir $r
        if (Test-Path $p) { return $p }
    }
    return $null
}

$benchC = Find-Artifact @(
    "src\EventProp.Native.Benchmarks\build\Release\bench_c.exe",
    "src\EventProp.Native.Benchmarks\build\Debug\bench_c.exe",
    "src\EventProp.Native.Benchmarks\build\bench_c.exe"
)
$benchCpp = Find-Artifact @(
    "src\EventProp.Native.Benchmarks\build\Release\bench_cpp.exe",
    "src\EventProp.Native.Benchmarks\build\Debug\bench_cpp.exe",
    "src\EventProp.Native.Benchmarks\build\bench_cpp.exe"
)
if (-not $benchC -or -not $benchCpp) {
    $missing += "native benchmark binaries"
    Write-Host "  MISSING: bench_c.exe / bench_cpp.exe" -ForegroundColor Red
    Write-Host "           Build: .\build.ps1 release   (or .\run_benchmarks.ps1 -Build)" -ForegroundColor DarkGray
    # bench_cpp links the Eigen core; bench_c links the C library. Both are built by
    # the same preset, so a missing exe means the preset never ran, not that one
    # target failed.
}

$dotnetDll = Find-Artifact @(
    "src\EventProp.Benchmarks\bin\Release\net10.0\EventProp.Benchmarks.dll",
    "src\EventProp.Benchmarks\bin\Debug\net10.0\EventProp.Benchmarks.dll"
)
if (-not $dotnetDll) {
    $missing += ".NET benchmark dll"
    Write-Host "  MISSING: src\EventProp.Benchmarks\bin\Release\net10.0\EventProp.Benchmarks.dll" -ForegroundColor Red
    Write-Host "           Build: .\build.ps1 release   (it builds EventProp.Benchmarks)" -ForegroundColor DarkGray
}

# The C library .dll, at the path the Cython extension resolves. run_all.py exports
# DYLD_LIBRARY_PATH / LD_LIBRARY_PATH / PATH; on Windows the PATH entry is for
# non-Python consumers only, and the import below is what proves the extension loads.
$cDll = Find-Artifact @(
    "src\EventProp.C\build\Release\eventprop_c_api.dll",
    "src\EventProp.C\build\Debug\eventprop_c_api.dll",
    "src\EventProp.C\build\eventprop_c_api.dll"
)
if (-not $cDll) {
    $missing += "C library"
    Write-Host "  MISSING: eventprop_c_api.dll (looked in build\, build\Release\, build\Debug\)" -ForegroundColor Red
    Write-Host "           Build: .\build.ps1 release" -ForegroundColor DarkGray
}

if ($missing.Count -gt 0 -and -not $ReportOnly) {
    Write-Host ""
    Write-Host "$($missing.Count) prerequisite(s) missing. Nothing was run." -ForegroundColor Red
    Write-Host "Run .\build.ps1 release first, or: .\run_benchmarks.ps1 -Build" -ForegroundColor Yellow
    exit 1
}
if ($missing.Count -eq 0) { Write-Host "  artifacts: ok" -ForegroundColor DarkGray }

# --- 2. Python ----------------------------------------------------------------
# The reference library (pybind11) and the Cython layer are both imported by the
# python suite, with PYTHONPATH set by run_all.py. Import them the way the runner
# will, so a pass here means the suite can start.
if (-not $ReportOnly) {
    Write-Host "=== checking python ===" -ForegroundColor Cyan
    $pyPaths = @(
        (Join-Path $ScriptDir "src\EventProp.Python"),
        (Join-Path $ScriptDir "src\EventProp.Native.Python")
    ) -join ";"
    $probe = "import numpy, pytest, pytest_benchmark, eventprop, eventprop_c; " +
             "import eventprop_c._native; print('  python packages + both bindings import ok')"
    $env:PYTHONPATH = $pyPaths
    & $PythonExe -c $probe
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  The python suite cannot run. Exit code $LASTEXITCODE." -ForegroundColor Red
        Write-Host "  Packages:  $PythonExe -m pip install numpy pytest pytest-benchmark" -ForegroundColor Yellow
        Write-Host "  Cython:    .\build.ps1 release   (pip install -e src\EventProp.Native.Python)" -ForegroundColor Yellow
        Write-Host "  Reference: .\build.ps1 release   (pip install -e src\EventProp.Python)" -ForegroundColor Yellow
        Write-Host "  If the failure is 'DLL load failed while importing _native': the extension" -ForegroundColor Yellow
        Write-Host "  needs eventprop_c_api.dll found via os.add_dll_directory, which" -ForegroundColor Yellow
        Write-Host "  eventprop_c\_library_path.py registers from src\EventProp.C\build\Release." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  python: $(& $PythonExe --version)" -ForegroundColor DarkGray
}

# --- 3. Run the pipeline ------------------------------------------------------
$runner = Join-Path $ScriptDir "scripts\bench\run_all.py"
if (-not (Test-Path $runner)) {
    Write-Host "runner not found: $runner" -ForegroundColor Red
    exit 1
}

# NOTE: do NOT name this $args - that is a PowerShell automatic variable (the
# call-site argument array). Shadowing it works in a script but is fragile and
# reads as a bug.
$runnerArgs = @($runner)
if ($Full) {
    Write-Host "=== running FULL suite (includes XL; expect tens of minutes) ===" -ForegroundColor Cyan
} else {
    $runnerArgs += "--quick"
    Write-Host "=== running QUICK suite (Tiny/Small native; skips XL) ===" -ForegroundColor Cyan
}
if ($ReportOnly)    { $runnerArgs += "--report-only" }
if ($Sweeps)        { $runnerArgs += "--sweeps" }
if ($Suites)        { $runnerArgs += @("--suites") + $Suites }
if ($Tag)           { $runnerArgs += @("--tag", $Tag) }
if ($Out)           { $runnerArgs += @("--out", $Out) }
if ($ReportsDir)    { $runnerArgs += @("--reports-dir", $ReportsDir) }
if ($BenchmarksMd)  { $runnerArgs += @("--benchmarks-md", $BenchmarksMd) }

# Suppressing the committed outputs. run_all.py documents "pass '' to skip", but
# PowerShell DROPS empty strings from a native command's argument array, so
# -ReportsDir '' arrives as a bare --reports-dir and the runner silently uses the
# default - which overwrites the committed BENCHMARKS.md with a single-suite report.
# Measured: -ReportsDir '' -BenchmarksMd '' wrote both bench-reports\report_*.md and
# BENCHMARKS.md. The `--opt=` form is one token, so it survives the handoff and
# argparse receives the empty value it was designed for.
if ($NoReports)       { $runnerArgs += "--reports-dir=" }
if ($NoBenchmarksMd)  { $runnerArgs += "--benchmarks-md=" }

# A subset run rewrites the COMMITTED report from whatever records are in -Out, so a
# native-only run replaces the three-suite report with a one-suite report AT THE SAME
# PATH. Measured on this machine: a scratch `--report-only -Out <scratch>` overwrote
# bench-reports\report_windows-amd64.md and BENCHMARKS.md with a native-only report and
# exited 0 - a green run that destroyed the committed artifact. The -NoReports flag is the
# escape hatch; say so before the run rather than after.
if (-not $NoReports -and ($ReportOnly -or $Suites)) {
    Write-Host "  NOTE: partial run. This WILL rewrite the committed" -ForegroundColor Yellow
    Write-Host "  bench-reports\report_<tag>.md and BENCHMARKS.md from a SUBSET of the suites." -ForegroundColor Yellow
    Write-Host "  For a scratch run add:  -NoReports -NoBenchmarksMd" -ForegroundColor Yellow
    Write-Host ""
}

# Progress expectation: the runner streams each suite's own output (BenchmarkDotNet
# prints a live table per benchmark, pytest prints dots) AND emits a heartbeat to
# stderr every 60s, "done in Nm SSs" per command, and "[suite i/N]" markers. So a
# healthy run is visibly progressing; silence between heartbeats is normal, and NO
# heartbeat for several minutes means that command is genuinely stuck.
# Measured on an M3 MacBook Air, full suite: native ~9 min, .NET ~21 min, python
# ~2 min. XL is not the slow part - Google Benchmark's min_time is relative, so a 14 s
# iteration takes ONE iteration. The ~30 min is .NET's 75 benchmarks. Windows timings
# are not measured yet; this is the first run that will produce them.
Write-Host ""
Write-Host "Progress: each suite streams its own output, plus a 60s heartbeat and"
Write-Host "'done in Nm SSs' per command on stderr."
Write-Host ""

& $PythonExe @runnerArgs
$code = $LASTEXITCODE
Write-Host ""

# Derive the tag the same way run_all.py does (platform.system/machine lowercased), so
# this message cannot disagree with the filenames the runner actually wrote.
$derivedTag = & $PythonExe -c "import platform;print(f'{platform.system().lower()}-{platform.machine().lower()}')"
if (-not $derivedTag) { $derivedTag = "windows-amd64" }
if ($Tag) { $derivedTag = $Tag }

if ($code -ne 0) {
    # A suite failing is not fatal by design: the runner reports it, continues with the
    # other suites, and exits non-zero. A partial result is the expected outcome of a
    # first run on a new machine. Do NOT print the green report line here - on a
    # failed run there may be no report at all, and pointing at a file that does not
    # exist (or at a stale one from an earlier run) is how a bad run gets quoted.
    Write-Host "Pipeline exited non-zero ($code). Partial results are expected on a" -ForegroundColor Yellow
    Write-Host "first run on a new machine - read the suite output above before re-running." -ForegroundColor Yellow
    $reportPath = Join-Path $ScriptDir "bench-reports\report_${derivedTag}.md"
    if (Test-Path $reportPath) {
        Write-Host "A report exists at bench-reports\report_${derivedTag}.md, but it is from" -ForegroundColor Yellow
        Write-Host "this FAILED run - check its provenance header (vcs_rev, vcs_dirty) before" -ForegroundColor Yellow
        Write-Host "quoting it, and note it may cover fewer suites than a full run." -ForegroundColor Yellow
    }
    exit $code
}

$reportPath = Join-Path $ScriptDir "bench-reports\report_${derivedTag}.md"
if ($NoReports) {
    Write-Host "Reports suppressed (-NoReports): nothing written to bench-reports\ or BENCHMARKS.md." -ForegroundColor DarkGray
} elseif (Test-Path $reportPath) {
    Write-Host "Report (committed): bench-reports\report_${derivedTag}.md" -ForegroundColor Green
} else {
    Write-Host "Runner exited 0 but bench-reports\report_${derivedTag}.md is absent." -ForegroundColor Yellow
    Write-Host "Check --reports-dir (pass '' to suppress reports) and the run output above." -ForegroundColor Yellow
}
Write-Host "Raw records (ignored): bench-results\records_${derivedTag}_*.json" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Compare this report's per-pair GEOMEANS against report_darwin-arm64.md." -ForegroundColor Yellow
Write-Host "Absolute us columns are NOT comparable across machines (merge.py enforces this)." -ForegroundColor Yellow
exit $code
