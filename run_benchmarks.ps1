# run_benchmarks.ps1 - Run the cross-library benchmark pipeline on Windows.
#
# Usage:
#   .\run_benchmarks.ps1                 # quick run (Tiny/Small native, skips XL)
#   .\run_benchmarks.ps1 -Full           # full run, includes XL (long: ~30+ min)
#   .\run_benchmarks.ps1 -ReportOnly     # re-merge records already in bench-results\
#   .\run_benchmarks.ps1 -Build          # run .\build.ps1 release first, then benchmark
#
# What it does:
#   1. Verifies the build prerequisites exist (native benchmark binaries, the C
#      library, the .NET benchmark dll, the Cython extension). It does NOT build
#      them silently - a build failure must surface as a build failure, not as a
#      benchmark result. Pass -Build to have it invoke build.ps1 first.
#   2. Runs scripts\bench\run_all.py, which runs the three suites, normalizes each
#      tool's output into the common record schema, merges them, and writes:
#        bench-results\records_windows-amd64_*.json   (gitignored)
#        bench-results\merged_windows-amd64.json      (gitignored)
#        bench-reports\report_windows-amd64.md        (COMMITTED)
#        BENCHMARKS.md                                (COMMITTED, "latest run")
#
# Prerequisites (see AGENTS.md "TEMPORARY - cross-platform status"):
#   - Visual Studio / MSVC + CMake, Python 3.10+ with numpy, pytest,
#     pytest-benchmark, and .NET 10 SDK.
#   - Run .\build.ps1 release once first (or pass -Build).
#
# Cross-platform note: merge.py refuses to merge records from different hosts, so
# this produces a SEPARATE report from the macOS/Linux runs. Compare per-pair
# geomeans across reports, never the absolute us columns.

param(
    [switch]$Full,
    [switch]$ReportOnly,
    [switch]$Build,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Test-Prereq($path, $hint) {
    if (Test-Path $path) { return $true }
    Write-Host "  MISSING: $path" -ForegroundColor Red
    Write-Host "           $hint" -ForegroundColor DarkGray
    return $false
}

# --- 0. Optional build --------------------------------------------------------
if ($Build) {
    Write-Host "=== build (build.ps1 release) ===" -ForegroundColor Cyan
    & "$ScriptDir\build.ps1" -BuildType release
    if ($LASTEXITCODE -ne 0) {
        Write-Host "build.ps1 failed - stopping." -ForegroundColor Red
        exit 1
    }
}

# --- 1. Verify prerequisites --------------------------------------------------
# MSVC's CMake generator is multi-config, so the native binaries land in
# build\Release\, not build\. run_all.py probes both; this check must match.
Write-Host "=== checking prerequisites ===" -ForegroundColor Cyan
$missing = @()

$benchC = @(
    "src\EventProp.Native.Benchmarks\build\Release\bench_c.exe",
    "src\EventProp.Native.Benchmarks\build\bench_c.exe"
) | Where-Object { Test-Path (Join-Path $ScriptDir $_) }
if (-not $benchC) {
    $missing += "native benchmark binaries (bench_c.exe / bench_cpp.exe)"
    Write-Host "  MISSING: native benchmark binaries" -ForegroundColor Red
    Write-Host "           Build: .\build.ps1 release   (or .\run_benchmarks.ps1 -Build)" -ForegroundColor DarkGray
}

$dotnetDll = "src\EventProp.Benchmarks\bin\Release\net10.0\EventProp.Benchmarks.dll"
if (-not (Test-Path (Join-Path $ScriptDir $dotnetDll))) {
    $missing += ".NET benchmark dll"
    Write-Host "  MISSING: $dotnetDll" -ForegroundColor Red
    Write-Host "           Build: dotnet build src\EventProp.Benchmarks -c Release" -ForegroundColor DarkGray
}

# The Cython extension is what the Python suite imports. Its .pyd links against
# eventprop_c_api.lib, which setup.py locates by globbing build\*\*.lib.
$cython = Get-ChildItem -Path (Join-Path $ScriptDir "src\EventProp.Native.Python\eventprop_c") `
              -Filter "_native*.pyd" -ErrorAction SilentlyContinue
if (-not $cython) {
    $missing += "Cython extension (eventprop_c\_native*.pyd)"
    Write-Host "  MISSING: src\EventProp.Native.Python\eventprop_c\_native*.pyd" -ForegroundColor Red
    Write-Host "           Build: .\build.ps1 release   (it pip-installs the Cython layer)" -ForegroundColor DarkGray
}

# The C library .dll must be findable at RUN TIME by the Cython extension.
# On Windows there is no rpath equivalent in play here: the loader searches PATH
# and the extension's own directory. run_all.py prepends src\EventProp.C\build to
# PATH, but MSVC puts the .dll in build\Release\, so check both and say so.
$cDll = @(
    "src\EventProp.C\build\Release\eventprop_c_api.dll",
    "src\EventProp.C\build\eventprop_c_api.dll"
) | Where-Object { Test-Path (Join-Path $ScriptDir $_) }
if (-not $cDll) {
    $missing += "C library (eventprop_c_api.dll)"
    Write-Host "  MISSING: eventprop_c_api.dll (looked in build\ and build\Release\)" -ForegroundColor Red
    Write-Host "           Build: .\build.ps1 release" -ForegroundColor DarkGray
}

if ($missing.Count -gt 0 -and -not $ReportOnly) {
    Write-Host ""
    Write-Host "$($missing.Count) prerequisite(s) missing. Nothing was run." -ForegroundColor Red
    Write-Host "Run .\build.ps1 release first, or: .\run_benchmarks.ps1 -Build" -ForegroundColor Yellow
    exit 1
}

# --- 2. Python availability ---------------------------------------------------
if (-not $ReportOnly) {
    Write-Host "=== checking python packages ===" -ForegroundColor Cyan
    & $PythonExe -c "import numpy, pytest, pytest_benchmark" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  python packages missing (numpy, pytest, pytest-benchmark)" -ForegroundColor Red
        Write-Host "  Install: $PythonExe -m pip install numpy pytest pytest-benchmark" -ForegroundColor Yellow
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

# BENCHMARKS.md is the committed "latest run" copy. The per-platform report
# (bench-reports\report_<tag>.md) is written by the runner itself and is the
# durable artifact; it coexists with report_darwin-arm64.md by design.
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
if ($ReportOnly) { $runnerArgs += "--report-only" }

& $PythonExe @runnerArgs
$code = $LASTEXITCODE

Write-Host ""
if ($code -ne 0) {
    # A suite failing is not fatal by design: the runner reports it, continues with
    # the other suites, and exits non-zero. A partial cross-platform result is the
    # expected outcome of a first run on a new machine.
    Write-Host "Pipeline exited non-zero ($code). Partial results are expected on a" -ForegroundColor Yellow
    Write-Host "first run on a new machine - read the suite output above before re-running." -ForegroundColor Yellow
}

# Derive the tag the same way run_all.py does (platform.system/machine lowercased),
# so this message cannot disagree with the filenames the runner actually wrote.
$tag = & $PythonExe -c "import platform;print(f'{platform.system().lower()}-{platform.machine().lower()}')"
if (-not $tag) { $tag = "windows-amd64" }
Write-Host "Report (committed): bench-reports\report_${tag}.md" -ForegroundColor Green
Write-Host "Raw records (ignored): bench-results\records_${tag}_*.json" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Compare this report's per-pair GEOMEANS against report_darwin-arm64.md." -ForegroundColor Yellow
Write-Host "Absolute us columns are NOT comparable across machines (merge.py enforces this)." -ForegroundColor Yellow
exit $code
