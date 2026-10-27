# clean.ps1 — Remove build outputs from EventProp projects (Windows).
#
# Usage:
#   .\clean.ps1                                  # everything
#   .\clean.ps1 c                                # one target
#   .\clean.ps1 c cpp parity                     # targets COMPOSE (same as clean.sh)
#   .\clean.ps1 -Full                            # also delete the reference binary
#   .\clean.ps1 -DryRun results                  # list what would be removed
#   .\clean.ps1 -PythonExe python3.11 python     # which interpreter's pip uninstall runs
#
#   targets: all | c | cpp | python | dotnet | parity | benchmarks | results
#
# This is the same contract as clean.sh: composing targets, exit 2 on an unknown
# target, every platform suffix for the compiled extensions, _native.c removed, real
# failure reporting, and a real exit code. The previous version drifted from it in four
# ways, each measured:
#
#   * `-Target bogus` printed "Cleaning… Done." and exited 0 while removing nothing.
#     clean.sh's exit-2 guard never landed here, so a typo read as a successful clean.
#   * The dotnet list contained `EventProp.Cpp.Native` (NOT a project — the directory
#     does not exist) and omitted `EventProp.Abstractions`, which has real bin/obj.
#     `clean.ps1 dotnet`, and `all`, left Abstractions' build output behind forever.
#   * `-Target` was a single string, so targets did not compose.
#   * Remove-Item errors are non-terminating by default, so a locked build\ directory —
#     the normal Windows case while a benchmark exe has the C DLL loaded — printed an
#     error and the script still printed "Done." A build script whose last line is a
#     claim rather than a measurement; same class as the build.ps1 defect fixed in
#     8ef41a9.
#
# Why every platform suffix is listed: the compiled extension is _native.<abi>-<plat>.pyd
# on Windows, .so on Linux, .dylib on macOS. clean.sh deleted only `*.so` (and
# `eventprop_cpp.*.so/.dylib` for -full), which matches ZERO files on this machine — 2
# .pyd present, 0 .so — so `clean.sh python` left the extension in place and `-full`
# printed "→ Deleted reference binary" having deleted nothing. Each script was right on
# the platform the other covers.
#
# Why _native.c is removed: Cython generates it from _native.pyx and it is NOT tracked.
# build.ps1/build.sh regenerate it when _native.pyx is NEWER than _native.c, so a clean
# that leaves it behind makes the next build skip Cython and compile the stale generated
# C — the stale-extension class the ABI check in _native.pyx refuses to import on.
#
# What is NEVER touched: bench-reports\ and BENCHMARKS.md. They are committed artifacts,
# and a clean script that deletes them turns a cleanup into data loss. `results` removes
# regenerable measurements only, and stops at that boundary.
#
# A project .venv is NOT a build output and is not removed: the reference pybind11 build
# inside one is expensive to recreate, and the build/run scripts treat the interpreter as
# the user's choice.

# Parameter order is load-bearing. With $Targets declared LAST as
# ValueFromRemainingArguments, a bare word binds to the first positional parameter that
# can take it — which was $PythonExe. Measured: `.\clean.ps1 bogus` set PythonExe=bogus,
# left $Targets empty (so it defaulted to `all`), and ran a FULL clean, removing 39
# paths including the compiled extension, _native.c and the native baselines. A script
# whose usage line says `.\clean.ps1 c` must not treat `c` as the interpreter. Declaring
# $Targets as the first positional, with remaining-arguments binding, sends bare words to
# the target list and leaves -PythonExe named-only.
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Targets,
    [switch]$Full,
    [switch]$DryRun,
    [string]$PythonExe = "python"
)

# Native tools write to stderr; on Windows PowerShell 5.1 that is a terminating
# NativeCommandError when EAP is Stop, which would kill the script at the first pip
# warning. PS 7.3+ has an explicit switch; setting it is a no-op on 5.1.
$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$known = @("all", "c", "cpp", "python", "dotnet", "parity", "benchmarks", "results")
if (-not $Targets -or $Targets.Count -eq 0) { $Targets = @("all") }
foreach ($t in $Targets) {
    if ($known -notcontains $t.ToLower()) {
        Write-Error "clean.ps1: unknown target '$t'"
        Write-Host "Usage: .\clean.ps1 [-Full] [-DryRun] [-PythonExe EXE] [all|c|cpp|python|dotnet|parity|benchmarks|results ...]"
        exit 2
    }
}
$Targets = @($Targets | ForEach-Object { $_.ToLower() })
if ($Targets -contains "all") { $Targets = @("all") }

# Fail fast on a bad interpreter: discovering it mid-clean produces a CommandNotFoundException
# in the middle of a destructive script, which is how a typo becomes a partial clean.
if (-not $DryRun) {
    $pyCmd = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if (-not $pyCmd) {
        Write-Error "clean.ps1: PythonExe '$PythonExe' is not a command on PATH"
        Write-Host "Pass -PythonExe <exe> for the interpreter that has the packages, or -DryRun to inspect."
        exit 2
    }
}

$Failures = New-Object System.Collections.Generic.List[string]
$script:Removed = 0

function Should-Run($component) {
    return ($Targets -contains "all") -or ($Targets -contains $component)
}

# Remove a path, record a failure, keep going. -ErrorAction Stop + try/catch is the
# point: a locked directory must land in the failure list, not scroll past.
#
# Removing and REMOVED are different claims, and on Windows only the second one is a
# measurement. Remove-Item can return without error while the directory is still
# there: a file another process opened with FILE_SHARE_DELETE — which is what the
# MSBuild/Roslyn build nodes do to obj\ and bin\ — enters Windows delete-pending, so
# the unlink is queued, the call succeeds, and the path stays visible until the last
# handle closes. Measured on this machine: `clean.ps1 -Full` printed
# "Clean complete (27 path(s) removed)" and exited 0 with 8 .NET bin/obj paths still
# on disk (files included, not just empty shells); a second pass removed all 8. An
# unverified post-condition in a destructive script is the defect this script was
# rewritten to remove, so the path is re-tested after every removal.
function Remove-Tree($path) {
    if (-not (Test-Path $path)) { return }
    if ($DryRun) {
        Write-Host "  [dry-run] would remove $(Resolve-Path -Relative $path)"
        return
    }
    try {
        Remove-Item -Recurse -Force -ErrorAction Stop $path
    } catch {
        Write-Host "  FAILED to remove $path : $($_.Exception.Message)" -ForegroundColor Red
        $Failures.Add("rm $path")
        return
    }
    if (Test-Path -LiteralPath $path) {
        Write-Host "  FAILED to remove $path (still present - a process holds a handle)" -ForegroundColor Red
        $Failures.Add("rm $path (delete pending: path still present)")
        return
    }
    $script:Removed++
}

# Remove files matching -Path patterns under $dir. Same post-condition as Remove-Tree,
# and the count now reports removals against matches rather than matches dressed up
# as removals.
function Remove-Matched($dir, [string[]]$patterns) {
    if (-not (Test-Path $dir)) { return }
    $files = @(Get-ChildItem -Path (Join-Path $dir "*") -Include $patterns -File -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return }
    $gone = 0
    foreach ($f in $files) {
        if ($DryRun) {
            Write-Host "  [dry-run] would remove $($f.Name)"
            continue
        }
        try {
            Remove-Item -Force -ErrorAction Stop $f.FullName
        } catch {
            Write-Host "  FAILED to remove $($f.FullName) : $($_.Exception.Message)" -ForegroundColor Red
            $Failures.Add("rm $($f.Name)")
            continue
        }
        if (Test-Path -LiteralPath $f.FullName) {
            Write-Host "  FAILED to remove $($f.Name) (still present - a process holds a handle)" -ForegroundColor Red
            $Failures.Add("rm $($f.Name) (delete pending: file still present)")
            continue
        }
        $gone++
    }
    if ($DryRun) { Write-Host "  $($files.Count) file(s) matched" }
    else { Write-Host "  $gone/$($files.Count) file(s) removed" }
}

function Clean-PyCache($dir) {
    if (-not (Test-Path $dir)) { return }
    # NOT inside a .venv: __pycache__ there belongs to installed packages (numpy's,
    # pytest's), not to this project's build output. Deleting it is a surprise and costs
    # a rebuild. The .venv itself is the user's choice and is never removed.
    Get-ChildItem -Path $dir -Recurse -Directory -Include "__pycache__", ".pytest_cache" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '[\\/]\.venv[\\/]' -and $_.FullName -notmatch '[\\/]\.git[\\/]' } |
        ForEach-Object { Remove-Tree $_.FullName }
}

function Uninstall-Py($dist) {
    if ($DryRun) {
        Write-Host "  [dry-run] would run: $PythonExe -m pip uninstall -y $dist"
        return
    }
    # The interpreter that will RUN the suite, not the ambient `pip`: on this machine
    # `pip` is 3.14 while the suite runs on 3.11, so an uninstall aimed at `pip` can
    # report success against an interpreter the suite never uses.
    & $PythonExe -m pip uninstall -y --quiet $dist 2>$null | Out-Null
}

Write-Host "Cleaning EventProp build outputs (targets: $($Targets -join ' '), full: $($Full.IsPresent), dry-run: $($DryRun.IsPresent), python: $PythonExe)..."

# ── 1. C library ─────────────────────────────────────────────────────────────
if (Should-Run "c") {
    Write-Host ""; Write-Host "[1/9] EventProp.C"
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.C\build")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.C\build-werror")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.C.Tests\build")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.C.Tests\build-werror")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.C\.cache")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.C\compile_commands.json")
}

# ── 2. Eigen core ────────────────────────────────────────────────────────────
if (Should-Run "cpp") {
    Write-Host ""; Write-Host "[2/9] EventProp.Cpp.Wrapper"
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.Cpp.Wrapper\build")
}

# ── 3. Reference Python library (pybind11 submodule) ─────────────────────────
if (Should-Run "python") {
    Write-Host ""; Write-Host "[3/9] EventProp.Python (reference library)"
    Push-Location (Join-Path $ScriptDir "src\EventProp.Python")
    Uninstall-Py "eventprop"
    Remove-Tree "build"
    Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | ForEach-Object { Remove-Tree $_.FullName }
    Clean-PyCache "."
    if ($Full) {
        Write-Host "  reference binary:"
        Remove-Matched "eventprop" @("eventprop_cpp*.pyd", "eventprop_cpp*.dll", "eventprop_cpp*.so", "eventprop_cpp*.dylib")
    }
    Pop-Location
}

# ── 4. Cython bindings ───────────────────────────────────────────────────────
if (Should-Run "python") {
    Write-Host ""; Write-Host "[4/9] EventProp.Native.Python (Cython bindings)"
    Push-Location (Join-Path $ScriptDir "src\EventProp.Native.Python")
    Uninstall-Py "eventprop_c"
    Remove-Tree "build"
    Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | ForEach-Object { Remove-Tree $_.FullName }
    Write-Host "  compiled extension:"
    Remove-Matched "eventprop_c" @("_native*.pyd", "_native*.dll", "_native*.so", "_native*.dylib")
    # Generated by Cython; build.ps1's "regenerate when the .pyx is newer" rule is only
    # meaningful if a clean removes it.
    Remove-Matched "eventprop_c" @("_native.c")
    Remove-Matched "eventprop_c" @("*.obj", "*.exp", "*.lib")
    Clean-PyCache "."
    Pop-Location
}

# ── 5. Python benchmarks ─────────────────────────────────────────────────────
if (Should-Run "python") {
    Write-Host ""; Write-Host "[5/9] EventProp.Python.Benchmarks"
    Push-Location (Join-Path $ScriptDir "src\EventProp.Python.Benchmarks")
    # .venv deliberately NOT removed: not a build output, expensive to recreate.
    Remove-Tree "build"
    Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | ForEach-Object { Remove-Tree $_.FullName }
    Clean-PyCache "."
    Pop-Location
}

# ── 6. .NET ──────────────────────────────────────────────────────────────────
if (Should-Run "dotnet") {
    Write-Host ""; Write-Host "[6/9] .NET"
    # The handles that cause the delete-pending case above belong to MSBuild's and
    # Roslyn's persistent build nodes (`dotnet build-server`, plus whatever Visual
    # Studio is holding open). Releasing them first is what makes the FIRST pass
    # sufficient instead of requiring a second clean run; the next build pays one
    # node start-up. Best effort — a machine without dotnet on PATH must still clean
    # everything else, so a failure here is a note, never an exit.
    if (Get-Command dotnet -ErrorAction SilentlyContinue) {
        & dotnet build-server shutdown 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  build servers shut down (they hold handles in bin\ and obj\)" -ForegroundColor DarkGray
        } else {
            Write-Host "  note: dotnet build-server shutdown exited $LASTEXITCODE; a locked bin\ or obj\ may need a second pass" -ForegroundColor DarkGray
        }
    }
    Push-Location (Join-Path $ScriptDir "src")
    # EventProp.Abstractions is a real project (shared Layer contract) with real bin/obj.
    # EventProp.Cpp.Native is not a project and never was.
    foreach ($proj in @("EventProp", "EventProp.Abstractions", "EventProp.Native", "EventProp.Tests", "EventProp.Benchmarks")) {
        Remove-Tree (Join-Path $proj "bin")
        Remove-Tree (Join-Path $proj "obj")
    }
    Pop-Location
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.Tests\TestResults")
    Remove-Tree (Join-Path $ScriptDir "coverage-report")
}

# ── 7. Parity tests ──────────────────────────────────────────────────────────
if (Should-Run "parity") {
    Write-Host ""; Write-Host "[7/9] EventProp.Native.ParityTests"
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.Native.ParityTests\build")
}

# ── 8. Native benchmarks ─────────────────────────────────────────────────────
if (Should-Run "benchmarks") {
    Write-Host ""; Write-Host "[8/9] EventProp.Native.Benchmarks"
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.Native.Benchmarks\build")
}

# ── 9. Benchmark pipeline outputs ────────────────────────────────────────────
# Regenerable measurements only. bench-reports\ and BENCHMARKS.md are COMMITTED and are
# never removed by any target, including this one.
if (Should-Run "results") {
    Write-Host ""; Write-Host "[9/9] benchmark pipeline outputs"
    Remove-Tree (Join-Path $ScriptDir "bench-results")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.Benchmarks\BenchmarkDotNet.Artifacts")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.Benchmarks\bin\Release\net10.0\BenchmarkDotNet.Artifacts")
    Remove-Matched (Join-Path $ScriptDir "src\EventProp.Native.Benchmarks\baselines") @("*.json")
    Remove-Tree (Join-Path $ScriptDir "scripts\bench\__pycache__")
    Remove-Tree (Join-Path $ScriptDir "src\EventProp.Python.Benchmarks\.benchmarks")
}

Write-Host ""
Write-Host "============================================"
if ($Failures.Count -gt 0) {
    Write-Host " Clean INCOMPLETE - $($Failures.Count) removal(s) failed:" -ForegroundColor Red
    foreach ($f in $Failures) { Write-Host "   - $f" -ForegroundColor Red }
    Write-Host "============================================"
    Write-Host "A locked directory is usually a process holding the C library: close the running"
    Write-Host "benchmark / dotnet test / python session, then re-run."
    exit 1
}
if ($DryRun) {
    Write-Host " Dry run - nothing removed. Re-run without -DryRun to apply."
} else {
    Write-Host " Clean complete ($($script:Removed) path(s) removed)"
}
Write-Host "============================================"
Write-Host " bench-reports\ and BENCHMARKS.md are committed and were not touched."
exit 0
