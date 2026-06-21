# clean.ps1 — Remove all build outputs from EventProp projects.
# Usage: .\clean.ps1 [-Full] [all|c|cpp|python|dotnet|parity|benchmarks]

param(
    [switch]$Full,
    [string]$Target = "all"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Cleaning EventProp build outputs (target: $Target, full: $($Full.IsPresent))..."

function Should-Run($component) {
    return ($Target -eq "all") -or ($Target -ieq $component)
}

# ── C Library (EventProp.C) ────────────────────────────────
if (Should-Run "c") {
    Write-Host "[1/8] Cleaning EventProp.C..."
    $path = Join-Path $ScriptDir "src\EventProp.C\build"
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}

# ── C++ Wrapper (EventProp.Cpp.Wrapper) ────────────────────
if (Should-Run "cpp") {
    Write-Host "[2/8] Cleaning EventProp.Cpp.Wrapper..."
    $path = Join-Path $ScriptDir "src\EventProp.Cpp.Wrapper\build"
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}

# ── Python Package (EventProp.Python) ──────────────────────
if (Should-Run "python") {
    Write-Host "[3/8] Cleaning EventProp.Python..."
    Push-Location (Join-Path $ScriptDir "src\EventProp.Python")
    pip uninstall eventprop -y --quiet 2>$null
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem -File -Filter "*.pyc" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    if ($Full) {
        Get-ChildItem "eventprop\eventprop_cpp.*.pyd", "eventprop\eventprop_cpp.*.dll" -ErrorAction SilentlyContinue | Remove-Item -Force
        Write-Host "  → Deleted reference binary (use build.ps1 to rebuild)"
    }
    Pop-Location
}

# ── Python Cython Bindings (EventProp.Native.Python) ───────
if (Should-Run "python") {
    Write-Host "[4/8] Cleaning EventProp.Native.Python..."
    Push-Location (Join-Path $ScriptDir "src\EventProp.Native.Python")
    pip uninstall eventprop_c -y --quiet 2>$null
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem "eventprop_c\*.pyd", "eventprop_c\*.dll" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Pop-Location
}

# ── Python Benchmarks (EventProp.Python.Benchmarks) ────────
if (Should-Run "python") {
    Write-Host "[5/8] Cleaning EventProp.Python.Benchmarks..."
    Push-Location (Join-Path $ScriptDir "src\EventProp.Python.Benchmarks")
    if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    Get-ChildItem -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem -Directory -Filter "__pycache__" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Pop-Location
}

# ── .NET Projects ──────────────────────────────────────────
if (Should-Run "dotnet") {
    Write-Host "[6/8] Cleaning .NET projects..."
    Push-Location (Join-Path $ScriptDir "src")
    foreach ($proj in @("EventProp", "EventProp.Native", "EventProp.Cpp.Native", "EventProp.Tests", "EventProp.Benchmarks")) {
        $bin = Join-Path $proj "bin"
        $obj = Join-Path $proj "obj"
        if (Test-Path $bin) { Remove-Item -Recurse -Force $bin }
        if (Test-Path $obj) { Remove-Item -Recurse -Force $obj }
    }
    Pop-Location
}

# ── Parity Tests (C vs Eigen) ──────────────────────────────
if (Should-Run "parity") {
    Write-Host "[7/8] Cleaning parity tests..."
    $path = Join-Path $ScriptDir "src\EventProp.Native.ParityTests\build"
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}

# ── C/C++ Benchmark Projects ───────────────────────────────
if (Should-Run "benchmarks") {
    Write-Host "[8/8] Cleaning benchmark projects..."
    $path = Join-Path $ScriptDir "src\EventProp.Native.Benchmarks\build"
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}

Write-Host "Done."
