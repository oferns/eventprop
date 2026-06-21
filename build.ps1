# build.ps1 - Build all EventProp projects (C, C++, .NET, Python).
# Usage: .\build.ps1 [debug|release]

param(
    [string]$BuildType = "release"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($BuildType -ieq "debug") {
    $CMakeBuildType = "Debug"
    $DotNetConfig = "Debug"
} elseif ($BuildType -ieq "release") {
    $CMakeBuildType = "Release"
    $DotNetConfig = "Release"
} else {
    Write-Host "Usage: .\build.ps1 [debug|release]"
    exit 1
}

Write-Host "============================================"
Write-Host " EventProp Build - $BuildType mode"
Write-Host "============================================"

# Helper: run cmake configure + build in a subdirectory
function Invoke-CMakeBuild($projectPath, $outputName) {
    Push-Location $projectPath
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    New-Item -ItemType Directory -Name "build" | Out-Null
    Set-Location "build"
    cmake .. "-DCMAKE_BUILD_TYPE=$CMakeBuildType"
    cmake --build . --config $CMakeBuildType --parallel
    Write-Host "  -> $outputName"
    Pop-Location
}

# Helper: activate venv and run pip install
function Invoke-VenvInstall($projectPath) {
    Push-Location $projectPath
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        & ".venv\Scripts\Activate.ps1"
        pip install -e . --quiet > $null 2>&1
    } elseif (Test-Path ".venv\Scripts\python.exe") {
        # Use venv's Python directly - no activation needed
        & ".venv\Scripts\python.exe" -m pip install -e . --quiet > $null 2>&1
    }
    Pop-Location
}

# - 1. C Library (EventProp.C)
Write-Host ""
Write-Host "[1/9] Building C library (EventProp.C)..."
Invoke-CMakeBuild (Join-Path $ScriptDir "src\EventProp.C") "eventprop_c_api.dll"

# - 2. C++ Wrapper Library (EventProp.Cpp.Wrapper)
Write-Host ""
Write-Host "[2/9] Building C++ wrapper library (EventProp.Cpp.Wrapper)..."
Invoke-CMakeBuild (Join-Path $ScriptDir "src\EventProp.Cpp.Wrapper") "eventprop_cpp_wrapper.dll"

# - 3. Python Package (EventProp.Python) pybind11 bindings - reference library only
Write-Host ""
Write-Host "[3/9] Building Python package (EventProp.Python)..."
Push-Location (Join-Path $ScriptDir "src\EventProp.Python")

# Check if the compiled extension already exists (one-time build)
$foundExt = Get-ChildItem "eventprop\eventprop_cpp.*.pyd", "eventprop\eventprop_cpp.*.dll" -ErrorAction SilentlyContinue
if ($foundExt.Count -eq 0) {
    # Reset patched files to upstream state, apply platform patch, then build
    git checkout -- eventprop/loss_layer.py setup.py 2>$null
    $patchFile = Join-Path $ScriptDir "submodule-windows.patch"
    if (Test-Path $patchFile) {
        Get-Content $patchFile | patch -p1 > $null 2>&1
        Write-Host "  -> Applied submodule-windows.patch"
    } else {
        Write-Host "WARNING: No Windows patch found - skipping submodule patch"
    }

    # Install required Python dependencies and build
    python -m pip install setuptools numpy pytest > $null 2>&1
    python -m pip install -e . > $null 2>&1
    Write-Host "  -> eventprop.eventprop_cpp (editable install)"
} else {
    Write-Host "  -> eventprop.eventprop_cpp already built (skipping)"
}
Pop-Location

# - 4. Python Benchmarks
Write-Host ""
Write-Host "[4/9] Building Python benchmarks (EventProp.Python.Benchmarks)..."
Invoke-VenvInstall (Join-Path $ScriptDir "src\EventProp.Python.Benchmarks")
Write-Host "  -> eventprop-python-benchmarks (editable install)"

# - 5. Python Cython bindings
Write-Host ""
Write-Host "[5/9] Building Python cython bindings (EventProp.Native.Python)..."
Invoke-VenvInstall (Join-Path $ScriptDir "src\EventProp.Native.Python")
Write-Host "  -> eventprop_c (editable install)"

# - 6. .NET Libraries (EventProp + EventProp.Native)
Write-Host ""
Write-Host "[6/9] Building .NET libraries..."
Push-Location (Join-Path $ScriptDir "src")
foreach ($dir in @("bin", "obj")) {
    if (Test-Path "EventProp\$dir") { Remove-Item -Recurse -Force "EventProp\$dir" }
    if (Test-Path "EventProp.Native\$dir") { Remove-Item -Recurse -Force "EventProp.Native\$dir" }
}

dotnet build EventProp\EventProp.csproj -c $DotNetConfig --nologo
dotnet build EventProp.Native\EventProp.Native.csproj -c $DotNetConfig --nologo
Write-Host "  -> CW.EventProp.dll + EventProp.Native.dll"
Pop-Location

# - 7. .NET Tests (EventProp.Tests)
Write-Host ""
Write-Host "[7/9] Building .NET tests..."
Push-Location (Join-Path $ScriptDir "src")
foreach ($dir in @("bin", "obj")) {
    if (Test-Path "EventProp.Tests\$dir") { Remove-Item -Recurse -Force "EventProp.Tests\$dir" }
}

dotnet build EventProp.Tests\EventProp.Tests.csproj -c $DotNetConfig --nologo
Write-Host "  -> CW.EventProp.Tests.dll"
Pop-Location

# - 8. Parity Tests (C vs Eigen)
Write-Host ""
Write-Host "[8/9] Building parity tests (C vs Eigen)..."
Invoke-CMakeBuild (Join-Path $ScriptDir "src\EventProp.Native.ParityTests") "parity_test.exe"

# - 9. Native Benchmarks (C vs C++)
Write-Host ""
Write-Host "[9/9] Building native benchmarks (C vs C++)..."
Invoke-CMakeBuild (Join-Path $ScriptDir "src\EventProp.Native.Benchmarks") "bench_c.exe + bench_cpp.exe"

# - Done
Write-Host ""
Write-Host "============================================"
Write-Host " Build complete - $BuildType mode"
Write-Host "============================================"
