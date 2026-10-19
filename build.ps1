# build.ps1 - Build all EventProp projects (C, C++, .NET, Python) on Windows.
#
# Usage:
#   .\build.ps1                # release
#   .\build.ps1 debug
#   .\build.ps1 -SkipPython    # skip the two Python layers (reference + Cython)
#   .\build.ps1 -Clean         # delete each build dir first (flag changes, cache leaks)
#
# Why presets, not `mkdir build && cmake ..`
#   The five CMake projects ship CMakePresets.json, and AGENTS.md names presets as
#   the supported path. The hand-typed form skipped them, which loses:
#     - ENABLE_ASAN=OFF stated explicitly. It is a cache variable, so a previous
#       `--preset asan` leaks and the release build silently produces an
#       ASan-instrumented STATIC library, which breaks the parity tests (they import
#       a SHARED library) and poisons benchmark timings.
#     - the per-project settings the presets own (warning flags, /experimental:c11atomics
#       probing, the SIMD level).
#   Every preset writes to ${sourceDir}/build, which is what the cross-project
#   imports hardcode (C.Tests, ParityTests and Native.Benchmarks each consume a
#   PRE-BUILT library out of EventProp.C/build and EventProp.Cpp.Wrapper/build), so
#   the order below is load-bearing, not cosmetic.
#
# Why every step's exit code is checked
#   The previous version checked none of them, so a failed step printed nothing and
#   the script ended with "Build complete". A build script whose final line is a
#   claim rather than a measurement is how a compile failure reaches a benchmark
#   number.
#
# What it builds that the previous version did not
#   - EventProp.C.Tests (the 220-test Google Test suite)
#   - EventProp.Benchmarks (the .NET benchmark dll)
#   The second is a prerequisite run_benchmarks.ps1 checks for, so `-Build` could
#   never satisfy the check its own script performs.

param(
    [string]$BuildType = "release",
    [string]$PythonExe = "python",
    [switch]$SkipSubmodule,
    [switch]$SkipPython,
    [switch]$Clean
)

# Native tools (cmake, dotnet, pip, MSBuild) write progress to stderr. With
# $ErrorActionPreference = "Stop", Windows PowerShell 5.1 turns that into a
# terminating NativeCommandError at the first warning, so the build dies in the
# middle and the friendly per-step reporting never runs. PS 7.3+ has an explicit
# switch for this; setting it is a no-op on 5.1 (unknown variable).
$PSNativeCommandUseErrorActionPreference = $false
$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if ($BuildType -ieq "debug") {
    $Preset = "debug"; $DotNetConfig = "Debug"
} elseif ($BuildType -ieq "release") {
    $Preset = "release"; $DotNetConfig = "Release"
} else {
    Write-Host "Usage: .\build.ps1 [debug|release]   (or -BuildType debug)"
    exit 1
}

$Failures = New-Object System.Collections.Generic.List[string]
$Skips = New-Object System.Collections.Generic.List[string]

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "[$n] $msg" -ForegroundColor Cyan
}

# Run a block, record the failure, keep going. Returns $true on success.
function Invoke-Step($name, [scriptblock]$body) {
    try {
        & $body
    } catch {
        Write-Host "  EXCEPTION: $_" -ForegroundColor Red
        $script:Failures.Add("$name (exception)")
        return $false
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED (exit $LASTEXITCODE): $name" -ForegroundColor Red
        $script:Failures.Add("$name (exit $LASTEXITCODE)")
        return $false
    }
    return $true
}

function Invoke-CMakePreset($dir, $preset, $label) {
    $full = Join-Path $ScriptDir $dir
    if (-not (Test-Path (Join-Path $full "CMakePresets.json"))) {
        Write-Host "  no CMakePresets.json in $dir" -ForegroundColor Yellow
        $script:Skips.Add("$dir (no presets)")
        return $false
    }
    Push-Location $full
    if ($Clean -and (Test-Path "build")) {
        Write-Host "  -Clean: removing build\" -ForegroundColor DarkGray
        Remove-Item -Recurse -Force "build"
    }
    $ok = Invoke-Step "cmake --preset $preset ($dir)" { cmake --preset $preset }
    if ($ok) {
        $ok = Invoke-Step "cmake --build --preset $preset ($dir)" { cmake --build --preset $preset }
    }
    Pop-Location
    if ($ok) { Write-Host "  -> $label" -ForegroundColor Green }
    return $ok
}

function Invoke-DotNet($proj, $config) {
    return Invoke-Step "dotnet build $proj" {
        dotnet build $proj -c $config --nologo
    }
}

Write-Host "============================================"
Write-Host " EventProp Build - $BuildType (preset '$Preset')"
Write-Host "============================================"

# Resolve the bash that ships with Git for Windows. Do NOT take `bash` from PATH:
# `where bash` on this machine returns C:\WINDOWS\System32\bash.exe (WSL) first, and
# WSL bash is a Linux userland that cannot see G:\ as a Windows working tree, so
# "bash exists" is not evidence the script can run. That is exactly how this showed
# up: the same call succeeded from Git Bash and failed with exit 1 from a PATH where
# System32 precedes Git's usr\bin.
function Get-GitBash {
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if (-not $git) { return $null }
    $root = Split-Path (Split-Path $git.Source -Parent) -Parent
    foreach ($c in @((Join-Path $root "usr\bin\bash.exe"), (Join-Path $root "bin\bash.exe"))) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Get-RegistryPathValue {
    <#
    .SYNOPSIS
        Read a PATH value straight from the registry, unexpanded.

    The process PATH is a snapshot taken when the shell started. On this machine that
    mattered: the machine PATH had a broken entry (an unbalanced quote) that was fixed
    in the registry, and every shell opened before the sign-in that applies it kept the
    broken one. vcvarsall dies parsing it - "\NVIDIA was unexpected at this time", exit
    255 - so a script that inherits PATH inherits the defect. Reading the registry gets
    the state the machine is actually in.
    #>
    param([Parameter(Mandatory)] [string]$Key)

    $out = & reg.exe query $Key /v Path 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
    foreach ($line in $out) {
        if ($line -match 'REG_(EXPAND_)?SZ\s+(.*)$') { return $matches[2].Trim() }
    }
    return $null
}

function Enter-VisualStudioEnvironment {
    param([string]$Arch = "x86_amd64")

    # 1. PATH from the registry, machine then user, expanded.
    $machinePath = Get-RegistryPathValue -Key 'HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
    $userPath    = Get-RegistryPathValue -Key 'HKCU\Environment'
    if ($machinePath) {
        $env:PATH = [Environment]::ExpandEnvironmentVariables($machinePath)
        Write-Host "  PATH read from the registry (machine), not inherited from this shell" -ForegroundColor DarkGray
    }
    if ($userPath) {
        $env:PATH = $env:PATH + ';' + [Environment]::ExpandEnvironmentVariables($userPath)
    }

    # 2. vcvarsall, located by vswhere (the VS Installer), not by a hardcoded path: the
    #    directory name carries the product year and the Insider channel.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vcvars = $null
    if (Test-Path $vswhere) {
        $found = & $vswhere -latest -prerelease -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -find "VC\Auxiliary\Build\vcvarsall.bat" 2>$null
        if ($found) { $vcvars = @($found)[0] }
    }
    if (-not $vcvars -or -not (Test-Path $vcvars)) {
        Write-Host "  vcvarsall.bat not found via vswhere - leaving the environment as-is" -ForegroundColor Yellow
        Write-Host "  the CMake steps may still work; the Python extension build will not" -ForegroundColor Yellow
        return $false
    }

    # 3. Run it through a temp .bat and capture the environment it produces. Passing
    #    `call "..." && set` as one quoted string to cmd.exe is the classic quoting trap
    #    (exit 255, "The syntax of the command is incorrect"); a file has no quoting at
    #    all. `set` after the call prints the environment vcvars built.
    $dumpScript = Join-Path $env:TEMP "eventprop_vcvars_dump.bat"
    Set-Content -Path $dumpScript -Value "@echo off`r`ncall `"$vcvars`" $Arch`r`nset" -Encoding Ascii
    $dump = & cmd.exe /c "`"$dumpScript`""
    $dumpExit = $LASTEXITCODE
    Remove-Item $dumpScript -ErrorAction SilentlyContinue
    if ($dumpExit -ne 0) {
        Write-Host "  vcvarsall $Arch failed (exit $dumpExit)" -ForegroundColor Yellow
        Write-Host "  PATH it was given starts: $($env:PATH.Substring(0, [Math]::Min(80, $env:PATH.Length)))" -ForegroundColor DarkGray
        return $false
    }
    foreach ($line in $dump) {
        $i = $line.IndexOf('=')
        if ($i -lt 1) { continue }
        Set-Item -Path "Env:$($line.Substring(0, $i))" -Value $line.Substring($i + 1) -ErrorAction SilentlyContinue
    }
    # Strip Platform for the WHOLE script. MSBuild's $(Platform) inference reads the
    # environment, so with Platform=X64 imported from vcvarsall the `dotnet build` steps in
    # this SAME process wrote bin\x64\Release\net10.0\ instead of bin\Release\net10.0\.
    # Measured: EventProp.Benchmarks and EventProp.Tests both grew a bin\x64\ tree, while
    # run_benchmarks.ps1's prerequisite check and scripts/bench/run_all.py name the
    # framework-dependent path — so the pipeline reported the benchmark binary MISSING after
    # a green build, and the only reason it worked before is that a pre-existing
    # bin\Release\net10.0\ from an IDE build was still on disk. Platform is a VS/MSBuild
    # concept; the framework-dependent layout is what every consumer here expects, and the
    # CMake presets do not want it either.
    Remove-Item Env:Platform -ErrorAction SilentlyContinue
    Write-Host "  MSVC $env:VCToolsVersion ($Arch) environment imported; Platform stripped for MSBuild" -ForegroundColor Green
    return $true
}

# --- MSVC environment ---------------------------------------------------------
# CMake finds the compiler by itself (the generator probes the registry), so the CMake
# steps work in a plain shell. The Python extension build does NOT: setuptools' MSVC
# probe needs the vcvars environment, and without it `pip install -e .` dies with
#
#   DistutilsPlatformError: Microsoft Visual C++ 14.0 or greater is required
#
# Measured: `build.ps1 release` passed steps 2-6 and failed step 8 with exactly that
# message, while the same pip command succeeded in a shell where vcvarsall had been
# called. The script therefore imports the vcvars environment up front, so every step -
# CMake and pip alike - sees one compiler.

# --- 1. Submodule state -------------------------------------------------------
# The reference submodule is read-only and needs local modifications to build.
# scripts/submodule-state.sh is the single owner of that state, so call it rather
# than re-implementing it here. The previous version piped the patch through
# `patch -p1` and discarded the result: `patch` is more forgiving than `git apply`
# (it applies the good hunks and aborts at a malformed one), and the output went to
# $null, so a truncated patch left the submodule half-patched and the build failed
# far from the cause.
Write-Step "1/9" "submodule state (scripts/submodule-state.sh apply)"
if ($SkipSubmodule) {
    Write-Host "  -SkipSubmodule" -ForegroundColor DarkGray
    $Skips.Add("submodule state (-SkipSubmodule)")
} else {
    $sh = Join-Path $ScriptDir "scripts\submodule-state.sh"
    if (-not (Test-Path $sh)) {
        Write-Host "  scripts\submodule-state.sh not found" -ForegroundColor Red
        $Failures.Add("submodule-state.sh missing")
    } else {
        $bash = Get-GitBash
        if (-not $bash) {
            Write-Host "  Git for Windows bash not found (not using System32\bash.exe - that is WSL)" -ForegroundColor Yellow
            Write-Host "  applying submodule-windows.patch with git apply instead" -ForegroundColor Yellow
            Push-Location (Join-Path $ScriptDir "src\EventProp.Python")
            $patch = Join-Path $ScriptDir "submodule-windows.patch"
            git apply --check $patch 2>&1 | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
            if ($LASTEXITCODE -eq 0) {
                git apply $patch
                Write-Host "  -> submodule-windows.patch applied" -ForegroundColor Green
            } else {
                Write-Host "  patch does not apply cleanly: already applied, or upstream moved." -ForegroundColor Yellow
                Write-Host "  Check: bash scripts/submodule-state.sh status" -ForegroundColor Yellow
            }
            Pop-Location
        } else {
            # Git bash needs its own usr\bin on PATH for coreutils (the script uses
            # dirname). Invoking bash.exe directly from a Windows process inherits the
            # Windows PATH, which does not contain it, so the script dies with
            # "dirname: command not found" and a null directory. Opening "Git Bash"
            # hides this because the launcher adds those directories.
            $prevPath = $env:PATH
            $env:PATH = (Split-Path $bash) + ";" + $prevPath
            try {
                Invoke-Step "submodule-state.sh apply" { & $bash "scripts/submodule-state.sh" apply } | Out-Null
            } finally {
                $env:PATH = $prevPath
            }
        }
    }
}

# --- 2-3. C library, then the Eigen core --------------------------------------
# Order matters: the consumers import pre-built artifacts from these two.
Write-Host ""
Write-Host "[env] MSVC environment for the Python extension build" -ForegroundColor Cyan
Enter-VisualStudioEnvironment | Out-Null

Write-Step "2/9" "C library (EventProp.C)"
Invoke-CMakePreset "src\EventProp.C" $Preset "eventprop_c_api.dll (build\$Preset on MSVC)" | Out-Null

Write-Step "3/9" "Eigen core (EventProp.Cpp.Wrapper)"
Invoke-CMakePreset "src\EventProp.Cpp.Wrapper" $Preset "eventprop_eigen_core + eventprop_cpp_wrapper" | Out-Null

# --- 4. C unit tests ----------------------------------------------------------
# EventProp.C.Tests ships `release` and `asan` presets only, so there is nothing to
# do in debug. Say so: the previous version did not build this project at all, and a
# silent omission reads as "tests passed".
Write-Step "4/9" "C unit tests (EventProp.C.Tests, 220 tests)"
if ($Preset -eq "release") {
    Invoke-CMakePreset "src\EventProp.C.Tests" "release" "eventprop_tests.exe  (ctest --preset release)" | Out-Null
} else {
    Write-Host "  no 'debug' preset (presets are release + asan) - skipped" -ForegroundColor Yellow
    $Skips.Add("EventProp.C.Tests (no debug preset)")
}

# --- 5-6. Parity tests, native benchmarks -------------------------------------
Write-Step "5/9" "parity tests (C vs Eigen)"
Invoke-CMakePreset "src\EventProp.Native.ParityTests" $Preset "parity_test.exe" | Out-Null

Write-Step "6/9" "native benchmarks (C vs C++)"
Invoke-CMakePreset "src\EventProp.Native.Benchmarks" $Preset "bench_c.exe + bench_cpp.exe" | Out-Null

# --- 7. Reference Python library (pybind11 submodule) -------------------------
# Which interpreter? The one that will RUN the suite. scripts/bench/run_all.py invokes
# every tool with sys.executable, i.e. the interpreter that launched it, and
# run_benchmarks.ps1 defaults to `python`. Installing into a .venv that nobody runs
# produces a tree where the extension is built, the import works under the venv, and
# the benchmark suite still cannot import it. So: same interpreter as the runner,
# stated out loud, with a note when a venv exists and is not being used.
function Get-PyFor($projectPath) {
    $venvPy = Join-Path $projectPath ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        Write-Host "  note: a .venv exists here ($venvPy)" -ForegroundColor DarkGray
        Write-Host "        the benchmark runner uses '$PythonExe', so the install targets that." -ForegroundColor DarkGray
        Write-Host "        To build for the venv: .\build.ps1 -PythonExe `"$venvPy`"" -ForegroundColor DarkGray
    }
    return $PythonExe
}

Write-Step "7/9" "reference Python library (EventProp.Python)"
if ($SkipPython) {
    Write-Host "  -SkipPython" -ForegroundColor DarkGray
    $Skips.Add("EventProp.Python (-SkipPython)")
} else {
    Push-Location (Join-Path $ScriptDir "src\EventProp.Python")
    $built = Get-ChildItem "eventprop\eventprop_cpp*.pyd", "eventprop\eventprop_cpp*.dll", "eventprop\eventprop_cpp*.so" -ErrorAction SilentlyContinue
    if ($built) {
        Write-Host "  -> already built ($($built[0].Name)); rebuild by deleting it" -ForegroundColor Green
    } else {
        Invoke-Step "pip install -e . (EventProp.Python)" {
            & $PythonExe -m pip install -e . --no-build-isolation
        } | Out-Null
    }
    Pop-Location
}

# --- 8. Cython bindings -------------------------------------------------------
Write-Step "8/9" "Cython bindings (EventProp.Native.Python)"
if ($SkipPython) {
    Write-Host "  -SkipPython" -ForegroundColor DarkGray
    $Skips.Add("EventProp.Native.Python (-SkipPython)")
} else {
    Push-Location (Join-Path $ScriptDir "src\EventProp.Native.Python")
    $py = Get-PyFor (Get-Location)
    Write-Host "  interpreter: $py  $(& $py --version 2>&1)" -ForegroundColor DarkGray

    # The extension compiles _native.c, which Cython generates from _native.pyx, and
    # _native.c is NOT tracked - it is a build artifact. Two consequences: a fresh clone
    # has no _native.c at all, and an edited .pyx does not rebuild. The second is the
    # stale-extension class this project measured (a .pyd compiled from old headers reads
    # struct fields at the wrong offsets, and the ABI check in _native.pyx now refuses to
    # import when that happens), so regenerate whenever the .pyx is newer than the .c.
    $pyx = Join-Path (Get-Location) "eventprop_c\_native.pyx"
    $cfile = Join-Path (Get-Location) "eventprop_c\_native.c"
    $needCython = -not (Test-Path $cfile)
    if ((Test-Path $cfile) -and (Test-Path $pyx)) {
        $needCython = (Get-Item $pyx).LastWriteTime -gt (Get-Item $cfile).LastWriteTime
    }
    if ($needCython) {
        Invoke-Step "cython _native.pyx" { & $py -m cython "eventprop_c\_native.pyx" } | Out-Null
    } else {
        Write-Host "  _native.c is newer than _native.pyx - skipping cython" -ForegroundColor DarkGray
    }

    Invoke-Step "pip install -e . (EventProp.Native.Python)" {
        & $py -m pip install -e . --no-build-isolation
    } | Out-Null
    Pop-Location
}

# --- 9. .NET ------------------------------------------------------------------
Write-Step "9/9" ".NET (libraries, tests, benchmarks)"
Push-Location (Join-Path $ScriptDir "src")
foreach ($p in @("EventProp", "EventProp.Native", "EventProp.Tests", "EventProp.Benchmarks")) {
    Invoke-DotNet "$p\$p.csproj" $DotNetConfig | Out-Null
}
Pop-Location

# EventProp.Native/EventProp.Native.csproj:24 copies
# ..\EventProp.C\build\Release\eventprop_c_api.dll into its output, so a debug build
# of the C library is not the one the .NET native tests load. Not a defect to fix
# here - it is a trap to name, because the failure shows up as a P/Invoke load error
# in a debug run that looks unrelated to the build.
if ($DotNetConfig -eq "Debug") {
    Write-Host "  note: EventProp.Native.csproj hardcodes ..\EventProp.C\build\Release\eventprop_c_api.dll" -ForegroundColor Yellow
    Write-Host "        a debug run loads the RELEASE C library unless that file is a debug build." -ForegroundColor Yellow
}

# --- Summary ------------------------------------------------------------------
Write-Host ""
Write-Host "============================================"
if ($Failures.Count -eq 0) {
    Write-Host " Build complete - $BuildType mode" -ForegroundColor Green
} else {
    Write-Host " Build INCOMPLETE - $($Failures.Count) step(s) failed:" -ForegroundColor Red
    foreach ($f in $Failures) { Write-Host "   - $f" -ForegroundColor Red }
}
if ($Skips.Count -gt 0) {
    Write-Host " skipped:" -ForegroundColor Yellow
    foreach ($s in $Skips) { Write-Host "   - $s" -ForegroundColor Yellow }
}
Write-Host "============================================"
Write-Host "MSVC is multi-config: artifacts land in build\$Preset\, not build\." -ForegroundColor DarkGray
Write-Host "The parity test exe lands in src\EventProp.C\build\$Preset\$Preset\ (its" -ForegroundColor DarkGray
Write-Host "RUNTIME_OUTPUT_DIRECTORY plus the generator's config dir)." -ForegroundColor DarkGray
exit $(if ($Failures.Count -eq 0) { 0 } else { 1 })
