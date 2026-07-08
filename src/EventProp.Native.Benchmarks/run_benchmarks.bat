@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM run_benchmarks.bat - Run C vs C++ benchmarks and compare
REM
REM Usage:
REM   run_benchmarks.bat                  Run both, save to baselines/, show comparison
REM   run_benchmarks.bat c                Run only C benchmark
REM   run_benchmarks.bat cpp              Run only C++ benchmark
REM   run_benchmarks.bat compare          Compare last saved results (no re-run)
REM   run_benchmarks.bat compare A.json B.json  Compare two specific JSON files
REM
REM Options (set before calling or inline):
REM   MIN_TIME=0.1s    Minimum benchmark time per test (default: 0.1s for quick runs)
REM   FILTER=ComputeSpikes/Tiny  Run only matching benchmarks (--benchmark_filter)
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "BUILD_DIR=%SCRIPT_DIR%build\Release"
set "BASELINES_DIR=%SCRIPT_DIR%baselines"
set "TOOLS_DIR=%SCRIPT_DIR%build\_deps\benchmark-src\tools"

if not exist "!BUILD_DIR!\bench_c.exe" (
    echo ERROR: bench_c.exe not found in !BUILD_DIR!
    echo Run CMake build first: cmake --build build --config Release
    exit /b 1
)
if not exist "!BUILD_DIR!\bench_cpp.exe" (
    echo ERROR: bench_cpp.exe not found in !BUILD_DIR!
    echo Run CMake build first: cmake --build build --config Release
    exit /b 1
)

REM Default benchmark options
if not defined MIN_TIME set "MIN_TIME=0.1s"
set "BENCH_OPTS=--benchmark_min_time=!MIN_TIME!"
if defined FILTER set "BENCH_OPTS=!BENCH_OPTS! --benchmark_filter=!FILTER!"

mkdir "!BASELINES_DIR!" 2>nul

REM Unique ID for output files
set "DATE_STAMP=%RANDOM%%RANDOM%"

REM ============================================================
REM Determine action from first argument
REM ============================================================

if /i "%1"=="compare" (
    goto :do_compare
) else if /i "%1"=="c" (
    set "RUN_C=1"
    set "RUN_CPP="
    set "DO_COMPARE="
) else if /i "%1"=="cpp" (
    set "RUN_C="
    set "RUN_CPP=1"
    set "DO_COMPARE="
) else (
    REM Default: run both and compare
    set "RUN_C=1"
    set "RUN_CPP=1"
    set "DO_COMPARE=1"
)

REM ============================================================
REM Run C benchmark
REM ============================================================
if defined RUN_C (
    echo.
    echo ========================================
    echo Running C benchmark...
    echo ========================================

    copy /y "!BUILD_DIR!\eventprop_c_api.dll" "!BUILD_DIR!\" >nul 2>&1

    set "C_OUTPUT=!BASELINES_DIR!\bench_c_!DATE_STAMP!.json"
    echo Output: !C_OUTPUT!

    cd /d "!BUILD_DIR!"
    bench_c.exe --benchmark_format=json --benchmark_out="!C_OUTPUT!" !BENCH_OPTS!

    if errorlevel 1 (
        echo ERROR: C benchmark failed!
        exit /b 1
    )
)

REM ============================================================
REM Run C++ benchmark
REM ============================================================
if defined RUN_CPP (
    echo.
    echo ========================================
    echo Running C++ benchmark...
    echo ========================================

    set "CPP_OUTPUT=!BASELINES_DIR!\bench_cpp_!DATE_STAMP!.json"
    echo Output: !CPP_OUTPUT!

    cd /d "!BUILD_DIR!"
    bench_cpp.exe --benchmark_format=json --benchmark_out="!CPP_OUTPUT!" !BENCH_OPTS!

    if errorlevel 1 (
        echo ERROR: C++ benchmark failed!
        exit /b 1
    )
)

REM ============================================================
REM Compare results
REM ============================================================
if defined DO_COMPARE (
    goto :do_compare
) else (
    echo.
    echo Done. Use run_benchmarks.bat compare to view comparison.
    exit /b 0
)

goto :end

:do_compare
    REM Find the two most recent JSON files if not specified
    set "BASELINE="
    set "CONTENDER="

    REM When called as: run_benchmarks.bat compare [file1] [file2]
    REM %1=compare, %2=file1 (optional), %3=file2 (optional)
    if "%2"=="" (
        for /f "tokens=* delims=" %%F in ('dir /b /o:-d "!BASELINES_DIR!\bench_c_*.json" 2^>nul') do if not defined BASELINE set "BASELINE=%%F"
        for /f "tokens=* delims=" %%F in ('dir /b /o:-d "!BASELINES_DIR!\bench_cpp_*.json" 2^>nul') do if not defined CONTENDER set "CONTENDER=%%F"

        if not defined BASELINE (
            echo ERROR: No C benchmark results found in !BASELINES_DIR!
            exit /b 1
        )
        if not defined CONTENDER (
            echo ERROR: No C++ benchmark results found in !BASELINES_DIR!
            exit /b 1
        )
    ) else if "%3"=="" (
        set "BASELINE=%~2"
        for /f "tokens=* delims=" %%F in ('dir /b /o:-d "!BASELINES_DIR!\bench_cpp_*.json" 2^>nul') do set "CONTENDER=%%F"
    ) else (
        set "BASELINE=%~2"
        set "CONTENDER=%~3"
    )

    echo.
    echo ========================================
    echo Comparing: !BASELINE! vs !CONTENDER!
    echo ========================================

    REM Resolve file paths: try baselines/ first, then SCRIPT_DIR
    if exist "!BASELINES_DIR!\!BASELINE!" (
        set "FULL_BASELINE=!BASELINES_DIR!\!BASELINE!"
    ) else if exist "!SCRIPT_DIR!!BASELINE!" (
        set "FULL_BASELINE=!SCRIPT_DIR!!BASELINE!"
    ) else (
        echo ERROR: Cannot find !BASELINE!
        exit /b 1
    )
    if exist "!BASELINES_DIR!\!CONTENDER!" (
        set "FULL_CONTENDER=!BASELINES_DIR!\!CONTENDER!"
    ) else if exist "!SCRIPT_DIR!!CONTENDER!" (
        set "FULL_CONTENDER=!SCRIPT_DIR!!CONTENDER!"
    ) else (
        echo ERROR: Cannot find !CONTENDER!
        exit /b 1
    )

    cd /d "!TOOLS_DIR!"
    python compare.py benchmarks "!FULL_BASELINE!" "!FULL_CONTENDER!" --no-utest

    if errorlevel 1 (
        echo.
        echo WARNING: compare.py exited with code !errorlevel!
        echo This may be due to benchmark name mismatches between runs.
    )

:end
cd /d "!SCRIPT_DIR!"
exit /b 0
