cmake_minimum_required(VERSION 3.16)
project(eventprop_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)



# Detect SIMD support at compile time (priority: AVX-512 > AVX2 > NEON)
# Use runtime CPU feature detection, not just compiler flag acceptance
set(_HAS_AVX512 -1)
set(_HAS_AVX2 -1)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # Check actual CPU support via /proc/cpuinfo (Linux) or sysctl (macOS)
    if(EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" _CPUINFO)
        string(FIND "${_CPUINFO}" "avx512f" _HAS_AVX512)
        string(FIND "${_CPUINFO}" "avx2" _HAS_AVX2)
    else()
        # macOS fallback: check via sysctl
        execute_process(COMMAND sysctl -n machdep.cpu.features machdep.cpu.leaf7_features
                        OUTPUT_VARIABLE _CPU_FEATURES RESULT_VARIABLE _SYSCTL_FAIL)
        string(FIND "${_CPU_FEATURES}" "AVX512F" _HAS_AVX512)
        string(FIND "${_CPU_FEATURES}" "AVX2" _HAS_AVX2)
    endif()
endif()

if(NOT _HAS_AVX512 EQUAL -1)
    message(STATUS "AVX-512 detected — building with AVX-512 support")
    add_compile_definitions(HAVE_AVX512F)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
elseif(NOT _HAS_AVX2 EQUAL -1)
    message(STATUS "AVX2 detected — building with AVX2 support")
    add_compile_definitions(HAVE_AVX2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "ARM64 detected — NEON enabled by default")
    add_compile_definitions(__ARM_NEON)
endif()

# Pure C shared library (no external dependencies)
# Optional OpenMP for parallel neuron processing in ComputeSpikes
find_package(OpenMP QUIET)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(eventprop_c_api SHARED
    eventprop.c
    eventprop_diag.c
    eventprop_batch.c
)

# CPU-specific optimization and aggressive FP math optimizations
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang flags
    target_compile_options(eventprop_c_api PRIVATE -march=native)
    target_compile_options(eventprop_c_api PRIVATE -ffp-contract=fast)
    # Aggressive FP math optimizations — matches EventProp.Python's C++ flags.
    # Subset of -ffast-math but without -ffinite-math-only (avoids NaN/infinity UB).
    target_compile_options(eventprop_c_api PRIVATE
        -freciprocal-math   # x / y → x * (1/y) — many exp-decay divisions benefit
        -fno-signed-zeros   # treat ±0 as identical, enables more FP simplifications
        -funroll-loops      # unroll small per-neuron/per-spike loops
    )
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    # MSVC equivalent optimizations:
    # /O2 = Optimize for speed
    # /Ob2 = Inline functions
    # /Oi = Enable intrinsic functions
    # /Ot = Favor fast code
    # /fp:fast = Fast floating-point math (equivalent to -ffp-contract=fast, -freciprocal-math, -fno-signed-zeros)
    target_compile_options(eventprop_c_api PRIVATE /O2 /Ob2 /Oi /Ot /fp:fast)
endif()
# libm is separate on Unix, built-in on Windows
if(NOT WIN32)
    target_link_libraries(eventprop_c_api PRIVATE m)
endif()
if(OpenMP_C_FOUND)
    target_link_libraries(eventprop_c_api PRIVATE OpenMP::OpenMP_C)
    message(STATUS "OpenMP found — enabling parallel neuron processing")
else()
    # macOS fallback: use Homebrew libomp
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
    if(_BREW_FAIL EQUAL 0 AND EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
        target_compile_options(eventprop_c_api PRIVATE "-Xpreprocessor" "-fopenmp")
        target_include_directories(eventprop_c_api PRIVATE "${_OPENMP_PREFIX}/include")
        target_link_libraries(eventprop_c_api PRIVATE "${_OPENMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "OpenMP (Homebrew libomp) found — enabling parallel neuron processing")
    else()
        message(STATUS "OpenMP not found — running sequentially (Eigen uses OpenMP for parallel neuron processing)")
    endif()
endif()
target_include_directories(eventprop_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Install
install(TARGETS eventprop_c_api LIBRARY DESTINATION lib)
install(FILES eventprop.h eventprop_diag.h eventprop_batch.h DESTINATION include)

# Unit tests (see EventProp.C.Tests for Google Test suite)
