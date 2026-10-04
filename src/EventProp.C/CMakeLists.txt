cmake_minimum_required(VERSION 3.16)
project(eventprop_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# AddressSanitizer — enables memory leak detection, use-after-free, buffer overflow checks.
# Usage: cmake .. -DENABLE_ASAN=ON (Debug builds only)
# Note: on macOS this must be applied to both library and tests together. See Tests/CMakeLists.txt.
option(ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
if(ENABLE_ASAN AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "AddressSanitizer enabled — rebuilds library + tests with instrumentation")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# Generate compile_commands.json for clangd/IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Strict warning flags — catch real issues without drowning numerical code in noise.
# -Wall: unused vars, implicit decls, format mismatches, etc.
# -Wextra: missing field initializers, type limits, unused params (with -Wunused)
# -Wpedantic: strict C11 conformance diagnostics
# -Wshadow: variable shadowing bugs
# -Wimplicit-fallthrough: missing break/return in switch cases
# -Wformat=2: stricter format string checks (signedness, %d vs %ld)
# -Wnull-dereference: potential NULL pointer dereferences before checks
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wimplicit-fallthrough
                        -Wformat=2 -Wnull-dereference -Wsign-conversion)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        # GCC-only diagnostics (not available on Clang/AppleClang)
        add_compile_options(-Wduplicated-cond   # duplicate conditions in if/else chains
                            -Wmisleading-indentation  # indentation that doesn't match control flow
                            -Wlogical-op)          # suspicious & vs &&, || vs |
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    # MSVC equivalents (enabled by default: /W3, /W4 for extra warnings)
    add_compile_options(/W4)
endif()



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
# ASan on macOS can't instrument shared libraries (AppleClang linker bug).
# Build as static instead when ASan is enabled — tests link directly, no dylib issues.
if(ENABLE_ASAN AND APPLE)
    message(STATUS "macOS + ASan detected: building STATIC library for memory testing")
    set(LIB_TYPE STATIC)
else()
    set(LIB_TYPE SHARED)
endif()

find_package(OpenMP QUIET)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(eventprop_c_api ${LIB_TYPE}
    eventprop_validation.c
    eventprop_simd.c
    eventprop_memory.c
    eventprop_sums.c
    eventprop_maxima.c
    eventprop_spikes.c
    eventprop_backward.c
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
        -fassociative-math  # allow reassociation of FP operations (e.g., (a+b)+c → a+(b+c))
        -fno-trapping-math  # assume FP operations will not generate traps
        -fno-math-errno     # do not set errno on FP errors
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
