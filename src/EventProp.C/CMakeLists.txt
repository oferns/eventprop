cmake_minimum_required(VERSION 3.21)
project(eventprop_c LANGUAGES C)

# The configure presets set CMAKE_BUILD_TYPE, and on a multi-config generator (MSVC,
# Xcode) nothing consumes it — so CMake prints "Manually-specified variables were not
# used by the project: CMAKE_BUILD_TYPE" on every Windows configure. The presets cannot
# simply stop setting it: single-config generators (Unix Makefiles, Ninja — Linux and
# macOS) take their optimisation level from it, so dropping it there would silently
# build -O0 and poison every benchmark number. Reading it here is what tells CMake the
# variable is used; the default is the honest counterpart of that reading, because a
# single-config generator with no build type is an unoptimised build nobody chose.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (single-config generators only)" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# C11 atomics — eventprop_batch.h includes <stdatomic.h> and the batch implementation
# sets has_errors with atomic_store_explicit from the OpenMP loops.
#
# MSVC defines __STDC_NO_ATOMICS__ (so vcruntime_c11_stdatomic.h hits its #error
# "C atomic support is not enabled") on every toolset tested up to 14.52 / VS 2026,
# INCLUDING with /std:c11 and /std:c17. The switch that enables them is
# /experimental:c11atomics. Probed by compiling, not by a compiler-version test:
# the probe is the record of what this toolset actually does, so a toolset that
# supports atomics unconditionally gets no flag and a future MSVC that drops the
# switch does not silently keep passing it.
include(CheckCSourceCompiles)
set(EVENTPROP_ATOMICS_PROBE "
#include <stdatomic.h>
atomic_int flag;
int main(void) {
    atomic_store_explicit(&flag, 1, memory_order_relaxed);
    return atomic_load_explicit(&flag, memory_order_relaxed) - 1;
}
")
# Compile-only probe: the flag question is a front-end question, and a link step
# would let an unrelated missing system library answer it.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
# The probe must use the library's own language standard, or MSVC rejects the
# header for a second, unrelated reason (__STDC_VERSION__ < 201112L) and reports
# "no atomics" about a toolset that has them.
if(MSVC)
    set(CMAKE_REQUIRED_FLAGS "/std:c11")
endif()
check_c_source_compiles("${EVENTPROP_ATOMICS_PROBE}" EVENTPROP_C11_ATOMICS_ENABLED)
if(MSVC AND NOT EVENTPROP_C11_ATOMICS_ENABLED)
    set(CMAKE_REQUIRED_FLAGS "/std:c11 /experimental:c11atomics")
    check_c_source_compiles("${EVENTPROP_ATOMICS_PROBE}" EVENTPROP_C11_ATOMICS_EXPERIMENTAL)
    if(EVENTPROP_C11_ATOMICS_EXPERIMENTAL)
        set(EVENTPROP_ATOMICS_COMPILE_OPTIONS "/experimental:c11atomics")
        message(STATUS "C11 atomics: this MSVC toolset needs /experimental:c11atomics — enabling for eventprop_c_api")
    endif()
endif()
unset(CMAKE_REQUIRED_FLAGS)
unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
if(NOT EVENTPROP_C11_ATOMICS_ENABLED AND NOT EVENTPROP_C11_ATOMICS_EXPERIMENTAL)
    message(FATAL_ERROR
        "The C compiler cannot compile <stdatomic.h> in C11 mode, and "
        "eventprop_batch.h requires C11 atomics for the batch has_errors flag. "
        "Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} "
        "(${CMAKE_C_COMPILER})")
endif()

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
    # UCRT's <math.h> defines INFINITY as ((float)(_HUGE_ENUF * _HUGE_ENUF)), so every
    # use of it folds to a constant expression that overflows and MSVC reports C4756
    # "overflow in constant arithmetic" at the statement that used it. The library's
    # INFINITY uses are the forward pass's "no root in this interval" sentinel, not
    # arithmetic on a huge value; the diagnostic is about the header's own expansion
    # and cannot be avoided from source. Suppressed explicitly so the werror preset's
    # zero-warning rule stays meaningful for everything else.
    add_compile_options(/wd4756)
endif()



# SIMD level — shared with EventProp.Cpp.Wrapper so the C library and the Eigen
# reference core the parity suite compares against move together. See
# cmake/EventPropSimd.cmake for why the decision is shared, why it is a CPUID+XCR0
# probe, and why the flag spelling is per-compiler.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/EventPropSimd.cmake")
eventprop_simd_level(EVENTPROP_SIMD_LEVEL)
eventprop_simd_definitions(EVENTPROP_SIMD_DEFINITIONS LEVEL "${EVENTPROP_SIMD_LEVEL}")
eventprop_simd_compile_options(EVENTPROP_SIMD_COMPILE_OPTIONS LEVEL "${EVENTPROP_SIMD_LEVEL}")
if(EVENTPROP_SIMD_DEFINITIONS)
    add_compile_definitions(${EVENTPROP_SIMD_DEFINITIONS})
endif()
message(STATUS "SIMD level: ${EVENTPROP_SIMD_LEVEL} (definitions: ${EVENTPROP_SIMD_DEFINITIONS}; "
               "arch flag applied only where the project does not already set a "
               "superset — see the target flags below)")

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

# The validator's NaN guard is written with an explicit isnan() so it is
# NaN-correct under /fp:fast, and no per-file /fp:precise override is needed.
# History: it used to be `!(v_th > 0.0)` alone, which is NaN-correct only under
# strict FP semantics; MSVC's /fp:fast rewrote it to `v_th <= 0.0` (false for NaN)
# and accepted a NaN spike threshold. The first fix compiled this one translation
# unit /fp:precise. That worked, but the fix lived in CMake — a reader of this file
# saw a guard whose stated NaN-correctness was false on MSVC unless they knew about
# a flag 40 files away. The isnan() form is correct wherever it is compiled, so the
# whole library keeps /fp:fast. See the comment in eventprop_validation.c.

find_package(OpenMP QUIET)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(eventprop_c_api ${LIB_TYPE}
    eventprop_validation.c
    eventprop_abi.c
    eventprop_simd.c
    eventprop_memory.c
    eventprop_sums.c
    eventprop_maxima.c
    eventprop_spikes.c
    eventprop_backward.c
    eventprop_diag.c
    eventprop_batch.c
)

if(EVENTPROP_ATOMICS_COMPILE_OPTIONS)
    target_compile_options(eventprop_c_api PRIVATE ${EVENTPROP_ATOMICS_COMPILE_OPTIONS})
endif()

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
    # The GCC/Clang branch above already gets the ISA from -march=native, so the
    # probe's arch flag is only needed on MSVC, which has no -march=native. Without
    # it MSVC compiles the x86 baseline: no __AVX2__ (x64 MSVC predefines neither
    # __SSE2__ nor __AVX2__), so eventprop_simd.c selects its scalar fallback and
    # fma_row_simd — the ComputeSums prefix-sum inner loop and the backward pass's
    # lambda_I jump — runs unvectorised. Passing -mavx2 to cl instead is ignored with
    # only "D9002: ignoring unknown option", which is why the flag is mapped here
    # rather than spelled once.
    if(EVENTPROP_SIMD_COMPILE_OPTIONS)
        target_compile_options(eventprop_c_api PRIVATE ${EVENTPROP_SIMD_COMPILE_OPTIONS})
    endif()
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
