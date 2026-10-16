# EventPropSimd.cmake — one SIMD level decision, shared by the projects that must
# agree on it.
#
# Why this is a shared module: the C library and the Eigen reference core are
# compared against each other by EventProp.Native.ParityTests at
# ABS_TOLERANCE 1e-10 / REL_TOLERANCE 1e-9 on sums, spike times and gradients.
# Scalar-vs-FMA arithmetic differs (single-rounding fmadd vs two-rounding a*b+c),
# so if one side is built vectorised and the other at the x86 baseline, the parity
# suite measures a code-generation difference instead of the algorithm. On Linux
# both sides are vectorised (-mavx2 for C, -march=native for Eigen) and on macOS
# both are NEON; on Windows MSVC neither side was, which was also symmetric. The
# asymmetry only appears if the arch flag is added to one project, so the decision
# lives here and both projects call it.
#
# Why a CPUID probe replaces the old /proc/cpuinfo + sysctl reading:
#   * Neither file/command exists on Windows, so the old probe silently reported
#     "no SIMD" there and the library ran its scalar fallback on an AVX2 CPU.
#   * The probe compiles and RUNS on the machine doing the build, so it answers the
#     question the old probe was asking ("can this machine run AVX2 code?") using
#     the same mechanism on every OS.
#   * It checks XCR0, not just the CPUID feature bit. The CPUID bit says the
#     hardware has AVX2; the OS has to preserve YMM state across context switches
#     for user code to use it. Linux's /proc/cpuinfo "avx2" flag already implies OS
#     support (the kernel checks XCR0 before setting the flag), so a CPUID-only
#     probe would have been WEAKER than the path it replaces.
#
# Cross-compiling: the probe cannot run, so the level is "scalar" and a message
# says so. The old probe read the BUILD host's feature strings, which describes the
# wrong machine; scalar is the honest answer.
#
# The result is a build-host decision with no runtime dispatch — the same contract
# the project already has on Linux (-mavx2) and macOS (-march=native): the binary
# requires the ISA it was built for.

include_guard(GLOBAL)

set(EVENTPROP_SIMD "auto" CACHE STRING
    "x86 SIMD level for the EventProp targets: auto | avx512 | avx2 | scalar")
set_property(CACHE EVENTPROP_SIMD PROPERTY STRINGS auto avx512 avx2 scalar)

# Valid as C and as C++: the C library probes with its C compiler and
# EventProp.Cpp.Wrapper is a LANGUAGES CXX-only project.
set(EVENTPROP_SIMD_PROBE_COMMON "
#if defined(_MSC_VER)
#include <intrin.h>
static void ep_cpuid(unsigned leaf, unsigned sub, unsigned* a, unsigned* b,
                     unsigned* c, unsigned* d) {
    int r[4];
    __cpuidex(r, (int)leaf, (int)sub);
    *a = (unsigned)r[0]; *b = (unsigned)r[1]; *c = (unsigned)r[2]; *d = (unsigned)r[3];
}
static unsigned long long ep_xcr0(void) { return _xgetbv(0); }
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static void ep_cpuid(unsigned leaf, unsigned sub, unsigned* a, unsigned* b,
                     unsigned* c, unsigned* d) {
    __get_cpuid_count(leaf, sub, a, b, c, d);
}
static unsigned long long ep_xcr0(void) {
    unsigned lo, hi;
    __asm__ __volatile__(\"xgetbv\" : \"=a\"(lo), \"=d\"(hi) : \"c\"(0));
    return ((unsigned long long)hi << 32) | (unsigned long long)lo;
}
#else
int main(void) { return 1; }
#endif
")

# CPUID.1:ECX bit 27 = OSXSAVE (XGETBV is privileged without it), XCR0 bits 1|2 =
# XMM+YMM state enabled by the OS, CPUID.7.0:EBX bit 5 = AVX2.
set(EVENTPROP_SIMD_PROBE_AVX2 "${EVENTPROP_SIMD_PROBE_COMMON}
int main(void) {
    unsigned a, b, c, d;
    ep_cpuid(1u, 0u, &a, &b, &c, &d);
    if (((c >> 27) & 1u) == 0u) return 1;      /* no OSXSAVE -> cannot read XCR0 */
    if ((ep_xcr0() & 0x6ull) != 0x6ull) return 1;  /* OS does not keep XMM+YMM */
    ep_cpuid(7u, 0u, &a, &b, &c, &d);
    return ((b >> 5) & 1u) ? 0 : 1;            /* AVX2 */
}
")

# XCR0 bits 1|2|5|6|7 = opmask + ZMM state, CPUID.7.0:EBX bit 16 = AVX512F.
set(EVENTPROP_SIMD_PROBE_AVX512F "${EVENTPROP_SIMD_PROBE_COMMON}
int main(void) {
    unsigned a, b, c, d;
    ep_cpuid(1u, 0u, &a, &b, &c, &d);
    if (((c >> 27) & 1u) == 0u) return 1;
    if ((ep_xcr0() & 0xE6ull) != 0xE6ull) return 1;  /* OS does not keep ZMM state */
    ep_cpuid(7u, 0u, &a, &b, &c, &d);
    return ((b >> 16) & 1u) ? 0 : 1;           /* AVX512F */
}
")

# Probe with whichever language this project enables.
macro(_eventprop_simd_run_probe _probe _var)
    if(CMAKE_C_COMPILER)
        include(CheckCSourceRuns)
        check_c_source_runs("${_probe}" ${_var})
    else()
        include(CheckCXXSourceRuns)
        check_cxx_source_runs("${_probe}" ${_var})
    endif()
endmacro()

# eventprop_simd_level(<out-var>)
#   -> avx512 | avx2 | neon | scalar
function(eventprop_simd_level OUT)
    set(_level "")

    if(NOT EVENTPROP_SIMD STREQUAL "auto")
        set(_level "${EVENTPROP_SIMD}")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(_level "neon")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i686")
        if(CMAKE_CROSSCOMPILING)
            # The probe runs on the build machine; its answer says nothing about
            # the target. Stay at the baseline rather than describe the wrong CPU.
            message(STATUS "SIMD: cross-compiling, CPUID probe cannot run — building scalar")
            set(_level "scalar")
        else()
            _eventprop_simd_run_probe("${EVENTPROP_SIMD_PROBE_AVX512F}" EVENTPROP_CPU_AVX512F)
            _eventprop_simd_run_probe("${EVENTPROP_SIMD_PROBE_AVX2}" EVENTPROP_CPU_AVX2)
            if(EVENTPROP_CPU_AVX512F)
                set(_level "avx512")
            elseif(EVENTPROP_CPU_AVX2)
                set(_level "avx2")
            else()
                set(_level "scalar")
            endif()
        endif()
    else()
        set(_level "scalar")
    endif()

    # A forced level is a request, not a fact: say what the machine actually has,
    # because a forced avx2 on a machine without it produces a binary that dies
    # with SIGILL/EXCEPTION_ILLEGAL_INSTRUCTION at the first fmadd.
    if(NOT EVENTPROP_SIMD STREQUAL "auto" AND NOT CMAKE_CROSSCOMPILING
       AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        _eventprop_simd_run_probe("${EVENTPROP_SIMD_PROBE_AVX2}" EVENTPROP_CPU_AVX2)
        if(NOT EVENTPROP_CPU_AVX2 AND NOT EVENTPROP_SIMD STREQUAL "scalar")
            message(WARNING
                "EVENTPROP_SIMD=${EVENTPROP_SIMD} but this CPU cannot run AVX2 "
                "(CPUID/XCR0 probe failed). The build will not run on this machine.")
        endif()
    endif()

    set(${OUT} "${_level}" PARENT_SCOPE)
endfunction()

# eventprop_simd_compile_options(<out-var> LEVEL <level>)
#   Compiler-correct spelling of the arch flag. The GCC/Clang spellings are
#   IGNORED by MSVC with only a "cl : Command line warning D9002 : ignoring
#   unknown option '-mavx2'", which is how a detection-only fix ends up defining
#   HAVE_AVX2 while the intrinsics compile against the SSE2 baseline.
function(eventprop_simd_compile_options OUT)
    cmake_parse_arguments(ARG "" "LEVEL" "" ${ARGN})
    set(_opts "")
    if(ARG_LEVEL STREQUAL "avx512")
        if(MSVC)
            set(_opts "/arch:AVX512")
        else()
            set(_opts "-mavx512f")
        endif()
    elseif(ARG_LEVEL STREQUAL "avx2")
        if(MSVC)
            set(_opts "/arch:AVX2")
        else()
            set(_opts "-mavx2")
        endif()
    endif()
    # neon: AArch64 baseline, no flag. scalar: baseline, no flag.
    set(${OUT} "${_opts}" PARENT_SCOPE)
endfunction()

# eventprop_simd_eigen_interface(<target>)
#   Attach the arch flag as a PUBLIC property of the target that compiles the Eigen
#   core — PUBLIC so it lands on the core's own TU *and* propagates to every consumer.
#   This is an ABI property, not a performance knob, and it is the reason the flag
#   cannot be PRIVATE:
#
#   Eigen's EIGEN_MAX_ALIGN_BYTES is 32 when AVX/AVX2 is enabled and 16 otherwise,
#   and aligned allocation is header-inlined (internal::handmade_aligned_malloc /
#   _free). The Eigen core is a STATIC library, and every consumer TU includes the
#   Eigen headers and destroys the matrices the core returns. If the core is
#   compiled at AVX2 and a consumer TU at the x86 baseline, the core allocates with
#   the 32-byte convention and the consumer frees with the 16-byte convention, which
#   corrupts heap metadata. Measured on Windows/x64 with /arch:AVX2 PRIVATE on the
#   core: STATUS_HEAP_CORRUPTION (0xC0000374) in RtlFreeHeap, first test, stack
#   showing the DenseStorage destructor inlined into the CALLING TU:
#
#       ntdll!RtlpHeapHandleError -> ucrtbase!free_base
#         Eigen::internal::handmade_aligned_free      (Memory.h:118)
#         Eigen::DenseStorage<double,-1,-1,-1,1>::~   (DenseStorage.h:452)
#         compute_sums   (eventprop_eigen_core_stripped.cpp:30, inlined at tests.cpp:51)
#
#   The same mismatch exists on Linux for -march=native PRIVATE: the core gets
#   EIGEN_MAX_ALIGN_BYTES 32, the consumer TU 16. It is silent there because glibc
#   does not always notice. So: PUBLIC, and the prebuilt-import consumers get it too.
function(eventprop_simd_eigen_interface TARGET)
    eventprop_simd_level(_level)
    eventprop_simd_compile_options(_opts LEVEL "${_level}")
    if(_opts)
        # PUBLIC, not INTERFACE: INTERFACE alone leaves the target's OWN translation
        # unit at the baseline, which is the same mismatch in the other direction
        # (measured: the core compiled without /arch:AVX2 while its consumers were
        # compiled with it, and the heap corruption reproduces unchanged).
        target_compile_options(${TARGET} PUBLIC ${_opts})
    endif()
endfunction()

# eventprop_simd_definitions(<out-var> LEVEL <level>)
#   eventprop_simd.c keys its dispatch on HAVE_AVX512F / HAVE_AVX2, falling back to
#   the compiler's own __AVX512F__ / __AVX2__ macros. The fallback is dead on MSVC:
#   x64 MSVC predefines neither __SSE2__ nor __AVX2__ (measured), so the explicit
#   define is the only signal that works there.
function(eventprop_simd_definitions OUT)
    cmake_parse_arguments(ARG "" "LEVEL" "" ${ARGN})
    set(_defs "")
    if(ARG_LEVEL STREQUAL "avx512")
        set(_defs "HAVE_AVX512F")
    elseif(ARG_LEVEL STREQUAL "avx2")
        set(_defs "HAVE_AVX2")
    elseif(ARG_LEVEL STREQUAL "neon")
        set(_defs "__ARM_NEON")
    endif()
    set(${OUT} "${_defs}" PARENT_SCOPE)
endfunction()
