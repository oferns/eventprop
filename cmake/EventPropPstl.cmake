# EventPropPstl.cmake — what a translation unit must LINK to use the C++17 parallel
# execution policies, on the toolchains where they need something.
#
# eventprop_eigen_core_stripped.cpp calls std::sort(std::execution::par_unseq, ...) and
# std::find_if(std::execution::par_unseq, ...) (lines 259 and 274). libstdc++ 12+
# implements those over Intel oneTBB: the header inlines task-scheduler calls
# (tbb::detail::r1::spawn / allocate / execute_and_wait / ...) and expects a library at
# link time that nothing in this repo used to name. Measured on Linux/GCC 13 with a
# six-line reproduction of exactly those two calls: the same failure the build produced,
# and it goes away with -ltbb.
#
# Two failure shapes come out of that, and the second is the one that matters:
#
#   1. Executables fail to link — eventprop_cpp_tests, parity_test and bench_cpp all
#      died at link. Loud, therefore harmless.
#   2. The SHARED wrapper libeventprop_cpp_wrapper.so LINKS ANYWAY, because a shared
#      library is allowed to keep undefined symbols on Linux. The build step prints
#      "Built target", and the artifact cannot be loaded:
#          OSError: .../libeventprop_cpp_wrapper.so: undefined symbol:
#                    _ZN3tbb6detail2r15spawnERNS0_2d14taskERNS2_18task_group_contextE
#      (confirmed here by dlopen'ing the library the failing build produced). A green
#      step that hands out an unloadable library is the defect class this repo keeps
#      meeting: it surfaces far from its cause, in a consumer.
#
# macOS and Windows are NOT affected, and this module must not change them. libc++ and
# the MSVC STL fall back to serial execution for these policies and require no extra
# link input — not as a claim about stdlib internals but as what this repository's own
# validated builds measure: the same source compiles AND links on macOS and on
# Windows/x64 (parity 105/105) with no TBB anywhere in the tree (git grep -i tbb: zero
# hits outside the read-only submodule). So the first thing this function does is return
# "nothing" on APPLE and WIN32, before probing, so that a platform which already links
# cannot be touched by anything below.
#
# Everything past that gate is decided by compiling, not by an OS or compiler-version
# test, because the question "does <execution> need a library here" has different
# answers on the same Linux box — GCC against libstdc++ (yes: TBB) and clang against
# libc++ (no: serial). An OS gate would be wrong for one of the two; a probe is right
# for both, and it also keeps working if a future libstdc++ drops the dependency.
#
# Scope note: the reference library is NOT involved. submodule-linux.patch strips these
# very execution policies out of EventProp.Python's src/eventprop_cpp.cpp, which is why
# the pybind11 build has never needed TBB. This module covers the convenience copy —
# EventProp.Cpp.Wrapper — which kept them, and the projects that consume its archive.

include_guard(GLOBAL)

# The probe mirrors BOTH constructs in the Eigen core, not one: find_if pulls different
# scheduler symbols than sort, so a sort-only probe could link while the core still did
# not. "The probe links" has to mean "the TU links".
set(EVENTPROP_PSTL_PROBE "
#include <algorithm>
#include <execution>
#include <utility>
#include <vector>
int main() {
    std::vector<std::pair<std::pair<double, double>, int>> spikes;
    spikes.push_back(std::make_pair(std::make_pair(0.5, 0.25), 3));
    std::sort(std::execution::par_unseq, spikes.begin(), spikes.end(),
              [](const auto& a, const auto& b) { return a.first.first < b.first.first; });
    auto it = std::find_if(std::execution::par_unseq, spikes.begin(), spikes.end(),
                           [](const auto& s) { return s.second == 3; });
    return static_cast<int>(it - spikes.begin());
}
")

# <execution> present, no parallel policy: separates "this stdlib has no <execution> at
# all" from "it has it and needs a library", which are opposite pieces of advice.
#
# This probe MUST be compiled without linking (CMAKE_TRY_COMPILE_TARGET_TYPE
# STATIC_LIBRARY), measured here: on libstdc++ 13 a TU that merely includes <execution>
# and calls a SERIAL std::sort fails to link with the same undefined tbb::detail::r1
# symbols as the parallel version. So a linking header probe reports "no <execution>"
# about the exact toolchain that has it and needs TBB — it would print the wrong advice
# on every Linux box this module exists to help.
set(EVENTPROP_PSTL_HEADER_PROBE "
#include <algorithm>
#include <execution>
#include <vector>
int main() { std::vector<int> v(4); std::sort(v.begin(), v.end()); return (int)v.size(); }
")

# Escape hatch for a TBB that is not on the default search path.
set(EVENTPROP_PSTL_LIBRARY "" CACHE STRING
    "Library to link for C++17 parallel execution policies when the toolchain needs one (empty = search for libtbb)")

# eventprop_pstl_link_deps(<out-var>)
#   -> the link items a target needs for <execution> parallel policies; empty when the
#      toolchain needs none. Callers attach the result PUBLIC to the Eigen core (so the
#      shared wrapper resolves its symbols and the in-tree tests inherit it) and add it
#      explicitly wherever the core is consumed as a PRE-BUILT archive, because an
#      IMPORTED target carries no usage requirements.
function(eventprop_pstl_link_deps OUT)
    if(APPLE OR WIN32)
        # Measured by this repo's own builds on both platforms: these policies already
        # link. Return before any probe or find_library so nothing here can alter them.
        set(${OUT} "" PARENT_SCOPE)
        return()
    endif()

    include(CheckCXXSourceCompiles)

    # std::execution only exists at C++17 and above, so the probe has to be compiled at
    # the standard the project actually builds at — a probe run at the compiler default
    # would report "no <execution>" about a project that does have it.
    #
    # CMAKE_FLAGS (the tidy way to set the standard for a check) is NOT accepted by
    # check_cxx_source_compiles in CMake 3.28 and is not in the documented signature at
    # all — measured here: "Unknown argument: CMAKE_FLAGS". Since the floor of every
    # project using this module is 3.21, the flag goes through CMAKE_REQUIRED_FLAGS
    # instead, spelled the GNU/Clang way. That is safe by construction: this line is
    # unreachable on APPLE and WIN32 (the gate above), so the only compilers that can
    # reach it take -std=.
    set(_std "${CMAKE_CXX_STANDARD}")
    if(NOT _std MATCHES "^(17|20|23|26)$")
        set(_std 17)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_REQUIRED_FLAGS "-std=c++${_std}")
    endif()

    # Re-measure on every configure. check_cxx_source_compiles caches its result, and the
    # failure caching produces here is the stale-cache class this repo keeps meeting: run
    # one configure before `apt install libtbb-dev`, cache FALSE, and every configure
    # after the install keeps insisting TBB is missing (or keeps linking a TBB that is no
    # longer needed). Two short compiles per configure buys an answer that is true now.
    unset(EVENTPROP_PSTL_LINKS_PLAIN CACHE)
    unset(EVENTPROP_PSTL_LINKS_TBB CACHE)
    unset(EVENTPROP_PSTL_HAS_HEADER CACHE)
    unset(EVENTPROP_PSTL_TBB_LIBRARY CACHE)

    # 1. Nothing extra. Asked FIRST, so a machine that needs no TBB never goes looking
    #    for one, and so CMAKE_REQUIRED_LIBRARIES is empty for this probe no matter what
    #    the caller had set (function scope, so the caller's value is untouched).
    set(CMAKE_REQUIRED_LIBRARIES "")
    check_cxx_source_compiles("${EVENTPROP_PSTL_PROBE}" EVENTPROP_PSTL_LINKS_PLAIN)
    if(EVENTPROP_PSTL_LINKS_PLAIN)
        message(STATUS "PSTL: <execution> parallel policies link with no extra library")
        set(${OUT} "" PARENT_SCOPE)
        return()
    endif()

    # 2. TBB. find_library rather than find_package(TBB) on purpose: a probe runs in its
    #    own try_compile project that cannot see an IMPORTED target created in this one,
    #    so the answer must be spelled identically in the probe and in the build.
    if(EVENTPROP_PSTL_LIBRARY)
        set(_tbb "${EVENTPROP_PSTL_LIBRARY}")
        message(STATUS "PSTL: using EVENTPROP_PSTL_LIBRARY=${_tbb}")
    else()
        find_library(EVENTPROP_PSTL_TBB_LIBRARY NAMES tbb libtbb)
        set(_tbb "${EVENTPROP_PSTL_TBB_LIBRARY}")
    endif()

    if(_tbb)
        set(CMAKE_REQUIRED_LIBRARIES "${_tbb}")
        check_cxx_source_compiles("${EVENTPROP_PSTL_PROBE}" EVENTPROP_PSTL_LINKS_TBB)
        if(EVENTPROP_PSTL_LINKS_TBB)
            message(STATUS "PSTL: linking ${_tbb} — this stdlib implements <execution> over TBB")
            set(${OUT} "${_tbb}" PARENT_SCOPE)
            return()
        endif()
    endif()

    # 3. Neither worked. Name which of the two problems this is, because the advice is
    #    the opposite in each case, and never let it through as a build that only looks
    #    finished: the executables would fail at link and the shared library would load
    #    as an undefined-symbol error.
    set(CMAKE_REQUIRED_LIBRARIES "")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    check_cxx_source_compiles("${EVENTPROP_PSTL_HEADER_PROBE}" EVENTPROP_PSTL_HAS_HEADER)
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    if(EVENTPROP_PSTL_HAS_HEADER)
        message(FATAL_ERROR
            "The C++17 parallel execution policies need a library on this toolchain and "
            "none resolved it.\n"
            "  Needed by: EventProp.Cpp.Wrapper/eventprop_eigen_core_stripped.cpp "
            "(std::sort / std::find_if with std::execution::par_unseq)\n"
            "  Why:       libstdc++ 12+ implements <execution> over Intel oneTBB\n"
            "  Install:   Debian/Ubuntu  sudo apt install libtbb-dev\n"
            "             Fedora         sudo dnf install tbb-devel\n"
            "             Arch           sudo pacman -S onetbb\n"
            "             macOS          brew install tbb (not needed there: libc++ is serial)\n"
            "  Or point at one:  -DEVENTPROP_PSTL_LIBRARY=/path/to/libtbb.so\n"
            "  Tried:       ${_tbb} (linking it did not resolve the policies)")
    endif()
    message(FATAL_ERROR
        "C++17 <execution> is not available from this standard library, so "
        "eventprop_eigen_core_stripped.cpp cannot compile here at all (this is not a "
        "missing-TBB problem). It needs libstdc++ >= 9 or libc++, at -std=c++17 or later. "
        "Configured standard: c++${_std}; compiler: ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION}; library: ${CMAKE_CXX_STANDARD_LIBRARIES}")
endfunction()
