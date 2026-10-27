cmake_minimum_required(VERSION 3.21)
project(eventprop_c_api LANGUAGES CXX)

# The configure presets set CMAKE_BUILD_TYPE, and on a multi-config generator (MSVC,
# Xcode) nothing consumes it — so CMake prints "Manually-specified variables were not
# used by the project: CMAKE_BUILD_TYPE" on every Windows configure. The presets cannot
# simply stop setting it: single-config generators (Unix Makefiles, Ninja — Linux and
# macOS) take their optimisation level from it, so dropping it there would silently
# build -O0 and poison every benchmark number. Reading it here is what tells CMake the
# variable is used; the default is the honest counterpart of that reading, because a
# single-config generator with no build type is an unoptimised build nobody chose.
# Same block in every preset-driven project (EventPropSimd.cmake answers the ISA
# question once for the same reason).
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (single-config generators only)" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Eigen (from EventProp.Python submodule) ─────────────────────
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")
if(NOT EXISTS "${EIGEN_DIR}")
    message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: git submodule update --init src/EventProp.Python/src/eigen")
endif()

# ── Boost (header-only: math) ───────────────────────────────────
if(WIN32)
    # Use vcpkg-installed Boost on Windows (4 levels up from src/EventProp.Cpp.Wrapper to repo root)
    set(BOOST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/vcpkg/installed/x64-windows/include")
else()
    find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp
              PATHS /opt/homebrew/include /usr/include NO_DEFAULT_PATH)
endif()
if(NOT BOOST_INCLUDE_DIR OR NOT EXISTS "${BOOST_INCLUDE_DIR}/boost/math/tools/roots.hpp")
    message(FATAL_ERROR "Boost not found at ${BOOST_INCLUDE_DIR}. Install libboost-dev (Linux) or boost (macOS Homebrew), or use vcpkg on Windows")
endif()

# ── OpenMP (optional, for parallelization) ─────────────────────
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
    if(_BREW_FAIL EQUAL 0 AND EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_FOUND TRUE)
        set(OpenMP_CXX_LIBRARIES "${_OPENMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_INCLUDE_DIRS "${_OPENMP_PREFIX}/include")
    endif()
else()
    find_package(OpenMP)
endif()

# ── Eigen core (stripped copy without pybind11) — build as static lib
add_library(eventprop_eigen_core STATIC eventprop_eigen_core_stripped.cpp)
set_target_properties(eventprop_eigen_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# CPU-specific optimization and aggressive FP math optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang flags
    # PUBLIC, not PRIVATE: -march=native sets EIGEN_MAX_ALIGN_BYTES for every TU that
    # includes the Eigen headers, and aligned allocation is header-inlined, so the
    # core and its consumers must agree. See eventprop_simd_eigen_interface in
    # cmake/EventPropSimd.cmake for the measured failure when it is PRIVATE.
    target_compile_options(eventprop_eigen_core PUBLIC -march=native)
    target_compile_options(eventprop_eigen_core PRIVATE -ffp-contract=fast)
    # Aggressive FP math optimizations — matches EventProp.Python's C++ flags.
    # Subset of -ffast-math but without -ffinite-math-only (avoids NaN/infinity UB).
    target_compile_options(eventprop_eigen_core PRIVATE
        -freciprocal-math   # x / y → x * (1/y) — many exp-decay divisions benefit
        -fno-signed-zeros   # treat ±0 as identical, enables more FP simplifications
        -fassociative-math  # allow reassociation of FP operations (e.g., (a+b)+c → a+(b+c))
        -fno-trapping-math  # assume FP operations will not generate traps
        -fno-math-errno     # do not set errno on FP errors
        -funroll-loops      # unroll small per-neuron/per-spike loops
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # MSVC equivalent optimizations:
    # /O2 = Optimize for speed
    # /Ob2 = Inline functions
    # /Oi = Enable intrinsic functions
    # /Ot = Favor fast code
    # /fp:fast = Fast floating-point math (equivalent to -ffp-contract=fast, -freciprocal-math, -fno-signed-zeros)
    target_compile_options(eventprop_eigen_core PRIVATE /O2 /Ob2 /Oi /Ot /fp:fast)
    # Parity symmetry + ABI. The GCC/Clang branch above gets the ISA from
    # -march=native, so on Linux this core and the C library are both vectorised; on
    # macOS both are NEON. On MSVC neither had an arch flag, which was also symmetric.
    # Adding /arch:AVX2 to EventProp.C alone would make Windows the one platform where
    # the parity suite compares vectorised C against SSE2 Eigen, at ABS_TOLERANCE
    # 1e-10 / REL_TOLERANCE 1e-9 on sums, spike times and gradients — scalar-vs-FMA is
    # a single-rounding difference that accumulates through the prefix-sum recurrence,
    # and the root finder amplifies a voltage delta into a time delta by 1/V'.
    # Same module, same probe, so the two projects cannot drift apart.
    #
    # Applied as a PUBLIC property (see eventprop_simd_eigen_interface): the flag is an
    # ABI property of this header-heavy static library, so it must reach the core's own
    # TU and every consumer. PRIVATE here, and INTERFACE here, both reproduced as heap
    # corruption in a consumer that destroys a matrix the core returned.
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/EventPropSimd.cmake")
    eventprop_simd_eigen_interface(eventprop_eigen_core)
    eventprop_simd_level(EVENTPROP_SIMD_LEVEL)
    eventprop_simd_compile_options(EVENTPROP_SIMD_COMPILE_OPTIONS LEVEL "${EVENTPROP_SIMD_LEVEL}")
    if(EVENTPROP_SIMD_COMPILE_OPTIONS)
        message(STATUS "Eigen core: ${EVENTPROP_SIMD_LEVEL} (${EVENTPROP_SIMD_COMPILE_OPTIONS}) as "
                       "PUBLIC — propagated to every consumer that includes Eigen "
                       "headers; matched to the C library for parity")
    else()
        message(STATUS "Eigen core: x86 baseline (${EVENTPROP_SIMD_LEVEL}) — matched to the C library")
    endif()
endif()
target_include_directories(eventprop_eigen_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

# OpenMP for parallelization (no CPU-specific flags — let compiler auto-detect)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_include_directories(eventprop_eigen_core PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    # AppleClang requires -Xpreprocessor -fopenmp to enable OpenMP pragmas
    target_compile_options(eventprop_eigen_core PRIVATE "-Xpreprocessor" "-fopenmp")
    # PUBLIC so consumers (tests, wrapper DLL) get the OpenMP runtime linked
    target_link_libraries(eventprop_eigen_core PUBLIC ${OpenMP_CXX_LIBRARIES})
elseif(OpenMP_CXX_FOUND)
    # PUBLIC so consumers (tests, wrapper DLL) get the OpenMP runtime linked
    target_link_libraries(eventprop_eigen_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ── C++17 parallel execution policies (Linux/libstdc++: TBB) ───
# PUBLIC for the same reason OpenMP and -march=native are: this is a STATIC library, so
# "what it needs to link" is a requirement its consumers inherit. eventprop_c_api links
# the core into a SHARED library, and a shared library on Linux happily keeps undefined
# symbols — measured before this block existed, libeventprop_cpp_wrapper.so linked green
# with 14 undefined tbb::detail::r1 symbols and then refused to load:
#   OSError: undefined symbol: _ZN3tbb6detail2r15spawnERNS0_2d14taskERNS2_18task_group_contextE
# The two test executables below failed at link instead, which was the loud version of
# the same missing input. Zero on macOS/Windows: libc++ and the MSVC STL run these
# policies serially and need nothing extra, and eventprop_pstl_link_deps() returns empty
# on APPLE/WIN32 before probing, so this cannot change either platform.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/EventPropPstl.cmake")
eventprop_pstl_link_deps(EVENTPROP_PSTL_LIBS)
if(EVENTPROP_PSTL_LIBS)
    target_link_libraries(eventprop_eigen_core PUBLIC ${EVENTPROP_PSTL_LIBS})
endif()

# ── C API wrapper as a shared library ───────────────────────────
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(eventprop_c_api SHARED eventprop_c_api.cpp)
set_target_properties(eventprop_c_api PROPERTIES OUTPUT_NAME "eventprop_cpp_wrapper")
target_include_directories(eventprop_c_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

target_link_libraries(eventprop_c_api PRIVATE
    eventprop_eigen_core
)

if(APPLE AND _BREW_FAIL EQUAL 0)
    set_target_properties(eventprop_c_api PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(OpenMP_CXX_FOUND)
    set_target_properties(eventprop_c_api PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# ── Unit tests (targets Eigen directly, not via C API) ─────────
add_executable(eventprop_cpp_tests tests.cpp)

# ── C API wrapper unit tests (mirrors src/EventProp.C/tests.c) ──
add_executable(eventprop_capi_tests tests_capi.cpp eventprop_c_api.cpp)
target_include_directories(eventprop_capi_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)
target_link_libraries(eventprop_capi_tests PRIVATE eventprop_eigen_core)

target_include_directories(eventprop_cpp_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)
target_link_libraries(eventprop_cpp_tests PRIVATE eventprop_eigen_core)

# Install targets
install(TARGETS eventprop_c_api LIBRARY DESTINATION lib)
install(FILES eventprop_c_api.h DESTINATION include)
