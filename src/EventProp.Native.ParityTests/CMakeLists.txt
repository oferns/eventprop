cmake_minimum_required(VERSION 3.21)
project(EventPropParityTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Eigen (from EventProp.Python submodule, header-only) ─────────
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")
if(NOT EXISTS "${EIGEN_DIR}")
    message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: git submodule update --init src/EventProp.Python/src/eigen")
endif()

# ── Boost (header-only: math) ───────────────────────────────────
if(WIN32)
    set(BOOST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/vcpkg/installed/x64-windows/include")
else()
    find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp PATHS /opt/homebrew/include /usr/include NO_DEFAULT_PATH)
endif()
if(NOT BOOST_INCLUDE_DIR OR NOT EXISTS "${BOOST_INCLUDE_DIR}/boost/math/tools/roots.hpp")
    message(FATAL_ERROR "Boost not found at ${BOOST_INCLUDE_DIR}. Install libboost-dev (Linux) or boost (macOS), or use vcpkg on Windows")
endif()

# ── Pre-built C library (imported) ─────────────────────────────
if(WIN32)
    set(C_LIBRARY_EXT ".dll")
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api${C_LIBRARY_EXT}")
elseif(APPLE)
    set(C_LIBRARY_EXT ".dylib")
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api${C_LIBRARY_EXT}")
else()
    set(C_LIBRARY_EXT ".so")
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api${C_LIBRARY_EXT}")
endif()
if(NOT EXISTS "${C_LIBRARY}")
    message(FATAL_ERROR "C library not found at ${C_LIBRARY}. Build EventProp.C first.")
endif()

add_library(eventprop_c_api SHARED IMPORTED)
if(WIN32)
    set_target_properties(eventprop_c_api PROPERTIES
        IMPORTED_LOCATION "${C_LIBRARY}"
        IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api.lib"
    )
else()
    set_target_properties(eventprop_c_api PROPERTIES
        IMPORTED_LOCATION "${C_LIBRARY}"
    )
endif()

# ── Pre-built Eigen core (imported from EventProp.Cpp.Wrapper) ─
if(WIN32)
    set(EIGEN_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/build/Release/eventprop_eigen_core.lib")
else()
    set(EIGEN_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/build/libeventprop_eigen_core.a")
endif()
if(NOT EXISTS "${EIGEN_LIBRARY}")
    message(FATAL_ERROR "Eigen core library not found at ${EIGEN_LIBRARY}. Build EventProp.Cpp.Wrapper first.")
endif()

add_library(eventprop_eigen_core STATIC IMPORTED)
set_target_properties(eventprop_eigen_core PROPERTIES
    IMPORTED_LOCATION "${EIGEN_LIBRARY}"
)

# ── Parity test executable ─────────────────────────────────────
find_package(OpenMP QUIET)
add_executable(parity_test parity_main.cpp)

# ISA level must match the pre-built Eigen core, because the imported target carries
# no usage requirements. Eigen's EIGEN_MAX_ALIGN_BYTES is 32 with AVX/AVX2 and 16
# without, and the aligned allocator is header-inlined; parity_main.cpp includes the
# Eigen headers and destroys matrices the core (a STATIC library) allocated. Built at
# different levels, the two sides use different aligned-free conventions on the same
# block: STATUS_HEAP_CORRUPTION (0xC0000374) in RtlFreeHeap. Measured on Windows/x64
# when /arch:AVX2 was applied to the core but not to the consumer TU. On Linux the
# core is built -march=native and parity_test was not, which is the same mismatch.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/EventPropSimd.cmake")
eventprop_simd_level(EVENTPROP_SIMD_LEVEL)
eventprop_simd_compile_options(EVENTPROP_SIMD_COMPILE_OPTIONS LEVEL "${EVENTPROP_SIMD_LEVEL}")
if(EVENTPROP_SIMD_COMPILE_OPTIONS)
    target_compile_options(parity_test PRIVATE ${EVENTPROP_SIMD_COMPILE_OPTIONS})
    message(STATUS "parity_test: ${EVENTPROP_SIMD_LEVEL} (${EVENTPROP_SIMD_COMPILE_OPTIONS}) — "
                   "matched to the pre-built Eigen core")
endif()

target_include_directories(parity_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C"
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper"
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

# libm is separate on Unix, built-in on Windows
if(WIN32)
    target_link_libraries(parity_test PRIVATE eventprop_c_api eventprop_eigen_core)
else()
    target_link_libraries(parity_test PRIVATE eventprop_c_api eventprop_eigen_core m)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(parity_test PRIVATE OpenMP::OpenMP_CXX)
else()
    # macOS fallback: Homebrew libomp
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
    if(_BREW_FAIL EQUAL 0 AND EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
        target_compile_options(parity_test PRIVATE "-Xpreprocessor" "-fopenmp")
        target_include_directories(parity_test PRIVATE "${_OPENMP_PREFIX}/include")
        target_link_libraries(parity_test PRIVATE "${_OPENMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

# Runtime library search path
if(WIN32)
    # Windows needs DLL in PATH or same directory as executable
    set_target_properties(parity_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release"
    )
elseif(APPLE)
    set_target_properties(parity_test PROPERTIES
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
