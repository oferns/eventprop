cmake_minimum_required(VERSION 3.21)
project(eventprop_cpp_vs_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Paths to pre-built libraries (built by steps 1 & 2 of build.sh) ──
# Benchmarks and C/C++ libs are siblings under src/
set(EVENTPROP_C_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build")
set(EVENTPROP_CPP_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/build")
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")

# Locate pre-built libraries
# On Windows MSVC, outputs go into build/Release or build/Debug subdirectories
find_library(LIB_EVENTPROP_C eventprop_c_api PATHS "${EVENTPROP_C_BUILD}" "${EVENTPROP_C_BUILD}/Release" "${EVENTPROP_C_BUILD}/Debug" NO_DEFAULT_PATH)
if(NOT LIB_EVENTPROP_C)
    message(FATAL_ERROR "Pre-built C library not found at ${EVENTPROP_C_BUILD}. Run build.sh first.")
endif()

find_library(LIB_EVENTPROP_EIGEN eventprop_eigen_core PATHS "${EVENTPROP_CPP_BUILD}" "${EVENTPROP_CPP_BUILD}/Release" "${EVENTPROP_CPP_BUILD}/Debug" NO_DEFAULT_PATH)
if(NOT LIB_EVENTPROP_EIGEN)
    message(FATAL_ERROR "Pre-built Eigen library not found at ${EVENTPROP_CPP_BUILD}. Run build.sh first.")
endif()

# ── Boost (header-only: math) ───────────────────────────────────
find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp PATHS /opt/homebrew/include /usr/include "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/vcpkg/installed/x64-windows/include" NO_DEFAULT_PATH)
if(NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "Boost not found. Install libboost-dev (Linux) or boost (macOS)")
endif()

# ── OpenMP (needed by both C and Eigen libraries) ─────────────
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
else()
    find_package(OpenMP)
endif()

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# MSVC: /EHsc required for Google Benchmark (C4530 warning treated as error)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()
FetchContent_MakeAvailable(benchmark)

# ============================================================
# C library benchmark executable (bench_c)
# Links against pre-built libeventprop_c_api.dylib from EventProp.C
# ============================================================

add_executable(bench_c bench_main.cpp)
target_include_directories(bench_c PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C
)
target_link_libraries(bench_c PRIVATE
    ${LIB_EVENTPROP_C}
    benchmark::benchmark_main
)

# CPU-specific optimization and aggressive FP math optimizations for bench_c
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_c PRIVATE -march=native)
    target_compile_options(bench_c PRIVATE -ffp-contract=fast)
    target_compile_options(bench_c PRIVATE
        -freciprocal-math
        -fno-signed-zeros
        -fassociative-math
        -fno-trapping-math
        -fno-math-errno
        -funroll-loops
    )
elseif(CMAKE_C_COMPILER_ID MATCHES "MSVC")
    target_compile_options(bench_c PRIVATE /O2 /Ob2 /Oi /Ot /fp:fast)
endif()

# Link OpenMP runtime (needed by the C library)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_link_libraries(bench_c PRIVATE ${_OPENMP_PREFIX}/lib/libomp.dylib)
    set_target_properties(bench_c PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(bench_c PRIVATE OpenMP::OpenMP_CXX)
endif()

# Copy C library to output directory (so it's found at runtime)
if(APPLE)
    add_custom_command(TARGET bench_c POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${LIB_EVENTPROP_C}
            $<TARGET_FILE_DIR:bench_c>
    )
endif()
if(WIN32)
    # MSVC OpenMP is built into the compiler, no separate library needed
    target_compile_options(bench_c PRIVATE /openmp)
    # Copy the actual DLL (not just the .lib import library) for runtime
    add_custom_command(TARGET bench_c POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${EVENTPROP_C_BUILD}/Release/eventprop_c_api.dll
            $<TARGET_FILE_DIR:bench_c>
    )
endif()

# ============================================================
# C++ (Eigen) benchmark executable (bench_cpp)
# Links against pre-built libeventprop_eigen_core.a from EventProp.Cpp.Wrapper
# ============================================================

add_executable(bench_cpp bench_main.cpp)
target_compile_definitions(bench_cpp PRIVATE USE_CPP_WRAPPER)
target_include_directories(bench_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR})

# CPU-specific optimization and aggressive FP math optimizations for bench_cpp
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_cpp PRIVATE -march=native)
    target_compile_options(bench_cpp PRIVATE -ffp-contract=fast)
    target_compile_options(bench_cpp PRIVATE
        -freciprocal-math
        -fno-signed-zeros
        -fassociative-math
        -fno-trapping-math
        -fno-math-errno
        -funroll-loops
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(bench_cpp PRIVATE /O2 /Ob2 /Oi /Ot /fp:fast)
endif()

# OpenMP headers for the Eigen library (it uses #pragma omp)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_include_directories(bench_cpp PRIVATE ${_OPENMP_PREFIX}/include)
    # AppleClang requires -Xpreprocessor -fopenmp to enable OpenMP pragmas
    target_compile_options(bench_cpp PRIVATE "-Xpreprocessor" "-fopenmp")
endif()

target_link_libraries(bench_cpp PRIVATE
    ${LIB_EVENTPROP_EIGEN}
    benchmark::benchmark_main
)

# Link OpenMP runtime (needed by the pre-built Eigen library)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_link_libraries(bench_cpp PRIVATE ${_OPENMP_PREFIX}/lib/libomp.dylib)
    set_target_properties(bench_cpp PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(bench_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# ============================================================
# Root-finder micro-benchmark (bench_rootfind)
# Links against pre-built libeventprop_eigen_core.a from EventProp.Cpp.Wrapper
# ============================================================

add_executable(bench_rootfind bench_rootfind.cpp)
target_include_directories(bench_rootfind PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR})

# CPU-specific optimization and aggressive FP math optimizations for bench_rootfind
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bench_rootfind PRIVATE -march=native)
    target_compile_options(bench_rootfind PRIVATE -ffp-contract=fast)
    target_compile_options(bench_rootfind PRIVATE
        -freciprocal-math
        -fno-signed-zeros
        -fassociative-math
        -fno-trapping-math
        -fno-math-errno
        -funroll-loops
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(bench_rootfind PRIVATE /O2 /Ob2 /Oi /Ot /fp:fast)
endif()

# OpenMP headers for the Eigen library (it uses #pragma omp)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_include_directories(bench_rootfind PRIVATE ${_OPENMP_PREFIX}/include)
    # AppleClang requires -Xpreprocessor -fopenmp to enable OpenMP pragmas
    target_compile_options(bench_rootfind PRIVATE "-Xpreprocessor" "-fopenmp")
endif()

target_link_libraries(bench_rootfind PRIVATE
    ${LIB_EVENTPROP_EIGEN}
    benchmark::benchmark_main
)

# Link OpenMP runtime (needed by the pre-built Eigen library)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_link_libraries(bench_rootfind PRIVATE ${_OPENMP_PREFIX}/lib/libomp.dylib)
    set_target_properties(bench_rootfind PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(bench_rootfind PRIVATE OpenMP::OpenMP_CXX)
endif()
