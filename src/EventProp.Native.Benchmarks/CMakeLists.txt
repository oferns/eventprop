cmake_minimum_required(VERSION 3.16)
project(eventprop_cpp_vs_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Paths to pre-built libraries (built by steps 1 & 2 of build.sh) ──
# Benchmarks and C/C++ libs are siblings under src/
set(EVENTPROP_C_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build")
set(EVENTPROP_CPP_BUILD "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/build")
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")

# Locate pre-built libraries
find_library(LIB_EVENTPROP_C eventprop_c_api PATHS "${EVENTPROP_C_BUILD}" NO_DEFAULT_PATH)
if(NOT LIB_EVENTPROP_C)
    message(FATAL_ERROR "Pre-built C library not found at ${EVENTPROP_C_BUILD}. Run build.sh first.")
endif()

find_library(LIB_EVENTPROP_EIGEN eventprop_eigen_core PATHS "${EVENTPROP_CPP_BUILD}" NO_DEFAULT_PATH)
if(NOT LIB_EVENTPROP_EIGEN)
    message(FATAL_ERROR "Pre-built Eigen library not found at ${EVENTPROP_CPP_BUILD}. Run build.sh first.")
endif()

# ── Boost (header-only: math) ───────────────────────────────────
find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp PATHS /opt/homebrew/include /usr/include NO_DEFAULT_PATH)
if(NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "Boost not found. Install libboost-dev (Linux) or boost (macOS)")
endif()

# ── OpenMP (for rpath on macOS) ────────────────────────────────
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
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
    benchmark::benchmark
    benchmark::benchmark_main
)

# Add libomp to rpath on macOS (needed for OpenMP parallelization in the C library)
if(APPLE AND _BREW_FAIL EQUAL 0)
    set_target_properties(bench_c PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
endif()

# Copy C library to output directory (so it's found at runtime)
if(APPLE)
    add_custom_command(TARGET bench_c POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${LIB_EVENTPROP_C}
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

# OpenMP headers for the Eigen library (it uses #pragma omp)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_include_directories(bench_cpp PRIVATE ${_OPENMP_PREFIX}/include)
endif()

target_link_libraries(bench_cpp PRIVATE
    ${LIB_EVENTPROP_EIGEN}
    benchmark::benchmark
    benchmark::benchmark_main
)

# Link OpenMP runtime (needed by the pre-built Eigen library)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_link_libraries(bench_cpp PRIVATE ${_OPENMP_PREFIX}/lib/libomp.dylib)
    set_target_properties(bench_cpp PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
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

# OpenMP headers for the Eigen library (it uses #pragma omp)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_include_directories(bench_rootfind PRIVATE ${_OPENMP_PREFIX}/include)
endif()

target_link_libraries(bench_rootfind PRIVATE
    ${LIB_EVENTPROP_EIGEN}
    benchmark::benchmark
    benchmark::benchmark_main
)

# Link OpenMP runtime (needed by the pre-built Eigen library)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_link_libraries(bench_rootfind PRIVATE ${_OPENMP_PREFIX}/lib/libomp.dylib)
    set_target_properties(bench_rootfind PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
endif()
