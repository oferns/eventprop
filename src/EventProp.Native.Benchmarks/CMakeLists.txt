cmake_minimum_required(VERSION 3.16)
project(eventprop_cpp_vs_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Eigen (from EventProp.Python submodule) ─────────────────────
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")

# ── Boost (header-only: math) ───────────────────────────────────
find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp PATHS /opt/homebrew/include NO_DEFAULT_PATH)
if(NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "Boost not found at /opt/homebrew/include")
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

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ============================================================
# C library benchmark executable (bench_c)
# ============================================================
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C eventprop_c_lib)

add_executable(bench_c bench_main.cpp)
target_include_directories(bench_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C)
target_link_libraries(bench_c PRIVATE eventprop_c_api benchmark::benchmark benchmark::benchmark_main)

# Add libomp to rpath on macOS (needed for OpenMP parallelization)
if(APPLE AND _BREW_FAIL EQUAL 0)
    set_target_properties(bench_c PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
endif()

# Copy C library to output
if(APPLE)
    add_custom_command(TARGET bench_c POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:eventprop_c_api>
            $<TARGET_FILE_DIR:bench_c>
    )
endif()

# ============================================================
# C++ (Eigen) benchmark executable (bench_cpp)
# Calls Eigen functions directly from EventProp.Python
# ============================================================

# Build Eigen core as static lib (stripped copy without pybind11)
add_library(eventprop_eigen_core STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/eventprop_eigen_core_stripped.cpp)
target_include_directories(eventprop_eigen_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

# OpenMP for parallelization
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_include_directories(eventprop_eigen_core PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(eventprop_eigen_core PRIVATE ${OpenMP_CXX_LIBRARIES})
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(eventprop_eigen_core PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(bench_cpp bench_main.cpp)
target_compile_definitions(bench_cpp PRIVATE USE_CPP_WRAPPER)
target_include_directories(bench_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper
    ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR})
target_link_libraries(bench_cpp PRIVATE eventprop_eigen_core benchmark::benchmark benchmark::benchmark_main)
