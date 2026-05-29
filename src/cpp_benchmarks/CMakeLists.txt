cmake_minimum_required(VERSION 3.16)
project(eventprop_cpp_vs_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
    if(_BREW_FAIL EQUAL 0 AND EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
        set_target_properties(bench_c PROPERTIES
            INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
            BUILD_WITH_INSTALL_RPATH FALSE
        )
    endif()
endif()

# Copy C library to output
if(APPLE)
    add_custom_command(TARGET bench_c POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:eventprop_c_api>
            $<TARGET_FILE_DIR:bench_c>
    )
endif()
