cmake_minimum_required(VERSION 3.16)
project(eventprop_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
# Disable tests/examples from benchmark itself
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Build the C API library if not already built
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. eventprop_lib)

# Benchmark executable
add_executable(eventprop_benchmarks eventprop_benchmarks.cpp)
target_include_directories(eventprop_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(eventprop_benchmarks PRIVATE
    eventprop_c_api
    benchmark::benchmark
    benchmark::benchmark_main
)

# Copy native library to build output for runtime (macOS .dylib)
if(APPLE)
    add_custom_command(TARGET eventprop_benchmarks POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:eventprop_c_api>
            $<TARGET_FILE_DIR:eventprop_benchmarks>
    )
endif()
