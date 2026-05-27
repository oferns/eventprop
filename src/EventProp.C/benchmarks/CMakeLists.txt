cmake_minimum_required(VERSION 3.16)
project(eventprop_c_benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect hardware thread count for multi-threading benchmarks (compile-time constant)
if(APPLE)
    execute_process(COMMAND sysctl -n hw.ncpu OUTPUT_VARIABLE EP_BENCH_THREADS OUTPUT_STRIP_TRAILING_WHITESPACE)
elseif(WIN32)
    set(EP_BENCH_THREADS $ENV{NUMBER_OF_PROCESSORS})
else()
    execute_process(COMMAND nproc OUTPUT_VARIABLE EP_BENCH_THREADS OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
if(NOT EP_BENCH_THREADS OR EP_BENCH_THREADS STREQUAL "")
    set(EP_BENCH_THREADS 8) # fallback
endif()
message(STATUS "Multi-threading benchmarks will use ${EP_BENCH_THREADS} threads")
add_compile_definitions(EP_BENCH_THREADS=${EP_BENCH_THREADS})

# Optional OpenMP for parallelism (multi-threading benchmarks)
if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp;/usr/local/opt/libomp" CACHE STRING "" FORCE)
endif()
find_package(OpenMP)

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

# Build the pure C API library if not already built
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. eventprop_c_lib)

# Benchmark executable — main algorithm benchmarks
add_executable(eventprop_c_benchmarks eventprop_benchmarks.cpp)
target_include_directories(eventprop_c_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(eventprop_c_benchmarks PRIVATE
    eventprop_c_api
    benchmark::benchmark
    benchmark::benchmark_main
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(eventprop_c_benchmarks PRIVATE OpenMP::OpenMP_CXX)
endif()

# Copy native library to build output for runtime (macOS .dylib)
if(APPLE)
    add_custom_command(TARGET eventprop_c_benchmarks POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:eventprop_c_api>
            $<TARGET_FILE_DIR:eventprop_c_benchmarks>
    )
endif()

# ============================================================
# Micro-benchmarks executable
# ============================================================
add_executable(micro_c_benchmarks micro_benchmarks.cpp)
target_include_directories(micro_c_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(micro_c_benchmarks PRIVATE
    eventprop_c_api
    benchmark::benchmark
    benchmark::benchmark_main
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(micro_c_benchmarks PRIVATE OpenMP::OpenMP_CXX)
endif()

if(APPLE)
    add_custom_command(TARGET micro_c_benchmarks POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:eventprop_c_api>
            $<TARGET_FILE_DIR:micro_c_benchmarks>
    )
endif()
