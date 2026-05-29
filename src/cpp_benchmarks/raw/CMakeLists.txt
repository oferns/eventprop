cmake_minimum_required(VERSION 3.20)
project(eventprop_raw_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_path(EIGEN_INCLUDE_DIR NAMES Eigen/Dense PATHS /opt/homebrew/include/eigen3 NO_DEFAULT_PATH)
if(NOT EIGEN_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found at /opt/homebrew/include/eigen3")
endif()

# Find Boost (header-only: format, math)
find_path(BOOST_INCLUDE_DIR NAMES boost/format.hpp PATHS /opt/homebrew/include NO_DEFAULT_PATH)
if(NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "Boost not found at /opt/homebrew/include")
endif()

# Fetch Google Benchmark
include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# Include C++ algorithm headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../EventProp.cpp ${EIGEN_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})

# Raw C++ benchmark executable — links directly against libeventprop_cpp.dylib
add_executable(raw_bench raw_main.cpp)
target_link_libraries(raw_bench PRIVATE benchmark::benchmark benchmark::benchmark_main)

# Link against the C++ algorithm library
target_link_libraries(raw_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../EventProp.cpp/build/bin/libeventprop_cpp.dylib)

# macOS: rpath so it finds the .dylib at runtime
set_target_properties(raw_bench PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Copy .dylib to output dir
if(APPLE)
    add_custom_command(TARGET raw_bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            /Users/oliverferns/repos/eventprop.net/src/EventProp.cpp/build/bin/libeventprop_cpp.dylib
            $<TARGET_FILE_DIR:raw_bench>
    )
endif()
