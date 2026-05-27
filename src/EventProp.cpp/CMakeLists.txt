cmake_minimum_required(VERSION 3.20)
project(EventPropCppWrapper LANGUAGES CXX)

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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})

# Build as shared library
add_library(eventprop_cpp_wrapper SHARED
    eventprop_types.h
    eventprop_reference.cpp
)

# Set output name
set_target_properties(eventprop_cpp_wrapper PROPERTIES
    PREFIX "" OUTPUT_NAME "libeventprop_cpp_wrapper"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# macOS-specific: rpath for the library
set_target_properties(eventprop_cpp_wrapper PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# OpenMP (optional, skip on Apple)
if(NOT APPLE)
    find_package(OpenMP REQUIRED)
    target_link_libraries(eventprop_cpp_wrapper OpenMP::OpenMP_CXX)
endif()

# Position-independent code (required for shared library)
set_target_properties(eventprop_cpp_wrapper PROPERTIES POSITION_INDEPENDENT_CODE ON)
