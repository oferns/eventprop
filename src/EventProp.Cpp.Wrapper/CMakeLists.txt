cmake_minimum_required(VERSION 3.16)
project(eventprop_c_api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# Optional OpenMP for parallelism (skip on Apple — EventProp.cpp handles this separately)
if(NOT APPLE)
    find_package(OpenMP)
endif()

# Include EventProp.cpp headers (for eventprop_types.h and function declarations)
set(EVENTPROP_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.cpp")

# Build the C API wrapper as a shared library
add_library(eventprop_cpp_wrapper_lib SHARED eventprop_c_api.cpp)
set_target_properties(eventprop_cpp_wrapper_lib PROPERTIES OUTPUT_NAME "eventprop_cpp_wrapper")
target_include_directories(eventprop_c_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EVENTPROP_CPP_DIR}
)

# Link against the EventProp.cpp shared library (contains compute_sums, compute_spikes, etc.)
set(EVENTPROP_CPP_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.cpp/build/bin/libeventprop_cpp.dylib")
if(NOT EXISTS "${EVENTPROP_CPP_LIB}")
    message(FATAL_ERROR "Cannot find ${EVENTPROP_CPP_LIB}. Build EventProp.cpp first.")
endif()

target_link_libraries(eventprop_c_api PRIVATE
    Eigen3::Eigen
    ${EVENTPROP_CPP_LIB}
)

if(OpenMP_CXX_FOUND AND NOT APPLE)
    target_link_libraries(eventprop_c_api PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set output name to match P/Invoke expectation
# CMake adds 'lib' prefix automatically, so OUTPUT_NAME is just the suffix
set_target_properties(eventprop_c_api PROPERTIES
    OUTPUT_NAME "eventprop_cpp_wrapper"
)

# Set rpath so the wrapper can find libeventprop_cpp.dylib at runtime
set_target_properties(eventprop_c_api PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install targets
install(TARGETS eventprop_c_api LIBRARY DESTINATION lib)
install(FILES eventprop_c_api.h DESTINATION include)
