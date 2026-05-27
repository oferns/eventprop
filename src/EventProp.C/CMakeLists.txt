cmake_minimum_required(VERSION 3.16)
project(eventprop_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Detect SIMD support at compile time
include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "AVX2 detected — building SIMD variant")
    add_compile_definitions(HAVE_AVX2)
endif()

# Pure C shared library (no external dependencies)
add_library(eventprop_c_api SHARED eventprop_scalar.c)
target_link_libraries(eventprop_c_api PRIVATE m)
target_include_directories(eventprop_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Install
install(TARGETS eventprop_c_api LIBRARY DESTINATION lib)
install(FILES eventprop.h DESTINATION include)
