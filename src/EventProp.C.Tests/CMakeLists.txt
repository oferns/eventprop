cmake_minimum_required(VERSION 3.21)
project(EventPropCTests LANGUAGES CXX)

# The configure presets set CMAKE_BUILD_TYPE, and on a multi-config generator (MSVC,
# Xcode) nothing consumes it — so CMake prints "Manually-specified variables were not
# used by the project: CMAKE_BUILD_TYPE" on every Windows configure. The presets cannot
# simply stop setting it: single-config generators (Unix Makefiles, Ninja — Linux and
# macOS) take their optimisation level from it, so dropping it there would silently
# build -O0 and poison every benchmark number. Reading it here is what tells CMake the
# variable is used; the default is the honest counterpart of that reading, because a
# single-config generator with no build type is an unoptimised build nobody chose.
# Same block in every preset-driven project (EventPropSimd.cmake answers the ISA
# question once for the same reason).
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (single-config generators only)" FORCE)
endif()

# C++20, not 17: the C-API fixtures use designated initializers
# (BackwardsBatch{ .error_no = ..., .batch_size = ... }), which are standard only from
# C++20. Clang and GCC accepted them as an extension at 17; MSVC makes it a hard error
# (C7555 "use of designated initializers requires at least '/std:c++20'").
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# -- Pre-built C library (imported from EventProp.C build) -----
# When ASan is enabled on macOS, the library is built STATIC instead of SHARED.
if(ENABLE_ASAN AND APPLE)
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api.a")
else()
    if(WIN32)
        set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api.dll")
        set(C_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api.lib")
    elseif(APPLE)
        set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api.dylib")
    else()
        set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api.so")
    endif()
endif()

if(ENABLE_ASAN AND APPLE)
    add_library(eventprop_c_api STATIC IMPORTED)
else()
    add_library(eventprop_c_api SHARED IMPORTED)
endif()
set_target_properties(eventprop_c_api PROPERTIES
    IMPORTED_LOCATION "${C_LIBRARY}"
)
# Windows: a SHARED IMPORTED target needs the import library, not just the DLL, and
# the Visual Studio generator is multi-config, so it resolves imported paths per
# configuration. Setting only IMPORTED_LOCATION fails generation with
#   IMPORTED_IMPLIB not set for imported target "eventprop_c_api" configuration "MinSizeRel"
# All four configuration names point at the same Release artifacts, which is what the
# hardcoded build/Release paths above already mean; naming them explicitly is what
# CMake's per-config lookup requires.
if(WIN32)
    set_target_properties(eventprop_c_api PROPERTIES
        IMPORTED_CONFIGURATIONS "RELEASE;DEBUG;RELWITHDEBINFO;MINSIZEREL"
        IMPORTED_IMPLIB_RELEASE "${C_IMPLIB}"
        IMPORTED_LOCATION_RELEASE "${C_LIBRARY}"
        IMPORTED_IMPLIB_DEBUG "${C_IMPLIB}"
        IMPORTED_LOCATION_DEBUG "${C_LIBRARY}"
        IMPORTED_IMPLIB_RELWITHDEBINFO "${C_IMPLIB}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${C_LIBRARY}"
        IMPORTED_IMPLIB_MINSIZEREL "${C_IMPLIB}"
        IMPORTED_LOCATION_MINSIZEREL "${C_LIBRARY}"
    )
endif()
target_include_directories(eventprop_c_api SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C"
)

# -- Google Test (FetchContent) -------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# googletest's own source emits one warning under clang/AppleClang:
#   googletest/include/gtest/gtest-printers.h:524:35: implicit conversion from 'char8_t' to
#   'char32_t' may change the meaning of the represented code unit [-Wcharacter-conversion]
# It appears only while compiling googletest's sources: the test TUs include those headers as
# SYSTEM includes and are warning-free, so marking includes SYSTEM cannot reach this one —
# there is no consumer involved. Suppressed by name on the googletest targets alone, so the
# repository's zero-warning rule keeps meaning what it is meant to mean: first-party code.
# The other lever is a googletest version bump, which changes a dependency for every platform
# at once, so the scoped suppression is the cheaper one.
#
# The probe tests the POSITIVE flag on purpose. Compilers accept an unknown `-Wno-…` spelling
# without failing, so probing the negated form would report "supported" for a compiler that has
# never heard of the warning: GCC has no -Wcharacter-conversion, and MSVC takes no -W flags at
# all — both get nothing from this block.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wcharacter-conversion" EVENTPROP_CXX_WARN_CHARACTER_CONVERSION)
if(EVENTPROP_CXX_WARN_CHARACTER_CONVERSION)
    foreach(_gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
        if(TARGET ${_gtest_target})
            target_compile_options(${_gtest_target} PRIVATE -Wno-character-conversion)
        endif()
    endforeach()
endif()

# -- Test executable ------------------------------------------
enable_testing()
add_executable(eventprop_tests
    test/test_main.cpp
    test/compute_sums.cpp
    test/compute_spikes.cpp
    test/compute_maxima.cpp
    test/backward_spikes.cpp
    test/backward_maxima.cpp
    test/batch_functions.cpp
    test/validate_input.cpp
    test/error_strings.cpp
    test/abi_version.cpp
    test/free_functions.cpp
    test/alloc_fault_injection.cpp
    test/diag_functions.cpp
    test/matrix_view.cpp
)
target_compile_definitions(eventprop_tests PRIVATE EVENTPROP_VALIDATE_INPUTS)
target_link_libraries(eventprop_tests PRIVATE eventprop_c_api GTest::gtest_main)
if(NOT WIN32)
    target_link_libraries(eventprop_tests PRIVATE m)
endif()

# When using static library with ASan, OpenMP runtime must be linked explicitly.
if(ENABLE_ASAN AND APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _BREW_FAIL OUTPUT_VARIABLE _OPENMP_PREFIX)
    if(_BREW_FAIL EQUAL 0 AND EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
        target_link_libraries(eventprop_tests PRIVATE "${_OPENMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

# AddressSanitizer - must match library build (both ON or both OFF).
# Usage: cmake .. -DENABLE_ASAN=ON from EventProp.C/build, then rebuild tests.
# Note: on macOS AppleClang, ASan in shared libraries has known issues.
# Linux is the recommended platform for full ASan testing.
option(ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
if(ENABLE_ASAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "AddressSanitizer enabled for tests - run with ASAN_OPTIONS=detect_leaks=1")
    target_compile_options(eventprop_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(eventprop_tests PRIVATE -fsanitize=address)
endif()

# Copy the C library DLL alongside the test executable at build time
if(WIN32)
    add_custom_command(TARGET eventprop_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${C_LIBRARY}"
            $<TARGET_FILE_DIR:eventprop_tests>
    )
elseif(APPLE)
    set_target_properties(eventprop_tests PROPERTIES
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

include(GoogleTest)
gtest_discover_tests(eventprop_tests)
