cmake_minimum_required(VERSION 3.16)
project(EventPropCTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# ── Pre-built C library (imported from EventProp.C build) ─────
if(WIN32)
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api.dll")
    set(C_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api.lib")
elseif(APPLE)
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api.dylib")
else()
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api.so")
endif()

add_library(eventprop_c_api SHARED IMPORTED)
if(WIN32)
    set_target_properties(eventprop_c_api PROPERTIES
        IMPORTED_LOCATION "${C_LIBRARY}"
        IMPORTED_IMPLIB "${C_IMPLIB}"
    )
else()
    set_target_properties(eventprop_c_api PROPERTIES
        IMPORTED_LOCATION "${C_LIBRARY}"
    )
endif()
target_include_directories(eventprop_c_api SYSTEM INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C"
)

# ── Google Test (FetchContent) ───────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── Test executable ──────────────────────────────────────────
enable_testing()
add_executable(eventprop_tests
    test/test_main.cpp
    test/compute_sums.cpp
    test/compute_spikes.cpp
    test/compute_maxima.cpp
    test/backward_spikes.cpp
    test/backward_maxima.cpp
    test/validate_input.cpp
    test/free_functions.cpp
)
target_compile_definitions(eventprop_tests PRIVATE EVENTPROP_VALIDATE_INPUTS)
target_link_libraries(eventprop_tests PRIVATE eventprop_c_api GTest::gtest_main)
if(NOT WIN32)
    target_link_libraries(eventprop_tests PRIVATE m)
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
