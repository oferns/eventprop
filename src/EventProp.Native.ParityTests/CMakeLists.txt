cmake_minimum_required(VERSION 3.20)
project(EventPropParityTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Eigen (from EventProp.Python submodule, header-only) ─────────
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")
if(NOT EXISTS "${EIGEN_DIR}")
    message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: git submodule update --init src/EventProp.Python/src/eigen")
endif()

# ── Boost (header-only: math) ───────────────────────────────────
if(WIN32)
    set(BOOST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/vcpkg/installed/x64-windows/include")
else()
    find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp PATHS /opt/homebrew/include /usr/include NO_DEFAULT_PATH)
endif()
if(NOT BOOST_INCLUDE_DIR OR NOT EXISTS "${BOOST_INCLUDE_DIR}/boost/math/tools/roots.hpp")
    message(FATAL_ERROR "Boost not found at ${BOOST_INCLUDE_DIR}. Install libboost-dev (Linux) or boost (macOS), or use vcpkg on Windows")
endif()

# ── Pre-built C library (imported) ─────────────────────────────
if(WIN32)
    set(C_LIBRARY_EXT ".dll")
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api${C_LIBRARY_EXT}")
elseif(APPLE)
    set(C_LIBRARY_EXT ".dylib")
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api${C_LIBRARY_EXT}")
else()
    set(C_LIBRARY_EXT ".so")
    set(C_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/libeventprop_c_api${C_LIBRARY_EXT}")
endif()
if(NOT EXISTS "${C_LIBRARY}")
    message(FATAL_ERROR "C library not found at ${C_LIBRARY}. Build EventProp.C first.")
endif()

add_library(eventprop_c_api SHARED IMPORTED)
if(WIN32)
    set_target_properties(eventprop_c_api PROPERTIES
        IMPORTED_LOCATION "${C_LIBRARY}"
        IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release/eventprop_c_api.lib"
    )
else()
    set_target_properties(eventprop_c_api PROPERTIES
        IMPORTED_LOCATION "${C_LIBRARY}"
    )
endif()

# ── Pre-built Eigen core (imported from EventProp.Cpp.Wrapper) ─
if(WIN32)
    set(EIGEN_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/build/Release/eventprop_eigen_core.lib")
else()
    set(EIGEN_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/build/libeventprop_eigen_core.a")
endif()
if(NOT EXISTS "${EIGEN_LIBRARY}")
    message(FATAL_ERROR "Eigen core library not found at ${EIGEN_LIBRARY}. Build EventProp.Cpp.Wrapper first.")
endif()

add_library(eventprop_eigen_core STATIC IMPORTED)
set_target_properties(eventprop_eigen_core PROPERTIES
    IMPORTED_LOCATION "${EIGEN_LIBRARY}"
)

# ── Parity test executable ─────────────────────────────────────
find_package(OpenMP QUIET)
add_executable(parity_test parity_main.cpp)

target_include_directories(parity_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C"
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper"
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

# libm is separate on Unix, built-in on Windows
if(WIN32)
    target_link_libraries(parity_test PRIVATE eventprop_c_api eventprop_eigen_core)
else()
    target_link_libraries(parity_test PRIVATE eventprop_c_api eventprop_eigen_core m)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(parity_test PRIVATE OpenMP::OpenMP_CXX)
endif()

# Runtime library search path
if(WIN32)
    # Windows needs DLL in PATH or same directory as executable
    set_target_properties(parity_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build/Release"
    )
elseif(APPLE)
    set_target_properties(parity_test PROPERTIES
        INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C/build"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
