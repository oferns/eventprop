cmake_minimum_required(VERSION 3.20)
project(EventPropParityTest LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Eigen (from EventProp.Python submodule) ───────────────────────────────
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")
if(NOT EXISTS "${EIGEN_DIR}")
    message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: git submodule update --init src/EventProp.Python/src/eigen")
endif()

# ── Boost (header-only: math) ────────────────────────────────────────────
find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp PATHS /opt/homebrew/include NO_DEFAULT_PATH)
if(NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "Boost not found at /opt/homebrew/include")
endif()

# ── Eigen core (from EventProp.Python) — build as static lib ────────────
add_library(eventprop_eigen_core STATIC eventprop_eigen_core.cpp)
target_include_directories(eventprop_eigen_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

# macOS-specific: no OpenMP, use apple-m3
if(APPLE)
    target_compile_options(eventprop_eigen_core PRIVATE -mcpu=apple-m3)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(eventprop_eigen_core OpenMP::OpenMP_CXX)
endif()

# ── Parity test executable (loads C library via dlopen at runtime) ──────
add_executable(parity_test parity_main.cpp)
target_include_directories(parity_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src"
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C"
)

target_link_libraries(parity_test PRIVATE eventprop_eigen_core m dl)

# macOS rpath
if(APPLE)
    set_target_properties(parity_test PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
