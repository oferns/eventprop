cmake_minimum_required(VERSION 3.20)
project(EventPropParityTest LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Eigen (from EventProp.Python submodule) ─────────────────────
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src/eigen")
if(NOT EXISTS "${EIGEN_DIR}")
    message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: git submodule update --init src/EventProp.Python/src/eigen")
endif()

# ── Boost (header-only: math) ───────────────────────────────────
find_path(BOOST_INCLUDE_DIR NAMES boost/math/tools/roots.hpp PATHS /opt/homebrew/include NO_DEFAULT_PATH)
if(NOT BOOST_INCLUDE_DIR)
    message(FATAL_ERROR "Boost not found at /opt/homebrew/include")
endif()

# ── OpenMP (optional, for parallelization) ─────────────────────
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
    if(_BREW_FAIL EQUAL 0 AND EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_FOUND TRUE)
        set(OpenMP_CXX_LIBRARIES "${_OPENMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_INCLUDE_DIRS "${_OPENMP_PREFIX}/include")
    endif()
else()
    find_package(OpenMP)
endif()

# ── Eigen core (stripped copy without pybind11) — build as static lib
add_library(eventprop_eigen_core STATIC ${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Cpp.Wrapper/eventprop_eigen_core_stripped.cpp)
target_include_directories(eventprop_eigen_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

# OpenMP for parallelization (no CPU-specific flags — let compiler auto-detect)
if(APPLE AND _BREW_FAIL EQUAL 0)
    target_include_directories(eventprop_eigen_core PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(eventprop_eigen_core PRIVATE ${OpenMP_CXX_LIBRARIES})
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(eventprop_eigen_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# ── Parity test executable (loads C library via dlopen at runtime)
add_executable(parity_test parity_main.cpp)
target_include_directories(parity_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.Python/src"
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/../EventProp.C"
)

target_link_libraries(parity_test PRIVATE eventprop_eigen_core m dl)

# macOS rpath
if(APPLE AND _BREW_FAIL EQUAL 0)
    set_target_properties(parity_test PROPERTIES
        INSTALL_RPATH "${_OPENMP_PREFIX}/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()
