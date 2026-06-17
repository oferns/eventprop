cmake_minimum_required(VERSION 3.16)
project(eventprop_c_api LANGUAGES CXX)

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
add_library(eventprop_eigen_core STATIC eventprop_eigen_core_stripped.cpp)
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

# ── C API wrapper as a shared library ───────────────────────────
add_library(eventprop_c_api SHARED eventprop_c_api.cpp)
set_target_properties(eventprop_c_api PROPERTIES OUTPUT_NAME "eventprop_cpp_wrapper")
target_include_directories(eventprop_c_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

target_link_libraries(eventprop_c_api PRIVATE
    eventprop_eigen_core
)

if(APPLE AND _BREW_FAIL EQUAL 0)
    set_target_properties(eventprop_c_api PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(OpenMP_CXX_FOUND)
    set_target_properties(eventprop_c_api PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# ── Unit tests (targets Eigen directly, not via C API) ─────────
add_executable(eventprop_cpp_tests tests.cpp)

# ── C API wrapper unit tests (mirrors src/EventProp.C/tests.c) ──
add_executable(eventprop_capi_tests tests_capi.cpp eventprop_c_api.cpp)
target_include_directories(eventprop_capi_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)
target_link_libraries(eventprop_capi_tests PRIVATE eventprop_eigen_core)

target_include_directories(eventprop_cpp_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)
target_link_libraries(eventprop_cpp_tests PRIVATE eventprop_eigen_core)

# Install targets
install(TARGETS eventprop_c_api LIBRARY DESTINATION lib)
install(FILES eventprop_c_api.h DESTINATION include)
