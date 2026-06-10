cmake_minimum_required(VERSION 3.16)
project(eventprop_c LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Detect SIMD support at compile time (priority: AVX-512 > AVX2 > NEON)
include(CheckCCompilerFlag)

# AVX-512F (x86_64 only)
check_c_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "AVX-512 detected — building with AVX-512 support")
    add_compile_definitions(HAVE_AVX512F)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f")
# AVX2 (x86_64 only)
elif(COMPILER_SUPPORTS_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "AVX2 detected — building with AVX2 support")
    add_compile_definitions(HAVE_AVX2)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
# ARM NEON (ARM64 only)
elif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "ARM64 detected — NEON enabled by default")
    add_compile_definitions(__ARM_NEON)
endif()

# Pure C shared library (no external dependencies)
# Optional OpenMP for parallel neuron processing in ComputeSpikes
find_package(OpenMP QUIET)
add_library(eventprop_c_api SHARED
    eventprop.c
    eventprop_diag.c
)
# CPU-specific optimization (kept for performance)
target_compile_options(eventprop_c_api PRIVATE -march=native)
target_compile_options(eventprop_c_api PRIVATE -ffp-contract=fast)
target_link_libraries(eventprop_c_api PRIVATE m)
if(OpenMP_C_FOUND)
    target_link_libraries(eventprop_c_api PRIVATE OpenMP::OpenMP_C)
    message(STATUS "OpenMP found — enabling parallel neuron processing")
else()
    # macOS fallback: use Homebrew libomp
    execute_process(COMMAND brew --prefix libomp OUTPUT_STRIP_TRAILING_WHITESPACE
                    OUTPUT_VARIABLE _OPENMP_PREFIX RESULT_VARIABLE _BREW_FAIL)
    if(_BREW_FAIL EQUAL 0 AND EXISTS "${_OPENMP_PREFIX}/lib/libomp.dylib")
        target_compile_options(eventprop_c_api PRIVATE "-Xpreprocessor" "-fopenmp")
        target_include_directories(eventprop_c_api PRIVATE "${_OPENMP_PREFIX}/include")
        target_link_libraries(eventprop_c_api PRIVATE "${_OPENMP_PREFIX}/lib/libomp.dylib")
        message(STATUS "OpenMP (Homebrew libomp) found — enabling parallel neuron processing")
    else()
        message(STATUS "OpenMP not found — running sequentially (Eigen uses OpenMP for parallel neuron processing)")
    endif()
endif()
target_include_directories(eventprop_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Install
install(TARGETS eventprop_c_api LIBRARY DESTINATION lib)
install(FILES eventprop.h eventprop_diag.h DESTINATION include)

# Unit tests
add_executable(test_runner tests.c)
target_link_libraries(test_runner PRIVATE eventprop_c_api m)
add_test(NAME EventPropCTests COMMAND test_runner)
