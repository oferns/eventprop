cmake_minimum_required(VERSION 3.16)
project(eventprop_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)

# Optional OpenMP for parallelism
find_package(OpenMP)

add_library(eventprop_c_api SHARED eventprop_c_api.cpp)
target_link_libraries(eventprop_c_api PRIVATE Eigen3::Eigen)
target_include_directories(eventprop_c_api PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(eventprop_c_api PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the library
install(TARGETS eventprop_c_api LIBRARY DESTINATION lib)
install(FILES eventprop_c_api.h DESTINATION include)
