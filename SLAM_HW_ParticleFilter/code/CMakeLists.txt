cmake_minimum_required(VERSION 3.4...3.18)
project(ray_casting_lib)

add_library(ray_casting_fast STATIC src/ray_casting.cpp)
set_target_properties(ray_casting_fast PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(pybind11)
pybind11_add_module(ray_casting_lib src/binder.cpp)
target_link_libraries(ray_casting_lib PRIVATE ray_casting_fast)
