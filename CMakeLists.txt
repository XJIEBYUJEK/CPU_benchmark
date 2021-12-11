cmake_minimum_required(VERSION 3.20)
project(cpu_benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-march=native")

add_executable(cpu_benchmark main.cpp)
