cmake_minimum_required(VERSION 3.3)
project(Paralell_computations)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c99")

set(SOURCE_FILES ov3/debug/reverseParams.c )
add_executable(Paralell_computations ${SOURCE_FILES})