cmake_minimum_required(VERSION 3.24)
project(multithread_jacobi)

set(CMAKE_CXX_STANDARD 23)

add_executable(multithread_jacobi main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
