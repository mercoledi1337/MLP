cmake_minimum_required(VERSION 4.0)
project(MLP)

set(CMAKE_CXX_STANDARD 20)

add_executable(MLP main.cpp
        Layer.h
        Layer.cpp
        MLP.h
        MLP.cpp)
