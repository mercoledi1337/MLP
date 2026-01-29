cmake_minimum_required(VERSION 4.0)
project(MLP)

set(CMAKE_CXX_STANDARD 20)

add_executable(MLP main.cpp
        Layer.h
        Layer.cpp
        MLP.h
        MLP.cpp)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
target_link_libraries(MLP PRIVATE ${OpenCV_LIBS})
target_link_libraries(MLP PUBLIC OpenMP::OpenMP_CXX)