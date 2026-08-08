cmake_minimum_required(VERSION 4.3)
project(DigitClassification)

set(CMAKE_CXX_STANDARD 23)

add_executable(DigitClassification main.cpp
        math_vector.h
        matrix.h
        neural_network.h
        neural_network.cpp
        digits.cpp)
