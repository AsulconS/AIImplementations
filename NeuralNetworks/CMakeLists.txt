cmake_minimum_required(VERSION 3.9)
project(Lab09_AI)
add_executable(main src/main.cpp src/perceptron.cpp)
target_include_directories(main PRIVATE include)
