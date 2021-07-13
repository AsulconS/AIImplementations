cmake_minimum_required(VERSION 3.9)
project(DecisionTree)
add_executable(main src/main.cpp src/mnistParsing.cpp)
target_include_directories(main PRIVATE include)
