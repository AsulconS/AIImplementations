cmake_minimum_required(VERSION 3.9)
project(KDay VERSION 0.0.1 DESCRIPTION "CMakeLists File for compiling C++ KDay")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_SOURCE_DIR}>)
set(CMAKE_CXX_STANDARD 17)

add_library(image SHARED src/image.cpp)
set_target_properties(image PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER include/image.hpp
)
target_compile_options(image PRIVATE /W0)
target_compile_definitions(image PRIVATE __CMAKE_SHARED_MODE __STDC_LIB_EXT1__)
target_include_directories(image PRIVATE include)

add_executable(kday src/kday.cpp src/algorithm.cpp src/kdtree.cpp)
target_include_directories(kday PRIVATE include)
target_compile_options(kday PRIVATE /W0)
target_compile_definitions(kday PRIVATE __CMAKE_SHARED_MODE)
target_link_libraries(kday PUBLIC image)

add_executable(kscene src/kscene.cpp src/algorithm.cpp src/kdtree.cpp)
target_include_directories(kscene PRIVATE include)
target_compile_options(kscene PRIVATE /W0)
target_compile_definitions(kscene PRIVATE __CMAKE_SHARED_MODE)
target_link_libraries(kscene PUBLIC image)
