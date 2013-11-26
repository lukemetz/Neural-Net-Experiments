cmake_minimum_required (VERSION 2.6)

project (net)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
add_definitions("-g")
add_definitions("-Wall")
add_definitions("-pendantic")
add_definitions("-std=c++11")

add_executable(tests tests.cpp main.cpp)
target_link_libraries(tests gtest)
