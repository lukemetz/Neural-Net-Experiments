cmake_minimum_required (VERSION 2.8)
#set(CMAKE_VERBOSE_MAKEFILE ON)

project (net)

find_package(CUDA REQUIRED)
set(CUDA_BIN_PATH /opt/cuda/bin/)
# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ccbin /usr/bin -arch=sm_20)
#set(CUDA_HOST_COMPILER /usr/bin/clang++)
#set(CMAKE_C_COMPILER /usr/bin/clang)
#set(CMAKE_CXX_COMPILER /usr/bin/clang++)
#set(CUDA_HOST_COMPILER /opt/cuda/bin/nvcc)
#cuda_add_library(cuda_lib net.cu)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
add_definitions("-g")
add_definitions("-Wall")
add_definitions("-std=c++11")
add_definitions("-O1")

cuda_compile(CUDA_O src/net_gpu_impl.cu)
cuda_add_executable(tests tests/net_cpu.cpp tests/main.cpp ${CUDA_O} tests/net_gpu.cpp tests/net_raw_utils.cpp)

target_link_libraries(tests armadillo )
target_link_libraries(tests gtest)

add_executable(mnist samples/mnist.cpp)
include_directories(src)
target_link_libraries(mnist armadillo)

add_executable(convert_to_arma tools/convert_to_arma.cpp)
target_link_libraries(convert_to_arma armadillo)
