cmake_minimum_required (VERSION 2.8)
#set(CMAKE_VERBOSE_MAKEFILE ON)

project (net)

find_package(CUDA REQUIRED)
#fix the odd linking issue
set(CUDA_BIN_PATH /opt/cuda/bin/)
#-fPIC needed for shared library
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ccbin /usr/bin -arch=sm_20 -Xcompiler -fPIC)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
add_definitions("-g")
add_definitions("-Wall")
add_definitions("-std=c++11")
add_definitions("-O3")

cuda_compile(CUDA_O src/net_gpu_impl.cu)
cuda_add_library(net SHARED ${CUDA_O})

cuda_add_executable(tests
  tests/net_cpu.cpp
  tests/main.cpp
  tests/net.cpp
  tests/net_gpu.cpp
  tests/net_raw_utils.cpp
  )

target_link_libraries(tests armadillo net gtest )

include_directories(src)
add_executable(gpu_mnist samples/gpu_mnist.cpp)
add_executable(cpu_mnist samples/cpu_mnist.cpp)
target_link_libraries(cpu_mnist armadillo net)
target_link_libraries(gpu_mnist armadillo net)

#build the helper tool
add_executable(convert_to_arma tools/convert_to_arma.cpp)
target_link_libraries(convert_to_arma armadillo)
