cmake_minimum_required(VERSION 3.18)
project(cuda-kernel-optimization LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Add kernel libraries
add_library(vector_add kernels/vector_add/vecAdd.cu)
add_library(reduction 
    kernels/reduction/naive.cu
    kernels/reduction/shared.cu
    kernels/reduction/warp.cu
)
add_library(matmul
    kernels/matmul/naive.cu
    kernels/matmul/tiled.cu
    kernels/matmul/optimized.cu
)
add_library(softmax kernels/softmax/softmax.cu)

# Enable all CUDA kernels for compilation
target_compile_options(vector_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
target_compile_options(reduction PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
target_compile_options(softmax PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
