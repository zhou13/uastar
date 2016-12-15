cmake_minimum_required(VERSION 2.8)
project(uastar)

find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS program_options system filesystem REQUIRED)

add_definitions("-std=c++1y")
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -lineinfo -gencode arch=compute_30,code=sm_30 --ptxas-options -v )
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -rdynamic -Wall -DDEBUG ")
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -O2 -gencode arch=compute_30,code=sm_30 --ptxas-options -v)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

include_directories(libs/moderngpu/include)
include_directories(libs/bitmap)
include_directories(src)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

set(
    moderngpu_source
    libs/moderngpu/src/mgpucontext.cu
    libs/moderngpu/src/mgpuutil.cpp
)

cuda_add_executable(
    uastar
    src/main.cpp
    src/pathway/pathway.cpp
    src/pathway/CPU-solver.cpp
    src/pathway/GPU-solver.cu
    src/pathway/input/custom.cpp
    src/pathway/input/zigzag.cpp
    src/puzzle/puzzle.cu
    src/puzzle/database.cpp
    ${moderngpu_source}
)

#cuda_add_executable(
#    demo-sort
#    utils/modern-sort.cu
#    ${moderngpu_source}
#)

target_link_libraries(
    uastar
    ${Boost_LIBRARIES}
)
