cmake_minimum_required(VERSION 2.8)
project(uastar)

find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS program_options system REQUIRED)

add_definitions("-std=c++1y")
add_definitions("-Wall")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -fsanitize=address")

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
set_source_files_properties(
    libs/moderngpu/src/mgpucontext.cu
    PROPERTIES COMPILE_FLAGS -Wno-all)
set_source_files_properties(
    libs/moderngpu/src/mgpuutil.cpp
    PROPERTIES COMPILE_FLAGS -Wno-all)

cuda_add_executable(
    uastar
    src/main.cpp
    src/pathway/pathway.cpp
    src/pathway/CPU-solver.cpp
    src/pathway/GPU-solver.cu
    src/pathway/input/custom.cpp
    src/puzzle/puzzle.cpp
    ${moderngpu_source}
)

cuda_add_executable(
    demo-sort
    utils/modern-sort.cu
    ${moderngpu_source}
)

target_link_libraries(
    uastar
    ${Boost_LIBRARIES}
)
