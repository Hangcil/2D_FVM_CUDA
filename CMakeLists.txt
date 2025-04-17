cmake_minimum_required(VERSION 3.18.0)
project(GRPRel LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --ptxas-options=-v")
set(CMAKE_CUDA_ARCHITECTURES 89) # Target Ada Lovelace (RTX 4060)

# Specify the CUDA host compiler explicitly
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/<version>/bin/Hostx64/x64/cl.exe")

add_executable(GRPRelaxation  "./src/basicFuns.h " "TestSimple.cu")
target_link_directories(GRPRelaxation PUBLIC CUDA::cudart)

