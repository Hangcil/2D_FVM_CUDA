cmake_minimum_required(VERSION 3.18.0)
project(GRPRel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --ptxas-options=-v")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 --ptxas-options=-v")
set(CMAKE_CUDA_ARCHITECTURES 89) # set your own architecture

# Specify the CUDA host compiler explicitly
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/<version>/bin/Hostx64/x64/cl.exe")

find_package(OpenCV REQUIRED PATHS "C:\\libs\\OpenCV")

aux_source_directory(./src MYSRC)
add_executable(GRPRelaxation  ${MYSRC} "test2D.cu")
include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include")
target_link_libraries(GRPRelaxation PRIVATE ${OpenCV_LIBS})
target_include_directories(GRPRelaxation PRIVATE ${OpenCV_INCLUDE_DIRS} PUBLIC CUDA::cudart )


