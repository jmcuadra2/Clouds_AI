cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61)
endif()
project(NubeCUDA LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
#find_package( glm REQUIRED )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}" )

set(NubeCUDA_SRCS
    src/Axis.cpp
    src/Cumulus.cpp
    src/Lsystem.cpp
    src/Morph.cpp
    src/PrecomputeCUDA.cu
    src/Camera.cpp
    src/Exception.cpp
    src/Main.cpp
    src/Mountain.cpp
    src/Shader.cpp
    src/Canvas.cpp
    src/FluidCPU.cpp
    src/Metacloud.cpp
    src/OBJLoader.cpp
    src/Tool.cpp
    src/Cloud.cpp
    src/FluidCUDA.cu
    src/Model.cpp
    src/PrecomputeCPU.cpp
    src/CumulusIA.cpp
    src/NeuralNet.cpp
    src/LSTMNet.cpp
    src/GRUNet.cpp
    )

SET(NubeCUDA_HDRS
    inc/Axis.h
    inc/Cumulus.h    
    inc/FluidCUDA.h
    inc/Metacloud.h
    inc/OBJLoader.h
    inc/Shader.h
    inc/Camera.h
    inc/Defines.h
    inc/Fluid.h
    inc/Model.h
    inc/PrecomputeCPU.h
    inc/Tool.h
    inc/Canvas.h
    inc/Exception.h
    inc/Lsystem.h
    inc/Morph.h
    inc/PrecomputeCUDA.h
    inc/Cloud.h
    inc/FluidCPU.h
    inc/Main.h
    inc/Mountain.h
    inc/Precompute.h
    inc/CumulusIA.h
    inc/NeuralNet.h
    inc/LSTMNet.h
    inc/GRUNet.h
)

add_executable(NubeCUDA ${NubeCUDA_SRCS})
include_directories(inc)
target_include_directories(NubeCUDA PUBLIC
                           ${PROJECT_BINARY_DIR}/inc /usr/include/glm)

target_link_libraries(NubeCUDA "${TORCH_LIBRARIES}" -lglut -lGL -lGLU -lGLEW)
set_property(TARGET NubeCUDA PROPERTY CXX_STANDARD 17)
