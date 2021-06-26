set(CMAKE_C_COMPILER /usr/bin/gcc-7)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

cmake_minimum_required(VERSION 2.8)
project(appProj)

find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS}
                    ${OPENGL_INCLUDE_DIRS} 
                    ${GLFW_INCLUDE_DIRS}
                    $(GLEW_INCLUDE_DIRS)
		    )

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -x cu -O3 -pg -g -std=c++11 -D DEBUG -Xlinker -framework,OpenGL -Xlinker -lGLEW -Xlinker -lglfw -Xlinker -lGL -Xlinker -ldl -Xlinker -lm" )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

file(GLOB_RECURSE SOURCES
	*.h
	*.cuh
	visualizer.h
	visualizer.cpp
	src/*.h
	src/*.cpp
	libs/*.h
	libs/*.cpp
)

set (CUDA_NVCC_LINK_EXECUTABLE "${CUDA_NVCC_LINK_EXECUTABLE} -x cu -O3 -pg -g -std=c++11 -D DEBUG -Xlinker -framework,OpenGL -Xlinker -lGLEW -Xlinker -lglfw -Xlinker -lGL -Xlinker -ldl -Xlinker -lm")
cuda_add_executable(Simulation main.cu)

add_library(MyLib ${SOURCES})

target_link_libraries(Simulation MyLib ${OPENGL_gl_LIBRARY} ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} -lglfw -lGL -lGLEW)
