cmake_minimum_required(VERSION 3.15)
project(VartalaapEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -mavx2 -march=native -Wall -Wextra)
endif()

# Include directories
include_directories(include)

# Native Vector Engine Source
add_executable(vartalaap_bench cpp/native_vector_engine.cpp)

message(STATUS "Configured VartaLaap Native SIMD Vector Engine with AVX2 Optimization.")
