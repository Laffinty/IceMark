cmake_minimum_required(VERSION 3.15)
project(IceMark VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(icemark
    src/main.cpp
    src/cpu_detector.cpp
    src/weight_generator.cpp
    src/inference_engine.cpp
    src/benchmark.cpp
    src/renderer.cpp
)

target_include_directories(icemark PRIVATE ${CMAKE_SOURCE_DIR}/include)