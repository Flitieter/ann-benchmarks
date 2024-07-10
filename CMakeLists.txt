cmake_minimum_required (VERSION 3.17)
project (ANN-Benchmark)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(ann_benchmarks/algorithms)

