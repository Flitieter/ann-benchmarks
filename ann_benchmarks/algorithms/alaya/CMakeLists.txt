cmake_minimum_required (VERSION 3.17)
project (HybridSearch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Code intelligence use
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# FetchContent_Declare(
#     fmt
#     URL https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip
#     DOWNLOAD_EXTRACT_TIMESTAMP true
# )
# FetchContent_MakeAvailable(fmt)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -pthread -g -Wall -pthread -mavx2 -mfma")
else()
    # AVX2
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -march=native -mtune=native -O3 -mavx2 -pthread -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
    # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -std=c++17 -fPIC -mavx -mavx2 -mfma")
endif()

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(pyglass)

# include_directories("pyglass")
# add_subdirectory("pyglass")

# option(BUILD_TEST "Build the executable test" OFF)
# option(BUILD_LIB "Build the shared library" ON)

# if (BUILD_LIB)
#     set(LIBRARY_NAME hybrid_search)

#     # 添加源文件
#     set(SOURCES baseline.cpp)

#     # 编译生成共享库
#     add_library(${LIBRARY_NAME} SHARED ${SOURCES})
#     target_link_libraries(${LIBRARY_NAME} glass)
# endif()

# if (BUILD_TEST)
#     add_executable(test_alaya test_alaya.cpp)
# endif()

# add_executable(hybrid_search baseline.cpp)
# target_link_libraries(hybrid_search glass)