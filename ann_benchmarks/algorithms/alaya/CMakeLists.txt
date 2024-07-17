cmake_minimum_required (VERSION 3.16)
project (AlayaDB)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-stdlib=libc++) 
  add_link_options(-stdlib=libc++)
  link_libraries(c++)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g -O0)
else()
  add_compile_options(-O3)
endif()


include(FetchContent)

FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable(pybind11)
# set(PYBIND11_FINDPYTHON ON)

find_package(OpenMP REQUIRED)

# add_subdirectory(pybind11)
# find_package(pybind11 CONFIG REQUIRED)

add_compile_options(-march=native)

add_subdirectory(pyglass)

# find_package(pybind11 REQUIRED)

add_library(AlayaDB SHARED bind.cpp)
target_link_libraries(AlayaDB glass)

pybind11_add_module(alaya pybinding.cpp)
target_link_libraries(alaya PRIVATE glass OpenMP::OpenMP_CXX)

add_executable(main main/main.cpp)
target_link_libraries(main glass OpenMP::OpenMP_CXX)


# if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -pthread -g -Wall -pthread -mavx2 -mfma")
# else()
#     # AVX2
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -march=native -mtune=native -O3 -mavx2 -pthread -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread -std=c++17 -fPIC -mavx -mavx2 -mfma")
# endif()

# find_package(OpenMP REQUIRED)
# if (OPENMP_FOUND)
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()


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