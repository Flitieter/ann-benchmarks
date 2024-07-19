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
  add_compile_options(-Ofast -g)
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

add_compile_options(-march=native -O3 -g)

add_subdirectory(pyglass)

# find_package(pybind11 REQUIRED)

add_library(AlayaDB SHARED bind.cpp)
target_link_libraries(AlayaDB glass)

pybind11_add_module(alaya pybinding.cpp)
target_link_libraries(alaya PRIVATE glass OpenMP::OpenMP_CXX)

add_executable(read_file main/read_file.cpp)

add_executable(main main/main.cpp)
target_link_libraries(main glass fmt::fmt OpenMP::OpenMP_CXX)