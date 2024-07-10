cmake_minimum_required (VERSION 3.17)
project(glass LANGUAGES CXX)


add_library(glass INTERFACE)
target_link_libraries(glass
    INTERFACE
)
target_include_directories(glass INTERFACE .) 

set(CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# SET(CMAKE_CXX_FLAGS  "-Wall -Wextra -O0 -lrt -std=c++20 -march=native -fpic -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

SET(CMAKE_CXX_FLAGS  "-Wextra -O3 -lrt -std=c++17 -march=native -fPIC -fopenmp -ftree-vectorize -fno-exceptions -fno-rtti" )

add_executable(main main/main.cpp)
#target_link_libraries(main glass fmt::fmt)

# add_executable(main examples/main.cc)
# target_link_libraries(main glass)

add_library(AlayaDB SHARED bind.cpp)
target_link_libraries(AlayaDB glass)
