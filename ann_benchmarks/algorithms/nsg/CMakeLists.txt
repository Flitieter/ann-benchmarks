cmake_minimum_required(VERSION 2.8)

project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fPIC")
add_definitions (-std=c++17 -O3 -lboost -march=native -Wall -DINFO)

add_subdirectory(src)
add_subdirectory(tests)

set(LIBRARY_NAME hybrid_search)

# 添加源文件
set(SOURCES tests/baseline.cpp)

# 编译生成共享库
add_library(${LIBRARY_NAME} SHARED ${SOURCES})
target_link_libraries(hybrid_search ${PROJECT_NAME})
