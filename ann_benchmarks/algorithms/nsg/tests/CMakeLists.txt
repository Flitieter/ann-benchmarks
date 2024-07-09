set(CMAKE_CXX_STANDARD 17)

add_executable(test_nsg_index test_nsg_index.cpp)
target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nndescent test_nndescent.cpp)
target_link_libraries(test_nndescent ${PROJECT_NAME})

add_executable(test_nndescent_refine test_nndescent_refine.cpp)
target_link_libraries(test_nndescent_refine ${PROJECT_NAME})


add_executable(nsg_index nsg_index.cpp)
target_link_libraries(nsg_index ${PROJECT_NAME})


add_executable(grouth_test grouth_test.cpp)
target_link_libraries(grouth_test ${PROJECT_NAME})
# add_executable(baseline baseline.cpp)
# target_link_libraries(baseline ${PROJECT_NAME})


