#!/bin/bash

# 检查命令是否成功执行的函数
check_command() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed to execute."
        exit 1
    fi
}

# 运行第一个程序
echo "Running build_nndescent..."
./build/tests/test_nndescent ~/dataset/sift/sift_base.fvecs sift_200nn.graph 200 200 10 10 100
check_command "test_nndescent"

echo "Running refine_nndescent..."
./build/tests/test_nndescent_refine ~/dataset/sift/sift_base.fvecs sift_200nn.graph sift_200nn.graph 200 200 10 10 100
check_command "refine_nndescent"

# 运行第二个程序
echo "Running build_nsg_index..."
./build/tests/test_nsg_index ~/dataset/sift/sift_base.fvecs sift_200nn.graph 40 50 500 sift.nsg
check_command "test_nsg_index"

# 运行第三个程序
echo "Running test_nsg_optimized_search..."
./build/tests/test_nsg_optimized_search ~/dataset/sift/sift_base.fvecs ~/dataset/sift/sift_query.fvecs sift.nsg 100 100 result.nsg ~/dataset/sift/sift_groundtruth.ivecs
check_command "test_nsg_search"

echo "All commands executed successfully."
