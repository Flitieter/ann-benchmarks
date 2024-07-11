rm -rf build
mkdir build && cd build
cmake -D BUILD_TEST=ON ..
make -j${nproc}
cd ..
./build/test_alaya \
    ~/dataset/sift/sift_base.fvecs \
    ~/dataset/sift/sift_query.fvecs \
    ~/dataset/sift/sift_groundtruth.ivecs \
    ./test_result.txt