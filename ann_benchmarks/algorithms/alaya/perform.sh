start=50
end=300
step=25 # 更改为更合理的步长

>result_MERGE_NSG.txt
# 开始循环
#for i in $(seq $start $step $end)
#do
# rerank=15
# if [ "$i" -gt 75 ]; then
#  rerank=20
# fi
# if [ "$i" -gt 200 ]; then
#rerank=30
# fi
#
# # 执行 nsg 命令
# echo "---------------------------"
# echo "Running with NSG"
# build/nsg /data/cohere-768-euclidean 32 200 10 $i $rerank
#
#done

for i in $(seq $start $step $end)
do
 rerank=15
 if [ "$i" -gt 75 ]; then
 rerank=20
 fi
 if [ "$i" -gt 200 ]; then
  rerank=30
 fi

 # 执行 main 命令
 echo "---------------------------"
 echo "Running with MERGE_NSG"
 /home/ann-benchmark/yitao/ann-bench/ann-benchmarks/ann_benchmarks/algorithms/alaya/cmake-build-release/main /data/cohere-768-euclidean/cohere-768-euclidean_base.fvecs /data/cohere-768-euclidean/cohere-768-euclidean_query.fvecs /data/cohere-768-euclidean/cohere-768-euclidean_gti.ivecs 10 $i $rerank
done